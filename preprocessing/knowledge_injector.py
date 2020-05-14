import os
import re
import gc
import csv
import sys
import time
import torch
import argparse
import numpy as np
import pandas as pd
from preprocessing.config import KG_PATHS
from preprocessing.read_kgs import read_kg

from re import findall
from string import capwords
from tqdm import trange, tqdm
from nltk import ngrams, pos_tag
from itertools import takewhile
from multiprocessing import Pool #.dummy for threads
from scipy.sparse import lil_matrix as sparse_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import BertModel, BertForSequenceClassification, BertTokenizer, AdamW
#from transformers.modeling_bert import *
from models.berts import BertKnowledgeEncoder, BertEncoder, BertForSequenceClassification

class KnowledgeInjector():
    def __init__(
        self, 
        kg_graph_or_name, 
        tokenizer,
        task,
        look_back = 6,
        max_relations = 4, 
        max_length = 128, 
        regex_split = "[A-z][a-z]+", 
        strip_punc = r"[!\"#&()*+,-.:;<=>?@\\^_`{|}~]",
        sample = False
    ):
        self.LOOK_BACK = look_back
        self.MAX_RELATIONS = max_relations
        self.MAX_LEN = max_length
        self.tokenizer = tokenizer
        self.task = task
        self.strip_punc = strip_punc
        self.entity_insertions = 0
        self.unique_insert = set()
        if type(kg_graph_or_name) == str:
            self.kg = read_kg(KG_PATHS[kg_graph_or_name], tokenizer, sample = sample)
        else:
            self.kg = kg_graph_or_name
        print(f"Knowledge graph has {len(self.kg)} relations")

    def _kbpedia_read(self, kg_path):
        kg = {}
        with open(kg_path, "r") as kg_csv:
            try:
                for line in csv.reader(kg_csv):
                    if line == ["subject", "relation", "object"]:
                        continue
                    subj, obj, rel = [" ".join(findall("[A-z][a-z]*", string = clause)).lower().split() for clause in line]
                    subj = " ".join(subj)
                    body = obj + rel
                    
                    if subj in kg:
                        kg[subj].append(body)
                    else:
                        kg[subj] = list(body)
            except:
                print(f"Bad encoding at line {line}")

        return kg
    
    def inject_knowledge(self, instance_and_corpus):
        # This implementation borrows conceptually from K-BERT (Liu et al.) implementation, but is more efficient and adapted to English
        token_emb_corpus = []
        segment_mask_corpus = []
        soft_ids_corpus = []
        visibility_matrix_corpus = []
        vm_values_corpus = []
        vm_indices_corpus = []
        corpus = instance_and_corpus

        for injection_iteration, text in enumerate(corpus):
            sentence_tree = []
            soft_ids = []
            hard_ids = []
            soft_tree = []
            hard_tree = []
            history = ["[START]"] * self.LOOK_BACK
            token_emb = []
            segment_mask = []
            hard_position = 0
            turn_off_injection = False
            for soft_position, token in enumerate(text):
                if token in ["a", "b", "c", "d"]:
                    #turn_off_injection = True
                    relations = []
                #if self.task == "qa" and token == "[SEP]":
                    #turn_off_injection = True
                else:#if not turn_off_injection:
                    relations = [entity for combination in range(self.LOOK_BACK + 1) for entity in self.kg.get(" ".join(history[combination:self.LOOK_BACK] + [token]), [])][:self.MAX_RELATIONS]
                # elif turn_off_injection:
                #     relations = []
                sentence_tree.append((token, relations))
                relations_hard_pos = []
                relations_soft_pos = []
                previous_hard = hard_position

                token_emb.append(token)
                soft_ids.append(soft_position)
                segment_mask.append(0)

                soft_position += 1
                hard_ids.append(hard_position)

                for j, relation in enumerate(relations):
                    self.unique_insert.add(" ".join(relation))
                    self.entity_insertions += 1
                    relations_soft_pos.append([soft_position + offset for offset in range(1, len(relation)+1)])
                    relations_hard_pos.append([hard_position + offset for offset in range(1, len(relation)+1)])
                    token_emb += relation
                    segment_mask += [1] * len(relation)
                    soft_ids += range(soft_position, len(relation) + soft_position)
                    hard_position += len(relation)

                hard_tree.append(([previous_hard], relations_hard_pos))
                hard_position += 1
                soft_tree.append(([soft_position], relations_soft_pos))
                history.append(token + " ")
                if len(history) > self.LOOK_BACK:
                    history.pop(0)
                
            # Calculate visible matrix
            sentence_len = len(soft_ids)
            visibility_matrix = sparse_matrix((sentence_len, sentence_len)) #np.zeros((sentence_len, sentence_len))#
            for token_idx, relation_ids in hard_tree:
                for idx in token_idx:
                    visible_abs_idx = hard_ids + [idx for ent in relation_ids for idx in ent]
                    visibility_matrix[idx, visible_abs_idx] = 1
                for ent in relation_ids:
                    for idx in ent:
                        visible_abs_idx = ent + token_idx
                        visibility_matrix[idx, visible_abs_idx] = 1

            if sentence_len < self.MAX_LEN:
                pad_num = self.MAX_LEN - sentence_len
                segment_mask += [0] * pad_num
                soft_ids += [self.MAX_LEN - 1] * pad_num
                #visibility_matrix = np.pad(visibility_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                segment_mask = segment_mask[:self.MAX_LEN]
                soft_ids = soft_ids[:self.MAX_LEN]
                visibility_matrix = visibility_matrix[:self.MAX_LEN, :self.MAX_LEN]

            #attention_mask_corpus.append([1 if token is not "[PAD]" else 0 for token in token_emb])
            #snapshot = tracemalloc.take_snapshot()
            #display_top(snapshot)
            visibility_matrix = visibility_matrix.tocoo()
            values = visibility_matrix.data
            indices = np.vstack((visibility_matrix.row, visibility_matrix.col))

            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(values)

            #visibility_matrix = torch.sparse.LongTensor(i, v, (self.MAX_LEN, self.MAX_LEN))
            
            #yield token_emb, segment_mask, soft_ids, visibility_matrix
            token_emb_corpus.append(token_emb)
            segment_mask_corpus.append(segment_mask)
            soft_ids_corpus.append(soft_ids)
            vm_indices_corpus.append(indices)
            vm_values_corpus.append(values)
            #visibility_matrix_corpus.append(visibility_matrix)

        return token_emb_corpus, segment_mask_corpus, soft_ids_corpus, vm_values_corpus, vm_indices_corpus
        
    def __len__(self):
        return len(self.kg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--kg_name", type = str, default = "kbpedia",
                        help = "Path of knowledge to be used")

    parser.add_argument("--no_kg_augment", action="store_true", 
                        help = "Disable the injection of knowledge from a knowledge graph")

    parser.add_argument("--num_processes", type=int, default = 1,
                        help = "Number of workers to help with data loading onto GPU and data preparation")
    
    parser.add_argument("--sequence_length", type=int, default=128,
                        help="Max sequence length of model.")

    parser.add_argument("--seed", type=int, default=1337, 
                        help="Random seed.")

    parser.add_argument("--train_path", type = str)
    parser.add_argument("--test_path", type = str)
    parser.add_argument("--knowledge_graph", type = str)


    # parser.add_argument("--num_labels", type=int, default=4,
    #                     help="Number of labels, for classication tasks only.")

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    kg = KnowledgeInjector(kg_graph_or_name = args.knowledge_graph, tokenizer = tokenizer.tokenize)

    ########################################################################
    ##############################    TRAIN   ##############################
    ########################################################################

    train = pd.read_csv(args.train_path, header = None, names = ["label", "headline", "text"])
    train = train.sample(frac=.1, random_state = args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    labels = train.label.values
    if 0 not in labels:
        labels -= 1

    cuda = torch.cuda.is_available()

    if not args.no_kg_augment:
        if args.num_processes > 1:
            train_split = np.array_split(train.text.values, args.num_processes)
            train_split = [(key, value) for key, value in zip(range(len(train_split)), train_split)]
            print(f"Starting knowledge injection with {args.num_processes} processes:")
            with Pool(processes=args.num_processes) as pool:
                pooled = pool.map(kg.inject_knowledge, train_split) #list(tqdm(pool.imap(kg.inject_knowledge, train_split), desc = "Processes done", total=args.num_processes))
            token_emb, segment_emb, soft_pos_emb, visibility_matrix = np.reshape(np.moveaxis(pooled, 1, 0).flatten(), newshape = (4, len(train)))
            print(f"Used {kg.entity_insertions} relation insertions from KG")
            print(f"Used {len(kg.unique_insert) / len(kg) * 100}% of relations in KG to inject knowledge")
            segment_emb = segment_emb.tolist()
            soft_pos_emb = soft_pos_emb.tolist()
            visibility_matrix = visibility_matrix.tolist()
        else:
            token_emb, segment_emb, soft_pos_emb, visibility_matrix = kg.inject_knowledge([1, train.text.values])

        tokenized_emb = tokenizer.batch_encode_plus(
            token_emb, 
            is_pretokenized = False, 
            return_special_token_masks = True, 
            max_length = args.sequence_length, 
            pad_to_max_length = True, 
            return_tensors = "pt", 
            return_token_type_ids = True
        )

        input_ids = tokenized_emb["input_ids"]
        attention_mask = tokenized_emb["attention_mask"]
        segment_emb = torch.LongTensor(segment_emb)
        soft_pos_emb = torch.LongTensor(soft_pos_emb)
        visibility_matrix = torch.LongTensor(visibility_matrix)
        labels = torch.LongTensor(labels)

        dataset = TensorDataset(input_ids, segment_emb, soft_pos_emb, visibility_matrix, attention_mask, labels)
    else:
        tokenized_emb = tokenizer.batch_encode_plus(
            train.text.values,
            is_pretokenized = False, 
            return_special_token_masks = True, 
            max_length = args.sequence_length, 
            pad_to_max_length = True, 
            return_tensors = "pt", 
            return_token_type_ids = True
        )
        labels = torch.LongTensor(labels)
        dataset = TensorDataset(tokenized_emb["input_ids"], tokenized_emb["attention_mask"], labels)

    train_dataset, validation_dataset = train_test_split(
        dataset, random_state=1337, test_size=0.2
    )

    ########################################################################
    ##############################    TEST   ###############################
    ########################################################################


    test = pd.read_csv(args.test_path, header = None, names = ["label", "headline", "text"])
    test_labels = test.label.values
    if 0 not in test_labels:
        test_labels -= 1

    if not args.no_kg_augment:
        if args.num_processes > 1:
            test_split = np.array_split(test.text.values, args.num_processes)
            test_split = [(key, value) for key, value in zip(range(len(test_split)), test_split)]
            print(f"Starting knowledge injection with {args.num_processes} processes:")
            with Pool(processes=args.num_processes) as pool:
                pooled = pool.map(kg.inject_knowledge, test_split) #list(tqdm(pool.imap(kg.inject_knowledge, test_split), desc = "Processes done", total=args.num_processes))
            token_emb, segment_emb, soft_pos_emb, visibility_matrix = np.reshape(np.moveaxis(pooled, 1, 0).flatten(), newshape = (4, len(test)))
            print(f"Used {kg.entity_insertions / len(kg) * 100}% of relations in KG to inject knowledge")
            segment_emb = segment_emb.tolist()
            soft_pos_emb = soft_pos_emb.tolist()
            visibility_matrix = visibility_matrix.tolist()
        else:
            token_emb, segment_emb, soft_pos_emb, visibility_matrix = kg.inject_knowledge(test.text.values)

        tokenized_emb = tokenizer.batch_encode_plus( # gør alt tokenizereriet for en, den kan også returne token type id som man bruge QA
            token_emb,  #brug den!!
            is_pretokenized = False, 
            return_special_token_masks = True, 
            max_length = args.sequence_length, 
            pad_to_max_length = True, 
            return_tensors = "pt", 
            return_token_type_ids = True
        )

        input_ids = tokenized_emb["input_ids"]
        attention_mask = tokenized_emb["attention_mask"]
        segment_emb = torch.LongTensor(segment_emb)
        soft_pos_emb = torch.LongTensor(soft_pos_emb)
        visibility_matrix = torch.LongTensor(visibility_matrix)
        test_labels = torch.LongTensor(test_labels)
        
        test_dataset = TensorDataset(input_ids, segment_emb, soft_pos_emb, visibility_matrix, attention_mask, test_labels)
    else:
        tokenized_emb = tokenizer.batch_encode_plus(
            test.text.values,
            is_pretokenized = False, 
            return_special_token_masks = True, 
            max_length = args.sequence_length, 
            pad_to_max_length = True, 
            return_tensors = "pt", 
            return_token_type_ids = True
        )
        test_labels = torch.LongTensor(test_labels)
        test_dataset = TensorDataset(tokenized_emb["input_ids"], tokenized_emb["attention_mask"], test_labels)


    BATCH_SIZE = 32

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                        shuffle = True, num_workers = args.num_processes)
    val_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE,
                        shuffle = True, num_workers = args.num_processes)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 4)
    model.cuda() if cuda else model.cpu()
    # Store our loss and accuracy for plotting
    train_loss_set = []

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    del kg

    ########################################################################
    #########################    training loop   ###########################
    ########################################################################
    
    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=2e-5, correct_bias=False)

    # Number of training EPOCHS (authors recommend between 2 and 4)
    EPOCHS = 4

    for _ in trange(EPOCHS, desc="Epoch"):
    
        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Set to training mode")
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # Train the data for one epoch
        for step, batch in enumerate(train_loader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            if not args.no_kg_augment:
                b_input_ids, b_segment_emb, b_soft_pos_emb, b_visibility_matrix, b_attention_mask, b_labels = batch
            else:
                b_input_ids, b_attention_mask, b_labels = batch
                b_visibility_matrix = None
                b_soft_pos_emb = None
                b_segment_emb = None
            loss, logits = model(
                b_input_ids, 
                position_ids = b_soft_pos_emb, 
                token_type_ids = b_segment_emb, 
                attention_mask = b_attention_mask, 
                encoder_attention_mask = b_visibility_matrix, 
                labels=b_labels
            )

            optimizer.zero_grad() # Gradients are accumulated, useful for RNN but not BERT-man
            # Forward pass
            train_loss_set.append(loss.item()) #item when training on cuda device
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            
            
            # Update tracking variables
            if cuda:
                tr_loss += loss.item()
            else:
                tr_loss += loss
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            gc.collect()

        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        
        
    # Validation
    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in val_loader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        if not args.no_kg_augment:
                b_input_ids, b_segment_emb, b_soft_pos_emb, b_visibility_matrix, b_attention_mask, b_labels = batch
        else:
            b_input_ids, b_attention_mask, b_labels = batch
            b_visibility_matrix = None
            b_soft_pos_emb = None
            b_segment_emb = None
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(
                b_input_ids, 
                position_ids = b_soft_pos_emb, 
                token_type_ids = b_segment_emb, 
                attention_mask=b_attention_mask, 
                encoder_attention_mask = b_visibility_matrix
            )[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            with open("outputs/val_predictions.csv", "w") as file:
                predwriter = csv.writer(file)
                predwriter.writerow("pred, label\n")
                predwriter.writerows(zip(np.argmax(logits, axis=1).flatten(), label_ids.flatten()))
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
            gc.collect()

    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                        shuffle = True, num_workers = args.num_processes)



    # Tracking variables 
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0

    # Evaluate data for one epoch
    for batch in test_loader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        if not args.no_kg_augment:
                b_input_ids, b_segment_emb, b_soft_pos_emb, b_visibility_matrix, b_attention_mask, b_labels = batch
        else:
            b_input_ids, b_attention_mask, b_labels = batch
            b_visibility_matrix = None
            b_soft_pos_emb = None
            b_segment_emb = None
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(
                b_input_ids, 
                position_ids = b_soft_pos_emb, 
                token_type_ids = b_segment_emb, 
                attention_mask=b_attention_mask, 
                encoder_attention_mask = b_visibility_matrix
            )[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            with open("outputs/test_predictions.csv", "w") as file:
                predwriter = csv.writer(file)
                predwriter.writerow("pred, label\n")
                predwriter.writerows(zip(np.argmax(logits, axis=1).flatten(), label_ids.flatten()))
            tmp_test_accuracy = flat_accuracy(logits, label_ids)
            
            test_accuracy += tmp_test_accuracy
            nb_test_steps += 1
            gc.collect()

    print("Test Accuracy: {}".format(test_accuracy/nb_test_steps))