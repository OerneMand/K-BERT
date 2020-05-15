import os
import csv
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from distutils.spawn import find_executable


from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import BertTokenizer, AdamW
from models.berts import BertForSequenceClassification, BertForQuestionAnswering
#from models.evaluation import accuracy
from torch.utils.data import DataLoader, IterableDataset
from preprocessing.config import set_seed
from preprocessing.knowledge_injector import KnowledgeInjector
from preprocessing.dataset import KBertDataset, load_data_from_path

def accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--knowledge_base", type = str, default = "probase",
                        help = "The name of the knowledge base, must be defined in read_kgs")

    parser.add_argument("--no_kg_augment", action="store_true", 
                        help = "Disable the injection of knowledge from a knowledge graph")

    parser.add_argument("--num_processes", type=int, default = 1,
                        help = "Number of workers to help with data loading and data preparation")

    parser.add_argument("--dataset_name", type = str, 
                        help = "The directory with the files, needs to defined in preprocessing.dataset and logic for loading in the data")
    parser.add_argument("--minified", action = "store_true", 
                        help = "Use a small version of the dataset, for testing purposes only")
    parser.add_argument("--plot_train_val", action = "store_true", 
                        help = "Whether to plot the training and validation error")

    parser.add_argument("--pretrained_lm", type = str, default = "bert-base-uncased",
                        help = "A pretrained model name compatible with transformers library")

    parser.add_argument("--sequence_length", type=int, default=128,
                        help="Max sequence length of model.")
    parser.add_argument("--num_labels", type=int, default = 4,
                        help="Number of labels, for classication tasks only.")
    parser.add_argument("--batch_size", type = int, default = 64,
                        help = "The amount of training instances the model should learn from per step")
    parser.add_argument("--seed", type=int, default = None,
                        help="Random seed.")    
    parser.add_argument("--epochs", type=int, default = 2,
                        help="Number of epochs model should train for.")
    parser.add_argument("--learning_rate", type=float, default = 1.3e-5,
                        help="Random seed.")
    parser.add_argument("--epsilon", type=float, default = 1e-06,
                        help="Random seed.")
    parser.add_argument("--run_name", type=str, 
                        help="The name of the job in SLURM.")
    parser.add_argument("--sample", type=float, default = None,
                        help="Sample a subset of the knowledge graph.")
    parser.add_argument("--turn_off_training", action = "store_true",
                        help="Dont open the weights to be trained on, and just run the model as normal.")


    args = parser.parse_args()
    print(f"Using dataset {args.dataset_name}")
    print(f"Using {args.knowledge_base} knowledge graph")
    print(f"Empty if no kg augment ({args.no_kg_augment})")

    if args.seed:
       set_seed(args.seed)

    if args.dataset_name in ["ag_news", "yahoo"]:
        task = "classification"
    elif args.dataset_name in ["squad", "complex_web_questions", "jeopardy", "qasc", "arc"]:
        task = "qa"
    elif args.dataset_name in ["ner"]:
        task = "ner"
    else:
        raise NotImplementedError("dataset_name is not recognized with an implemented method")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    if args.no_kg_augment:
        print("Not injecting knowledge into sentences")
        kg = None
    else: 
        kg = KnowledgeInjector(kg_graph_or_name = args.knowledge_base, tokenizer = tokenizer.tokenize, sample=args.sample, task = task, max_length=args.sequence_length)

    train_ds, val_ds, test_ds = load_data_from_path(args.dataset_name, kg, tokenizer, args.sequence_length, 
                                                    args.batch_size, minified = args.minified, no_kg_augment=args.no_kg_augment, task = task)

    def sparse_graph_batch_collate(batch):
        b_input_ids, b_segment_emb, b_soft_pos_emb, b_vm_values, b_vm_indices, b_attention_mask, b_labels = batch
        if b_vm_indices and b_vm_values:
            b_visibility_matrix = torch.stack([torch.sparse.LongTensor(vm_index, vm_value, (args.sequence_length,args.sequence_length)) 
                                                for vm_index, vm_value in zip(b_vm_indices, b_vm_values)])
        else:
            b_visibility_matrix = None
        return b_input_ids, b_segment_emb, b_soft_pos_emb, b_visibility_matrix, b_attention_mask, b_labels

    train_loader = DataLoader(train_ds, batch_size = None, num_workers = 0, collate_fn = sparse_graph_batch_collate)
    val_loader = DataLoader(val_ds, batch_size = None, num_workers = 0, collate_fn = sparse_graph_batch_collate)
    test_loader = DataLoader(test_ds, batch_size = None, num_workers = 0, collate_fn = sparse_graph_batch_collate)

    # Load pretrained model
    if task == "classification":
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = args.num_labels)
    elif task == "qa":
        model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")


    # Utilize GPU and CUDA code for better performance if available
    cuda = torch.cuda.is_available()
    model.cuda() if cuda else model.cpu() 

    if not os.path.exists(f"outputs/job.{args.run_name}"):
        os.makedirs(f"outputs/job.{args.run_name}")

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters,
                        lr=args.learning_rate, eps = args.epsilon, correct_bias=False)

    # BERT authors recommended 2<EPOCHS<4
    EPOCHS = args.epochs
    # For storing our loss and accuracy for plotting
    train_loss, train_accuracy = 0, 0
    train_accuracy_list = []
    train_loss_set = [] 
    
    if args.plot_train_val:
        eval_accuracy_list = []
        train_loader = zip(train_loader, val_loader)

    # Set our model to training mode (as opposed to evaluation mode)
    if not args.turn_off_training:
        model.train()
    for epoch in range(EPOCHS):
        # Training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_ds.reload_data()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        # Train the data for one epoch
        for step, batch in tqdm(enumerate(train_loader), desc = "Batches", position = 0, total = train_ds.length //  train_ds.batch_size):
            if args.plot_train_val:
                val_batch = batch[1]
                batch = batch[0]
            
            # Add batch to GPU
            batch = tuple(t.to(device) if t != None else None for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_segment_emb, b_soft_pos_emb, b_visibility_matrix, b_attention_mask, b_labels = batch
            if task == "classification":
                loss, logits = model(
                    b_input_ids, 
                    position_ids = b_soft_pos_emb, 
                    token_type_ids = b_segment_emb, 
                    attention_mask = b_attention_mask, 
                    encoder_attention_mask = b_visibility_matrix, 
                    labels = b_labels
                )
            elif task == "qa":
                loss, start_scores, end_scores = model(
                    b_input_ids, 
                    position_ids = b_soft_pos_emb, 
                    token_type_ids = b_segment_emb, 
                    attention_mask = b_attention_mask, 
                    encoder_attention_mask = b_visibility_matrix, 
                    start_positions = b_labels[:,0],
                    end_positions = b_labels[:,1]
                )

            optimizer.zero_grad() # Gradients are by default accumulated, useful for RNN but not BERT-man
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
            if task == "classification":
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                train_accuracy_list.append(accuracy(logits, label_ids))
            elif task == "qa":
                start = torch.argmax(start_scores, dim=1).detach().cpu().numpy()
                end = torch.argmax(end_scores, dim=1).detach().cpu().numpy()
                real_start = b_labels[:,0].detach().cpu().numpy()
                real_end = b_labels[:,1].detach().cpu().numpy()

                micro_st = f1_score(real_start, start, average = "micro")
                micro_en = f1_score(real_end, end, average = "micro")
                macro_st = f1_score(real_start, start, average = "macro")
                macro_en = f1_score(real_end, end, average = "macro")
                weighted_st = f1_score(real_start, start, average = "weighted")
                weighted_en = f1_score(real_end, end, average = "weighted")
                tmp_train_accuracy = [micro_st, micro_en, macro_st, macro_en, weighted_st, weighted_en]

            if args.plot_train_val:
                # Add batch to GPU
                val_batch = tuple(t.to(device) if t != None else None for t in val_batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_segment_emb, b_soft_pos_emb, b_visibility_matrix, b_attention_mask, b_labels = val_batch

                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    if task == "classification":
                        _, logits = model(
                            b_input_ids, 
                            position_ids = b_soft_pos_emb, 
                            token_type_ids = b_segment_emb, 
                            attention_mask = b_attention_mask, 
                            encoder_attention_mask = b_visibility_matrix, 
                            labels = b_labels
                        )
                        # Move logits and labels to CPU
                        logits = logits.detach().cpu().numpy()
                        label_ids = b_labels.to('cpu').numpy()

                        eval_accuracy_list.append(accuracy(logits, label_ids))
                        eval_accuracy += tmp_eval_accuracy
                        
                    elif task == "qa":
                        _, start_scores, end_scores = model(
                            b_input_ids, 
                            position_ids = b_soft_pos_emb, 
                            token_type_ids = b_segment_emb, 
                            attention_mask = b_attention_mask, 
                            encoder_attention_mask = b_visibility_matrix, 
                            start_positions = b_labels[:,0],
                            end_positions = b_labels[:,1]
                        )
                        start = torch.argmax(start_scores, dim=1).detach().cpu().numpy()
                        end = torch.argmax(end_scores, dim=1).detach().cpu().numpy()
                        real_start = b_labels[:,0].detach().cpu().numpy()
                        real_end = b_labels[:,1].detach().cpu().numpy()

                        micro_st = f1_score(real_start, start, average = "micro")
                        micro_en = f1_score(real_end, end, average = "micro")
                        macro_st = f1_score(real_start, start, average = "macro")
                        macro_en = f1_score(real_end, end, average = "macro")
                        weighted_st = f1_score(real_start, start, average = "weighted")
                        weighted_en = f1_score(real_end, end, average = "weighted")
                        tmp_eval_accuracy = [micro_st, micro_en, macro_st, macro_en, weighted_st, weighted_en]
                        eval_accuracy_list.append(tmp_eval_accuracy)


                    val_ds.reload_data()
            
        print(f" Epoch {epoch + 1} done.\nTrain loss: {tr_loss/nb_tr_steps}\n")

    if args.plot_train_val:
        # We'll just use the easy way of creating plots, could do the object approach but since its all sequential and neatly package its fine
        plt.plot(train_accuracy_list, linestyle="-", label="Training") 
        plt.plot(eval_accuracy_list, linestyle="-", label="Evaluation")
        plt.legend()
        plt.xlabel("Batch iterations")
        if task == "classification":
            plt.ylabel("Accuracy score")
        elif task == "qa":
            plt.ylabel("F1 score")
        plt.savefig(f"outputs/job.{args.run_name}/training_error.png")
        plt.clf()

    plt.plot(train_loss_set, label = "Training")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(f"outputs/job.{args.run_name}/loss.png")
    plt.clf()

    # Validation
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    val_ds.reload_data()

    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    eval_accuracy_list = []
    nb_eval_steps, nb_eval_examples = 0, 0

    f = open(f"outputs/job.{args.run_name}/val_predictions.csv", "w")
    predwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    if task == "classification":
        predwriter.writerow([f"class_{i+1}" for i in range(args.num_labels)] + ["label"] )
    elif task == "qa":
        predwriter.writerow(["pred_start", "pred_end", "real_start", "real_end"])

    # Evaluate data for one epoch
    for batch in tqdm(val_loader, desc = "Batches", position = 0, total = val_ds.length //  val_ds.batch_size):
        # Add batch to GPU
        batch = tuple(t.to(device) if t != None else None for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_segment_emb, b_soft_pos_emb, b_visibility_matrix, b_attention_mask, b_labels = batch
        
        # Dont compute gradients, saves memory and speeds up testing
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            if task == "classification":
                _, logits = model(
                    b_input_ids, 
                    position_ids = b_soft_pos_emb, 
                    token_type_ids = b_segment_emb, 
                    attention_mask = b_attention_mask, 
                    encoder_attention_mask = b_visibility_matrix, 
                    labels = b_labels
                )
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                predwriter.writerows(zip(*logits.T, label_ids.flatten()))
                tmp_eval_accuracy = accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                
            elif task == "qa":
                _, start_scores, end_scores = model(
                    b_input_ids, 
                    position_ids = b_soft_pos_emb, 
                    token_type_ids = b_segment_emb, 
                    attention_mask = b_attention_mask, 
                    encoder_attention_mask = b_visibility_matrix, 
                    start_positions = b_labels[:,0],
                    end_positions = b_labels[:,1]
                )
                start = torch.argmax(start_scores, dim=1).detach().cpu().numpy()
                end = torch.argmax(end_scores, dim=1).detach().cpu().numpy()
                real_start = b_labels[:,0].detach().cpu().numpy()
                real_end = b_labels[:,1].detach().cpu().numpy()

                predwriter.writerows(zip(start, end, real_start, real_end))

                micro_st = f1_score(real_start, start, average = "micro")
                micro_en = f1_score(real_end, end, average = "micro")
                macro_st = f1_score(real_start, start, average = "macro")
                macro_en = f1_score(real_end, end, average = "macro")
                weighted_st = f1_score(real_start, start, average = "weighted")
                weighted_en = f1_score(real_end, end, average = "weighted")
                tmp_eval_accuracy = [micro_st, micro_en, macro_st, macro_en, weighted_st, weighted_en]

            eval_accuracy_list.append(tmp_eval_accuracy)
            nb_eval_steps += 1
     
    f.close()

    if task == "classification":
        print(f"Validation Accuracy: {eval_accuracy/nb_eval_steps}")
    elif task == "qa":
        print('Micro: start: {} end: {} Macro: start: {} end: {} Weighted: start: {} end: {}'.format(*np.mean(eval_accuracy_list, 0)))

    # Tracking variables 
    test_loss, test_accuracy = 0, 0
    test_accuracy_list = []
    nb_test_steps, nb_test_examples = 0, 0

    f = open(f"outputs/job.{args.run_name}/test_predictions.csv", "w")
    predwriter = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    predwriter.writerow([f"class_{i+1}" for i in range(args.num_labels)] + ["label"] )
    # Evaluate data for one epoch
    for batch in tqdm(test_loader, desc = "Batches", position = 0, total = test_ds.length //  test_ds.batch_size):
        # Add batch to GPU
        batch = tuple(t.to(device) if t != None else None for t in batch)#sparse_graph_batch_collate(batch))
        # Unpack the inputs from our dataloader
        b_input_ids, b_segment_emb, b_soft_pos_emb, b_visibility_matrix, b_attention_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            if task == "classification":
                _, logits = model(
                    b_input_ids, 
                    position_ids = b_soft_pos_emb, 
                    token_type_ids = b_segment_emb, 
                    attention_mask = b_attention_mask, 
                    encoder_attention_mask = b_visibility_matrix, 
                    labels = b_labels
                )
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                predwriter.writerows(zip(*logits.T, label_ids.flatten()))
                tmp_test_accuracy = accuracy(logits, label_ids)
                test_accuracy += tmp_test_accuracy
                
            elif task == "qa":
                _, start_scores, end_scores = model(
                    b_input_ids, 
                    position_ids = b_soft_pos_emb, 
                    token_type_ids = b_segment_emb, 
                    attention_mask = b_attention_mask, 
                    encoder_attention_mask = b_visibility_matrix, 
                    start_positions = b_labels[:,0],
                    end_positions = b_labels[:,1]
                )
                start = torch.argmax(start_scores, dim=1).detach().cpu().numpy()
                end = torch.argmax(end_scores, dim=1).detach().cpu().numpy()
                real_start = b_labels[:,0].detach().cpu().numpy()
                real_end = b_labels[:,1].detach().cpu().numpy()

                predwriter.writerows(zip(start, end, real_start, real_end))

                micro_st = f1_score(real_start, start, average = "micro")
                micro_en = f1_score(real_end, end, average = "micro")
                macro_st = f1_score(real_start, start, average = "macro")
                macro_en = f1_score(real_end, end, average = "macro")
                weighted_st = f1_score(real_start, start, average = "weighted")
                weighted_en = f1_score(real_end, end, average = "weighted")
                tmp_test_accuracy = [micro_st, micro_en, macro_st, macro_en, weighted_st, weighted_en]
            
            test_accuracy_list.append(tmp_test_accuracy)
            nb_test_steps += 1
    f.close()

    
    if task == "classification":
        print(f"Validation Accuracy: {test_accuracy/nb_test_steps}")
    elif task == "qa":
        print('Micro: start: {} end: {} Macro: start: {} end: {} Weighted: start: {} end: {}'.format(*np.mean(test_accuracy_list, 0)))

    if not args.no_kg_augment:
        print(f"Used {kg.entity_insertions} relation insertions from KG")
        print(f"Used {len(kg.unique_insert) / len(kg) * 100}% of relations in KG to inject knowledge")