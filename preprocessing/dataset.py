import csv
import json
import torch
import jsonlines
import pandas as pd
from itertools import takewhile
from preprocessing.config import DATA_PATHS
from sklearn.model_selection import train_test_split

class KBertDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_type, dataframe, kg, tokenizer, sequence_length, no_kg_augment, task, batch_size = None, minified = False):
        super(KBertDataset).__init__()
        self.task = task
        self.tokenizer = tokenizer
        self.strip_punc = r"[!\"#&()*+,.:;<=>?@\\^_`{|}~]"
        self.length, self.text, self.labels = self.load_data(dataset_type, dataframe, minified)
        self.iter_text = iter(self.text)
        self.no_kg_augment = no_kg_augment
        self.sequence_length = sequence_length
        self.iter_labels = iter(self.labels)
        self.kg = kg
        if not batch_size or batch_size > self.length:
          self.batch_size = self.length
        else:
          self.batch_size = batch_size
        
    def load_data(self, dataset_type, data, minified):
        if minified:
            data = data.sample(frac=.001, random_state = 1337)
        if dataset_type == "ag_news" or dataset_type == "yahoo":
            text = (data.headline + " " + data.text).str.replace(self.strip_punc, " ").values
            print(f"{len(data)} lines")
            labels = data.label.values 
            if 0 not in labels:
                labels -= 1
            text = [["[CLS]"] + self.tokenizer.tokenize(sentence) + ["[SEP]"] for sentence in text]
            return len(data), text, labels
        elif dataset_type == "arc" or dataset_type == "qasc":
            data["question"] = data.question.apply(lambda row: ["[CLS]"] + self.tokenizer.tokenize(row) + ["[SEP]"])
            labels = data.AnswerKey.values
            text = data.question.values
            return len(data), text, labels

    def reload_data(self):
        self.iter_text = iter(self.text)
        self.iter_labels = iter(self.labels)

    def _row_parser(self, question, answer):
        split_point = 0
        while question[split_point] != "(" and question[split_point + 2] != ")": 
            split_point+=1
            #if split_point >= self.sequence_length:
            #    break
        answer = ["(", answer.lower(), ")"]
        answer_span = [(i, i+len(answer)) for i in range(len(question)) if question[i:i+len(answer)] == answer]
        if not answer_span:
            print(f"Bad line: {question}")
            return [1] * 3
        start_of_answer = answer_span[0][1]
        end_of_answer = start_of_answer + len(list(takewhile(lambda s: s != "(", question[start_of_answer:])))
        return split_point, start_of_answer, end_of_answer

    def get_batch_size(self):
        return self.batch_size

    def get_data(self, data):
        return [next(data) for _ in range(self.batch_size)]

    def get_normal_streams(self, token_emb):
        if self.task == "classification":
            tokenized_emb = self.tokenizer.batch_encode_plus(
                token_emb, 
                is_pretokenized = True, 
                return_special_token_masks = True, 
                max_length = self.sequence_length, 
                pad_to_max_length = True, 
                return_tensors = "pt", 
                return_token_type_ids = True
            )
            labels = torch.LongTensor(self.get_data(self.iter_labels))
        elif self.task == "qa":
            qa_token_emb = []
            span_ids = []
            labels = self.get_data(self.iter_labels)
            for sentence, label in zip(token_emb, labels):
                split_point, start, end = self._row_parser(sentence, label)
                qa_token_emb.append([sentence[:split_point], sentence[split_point:]])
                span_ids.append((start, end))

            tokenized_emb = self.tokenizer.batch_encode_plus(
                qa_token_emb, 
                is_pretokenized = True, 
                add_special_tokens = False,
                return_special_token_masks = True, 
                max_length = self.sequence_length, 
                pad_to_max_length = True, 
                return_tensors = "pt", 
                return_token_type_ids = True,
                truncation_strategy = "only_first"
            )
            #data.question.apply(lambda row: row.insert(len(list(takewhile(lambda s: s is not "(", row))), "[SEP]"))
            #span_ids = data.apply(row_parser, axis = 1)
            labels = torch.LongTensor(span_ids) 
        
        input_ids = tokenized_emb["input_ids"]
        segment_emb = tokenized_emb["token_type_ids"]
        attention_mask = tokenized_emb["attention_mask"]
        return input_ids, segment_emb, attention_mask, labels

    def get_streams(self):
        token_emb, segment_emb, soft_pos_emb, vm_values, vm_indices = self.kg.inject_knowledge(self.get_data(self.iter_text))
        soft_pos_emb = torch.LongTensor(soft_pos_emb)
        input_ids, segment_emb, attention_mask, labels = self.get_normal_streams(token_emb)


        return input_ids, segment_emb, soft_pos_emb, vm_values, vm_indices, attention_mask, labels

    def __next__(self):
        if self.no_kg_augment:
            input_ids, segment_emb, attention_mask, labels = self.get_normal_streams(self.get_data(self.iter_text))
            soft_pos_emb = None
            vm_values = None
            vm_indices = None

            return input_ids, segment_emb, soft_pos_emb, vm_values, vm_indices, attention_mask, labels
        else:
            return self.get_streams()

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

def load_data_from_path(dataset_name, kg, tokenizer, sequence_length, batch_size, minified, no_kg_augment, task):
    directory = DATA_PATHS[dataset_name]
    dataset_type = dataset_name
    if dataset_name == "ag_news":
        train = pd.read_csv(directory + "original_train.csv", header = None, names = ["label", "headline", "text"])
        train, validation = train_test_split(
            train, test_size=0.25
        )

        test = pd.read_csv(directory + "test.csv", header = None, names = ["label", "headline", "text"])

    elif dataset_name == "yahoo":
        train = pd.read_csv(directory + "train.csv", header = None, names = ["label", "headline", "question", "answer"])
        train["text"] = train.question.fillna(" ") + " " + train.answer.fillna(" ") 
        train, validation = train_test_split(
            train, test_size=0.25
        )

        test = pd.read_csv(directory + "test.csv", header = None, names = ["label", "headline", "question", "answer"])
        test["text"] = test.question.fillna(" ") + " " + test.answer.fillna(" ")

    
    elif dataset_name == "squad":
        train = pd.read_csv(directory + "squad_train.csv")
        train, validation = train_test_split(
            train, test_size=0.25
        )

        test = pd.read_csv(directory + "squad_test.csv")
    
    elif dataset_name == "complex_web_questions":
        train = pd.read_json(directory + "ComplexWebQuestions_train.json")
        extract_answers = lambda row: [answer["aliases"] + [answer["answer"]] 
                                if answer["aliases"] else [answer["answer"]] 
                                for answer in row]
        train = pd.concat([train.question, train.answers.apply(extract_answers)], axis = 1)
        train, validation = train_test_split(
            train, test_size=0.25
        )

        test = pd.read_json(directory + "ComplexWebQuestions_dev.json")
        test = pd.concat([test.question, test.answers.apply(extract_answers)], axis = 1)

    elif dataset_name == "jeopardy":
        data = pd.read_csv(directory + "JEOPARDY_CSV.csv")
        train, test = train_test_split(
            data, test_size=0.15
        )
        train, validation = train_test_split(
            train, test_size=0.25
        )
    
    elif dataset_name == "arc":
        data_1 = pd.read_csv(directory + "ARC-Challenge/ARC-Challenge-Train.csv")
        data_2 = pd.read_csv(directory + "ARC-Easy/ARC-Easy-Train.csv")
        data = pd.concat([data_1, data_2]).sample(frac = 1)
        train = data.loc[:, ["question", "AnswerKey"]].reset_index(drop=True)
        train["AnswerKey"] = train.AnswerKey.replace({"1": "A", "2": "B", "3": "C", "4":"D"})   
        train["question"] = train.question.str.replace("\(1\)", "(A)").str.replace("\(2\)", "(B)").str.replace("\(3\)", "(C)").str.replace("\(4\)", "(D)")
        
        data_1 = pd.read_csv(directory + "ARC-Challenge/ARC-Challenge-Dev.csv")
        data_2 = pd.read_csv(directory + "ARC-Easy/ARC-Easy-Dev.csv")
        data = pd.concat([data_1, data_2]).sample(frac = 1)
        validation = data.loc[:, ["question", "AnswerKey"]].reset_index(drop=True)
        validation["AnswerKey"] = validation.AnswerKey.replace({"1": "A", "2": "B", "3": "C", "4":"D"})
        validation["question"] = validation.question.str.replace("\(1\)", "(A)").str.replace("\(2\)", "(B)").str.replace("\(3\)", "(C)").str.replace("\(4\)", "(D)")  
        
        data_1 = pd.read_csv(directory + "ARC-Challenge/ARC-Challenge-Test.csv")
        data_2 = pd.read_csv(directory + "ARC-Easy/ARC-Easy-Test.csv")
        data = pd.concat([data_1, data_2]).sample(frac = 1)
        test = data.loc[:, ["question", "AnswerKey"]].reset_index(drop=True)
        test["AnswerKey"] = test.AnswerKey.replace({"1": "A", "2": "B", "3": "C", "4":"D"})
        test["question"] = test.question.str.replace("\(1\)", "(A)").str.replace("\(2\)", "(B)").str.replace("\(3\)", "(C)").str.replace("\(4\)", "(D)")
    
    elif dataset_name == "qasc":
        with jsonlines.open(directory + "train.jsonl") as f: 
            train = pd.DataFrame([line for line in f])
        train["question"] = train.formatted_question #train.fact2 + " " + 
        train.rename({"answerKey": "AnswerKey"}, inplace = True, axis = 1)
        train = train.loc[:, ["question", "AnswerKey"]]
        train, validation = train_test_split(
            train, test_size=0.25
        )
        
        with jsonlines.open(directory + "dev.jsonl") as f: 
            test = pd.DataFrame([line for line in f]) 
        test["question"] = test.formatted_question #test.fact2 + " " + 
        test.rename({"answerKey": "AnswerKey"}, inplace = True, axis = 1)
        test = test.loc[:, ["question", "AnswerKey"]]


    print(f"Loading {dataset_type} train dataset")
    train_ds = KBertDataset(dataset_type, train, kg, tokenizer, sequence_length, no_kg_augment, task, batch_size, minified)
    print(f"Loading {dataset_type} val dataset")
    val_ds = KBertDataset(dataset_type, validation, kg, tokenizer, sequence_length, no_kg_augment, task, batch_size, minified)
    print(f"Loading {dataset_type} test dataset")
    test_ds = KBertDataset(dataset_type, test, kg, tokenizer, sequence_length, no_kg_augment, task, batch_size, minified)
    return train_ds, val_ds, test_ds

def preprocess_squad():
    with open("../../data/squad/SQuAD train-v2.0.json", 'r') as f:
        datastore = json.load(f)

    with open("../../data/squad/squad_train.csv", 'w') as f:
        csvwriter = csv.writer(f, quoting = csv.QUOTE_MINIMAL)
        csvwriter.writerow(["question", "answer"])
        for article in datastore['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    question = qa['question']
                    answer = [a['text'] for a in qa['answers']]
                    csvwriter.writerow([question, answer])