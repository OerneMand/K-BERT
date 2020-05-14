import csv 
import time 
import itertools 
import numpy as np
import pandas as pd

from re import findall, sub
from nltk import ngrams
from transformers import BertTokenizer 
from preprocessing.config import KG_PATHS
from multiprocessing import Manager, Queue, Process

def read_kg(kg_name, tokenizer, num_processes = 6, sample = False):
    if kg_name == KG_PATHS["kbpedia"]:
        kg = kbpedia_read(kg_name)
    elif kg_name == KG_PATHS["yago"]:
        kg = parallel_read(kg_name, do_yago, tokenizer, num_processes, "utf-8", sample)
    elif kg_name == KG_PATHS["probase"]:
        kg = parallel_read(kg_name, do_probase, tokenizer, num_processes, sample)
    return kg
    
def do_yago(in_queue, kg, tokenizer):  
    # While there is still something in the queue, do processing of relations
    while in_queue:
        item = in_queue.get() 
        if not item:
            print("Process reached end of work")
            return
        line = item 
    
        _, subj, rel, obj, _ = line 
        subj = tokenizer(sub("[_><]", " ", subj)) #sub("[_><]", " ", subj).lower().split() 
        rel = " ".join(findall("[A-z][a-z]*", rel)).lower().split() 
        obj = tokenizer(sub("[_><]", " ", obj)) #.lower().split() 
        subj = " ".join(subj) 
        body = rel + obj 
                
        if subj in kg: 
            kg[subj].append(body) 
        else: 
            kg[subj] = [body] 
    
def do_probase(in_queue, kg, tokenizer):  
    # While there is still something in the queue, do processing of relations
    while in_queue:
        item = in_queue.get() 
        if not item:
            print("Process reached end of work")
            return
        line = item

        obj, subj, _ = line
        subj = " ".join(tokenizer(subj))
        body = tokenizer("is a " + obj)
        if subj in kg:
            kg[subj].append(body)
        else:
            kg[subj] = [body]

def parallel_read(kg_path, work_function, tokenizer, num_processes = 6, encoding = "ascii", sample = False):
    #kg_path = KG_PATHS[kg_name] 
    manager = Manager() 
    results = manager.dict() 
    work = manager.Queue(num_processes)
    start = time.time()
    
    # Prime workers     
    pool = [] 
    for _ in range(num_processes): 
        p = Process(target=work_function, args=(work, results, tokenizer)) 
        p.start() 
        pool.append(p) 
    
    # Open stream
    with open(kg_path, encoding = encoding) as f: 
        kg_tsv = csv.reader(f, delimiter = "\t") 
        iters = itertools.chain(kg_tsv, (None,)*num_processes) # To allow stopping and empty Queue
        for line in iters: 
            #should be a percent of how many lines to skip
            if sample and np.random.random() > sample:
                continue
            # Put streamed line in queue to be snagged by work_function
            work.put(line) 
    
    for p in pool: 
        p.join() 
            
    print(f"Reading kg was {time.time() - start:.2f}s") 
    return dict(results)
                                

def probase_read(kg_path, tokenizer = BertTokenizer.from_pretrained("bert-base-uncased").tokenize):
    kg = {}
    with open(kg_path) as kg_tsv:
        for i, line in enumerate(csv.reader(kg_tsv, delimiter = "\t")):
            obj, subj, _ = line
            if i % 5000000 == 0:
                print(f"Iteration {i//1000}k: through the file")
            body = "is a" + tokenizer(obj)
            subj = " ".join(tokenizer(subj))
            if subj in kg:
                kg[subj].append(body)
            else:
                kg[subj] = [body]
    return kg

def yago_read(kg_path, tokenizer = BertTokenizer.from_pretrained("bert-base-uncased").tokenize, sample = False):
    kg = {}
        
    with open(kg_path, "r") as kg_tsv:
        for i, line in enumerate(csv.reader(kg_tsv, delimiter = "\t")):
            #should be a percent of how many lines to skip
            if sample and np.random.random() > sample:
                continue
            if i < 1:
                continue
            if i % 5000000 == 0:
                print(f"Iteration {i//1000}k: {i / 12430701 * 100:.1f}% through the file")
            _, subj, rel, obj, _ = line
            subj = tokenizer(sub("[_><]", " ", subj)) #sub("[_><]", " ", subj).lower().split()
            rel = " ".join(findall("[A-z][a-z]*", rel)).lower().split()
            obj = tokenizer(sub("[_><]", " ", obj)) #.lower().split()
            subj = " ".join(subj)
            body = rel + obj
                        
            if subj in kg:
                kg[subj].append(body)
            else:
                kg[subj] = [body]
        return kg


def kbpedia_read(kg_path):
    kg = {}
    with open(kg_path, "r") as kg_csv:
        try:
            for line in csv.reader(kg_csv):
                if line == ["subject", "relation", "object"]:
                    continue
                subj, rel, obj = [" ".join(findall("[A-z][a-z]*", string = clause)).lower().split() for clause in line]
                subj = " ".join(subj)
                body = rel + obj
                
                if subj in kg:
                    kg[subj].append(body)
                else:
                    kg[subj] = [body]
        except:
            print(f"Bad encoding at line {line}")

    return kg

def probase_filter(kg_path, tokenizer): 
        kg = {} 
        already_written = set()
        with open(kg_path) as kg_tsv,  open("kgs/filtered/probase.tsv", "w") as filtered: 
            filter_writer = csv.writer(filtered, delimiter = "\t") 
            for i, line in enumerate(csv.reader(kg_tsv, delimiter = "\t")): 
                _, subj, _ = line 
                if i % 5000000 == 0: 
                    print(f"Iteration {i//1000}k: {i / 33377320 * 100:.1f}% through the file") 
                subj = " ".join(tokenizer(subj)) 
                dataset = pd.read_csv("../data/ag_news_csv", header = None, names = ["label", "headline", "text"])
                fuck = [tokenizer.tokenize(sent) for sent in dataset.text]
                jesus_fuck = [ngrams(sent, i+1) for sent in fuck for i in range(5)] 
                vocab = set()
                for i in jesus_fuck: 
                   for j in i: 
                       vocab.add(j) 
                real_vocab = {" ".join(ngram) for ngram in vocab} 
                if subj in real_vocab and subj not in already_written: 
                    filter_writer.writerow(line) 
                    already_written.add(subj) 
        return kg

