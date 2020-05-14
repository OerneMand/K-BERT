import os
import numpy as np
import torch

FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))

KG_PATHS = {
    "kbpedia": "../kgs/kbpedia_kg.csv",
    "yago": "../kgs/small_yago.tsv",
    "probase": "../kgs/filtered/probase.tsv",
    "hownet": "../../../K-BERT original/brain/kgs/HowNet.spo",
    "medical": "../../../K-BERT original/brain/kgs/Medical.spo"
}
DATA_PATHS = {
    "ag_news": "../data/ag_news_csv/",
    "squad": "../data/squad/",
    "complex_web_questions": "../data/complexWeb/",
    "arc": "../data/arc/ARC-V1-Feb2018-2/",
    "jeopardy": "../data/",
    "yahoo": "../data/yahoo_answers_csv/"
}

def read_kg():
    pass

def set_seed(seed=1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
