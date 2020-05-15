# K-BERT
This work is a re-implementation of the [Liu et. al](https://arxiv.org/abs/1909.07606) and a part of my bachelor thesis.

The primary interface of this repository is with `run.py`.

A simple example can be run with

```
python3 run.py \ 
  --dataset_name qasc


The full list of parameter is as this

``` 
python3 run.py \ 
  --dataset_name [qasc, arc, ag_news] \ 
  --knowledge_base [kbpedia, yago, probase] \ 
  --num_processes [int] \
  --run_name [str] \
  --batch_size [int] \ 
  --plot_train_val \ 
  --sequence_length [int] \
  --epochs [int] \
  --no_kg_augment \
  --minified \
  --learning_rate [float] \
  

```

The `config.py` contains the directories to the data and the path to the knowledge graphs.

