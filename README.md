# BerTSP
## Formulating Neural Sentence Ordering as the Asymmetric Traveling Salesman Problem
**Authors**: Vishal Keswani & Harsh Jhamtani

This repository describes the code for our work on Sentence Ordering as ATSP presented at INLG 2021. 

## Abstract 
The task of Sentence Ordering refers to rearranging a set of given sentences in a coherent ordering. Prior work (Prabhumoye et al., 2020) models this as an optimal graph traversal (with sentences as nodes, and edges as local constraints) using topological sorting. However, such an approach has major limitations – it cannot handle presence of cycles in the resulting graphs and considers only the binary presence/absence of edges rather than a more granular score. In this work, we propose an alternate formulation of this task as a classic combinatorial optimization problem popular as the Travelling Salesman Problem (or TSP in short). Compared to previous approach of using topological sorting, our proposed technique gracefully handles presence of cycles and is more expressive since it takes into account real valued constraint/edge scores rather than just presence/absence of edges. The results as per various metrics reveal the superiority of our approach over previous graph-based formulation of this task. Additionally, we highlight how using a specific canonical ordering of sentences can inadvertently expose ground truth ordering to the models, leading to inflated scores when using such graphs-based formulations. Finally, we note that our approach requires only light-weight fine-tuning of a classification layer built on pre-trained BERT sentence encoder to identify local relationships.

**Full paper:** Link will be added soon.

## Datasets
SIND (only SIS is relevant): https://visionandlanguage.net/VIST/dataset.html <br>
NIPS, AAN, NSF abstracts: https://ojs.aaai.org/index.php/AAAI/article/view/11997 

## Requirements
transformers
torch

## Directory structure used
|___Sentence_Ordering  <br>
&emsp;&emsp;|___SIND  <br>
&emsp;&emsp;&emsp;&emsp;|___sis  <br>
&emsp;&emsp;&emsp;&emsp;|___sind_bert  <br>
&emsp;&emsp;&emsp;&emsp;|___sind_data  <br>
&emsp;&emsp;|___NIPS  <br>
&emsp;&emsp;&emsp;&emsp;|___nips  <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;|___split  <br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;|___txt_tokenized  <br>
&emsp;&emsp;&emsp;&emsp;|___nips_bert  <br>
&emsp;&emsp;&emsp;&emsp;|___nips_data  <br>
&emsp;&emsp;|___AAN  *(same as NIPS)*<br>
&emsp;&emsp;|___NSF  *(same as NIPS)*<br>
&emsp;&emsp;|___prepare_data.py  <br> 
&emsp;&emsp;|___model.py  <br>
&emsp;&emsp;|___graph_decoder.py  <br>

## Code
Given a set of unordered sentences, we calculate the probability of each ordered sentence-pair using BertForSequenceClassification. We construct a matrix with these probabilities which serves as input for the Traveling Salesman Problem. Since sentence A followed by sentence B has a different input representation than sentence B followed by sentence A, the matrix is asymmetric. We then solve the ATSP via exact and heuristic methods. 

1. The code is based on this [repo](https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering) by @shrimai.
2. The [trained BERT models](https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering#trained-models) for calculating sentence-pair probabilities are also available in this repo. 

### 1. Data preparation
Prepare data for training, development and testing: <br>
```
python prepare_data_modified.py --data_dir ../sis/ --out_dir ../sind_data/ --task_name sind
```
```
python prepare_data_modified.py --data_dir ../nips/ --out_dir ../nips_data/ --task_name nips
```
Output: train.tsv, dev.tsv, test_TopoSort.tsv, test_TSP.tsv <br>

When using pretrained models, prepare data for testing only: <br>
```
python prepare_data_modified.py --data_dir ../nips/ --out_dir ../nips_data/ --task_name nips --test_only
```
Output: test_TopoSort.tsv, test_TSP.tsv <br>

### 2. Training the sentence-pair classifier
Training custom models: <br>
```
mkdir ../sind_bert
```
```
python model.py --data_dir ../sind_data/ --output_dir ../sind_bert/ --do_train --do_eval --per_gpu_eval_batch_size 16
```
Output: checkpoint-2000, checkpoint-4000, etc <br>

### 3. Inference from the sentence-pair classifier
Running inference using pretrained models: <br>
```
python model.py --data_dir ../sind_data/ --output_dir ../sind_bert/ --do_test --per_gpu_eval_batch_size 16
```
Output: test_results_TopoSort.tsv, test_results_TSP.tsv <br>

Running inference using custom trained models: <br>
```
python model.py --data_dir ../sind_data/ --output_dir ../sind_bert/checkpoint-X/ --do_test --per_gpu_eval_batch_size 16
```
Output: test_results_TopoSort.tsv, test_results_TSP.tsv <br>

### 4. Decoding the order via graph traversal
Parameters: <br>
1. file_path:  (required) path to input data directory <br>
2. decoder:    (default - TopoSort) TopoSort/TSP <br>
3. indexing:   (default - reverse) correct/reverse/shuffled <br>
4. subset:     (default - all) cyclic/non_cyclic/all <br>
5. tsp_solver: (default - approx) approx/ensemble/exact <br>
6. exact_upto: (default - 8) if exact or ensemble is chosen, upto how many sentences (or sequence length) should exact tsp be used, recommended upto 8 in general (upto 10 for small datasets) <br>

Examples:
```
python graph_decoder.py --file_path ../nips_data/
``` 
```
python graph_decoder.py --file_path ../nips_data/ --decoder TSP
```
```
python graph_decoder.py --file_path ../nips_data/ --decoder TSP --indexing reverse --subset cyclic --tsp_solver ensemble --exact_upto 6
```

## BibTeX
Use the following to cite the paper or code.<br>
**Citation will be added soon.**

