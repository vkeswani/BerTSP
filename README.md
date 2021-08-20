# BerTSP
## Formulating Neural Sentence Ordering as the Asymmetric Traveling Salesman Problem
**Authors**: Vishal Keswani & Harsh Jhamtani

This repository describes the code for our work on Sentence Ordering as ATSP presented at INLG 2021. 

### Abstract 
The task of Sentence Ordering refers to rearranging a set of given sentences in a coherent ordering. Prior work (Prabhumoye et al., 2020) models this as an optimal graph traversal (with sentences as nodes, and edges as local constraints) using topological sorting. However, such an approach has major limitations – it cannot handle presence of cycles in the resulting graphs and considers only the binary presence/absence of edges rather than a more granular score. In this work, we propose an alternate formulation of this task as a classic combinatorial optimization problem popular as the Travelling Salesman Problem (or TSP in short). Compared to previous approach of using topological sorting, our proposed technique gracefully handles presence of cycles and is more expressive since it takes into account real valued constraint/edge scores rather than just presence/absence of edges. The results as per various metrics reveal the superiority of our approach over previous graph-based formulation of this task. Additionally, we highlight how using a specific canonical ordering of sentences can inadvertently expose ground truth ordering to the models, leading to inflated scores when using such graphs-based formulations. Finally, we note that our approach requires only light-weight fine-tuning of a classification layer built on pre-trained BERT sentence encoder to identify local relationships.

**Full paper:** Link will be added soon.

### Datasets
SIND (only SIS is relevant): https://visionandlanguage.net/VIST/dataset.html <br>
NIPS, AAN, NSF abstracts: https://ojs.aaai.org/index.php/AAAI/article/view/11997 
### Directory structure used
|____Sentence_Ordering  <br>
|________SIND  <br>
|____________sind  <br>
|________________sis?  <br>
|____________sind_bert  <br>
|____________sind_data  <br>
|_______NIPS  <br>
|___________nips  <br>
|_______________split  <br>
|_______________txt_tokenized  <br>
|____________nips_bert  <br>
|____________nips_data  <br>
|________AAN  *(same as NIPS)*<br>
|________NSF  *(same as NIPS)*<br>
|________prepare_data.py  <br> 
|________model.py  <br>
|________graph_decoder.py  <br>

### Code
Given a set of unordered sentences, we calculate the probability of each ordered sentence-pair using BertForSequenceClassification. We construct a matrix with these probabilities which serves as input for the Traveling Salesman Problem. Since sentence A followed by sentence B has a different input representation than sentence B followed by sentence A, the matrix is asymmetric. We then solve the ATSP via exact and heuristic methods. 

1. The code and trained BERT models for calculating sentence-pair probabilities are taken from https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering.
2. The code for solving ATSP (exact and heuristic) is provided here. 


prepare_data.py for topo and tsp separate,topological_sort.py,tsp.py args for exact and heuristics
#### Data preparation
Prepare data for training, development and testing: <br>
`python prepare_data.py --data_dir nips/ --out_dir nips_data/ --task_name nips` <br>
Output: train.tsv, dev.tsv, test_TopoSort.tsv, test_TSP.tsv <br>

When using pretrained models, prepare data for testing only: <br>
`python prepare_data.py --data_dir nips/ --out_dir nips_data/ --task_name nips --test_only` <br>
Output: test_TopoSort.tsv, test_TSP.tsv <br>

#### Training
Training custom models: <br>
`python model.py --data_dir ../sind_data/ --output_dir ../trained_models/sind_bert/ --do_train --per_gpu_eval_batch_size 16` <br>
Output: checkpoints ___ <br>

#### Inference from sentence-pair classifier
Running inference using pretrained models: <br>
`python model.py --data_dir ../sind_data/ --output_dir ../trained_models/sind_bert/ --do_test --per_gpu_eval_batch_size 16` <br>
Output: test_results_TopoSort.tsv, test_results_TSP.tsv <br>

Running inference using custom trained models: <br>
`python model.py --data_dir ../sind_data/ --output_dir ../trained_models/sind_bert/checkpoint-XXXX/ --do_test --per_gpu_eval_batch_size 16` <br>
Output: test_results_TopoSort.tsv, test_results_TSP.tsv <br>

`python topological_sort.py --file_path nips_data/test_results.tsv` <br>
#### Decoding the order via graph traversal
Different modes of decoding:  <br>
- Topological Sort: `--TopoSort`  <br>
--> Input indexing: `--correct` or `--reverse` or `--shuffled`  <br>
--> Subset of data: `--cyclic` or `--non_cyclic` or `--all`  <br>
e.g.: `python graph_decoder.py --file_path nips_data/ --TopoSort --reverse --all`  <br><br>
- TSP: `--TSP`  <br>
--> TSP solution: `--exact` or `--approx` or `--ensemble` <br>
(to specify sequence length upto which exact should be used, specify `--exact_upto`, beyond which approx is used)  <br>
--> Subset of data: `--cyclic` or `--non_cyclic` or `--all`  <br>
e.g.: `python graph_decoder.py --file_path nips_data/ --TSP --approx --cyclic` <br>


