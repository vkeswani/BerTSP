# BerTSP
## Formulating Sentence Ordering as the Asymmetric Travelling Salesman Problem
**Authors**: Vishal Keswani & Harsh Jhamtani

This repository describes the code for our work on Sentence Ordering as ATSP presented at INLG 2021. 

### Abstract 
The task of Sentence Ordering refers to rearranging a set of given sentences in a coherent ordering. Prior work (Prabhumoye et al., 2020) models this as an optimal graph traversal (with sentences as nodes, and edges as local constraints) using topological sorting. However, such an approach has major limitations – it cannot handle presence of cycles in the resulting graphs and considers only the binary presence/absence of edges rather than a more granular score. In this work, we propose an alternate formulation of this task as a classic combinatorial optimization problem popular as the Travelling Salesman Problem (or TSP in short). Compared to previous approach of using topological sorting, our proposed technique gracefully handles presence of cycles and is more expressive since it takes into account real valued constraint/edge scores rather than just presence/absence of edges. The results as per various metrics reveal the superiority of our approach over previous graph-based formulation of this task. Additionally, we highlight how using a specific canonical ordering of sentences can inadvertently expose ground truth ordering to the models, leading to inflated scores when using such graphs-based formulations. Finally, we note that our approach requires only light-weight fine-tuning of a classification layer built on pre-trained BERT sentence encoder to identify local relationships.

**Full paper:** Link will be added soon.

### Datasets
SIND (only SIS is relevant): https://visionandlanguage.net/VIST/dataset.html <br>
NIPS, AAN, NSF abstracts: https://ojs.aaai.org/index.php/AAAI/article/view/11997 
### Code
Given a set or unordered sentences, we calculate probability of each ordered sentence-pair using BertForSequenceClassification. We construct a matrix with these probabilities which serves as input for the Travelling Salesman Problem. Since sentence A followed by sentence B has a different input representation than sentence B followed by sentence A, the matrix is asymmetric. We then solve the ATSP via exact and heuristic methods. 

1. The code and trained BERT models for calculating sentence-pair probabilities are taken from https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering.
2. The code for solving ATSP (exact and heuristic) is provided here. 

Files:

prepare_data.py for topo and tsp separate,topological_sort.py,tsp.py args for exact and heuristics



