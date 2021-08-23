import torch
import numpy as np
from sys import maxsize
from itertools import permutations
from collections import defaultdict 
import csv
import argparse
import random

def TSP_exact(graph,source,V):
  vertices = []
  for i in range(V):
    if i != source:
      vertices.append(i)
  min_cost = maxsize
  min_cost_path=()
  all_permutations=permutations(vertices)
  
  for i in all_permutations:
    current_path_cost = 0
    k = source
    for j in i:
      current_path_cost += graph[k][j]
      k = j
    current_path_cost += graph[k][source]
    
    min_cost = min(min_cost,current_path_cost)
    if min_cost==current_path_cost:
      min_cost_path=i

  return list(min_cost_path) # min_cost, min_cost_path

def TSP_approx(graph, l):
  order=[0 for j in range(l)]
  for j1 in range(l):
    for j2 in range(l):
      order[j2]+=graph[j2][j1]
  return list(np.argsort(order))

index = 0
class Graph: 
    '''
    The code for this class is based on geeksforgeeks.com
    '''
    def __init__(self,vertices): 
        self.graph = defaultdict(list) 
        self.V = vertices 
  
    def addEdge(self, u, v, w): 
        self.graph[u].append([v, w]) 
    
    def topologicalSortUtil(self, v, visited, stack): 
  
        visited[v] = True
  
        for i in self.graph[v]: 
            if visited[i[0]] == False: 
                self.topologicalSortUtil(i[0], visited, stack) 
  
        stack.insert(0,v) 
  
    def topologicalSort(self): 
        visited = [False]*self.V 
        stack =[] 

        for i in range(self.V): 
            if visited[i] == False: 
                self.topologicalSortUtil(i, visited, stack) 
  
        return stack
        
    def isCyclicUtil(self, v, visited, recStack): 
  
        visited[v] = True
        recStack[v] = True
  
        for neighbour in self.graph[v]:
            if visited[neighbour[0]] == False: 
                if self.isCyclicUtil(
                    neighbour[0], visited, recStack) == True: 
                    return True
            elif recStack[neighbour[0]] == True: 
                self.graph[v].remove(neighbour)
                return True
  
        recStack[v] = False
        return False
  
    def isCyclic(self): 
        visited = [False] * self.V 
        recStack = [False] * self.V 
      
        for node in range(self.V):
            if visited[node] == False: 
                if self.isCyclicUtil(node, visited, recStack) == True: 
                    return True
        return False

class Stats(object):
    
    def __init__(self):
        self.n_samp = 0
        self.n_sent = 0
        self.n_pair = 0
        self.corr_samp = 0
        self.corr_sent = 0
        self.corr_pair = 0
        self.lcs_seq = 0
        self.tau = 0
        self.dist_window = [1, 2, 3]
        self.min_dist = [0]*len(self.dist_window)

        
    def pairwise_metric(self, order,gold_order):
        '''
        This  calculates the percentage of skip-bigrams for which the 
        relative order is predicted correctly. Rouge-S metric.
        '''
        # original code for this metric (Prabhumoya et al., 2020) does not 
        # work for shuffled orders, hence we modify it
        common = 0
        for i in range(len(order)):
          for j in range(i+1,len(order)):
            if gold_order[order[i]]<gold_order[order[j]]:
              common+=1
        
        return common
    
    def kendall_tau(self, porder, gorder):
        '''
        It calculates the number of inversions required by the predicted 
        order to reach the correct order.
        '''
        pred_pairs, gold_pairs = [], []
        for i in range(len(porder)):
            for j in range(i+1, len(porder)):
                pred_pairs.append((porder[i], porder[j]))
                gold_pairs.append((gorder[i], gorder[j]))
        common = len(set(pred_pairs).intersection(set(gold_pairs)))
        uncommon = len(gold_pairs) - common
        tau = 1 - (2*(uncommon/len(gold_pairs)))

        return tau
    
    def min_dist_metric(self, porder, gorder):
        '''
        It calculates the displacement of sentences within a given window.
        '''
        count = [0]*len(self.dist_window)
        for i in range(len(porder)):
            pidx = i
            pval = porder[i]
            gidx = gorder.index(pval)
            for w, window in enumerate(self.dist_window):
                if abs(pidx-gidx) <= window:
                    count[w] += 1
        return count
    
    def lcs(self, X , Y): 
        m = len(X) 
        n = len(Y) 

        L = [[None]*(n+1) for i in range(m+1)] 

        for i in range(m+1): 
            for j in range(n+1): 
                if i == 0 or j == 0 : 
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]: 
                    L[i][j] = L[i-1][j-1]+1
                else: 
                    L[i][j] = max(L[i-1][j] , L[i][j-1]) 

        return L[m][n] 
    
    def sample_match(self, order, gold_order):
        '''
        It calculates the percentage of samples for which the entire 
        sequence was correctly predicted. (PMR)
        '''
        return order == gold_order
    
    def sentence_match(self, order, gold_order):
        '''
        It measures the percentage of sentences for which their absolute 
        position was correctly predicted. (Acc)
        '''
        return sum([1 for x in range(len(order)) if order[x] == gold_order[x]])
    
    def update_stats(self, nvert, npairs, order, gold_order):
        self.n_samp += 1
        self.n_sent += nvert
        self.n_pair += npairs
        
        if self.sample_match(order, gold_order):
            self.corr_samp += 1
        self.corr_sent += self.sentence_match(order, gold_order)
        self.corr_pair += self.pairwise_metric(order,gold_order)
        self.lcs_seq += self.lcs(order, gold_order)
        self.tau += self.kendall_tau(order, gold_order)
        window_counts = self.min_dist_metric(order, gold_order)
        for w, wc in enumerate(window_counts):
            self.min_dist[w] += wc
        
    def print_stats(self):
        print("Perfect Match: " + str(self.corr_samp*100/self.n_samp))
        print("Sentence Accuracy: " + str(self.corr_sent*100/self.n_sent))
        print("Rouge-S: " + str(self.corr_pair*100/self.n_pair))
        print("LCS: " + str(self.lcs_seq*100/self.n_sent))
        print("Kendall Tau Ratio: " + str(self.tau/self.n_samp))
        for w, window in enumerate(self.dist_window):
            print("Min Dist Metric for window " + str(window) + ": " + \
                                    str(self.min_dist[w]*100/self.n_sent))
        print(index)
        #return [self.corr_samp*100/self.n_samp, self.corr_sent*100/self.n_sent,
        #self.corr_pair*100/self.n_pair,self.lcs_seq*100/self.n_sent,
        #self.tau/self.n_samp]

def convert_to_graph(data_TopoSort, data_TSP, decoder, indexing, subset, tsp_solver, exact_upto):

    stats = Stats()
    i1 = 0
    i2 = 0
    first = 0
    last  = 0

    while i1 < len(data_TopoSort):
        ids = data_TopoSort[i1][0]

        docid, nvert, npairs = ids.split('-')
        docid, nvert, npairs = int(docid), int(nvert), int(npairs)
        
        correct = list(range(nvert))
        reverse = list(reversed(correct))
        shuffled = random.sample(range(nvert),nvert)
        
        if indexing == 'correct': io = correct
        elif indexing == 'reverse': io = reverse
        elif indexing == 'shuffled': io = shuffled

        g = Graph(nvert) 
        for j in range(i1, i1+npairs):
            d = data_TopoSort
            pred = int(d[j][8])
            log0, log1 = float(d[j][6]), float(d[j][7])
            pos_s1, pos_s2 = io[int(d[j][4])], io[int(d[j][5])]
           
            if pred == 0:
                g.addEdge(pos_s2, pos_s1, log0)
            elif pred == 1:
                g.addEdge(pos_s1, pos_s2, log1) 
        
        flag = 0
        while g.isCyclic():
            flag=1
            g.isCyclic()
        
        if subset == 'cyclic' and flag==0: continue
        if subset == 'non_cyclic' and flag==1: continue
           
        if decoder == 'TopoSort': order = g.topologicalSort()        
        elif decoder == 'TSP':
          tensor_test_approx = torch.zeros((1,nvert,nvert))
          tensor_test_exact = torch.zeros((1,nvert+1,nvert+1))
          for j in range(i2, i2+2*npairs):
              d=data_TSP[j]
              if io[int(d[4])]==io[int(d[5])]:
                  continue
              else:
                  # softmax normalization
                  a=float(d[6])
                  b=float(d[7])
                  e=np.exp([a,b])
                  f=e/np.sum(e)
                  tensor_test_approx[0][ io[int(d[4])] ][ io[int(d[5])] ] = f[0]
                  tensor_test_exact[0][ io[int(d[4])] ][ io[int(d[5])] ] = f[0]
        
          if tsp_solver == 'approx':
            order = TSP_approx(tensor_test_approx[0], nvert)
          elif tsp_solver == 'ensemble':
            if nvert <= exact_upto: order = TSP_exact(tensor_test_exact[0], nvert, nvert)
            else: order = TSP_approx(tensor_test_approx[0], nvert)
          elif tsp_solver == 'exact': 
            if nvert <= exact_upto: order = TSP_exact(tensor_test_exact[0], nvert, nvert)
            else: continue
              
        gold_order = io
        
        if order[0] == gold_order[0]: first+=1
        if order[nvert-1] == gold_order[nvert-1]: last+=1

        stats.update_stats(nvert, npairs, order, gold_order)
    
        global index
        print(index)
        index = index + 1
        i1 += npairs
        i2 += 2*npairs
        
    print('First:',first*100/index,'Last:',last*100/index)
    return stats

def readf(filename):
    data = []
    with open(filename, "r") as inp:
        spam = csv.reader(inp, delimiter='\t')
        for row in spam:
            data.append(row)
    return data      

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file_path", type = str, required = True, \
                        help = "path to input data directory")
    parser.add_argument("--decoder", type = str, default = "TopoSort", \
                        help = "decoding algorithm:  TopoSort/TSP")
    parser.add_argument("--indexing", type = str, default = "reverse", \
                        help = "index ordering: correct/reverse/shuffled")
    parser.add_argument("--subset", type = str, default = "all", \
                        help = "subset of the dataset on which decoding is done: \
                        cyclic/non_cyclic/all")
    parser.add_argument("--tsp_solver", type = str, default = "approx", \
                        help = "solver to use if tsp is chosen: exact/approx/ensemble")
    parser.add_argument("--exact_upto", type = int, default = 8,\
                        help = "if exact or ensemble is chosen, upto how many \
                        sentences (or sequence length) should exact tsp be used, \
                        recommended upto 8 in general (upto 10 for small datasets)")
    args = parser.parse_args()
    
    file_path_TSP = args.file_path + "test_results_TSP.tsv"
    file_path_TopoSort = args.file_path + "test_results_TopoSort.tsv"

    data_TopoSort = readf(file_path_TopoSort)
    data_TSP = readf(file_path_TSP)
    
    stats = convert_to_graph(data_TopoSort, data_TSP, args.decoder, args.indexing, \
                             args.subset, args.tsp_solver, args.exact_upto)        
    stats.print_stats()

#random.seed(1001)
if __name__ == "__main__":
    main()