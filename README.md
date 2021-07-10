# MathEmb
A PyTorch implementation of *Searching for Mathematical Formulas Based on Graph Representation Learning (CICM 2021)*.

## Abstract
Significant advances have been witnessed in the area of representation learning. Recently, there have been some attempts of applying representation learning methods on mathematical formula retrieval. We introduce a new formula embedding model based on a kind of graph representation generated from hierarchical representation for mathematical formula. Such a representation characterizes structural features in a compact form by merging the same part of a mathematical formula. Following the approach of graph self-supervised learning, we pre-train Graph Neural Networks at the level of individual nodes to learn local representations and then produce a global representation for the entire graph. In this way, formulas can be embedded into a low-dimensional vector space, which allows efficient nearest neighbor search using cosine similarity. We use 579,628 formulas extracted from Wikipedia Corpus provided by NTCIR-12 Wikipedia Formula Browsing Task to train our model, leading to competitive results for full relevance on the task. Experiments with a preliminary implementation of the embedding model illustrate the feasibility and capability of graph representation learning in capturing structural similarities of mathematical formulas.

## Requirements
The codebase is implemented in Python 3.8.8. package versions used for development are just below.
```
torch             1.7.1
torch_geometric   1.6.3
torch_scatter     2.0.6
networkx          2.5
numpy             1.19.2
```

## Dataset
We use Wikipedia Corpus provided by NTCIR-12 Wikipedia Formula Browsing Task to train and evaluate our embedding model. `data_process.py` first extracts formulas from html files contained in Wikipedia Corpus and converts each formula to a string indicating a tree structure. Then each string is converted to a dict indicating a graph structure where keys and values are both indexes of vertices in the graph. The last step of getting the data ready is convert each graph to an instance of `torch_geometric.data.Data` in Pytorch Geometric. We can create different datasets due to choosing different discarding thresholds for node labels. The node labels corresponded to the discarding threshold of 0, 5, 10 can be found in w2c directory.

## Running
After getting the dataset ready, just run `pretrain.py` in the shell to train a model and the trained model will be saved in the model directory. Then `graph_rep.py` is used to generate dense vectors for all graphs (i.e. formulas). Finally, `retrieval.py` computes the cosine similarity between each query and each formula in the dataset, and returns a txt file containing the top k results. To evaluate the results, just use trec_eval tool with the judge file of NTCIR-12 task.
