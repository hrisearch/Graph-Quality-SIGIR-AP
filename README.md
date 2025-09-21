# Graph-Quality

Code to reproduce results for the paper 'On the Interplay Between Graph Quality, Traversal Strategies, and Performance of ANN Retrieval Methods' published at SIGIR AP 2025. Also modified popular PyTerrier library (pyterrier_dr) for below experiments. The modifications are also present in this repository.

## build_nn_graph.py

Indexes and builds corpus graphs for both nearest and farthest neighbors.

## retrieve_nodereplacement.py

Run ANN robustness experiments on poor graphs simulated using node replacement strategy. Variable ni determines the node replacement strategy with '' for approximate, 'n1' for random, 'n2' & 'n3' for malicious.

## retrieve_hnswgraph.py

Run ANN robustness experiments on poor graphs simulated by varying ef_construction value through hnsw-based graph strategy. ef_construction is varied from 40 to 10 degrading quality of graph step-wise.

## statsig_nodereplacement.py

Evaluates statistical significance of ANN effectiveness results on poor graphs simulated using node replacement strategy.

## statsig_hnswgraph.py

Evaluates statistical significance of ANN effectiveness results on poor graphs simulated by varying ef_construction value through hnsw-based graph strategy.

## graph_traversal.py

Evaluates effectiveness and robustness of heuristic search methods like Best First Search, A* and Hill Climbing in Adaptive LADR.

## topresults_analysis.py

Analyzes percentage of top results of ANN methods that come from BM25 seeds across different graph construction methods.

## construction_time.py

Measures time required for building different quality HNSW-based nearest neighbor graphs.
