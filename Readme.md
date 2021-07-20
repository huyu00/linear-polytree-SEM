
# Linear polytree structure equation modeling


This repository contains the inference and simulation codes accompying the paper: *X. Lou, Y. Hu, X. Li, Linear Polytree Structural Equation Models: Structural Learning and Inverse Correlation Estimation, 2021*.


## Installation
The majority of the code is written in Python 3 along with several assisting R scripts. 

For Python, it requires `netwowrkx` and `graphviz, pydot` (for visualizing graphs only) packages in addition to standard packages such as `numpy, scipy`.

For R, we use the package `bnlearn` for the implementation of the comparative hill-climbing algorithm and benchmark data.

The code also requires a `\data` and `\figure` sub-folders to store results.


## Main files        
`infer_polytree.py` contains the functions for learning the CPDAG from data based on a linear polytree model, as well as functions for randomly generate polytree models and visualizing the graphs.

`example_infer_vs_hc.py` gives a simple example of applying the polytree learning to a synthetic data, and compare the result with the hill climbing algorithm.

`polytree_simulation_vs_hc.py` test the performance of the polytree learning algorithm on randomly generated polytree models under various parameters `p`, `n`, `r_min` (rho_min), `din_max` (see the paper for details). The code produces Fig.1 of the paper (using pre-computed simulation data `run_id=5,6`). Note, the code may take a while to run with a large number of `n` and `ntrial`. 

`asia_data.py` applies the polytree learning to the benchmark data Asia, *Lauritzen and Spiegelhalter (1988)*. It produces Fig.2 of the paper.

`alarm_data.py` applies the polytree learning to the benchmark data ALARM *Beinlich et al. (1989)*. It produces Fig.3 of the paper.





## Acknowledgement

The DAG benchmark ALARM data is downloaded from the online resources of the paper [*The Max-Min Hill-Climbing Bayesian Network Structure Learning Algorithm 
I. Tsamardinos, L. E. Brown, C. F. Aliferis, Machine Learning, 2006*](https://pages.mtu.edu/~lebrown/supplements/mmhc_paper/mmhc_index.html).

We use a simple Pyhton implementation of the Kruskal MST algorithm from [Pedro Lobato@GitHub](https://gist.github.com/pedrolobato/e9ef04ac0525ed96a0a78956a1e9cd36), `kruskal.py`. We found it is faster than the buildt-in function from `networkx`.




## Citation
Please give citations to the paper: *X. Lou, Y. Hu, X. Li, Linear Polytree Structural Equation Models: Structural Learning and Inverse Correlation Estimation, 2021*.



 





