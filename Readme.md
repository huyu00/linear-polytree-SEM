
# Learning Linear Polytree Structural Equation Model


This repository contains the inference and simulation codes accompying the paper: [*X. Lou, Y. Hu, X. Li, Learning Linear Polytree Structural Equation Model, TMLR, 2025*](https://openreview.net/forum?id=N28FdYO2sH).


## Installation
The majority of the code is written in Python 3 along with several assisting R scripts. 

For Python, it requires `netwowrkx` and `graphviz, pydot` (for visualizing graphs only) packages in addition to standard packages such as `numpy, scipy`.

For R, we use the package `bnlearn` for the implementation of the comparative hill-climbing algorithm and benchmark data.

The code also requires a `\data` and `\figure` sub-folders to store results.


## Main files        
`infer_polytree.py` contains the functions for learning the CPDAG from data based on a linear polytree model, as well as functions for randomly generate polytree models and visualizing the graphs.

`example_infer_vs_hc.py` gives a simple example of applying the Chow-Liu algorithm to learn a polytree on a randomly generated SEM data, and compare the result with other algorithms (hill climbing, PC, polytree adapted PC).

`polytree_simulation_vs_hc_PC.py` test the performance of the Chow-Liu algorithm to learn randomly generated polytree models under various parameters `p`, `n`, `r_min` ($\rho_\min$), `din_max` (see the paper for details). The code produces Fig.1,2 of the paper (using pre-computed simulation data `run_id=3,4`). Note, the code may take a while to run with a large number of `n` and `ntrial`. 


`alarm_data.py` applies the Chow-Liu algorithm to the benchmark data ALARM *Beinlich et al. (1989)*, which is a DAG with 37 nodes and 46 edges. It produces Fig.3 of the paper.

`asia_data.py` applies the Chow-Liu algorithm to the benchmark data ASIA, *Lauritzen and Spiegelhalter (1988)*, which is a DAG with 8 nodes and 8 edges. It produces Fig.4 of the paper.

`earthquake_data.py` applies the Chow-Liu algorithm to the benchmark data EARTHQUAKE, *Korb and Nicholson, (2010)*. It produces Fig.5 of the paper.




## Acknowledgement

The DAG benchmark ALARM data is downloaded from the online resources of the paper [*The Max-Min Hill-Climbing Bayesian Network Structure Learning Algorithm 
I. Tsamardinos, L. E. Brown, C. F. Aliferis, Machine Learning, 2006*](https://pages.mtu.edu/~lebrown/supplements/mmhc_paper/mmhc_index.html).

The DAG benchmark AISA and EARTHQUAKE data are from the R package bnlearn [*bnlearn*](https://cran.r-project.org/web/packages/bnlearn/index.html) and its [*Bayesian Network Repository*](https://www.bnlearn.com/bnrepository/).



We use a simple Pyhton implementation of the Kruskal MST algorithm from [Pedro Lobato@GitHub](https://gist.github.com/pedrolobato/e9ef04ac0525ed96a0a78956a1e9cd36), `kruskal.py`. We found it is faster than the buildt-in function from `networkx`.




## Citation
Please give citations to the paper: [*X. Lou, Y. Hu, X. Li, Learning Linear Polytree Structural Equation Model, TMLR, 2025*](https://openreview.net/forum?id=N28FdYO2sH).


 





