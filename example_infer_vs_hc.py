import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint, uniform, multivariate_normal
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import (log, exp, sqrt, zeros, ones, eye, linspace, arange,
                    dot, outer, sign,diag)
import networkx as nx
import timeit

from infer_polytree import (rand_polytree, gen_X_SEM,
        gen_SEM_polytree, corr_X, inf_polytree, measure_CPDAG,
        plot_polytree, CPDAG_polytree, hc_R)




import random
# random.seed(1)
# np.random.seed(1)

p = 30
n = 1000
ommin = 0.1
din_max = 10
rmin = 0.1
rmax = 0.8
assert ommin+rmin**2*din_max<=1
assert rmax**2<=1-ommin
print('p,n:',p,n)
t0 = timeit.default_timer()

# randomly generated polytree
T0 = rand_polytree(p, din_max=din_max)

# # Define a specifice polytree example
# T0 = [[1,0], [1,2], [3,2],[4,3],[5,4],[5,6],[6,7],[7,8],
#    [8,9],[9,10],[9,11],[8,12],[12,13],[13,14],[14,15],
#    [14,16],[5,17],[17,18],[18,19]]

# print('True polytree: ', T0)

de0, ue0 = CPDAG_polytree(T0)
# plot_polytree(T0, 'test_polytree')
T0_sem = gen_SEM_polytree(T0,ommin,rmin,rmax)
X = gen_X_SEM(T0,T0_sem[0],T0_sem[1],n)
C = corr_X(X)
t1 = timeit.default_timer()
print('time generate X:', t1-t0)

# CL
t0 = timeit.default_timer()
de1, ue1 = inf_polytree(C,n, alpha=0.05)
t1 = timeit.default_timer()
print('time CL:', t1-t0)

# hc algorithm
tag = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
de2, ue2, runtime_R = hc_R(X,tag=tag)
print('time hc:', runtime_R)

# evaluation
diff1 = measure_CPDAG(de0,ue0,de1,ue1)
diff2 = measure_CPDAG(de0,ue0,de2,ue2)
print('true vs CL: missing, extra, wrong direction, fdr-sk, fdr-cpdag, jaccard-sk, jaccard_cpdag')
print(list(diff1[:3])+[round(x,2) for x in diff1[3:]])
print('true vs hc: missing, extra, wrong direction, fdr-sk, fdr-cpdag, jaccard-sk, jaccard_cpdag')
print(list(diff2[:3])+[round(x,2) for x in diff2[3:]])


# remove temp files
import os
os.remove("./data/X_"+str(tag)+".csv")
os.remove("./data/A_cpdag_"+str(tag)+".csv")
os.remove("./data/runtime_"+str(tag)+".txt")
