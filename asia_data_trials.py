import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint, uniform, multivariate_normal
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import (log, exp, sqrt, zeros, ones, eye, linspace, arange,
                    dot, outer, sign,diag)
from sympy.combinatorics.prufer import Prufer
from kruskal import inf_tree_m2
import networkx as nx
from infer_polytree import (inf_ploytree, plot_polytree, CPDAG_polytree,
        plot_CPDAG, hc_R, dA2edge, measure_CPDAG, plot_compare_CPDAG)

# needs to be defined locally not imported
def loadall_npz(file):
    with np.load(file) as data:
        for var in data:
            globals()[var] = data[var]


import random
# random.seed(1)
# np.random.seed(1)
import timeit


n = 5000
ntrial = 1000
alpha = 0.1

# hc, and save X and the true graph
import csv
import subprocess
import sys

# true cpdag
with open("./data/A_asia_cpdag.csv") as csvfile:
    data = list(csv.reader(csvfile))
A0 = np.array(data)
A = A0[1:,:]
A = A[:,1:]
A = A.astype(int) # adj matrix
de, ue = dA2edge(A)
# all data
from pandas import read_csv
df = read_csv(r"./data/X_asia.csv")
node_label = df.columns
node_label = node_label[1:]
X0 = df.to_numpy()
X0 = X0[:,1:]
n0,p = X0.shape
assert p == 8
assert n0 == 5000

print('p,n:', p,n)





# CL
diff_trial = zeros((7,ntrial))
from random import choices
t0 = timeit.default_timer()
for t in range(ntrial):
    X = X0[choices(range(n0),k=n),:]
    X = X - np.outer(ones(n), np.mean(X,axis=0))
    C = (X.T @ X) / (n-1)
    dC = np.diag(1/sqrt(np.diag(C)))
    C = dot(dot(dC, C), dC)
    de1, ue1 = inf_ploytree(C,n, alpha=alpha)
    diff_trial[:,t] = measure_CPDAG(de,ue,de1,ue1)
t1 = timeit.default_timer()
print('time CL:', t1-t0)



# evaluation
print('fraction best recorvery:', np.mean(np.logical_and(diff_trial[0,:]<=1, diff_trial[2,:]<=1)))
print('fraction better recorvery:', np.mean(np.logical_and(diff_trial[0,:]<=1, diff_trial[2,:]<=3)))
print('fraction good recorvery:', np.mean(np.logical_and(diff_trial[0,:]<=2, diff_trial[2,:]<=3)))
print('average missing:', np.mean(diff_trial[0,:]))
print('sd missing:', np.std(diff_trial[0,:]))

plt.figure()
ep = 0.2
plt.scatter(diff_trial[0,:]+ep*uniform(-1,1,ntrial),
    diff_trial[2,:]+ep*uniform(-1,1,ntrial), 1*ones(ntrial))
plt.xlabel('missing')
plt.ylabel('wrong direction')
plt.savefig('./figure/asia_data_diff.png', dpi=300)
