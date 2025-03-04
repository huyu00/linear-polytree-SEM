import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint, uniform, multivariate_normal
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import (log, exp, sqrt, zeros, ones, eye, linspace, arange,
                    dot, outer, sign,diag)
from sympy.combinatorics.prufer import Prufer
from kruskal import inf_tree_m2
import networkx as nx
from infer_polytree import *

# needs to be defined locally not imported
def loadall_npz(file):
    with np.load(file) as data:
        for var in data:
            globals()[var] = data[var]


import random
# seed = 7
# random.seed(seed)
# np.random.seed(seed)
import timeit


n = 1000
alpha_CL = 0.1
alpha_PC = 0.01

import csv
import subprocess
import sys

# all data
from pandas import read_csv
df = read_csv(r"./data/X_earthquake.csv")
node_label = df.columns
node_label = node_label[1:]
X0 = df.to_numpy()
X0 = X0[:,1:]
n0,p = X0.shape
assert p == 5
assert n0 == 100000

# true cpdag
with open("./data/A_earthquake_cpdag.csv") as csvfile:
    data = list(csv.reader(csvfile))
A0 = np.array(data)
A = A0[1:,:]
A = A[:,1:]
A = A.astype(int) # adj matrix
de, ue = dA2edge(A)


# subsample data
from random import choices
while 1:
    X = X0[choices(range(n0),k=n),:]
    X1 = np.copy(X) # noncentered
    X = X - np.outer(ones(n), np.mean(X,axis=0))
    C = (X.T @ X) / (n-1)
    if np.all(np.diag(C)>0):
        break
print("p,n", p,n)

# hc, and save numerical X and the true graph
tag_hc = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
# hc
de2, ue2, runtime_R = hc_R(X1, tag=tag_hc)


# CL
# load data and calculate corr matrix
dC = np.diag(1/sqrt(np.diag(C)))
C = dot(dot(dC, C), dC)

t0 = timeit.default_timer()
de1, ue1 = inf_polytree(C,n, alpha=alpha_CL)
t1 = timeit.default_timer()
print('time CL:', t1-t0)




# PC algorithm
tag_PC = np.random.randint(1000,9999)
de3, ue3, runtime_R = PC_R(C,n,alpha=alpha_PC,mmax=1,tag=tag_PC)
print('time PC:', runtime_R)


# PC early stopping
t0 = timeit.default_timer()
de4, ue4 = PC_earlystop(C,n, alpha=alpha_PC) # adapted to polytree
t1 = timeit.default_timer()
print('time PC early stop:', t1-t0)


# evaluation
diff1 = measure_CPDAG(de,ue,de1,ue1)
diff2 = measure_CPDAG(de,ue,de2,ue2)
diff3 = measure_CPDAG(de,ue,de3,ue3)
diff4 = measure_CPDAG(de,ue,de4,ue4)
print('total true edge:', len(de)+len(ue))
print('true vs CL: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff1[:3])+[round(x,2) for x in diff1[3:]])
print('true vs hc: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff2[:3])+[round(x,2) for x in diff2[3:]])
print('true vs PC: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff3[:3])+[round(x,2) for x in diff3[3:]])
print('true vs PC early stop: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff4[:3])+[round(x,2) for x in diff4[3:]])


pos = plot_CPDAG(de,ue,'earthquake_cpdag', p=p,node_label=node_label)
# exy = 10
# exy = 0
# pos_noise = {key:(value[0]+(uniform()-0.5)*exy, value[1]+(uniform()-0.5)*exy) for (key,value) in pos.items()}
# print(node_label)
pos_noise = pos.copy()
pos_noise[2] = (pos[2][0], pos[2][1]+20)
plot_CPDAG(de,ue,'earthquake_cpdag', p=p,node_label=node_label, pos=pos_noise,
    fig_size=(3,3),node_size=600,font_size=4)
plot_compare_CPDAG(de, ue, de1, ue1,'earthquake_cpdag_CL',p=p,node_label=node_label, pos=pos_noise,
    fig_size=(3,3),node_size=600,font_size=4)
plot_compare_CPDAG(de, ue, de2, ue2,'earthquake_cpdag_hc',p=p,node_label=node_label, pos=pos_noise,
    fig_size=(3,3),node_size=600,font_size=4)
plot_compare_CPDAG(de, ue, de3, ue3,'earthquake_cpdag_PC',p=p,node_label=node_label, pos=pos_noise,
    fig_size=(3,3),node_size=600,font_size=4)
plot_compare_CPDAG(de, ue, de4, ue4,'earthquake_cpdag_PCes',p=p,node_label=node_label, pos=pos_noise,
    fig_size=(3,3),node_size=600,font_size=4)


import os
os.remove("./data/A_cpdag_"+str(tag_hc)+".csv")
os.remove("./data/X_"+str(tag_hc)+".csv")
os.remove("./data/runtime_"+str(tag_hc)+".txt")
os.remove("./data/A_cpdag_"+str(tag_PC)+".csv")
os.remove("./data/C_"+str(tag_PC)+".csv")
os.remove("./data/runtime_"+str(tag_PC)+".txt")
