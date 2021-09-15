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
# random.seed(1)
# np.random.seed(1)
import timeit


n = 5000
alpha_CL = 0.1
alpha_PC = 0.01

# hc, and save X and the true graph
tag_hc = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
import csv
import subprocess
import sys
try:
    subprocess.check_call("Rscript hc_asia.R "+str(tag_hc) + " "+ str(n), shell=True)
except:
    assert 0
with open("./data/A_cpdag_"+str(tag_hc)+".csv") as csvfile:
    data = list(csv.reader(csvfile))
A0 = np.array(data)
A = A0[1:,:]
A = A[:,1:]
A = A.astype(int) # adj matrix
de2, ue2 = dA2edge(A)
with open("./data/runtime_"+str(tag_hc)+".txt") as txtfile:
    lines = txtfile.readlines()
    runtime_R = float(lines[0])
print('time hc:', runtime_R)

# true cpdag
with open("./data/A_asia_cpdag.csv") as csvfile:
    data = list(csv.reader(csvfile))
A0 = np.array(data)
A = A0[1:,:]
A = A[:,1:]
A = A.astype(int) # adj matrix
de, ue = dA2edge(A)




# CL
alpha_CL = 0.1
# load data and calculate corr matrix
from pandas import read_csv
df = read_csv(r"./data/X_"+str(tag_hc)+".csv")
node_label = df.columns
node_label = node_label[1:]
# node_label = []
X = df.to_numpy()
X = X[:,1:]
n1,p = X.shape
print('p,n:', p,n)
assert p == 8
assert n1 == n


X = X - np.outer(ones(n), np.mean(X,axis=0))
C = (X.T @ X) / (n-1)
dC = np.diag(1/sqrt(np.diag(C)))
C = dot(dot(dC, C), dC)

t0 = timeit.default_timer()
de1, ue1 = inf_polytree(C,n, alpha=alpha_CL)
t1 = timeit.default_timer()
print('time CL:', t1-t0)




# PC algorithm
tag_PC = np.random.randint(1000,9999)
t0 = timeit.default_timer()
de3, ue3, runtime_R = PC_R(C,n,alpha=alpha_PC,tag=tag_PC)
t1 = timeit.default_timer()
print('time PC:', runtime_R)




# evaluation
diff1 = measure_CPDAG(de,ue,de1,ue1)
diff2 = measure_CPDAG(de,ue,de2,ue2)
diff3 = measure_CPDAG(de,ue,de3,ue3)
print('total true edge:', len(de)+len(ue))
print('true vs CL: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff1[:3])+[round(x,2) for x in diff1[3:]])
print('true vs hc: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff2[:3])+[round(x,2) for x in diff2[3:]])
print('true vs PC: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff3[:3])+[round(x,2) for x in diff3[3:]])


pos = plot_CPDAG(de,ue,'asia_cpdag', p=p,node_label=node_label)
# exy = 10
exy = 0
pos_noise = {key:(value[0]+(uniform()-0.5)*exy, value[1]+(uniform()-0.5)*exy) for (key,value) in pos.items()}
plot_CPDAG(de,ue,'asia_cpdag', p=p,node_label=node_label, pos=pos_noise, fig_size=(3.5,3.5))
plot_compare_CPDAG(de, ue, de1, ue1,'asia_cpdag_CL',p=p,node_label=node_label, pos=pos_noise, fig_size=(3.5,3.5))
plot_compare_CPDAG(de, ue, de2, ue2,'asia_cpdag_hc',p=p,node_label=node_label, pos=pos_noise, fig_size=(3.5,3.5))
plot_compare_CPDAG(de, ue, de3, ue3,'asia_cpdag_PC',p=p,node_label=node_label, pos=pos_noise, fig_size=(3.5,3.5))


import os
os.remove("./data/A_cpdag_"+str(tag_hc)+".csv")
os.remove("./data/X_"+str(tag_hc)+".csv")
os.remove("./data/runtime_"+str(tag_hc)+".txt")
os.remove("./data/A_cpdag_"+str(tag_PC)+".csv")
os.remove("./data/C_"+str(tag_PC)+".csv")
os.remove("./data/runtime_"+str(tag_PC)+".txt")
