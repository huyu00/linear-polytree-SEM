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
import timeit

# ALARM data from https://pages.mtu.edu/~lebrown/supplements/mmhc_paper/mmhc_index.html

# load X data
# file_name_X = "./data/alarm_data/Alarm1_s500_v1"
file_name_X = "./data/alarm_data/Alarm1_s5000_v1"
X = []
with open(file_name_X+'.txt') as txtfile:
    data = txtfile.readlines()
    for line in data:
        X.append([int(s) for s in line.split()])
    X = np.array(X, dtype=int)
n,p = X.shape
assert p==37
# resave data in csv
np.savetxt(file_name_X+'.csv', X, delimiter=",")

# load true DAG
file_name_A = "./data/alarm_data/Alarm1_graph"
A = []
with open(file_name_A+'.txt') as txtfile:
    data = txtfile.readlines()
    for line in data:
        A.append([int(s) for s in line.split()])
    A = np.array(A, dtype=int)
    A = A.T # the adjacency matrix with the online data is transposed comparing usual matrix convention
p,_ = A.shape
assert p==37
# # resave data in csv
# np.savetxt(file_name_A+'.csv', A, delimiter=",")

# true CPDAG
A_CPDAG = A_DAG_CPDAG(A)
de, ue = dA2edge(A_CPDAG)
node_label = ["MINV", "PRSS", "PAP", "ECO2", "HRBP", "ERLO", "HRSA", "HREK",
    "ERCA", "BP", "HIST", "CO", "HR", "CCHL", "ACO2", "TPR", "VAPL", "SAO2",
    "PVS", "FIO2", "VALV", "VLNG", "VTUB", "DISC", "VMCH", "MVS", "KINK",
    "SHNT", "INT", "PMB", "ANES", "CVP", "PCWP", "STKV", "LVV", "LVF", "HYP"]


# CL
alpha_CL = 0.1
X = X - np.outer(ones(n), np.mean(X,axis=0))
C = (X.T @ X) / (n-1)
dC = np.diag(1/sqrt(np.diag(C)))
C = dot(dot(dC, C), dC)

t0 = timeit.default_timer()
de1, ue1 = inf_polytree(C,n, alpha=alpha_CL)
t1 = timeit.default_timer()

# evaluation
print('p,n:', p,n)
print('total true edge:', len(de)+len(ue))
diff1 = measure_CPDAG(de,ue,de1,ue1)
print('time CL:', t1-t0)
print('true vs CL: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff1[:3])+[round(x,2) for x in diff1[3:]])

pos = plot_CPDAG(de,ue,'alarm_cpdag', p=p,node_label=node_label)
exy = 10
pos_noise = {key:(value[0]+(uniform()-0.5)*exy, value[1]+(uniform()-0.5)*exy) for (key,value) in pos.items()}
plot_CPDAG(de,ue,'alarm_cpdag', p=p,node_label=node_label, pos=pos_noise, fig_size=(8,8))
# plot_CPDAG(de1,ue1,'alarm_cpdag_CL_self', p=p,node_label=node_label)
plot_compare_CPDAG(de, ue, de1, ue1,'alarm_cpdag_CL',p=p,node_label=node_label, pos=pos_noise, fig_size=(8,8))




# hc
tag = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
import csv
import subprocess
import sys
try:
    subprocess.check_call("Rscript hc_alarm.R "+str(tag) + " "+ file_name_X+'.csv', shell=True)
except:
    assert 0
with open("./data/A_cpdag_"+str(tag)+".csv") as csvfile:
    data = list(csv.reader(csvfile))
A0 = np.array(data)
A = A0[1:,:]
A = A[:,1:]
A = A.astype(int) # adj matrix
de2, ue2 = dA2edge(A)
with open("./data/runtime_"+str(tag)+".txt") as txtfile:
    lines = txtfile.readlines()
    runtime_R = float(lines[0])

diff2 = measure_CPDAG(de,ue,de2,ue2)
print('time hc:', runtime_R)
print('true vs hc: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff2[:3])+[round(x,2) for x in diff2[3:]])
plot_compare_CPDAG(de, ue, de2, ue2,'alarm_cpdag_hc',p=p,node_label=node_label, pos=pos_noise, fig_size=(8,8))

import os
os.remove("./data/A_cpdag_"+str(tag)+".csv")
os.remove("./data/runtime_"+str(tag)+".txt")



# PC
alpha_PC = 0.01
tag_PC = np.random.randint(1000,9999)
t0 = timeit.default_timer()
de3, ue3, runtime_R = PC_R(C,n, alpha=alpha_PC,tag=tag_PC)
t1 = timeit.default_timer()
diff3 = measure_CPDAG(de,ue,de3,ue3)
print('time PC:', runtime_R)
print('true vs PC: miss, extra, wrong-d, fdr-sk, fdr-cpdag, jac-sk, jac_cpdag')
print(list(diff3[:3])+[round(x,2) for x in diff3[3:]])
plot_compare_CPDAG(de, ue, de3, ue3,'alarm_cpdag_PC',p=p,node_label=node_label, pos=pos_noise, fig_size=(8,8))

import os
os.remove("./data/A_cpdag_"+str(tag_PC)+".csv")
os.remove("./data/C_"+str(tag_PC)+".csv")
os.remove("./data/runtime_"+str(tag_PC)+".txt")





def measure_single_latex_table_ext(measure, method_names):
    # 3,8
    # correct, missing, extra, wrong direction, fdr-sk, fdr-cpdag, jaccard-sk, jaccard_cpdag
    # n_round = [2,2,2,3,3,3,3]
    n_round = [2,2,2,2,2,2,2,2]
    opt_direction = [1,0,0,0,0,0,1,1] # min or max preferred
    print_order = [0,3,1,2,4,6,5,7]
    nmethod,_ = measure.shape
    id_best = zeros(8)
    for i in range(8):
        if opt_direction[i]==0:
            id_best[i] = np.argmin(measure[:,i])
        else:
            id_best[i] = np.argmax(measure[:,i])
    for k in range(nmethod):
        s = method_names[k]
        for i in print_order:
            m = measure[k,i]
            m = np.round(m,n_round[i])
            x = str(m)
            if k == id_best[i]:
                s += ' & \\textbf{' + x +'}'
            else:
                s += ' & ' + x
        s += ' \\\\'
        print(s)



print('Latex table:')
method_names = ['Polytree','Hill-climbing','PC']
n_edge_true = len(de) + len(ue)
diff_ext = zeros((3,8))
diff_ext[0,1:] = np.copy(diff1)
print(diff1)
diff_ext[1,1:] = np.copy(diff2)
diff_ext[2,1:] = np.copy(diff3)
diff_ext[:,0] = n_edge_true - diff_ext[:,1] - diff_ext[:,3]
print(diff_ext[0,:])
measure_single_latex_table_ext(diff_ext, method_names=method_names)
