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

flag_load_data = False
# data_file_name = 'asia_data_trial_n500.npz'
data_file_name = 'asia_data_trial_n5000.npz'

if flag_load_data:
    loadall_npz('./data/'+data_file_name)
else:
    n = 5000
    ntrial = 100
    alpha_CL = 0.1
    alpha_PC = 0.01
    mmax = -1 # switch between PC and PCm1

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


    tag_hc = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
    tag_PC = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
    diff_trial = zeros((4,7,ntrial))
    from random import choices
    total_time = zeros(4)
    t0_out = timeit.default_timer()
    for t in range(ntrial):
        while 1:
            X = X0[choices(range(n0),k=n),:]
            X = X - np.outer(ones(n), np.mean(X,axis=0))
            C = (X.T @ X) / (n-1)
            if np.all(np.diag(C)>0):
                break
        dC = np.diag(1/sqrt(np.diag(C)))
        C = dot(dot(dC, C), dC)
        # CL
        t0 = timeit.default_timer()
        de1, ue1 = inf_polytree(C,n, alpha=alpha_CL)
        t1 = timeit.default_timer()
        if t > 0:
            total_time[0] += t1 - t0
        diff_trial[0,:,t] = measure_CPDAG(de,ue,de1,ue1)
        # hc
        de_hc, ue_hc, runtime_R = hc_R(X, tag=tag_hc)
        total_time[1] += runtime_R
        diff_trial[1,:,t] = measure_CPDAG(de,ue,de_hc,ue_hc)
        # PC
        de_PC, ue_PC, runtime_R = PC_R(C,n=n,alpha=alpha_PC,mmax=mmax,tag=tag_PC)
        total_time[2] += runtime_R
        diff_trial[2,:,t] = measure_CPDAG(de,ue,de_PC,ue_PC)
        # PC early stopping
        t0 = timeit.default_timer()
        de_PCes, ue_PCes = PC_earlystop(C,n, alpha=alpha_PC) # adapted to polytree
        t1 = timeit.default_timer()
        total_time[3] += t1 - t0
        diff_trial[3,:,t] = measure_CPDAG(de,ue,de_PCes,ue_PCes)
    t1_out = timeit.default_timer()
    print('ntrial:', ntrial)
    print('total time simulation:', np.sum(total_time))
    print('total time:', t1_out - t0_out)


    with open('./data/asia_data_trial_n'+str(n)+'.npz','wb') as file1:
            np.savez(file1, n=n, diff_trial=diff_trial, total_time=total_time, ntrial=ntrial)
    import os
    os.remove("./data/A_cpdag_"+str(tag_hc)+".csv")
    os.remove("./data/X_"+str(tag_hc)+".csv")
    os.remove("./data/runtime_"+str(tag_hc)+".txt")
    os.remove("./data/A_cpdag_"+str(tag_PC)+".csv")
    os.remove("./data/C_"+str(tag_PC)+".csv")
    os.remove("./data/runtime_"+str(tag_PC)+".txt")


# evaluation
# def measure_latex_table(measure, method_names):
#     # 3,7,ntrial
#     # missing, extra, wrong direction, fdr-sk, fdr-cpdag, jaccard-sk, jaccard_cpdag
#     # n_round = [2,2,2,3,3,3,3]
#     n_round = [2,2,2,2,2,2,2]
#     print_order = [0,1,2,3,5,4,6]
#     opt_direction = [0,0,0,0,0,1,1] # min or max preferred
#     nmethod,_,ntrial = measure.shape
#     m_measure = np.mean(measure,axis=2)
#     id_best = zeros(7)
#     for i in range(7):
#         if opt_direction[i]==0:
#             id_best[i] = np.argmin(m_measure[:,i])
#         else:
#             id_best[i] = np.argmax(m_measure[:,i])
#     for k in range(nmethod):
#         s = method_names[k]
#         for i in print_order:
#             m = np.mean(measure[k,i,:])
#             # sd = np.std(measure[i,:]) / sqrt(ntrial)
#             sd = np.std(measure[k,i,:])
#             m = np.round(m,n_round[i])
#             sd = np.round(sd,n_round[i])
#             x = str(m)+'('+str(sd)+')'
#             if k == id_best[i]:
#                 s += ' & \\textbf{' + x +'}'
#             else:
#                 s += ' & ' + x
#         s += ' \\\\'
#         print(s)

def measure_latex_table_ext(measure, method_names,print_subset=list(range(8))):
    # 4,8,ntrial
    # correct, missing, extra, wrong direction, fdr-sk, fdr-cpdag, jaccard-sk, jaccard_cpdag
    # n_round = [2,2,2,3,3,3,3]
    n_round = [2,2,2,2,2,2,2,2]
    opt_direction = [1,0,0,0,0,0,1,1] # min or max preferred
    print_order = [0,3,1,2,4,6,5,7]
    nmethod,_,ntrial = measure.shape
    m_measure = np.mean(measure,axis=2)
    id_best = zeros(8)
    for i in range(8):
        if opt_direction[i]==0:
            id_best[i] = np.argmin(m_measure[:,i])
        else:
            id_best[i] = np.argmax(m_measure[:,i])
    for k in range(nmethod):
        s = method_names[k]
        for i_k in print_subset:
            i = print_order[i_k]
            m = np.mean(measure[k,i,:])
            # sd = np.std(measure[i,:]) / sqrt(ntrial)
            sd = np.std(measure[k,i,:])
            m = np.round(m,n_round[i])
            sd = np.round(sd,n_round[i])
            x = str(m)+'('+str(sd)+')'
            if k == id_best[i]:
                s += ' & \\textbf{' + x +'}'
            else:
                s += ' & ' + x
        s += ' \\\\'
        print(s)

def print_accuracy_info(method_id, method_names, diff_trial):
    print(method_names[method_id]+':')
    print('fraction best recovery:', np.mean(np.logical_and(diff_trial[method_id,0,:]<=1, diff_trial[method_id,2,:]<=1)))
    print('fraction better recovery:', np.mean(np.logical_and(diff_trial[method_id,0,:]<=1, diff_trial[method_id,2,:]<=3)))
    print('fraction good recovery:', np.mean(np.logical_and(diff_trial[method_id,0,:]<=2, diff_trial[method_id,2,:]<=3)))
    print('average recovered edge:', np.mean(n_edge_true - diff_trial[method_id,0,:]))
    print('sd recovered edge:',np.std(n_edge_true - diff_trial[method_id,0,:]))
    # print('average extra:', np.mean(diff_trial[method_id,1,:]))
    # print('sd extra:', np.std(diff_trial[method_id,1,:]))
    print('average FDR-sk:', np.mean(diff_trial[method_id,3,:]))
    print('sd FDR-sk:', np.std(diff_trial[method_id,3,:]))

def print_case_distribution(method_id, method_names, diff_trial):
    print(method_names[method_id]+':')
    plt.figure()
    ep = 0.2
    plt.scatter(diff_trial[method_id,0,:]+ep*uniform(-1,1,ntrial),
        diff_trial[method_id,2,:]+ep*uniform(-1,1,ntrial), 1*ones(ntrial))
    plt.xlabel('missing')
    plt.ylabel('wrong direction')
    plt.savefig('./figure/asia_data_diff_m'+str(method_id)+'.png', dpi=300)
    case_table = zeros((n_edge_true+1,n_edge_true+1))
    for t in range(ntrial):
        x = int(diff_trial[method_id,0,t]) # row=missing
        y = int(diff_trial[method_id,2,t]) # column=wrong direction
        case_table[x,y] += 1
    case_table /= ntrial
    case_table = np.round(case_table,2)
    print('case table:')
    print(case_table)


n_edge_true = 8
method_names = ['Chow-Liu','Hill-climbing','PC','PC early stopped']

print('running time:')
for i in range(len(method_names)):
    print(method_names[i], total_time[i])

# accuracy summary
for i in range(len(method_names)):
    print_accuracy_info(i, method_names, diff_trial)

# latex table
print('Latex table:')
diff_trial_ext = zeros((4,8,ntrial))
diff_trial_ext[:,1:,:] = np.copy(diff_trial)
diff_trial_ext[:,0,:] = n_edge_true - diff_trial[:,0,:] - diff_trial[:,2,:]
measure_latex_table_ext(diff_trial_ext, method_names=method_names, print_subset=list(range(4)))
measure_latex_table_ext(diff_trial_ext, method_names=method_names, print_subset=list(range(4,8)))

# case distribution
for i in range(len(method_names)):
    print_case_distribution(i, method_names, diff_trial)
