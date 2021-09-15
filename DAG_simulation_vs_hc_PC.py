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
# random.seed(7)
# np.random.seed(7)
import timeit


flag_load_data = True
run_id = 2
# pre-computed data:
# run_id 2: p=100,n_edge=110,125, ntrial=100, alpha_PC=0.01
tag_hc = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
tag_PC = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
tag_true = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
flag_CL_only = False

if not flag_load_data:
    alpha_CL = 0.1
    alpha_PC = 0.01
    positive_weight = False
    mmax = -1
    tx0 = timeit.default_timer()
    p_target = 100
    p_use_ls = [120, 109]
    n_edge_use_ls = [125,145]
    ncase = len(p_use_ls)
    n_edge_ls = zeros(ncase)
    ntrial = 32 # sets of samples from the same SEM
    ns = [50,100,200,400,600,800,1000]
    nn = len(ns)
    measure = zeros((3,7,ncase,nn,ntrial)) # edge set difference, 0:CL, 1:hc, 2:PC
    # missing, extra, wrong direction, fdr-sk, fdr-cpdag, jaccard-sk, jaccard_cpdag
    nls = zeros((ncase,nn))
    print('run id:', run_id)
    print('number of inference:', ncase*nn*ntrial)
    print('number of SEM:', ncase)
    parameters = []
    jac_sk_th = zeros(ncase)
    time_total = zeros(3)
    for i in range(ncase):
        p = p_target
        p1 = p_use_ls[i]
        ne1 = n_edge_use_ls[i]
        A0,G0 = rand_DAG_nedge_giant_component_target(p,p1,ne1)
        ne2 = len(G0)
        n_edge_ls[i] = ne2
        de0,ue0 = CPDAG_DAG(p_target,G0,tag=tag_true)
        G0_sem = gen_SEM_DAG(p,G0,positive_weight=positive_weight)
        args = (p,ne2,p1,ne1)  # actual p, n_edge and the ones used in rand_DAG
        parameters.append(args)
        nls[i,:] = ns
        n_component_G0 = 1 # connected
        n_edge_G0 = ne2
        jac_sk_th[i] = (p - n_component_G0) / (n_edge_G0+n_component_G0-1)
        for (j,n) in enumerate(ns):
            # print('n='+str(n))
            for t in range(ntrial):
                X = gen_X_SEM(G0,G0_sem[0],G0_sem[1],n)
                C = corr_X(X)
                # CL
                t0 = timeit.default_timer()
                de,ue = inf_polytree(C,n, alpha=alpha_CL)
                t1 = timeit.default_timer()
                time_total[0] += t1-t0
                measure[0,:,i,j,t] = measure_CPDAG(de0,ue0,de,ue)
                if not flag_CL_only:
                    # hc
                    de_hc, ue_hc, runtime_R = hc_R(X, tag=tag_hc)
                    time_total[1] += runtime_R
                    measure[1,:,i,j,t] = measure_CPDAG(de0,ue0,de_hc,ue_hc)
                    # PC
                    de_PC, ue_PC, runtime_R = PC_R(C,n=n,alpha=alpha_PC,mmax=mmax,tag=tag_PC)
                    time_total[2] += runtime_R
                    measure[2,:,i,j,t] = measure_CPDAG(de0,ue0,de_PC,ue_PC)
    tx1 = timeit.default_timer()
    total_sim_time = tx1-tx0
    print('inference time CL: ', time_total[0])
    print('inference time hc: ', time_total[1])
    print('inference time PC: ', time_total[2])
    print('total sim time: ', total_sim_time)
    with open('./data/DAG_sim_'+str(run_id)+'.npz','wb') as file1:
        np.savez(file1, p_target=p_target,n_edge_ls=n_edge_ls,
            p_use_ls=p_use_ls,n_edge_use_ls=n_edge_use_ls,jac_sk_th=jac_sk_th,
            ntrial=ntrial,nn=nn,parameters=parameters,
            ncase=ncase,measure=measure,nls=nls,time_total=time_total,
            total_sim_time=total_sim_time,run_id=run_id,
            alpha_CL=alpha_CL,alpha_PC=alpha_PC,positive_weight=positive_weight,
            mmax=mmax,flag_CL_only=flag_CL_only)
    if not flag_CL_only:
        import os
        os.remove("./data/X_"+str(tag_hc)+".csv")
        os.remove("./data/A_cpdag_"+str(tag_hc)+".csv")
        os.remove("./data/runtime_"+str(tag_hc)+".txt")
        os.remove("./data/A_cpdag_"+str(tag_PC)+".csv")
        os.remove("./data/C_"+str(tag_PC)+".csv")
        os.remove("./data/runtime_"+str(tag_PC)+".txt")
else:
    loadall_npz('./data/DAG_sim_'+str(run_id)+'.npz')
    print('run id:', run_id)
    print('number of inference:', ncase*nn*ntrial)
    print('number of SEM:', ncase*ntrial)
    print('inference time CL: ', time_total[0])
    print('inference time hc: ', time_total[1])
    print('inference time hc: ', time_total[2])
    print('total sim time: ', total_sim_time)
    print('ncase:', ncase)
    print('nn:', nn)
    print('ntrial:', ntrial)

    # ninfer = ncase*nn*ntrial
    # print('inference time CL: ', round(time_total[0]/ninfer,2))
    # print('inference time hc: ', round(time_total[1]/ninfer,2))
    # print('inference time PC: ', round(time_total[2]/ninfer,2))




flag_plot_comparison = True
if flag_CL_only:
    flag_plot_comparison = False
id_plot_measure = [3,4,5,6]
measure_label_short = ["miss", "extra","wrong-d","fdr-sk", "fdr-cpdag", "jac-sk","jac-cpdag"]
measure_label = ["missing", "extra","wrong direction",
    "FDR skeleton", "FDR CPDAG", "Jaccard index skeleton","Jaccard index CPDAG"]
flag_plot_sd = True
sd_sem = sqrt(ntrial) # plot standard error of the mean
# sd_sem = 1 # plot standard deviation
line_w = 1
capsize = 0
plt.rcParams.update({'font.size': 15})
flag_plot_nlogp = False
if not flag_plot_nlogp:
    fig_edge = plt.figure(figsize=(8,6))
else:
    fig_edge_nlogp = plt.figure(figsize=(8,6))

for i_m in id_plot_measure:
    plt.clf()
    for (i,args) in enumerate(parameters):
        p,n_edge,_,_ = args
        if True:
            line_label = 'p'+str(int(p))+'nedge'+str(int(round(n_edge)))
            ns = nls[i,:]
            # plot measure
            # measure = zeros((2,7,ncase,nn,ntrial))
            m = np.squeeze(measure[:,i_m,i,:,:])
            m_label = measure_label[i_m]
            m_label_short = measure_label_short[i_m]
            fraction = np.mean(m,axis=-1)
            f_sd = np.std(m,axis=-1) / sd_sem * 1.96
            # print('cpdag sd:', np.median(f_sd*sd_sem))
            if not flag_plot_nlogp:
                plt.figure(fig_edge.number)
                if flag_plot_sd:
                    line = plt.errorbar(ns, fraction[0,:], f_sd[0,:],
                            marker='.', linestyle='-',label='polytree:'+line_label)
                    caps = line[1]
                    for cap in caps:
                        cap.set_markeredgewidth(0.5)
                    if flag_plot_comparison:
                        (_, caps, _) = plt.errorbar(ns, fraction[1,:], f_sd[1,:],
                            marker='^',linestyle='--', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                        (_, caps, _) = plt.errorbar(ns, fraction[2,:], f_sd[2,:],
                            marker='s', markersize=4, linestyle='-.', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                    if i_m == 5: #jac-skeleton
                        jac_limit = jac_sk_th[i]
                        plt.plot(ns, jac_limit*ones(len(ns)), linestyle='-', color=line[0].get_color(), linewidth=0.5)
                else:
                    line = plt.plot(ns, fraction[0,:], marker='.', linestyle='-',label='polytree:'+line_label)
                    if flag_plot_comparison:
                        plt.plot(ns, fraction[1,:], marker='^', linestyle='--', color=line[0].get_color())
                        plt.plot(ns, fraction[2,:], marker='s', markersize=4, linestyle='-.', color=line[0].get_color())
            else:
                plt.figure(fig_edge_nlogp.number)
                if flag_plot_sd:
                    line = plt.errorbar(ns/log(p), fraction[0,:], f_sd[0,:],
                            marker='.', linestyle='-',label='polytree:'+line_label, capsize=capsize)
                    caps = line[1]
                    for cap in caps:
                        cap.set_markeredgewidth(0.5)
                    if flag_plot_comparison:
                        (_, caps, _) = plt.errorbar(ns/log(p), fraction[1,:], f_sd[1,:],
                            marker='^',linestyle='--', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                        (_, caps, _) = plt.errorbar(ns/log(p), fraction[2,:], f_sd[2,:],
                            marker='s',markersize=4,linestyle='-.', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                else:
                    line = plt.plot(ns/log(p), fraction[0,:], marker='.', linestyle='-',label='polytree:'+line_label)
                    if flag_plot_comparison:
                        plt.plot(ns/log(p), fraction[1,:], marker='^', linestyle='--', color=line[0].get_color())
                        plt.plot(ns/log(p), fraction[2,:], marker='s', markersize=4, linestyle='-.', color=line[0].get_color())
    if not flag_plot_nlogp:
        plt.figure(fig_edge.number)
        plt.xlabel(r'$n$')
    else:
        plt.figure(fig_edge_nlogp.number)
        plt.xlabel(r'$n/\log(p)$')
    # plt.ylabel(m_label)
    plt.ylim([0,1])
    # plt.legend()
    plt.title(m_label)
    # plt.title('polytree:solid, hc:dashed')
    if not flag_plot_nlogp:
        plt.savefig('./figure/DAG_sim_'+m_label_short+'_'+str(run_id), dpi=300)
    else:
        plt.savefig('./figure/DAG_sim_'+m_label_short+'_log_'+str(run_id), dpi=300)
plt.close()
