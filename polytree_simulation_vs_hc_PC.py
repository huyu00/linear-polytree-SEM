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
# random.seed(2)
# np.random.seed(2)
import timeit

flag_load_data = True
run_id = 56 # generate a single random polytree, then equipt with different rho_min at each trial
flag_no_label = True
# pre-computed data:
# run_id 55: p=100, din_max=10, ntrial=100, alpha_CL=0.1, alpha_PC=0.01, m.max=inf
# run_id 56: p=100, din_max=20, ntrial=100, alpha_CL=0.1, alpha_PC=0.01, m.max=inf
alpha_CL = 0.1
alpha_PC = 0.01
tag_hc = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
tag_PC = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
# tag_PCm1 = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
flag_CL_only = False

if not flag_load_data:
    tx0 = timeit.default_timer()
    p_ls = [100]
    ommin = 0.1
    # rmin_ls = [0.05, 0.1, 0.15, 0.3]
    rmin_ls = [0.05, 0.1, 0.15]
    rmax_ls = [0.8]
    assert max(rmax_ls)**2 <= 1-ommin
    ntrial = 100
    ns = [50,100,200,400,600,800,1000]
    # ns = [50,100]
    nn = len(ns)
    parameters = []
    for p in p_ls:
        for din_max in [10]:
            for rmin in rmin_ls:
                if ommin+rmin**2*din_max<=1:
                    for rmax in rmax_ls:
                        if (din_max<p-1) or (ommin+rmin**2*(p-2)+rmax**2<=1): # special check for star
                            args = (p,din_max,ommin,rmin,rmax)
                            parameters.append(args)
    ncase = len(parameters)
    measure = zeros((4,7,ncase,nn,ntrial)) # edge set difference, 0:CL, 1:hc, 2:PC, 3:PCm1/PCes
    # missing, extra, wrong direction, fdr-sk, fdr-cpdag, jaccard-sk, jaccard_cpdag
    nls = zeros((ncase,nn))
    print('run id:', run_id)
    print('number of inference:', nn*ncase*ntrial)
    print('number of SEM:', ncase*ntrial)
    print('number of graph:', ntrial)
    time_total = zeros(4)
    for t in range(ntrial):
        # generate polytree
        ntry = 0
        while 1:
            T0 = rand_polytree(p, din_max=din_max)
            ntry += 1
            if np.max(indegree(T0))==din_max:
                # print('n try', ntry)
                break
        de0,ue0 = CPDAG_polytree(T0)
        # generate SEM
        for (i,args) in enumerate(parameters):
            _,_,ommin,rmin,rmax = args
            T0_sem = gen_SEM_polytree(T0,ommin,rmin,rmax)
            # ns = [int(max(10,round(x))) for x in linspace(log(p)*8, log(p)*180, nn)]  #for small p
            # ns = [int(max(10,round(x))) for x in linspace(log(p)*150, log(p)*500, nn)]   #for large p
            # generate samples X and inference
            nls[i,:] = ns
            for (j,n) in enumerate(ns):
                # print('n='+str(n))
                X = gen_X_SEM(T0,T0_sem[0],T0_sem[1],n)
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
                    de_PC, ue_PC, runtime_R = PC_R(C,n=n,alpha=alpha_PC,mmax=-1,tag=tag_PC)
                    time_total[2] += runtime_R
                    measure[2,:,i,j,t] = measure_CPDAG(de0,ue0,de_PC,ue_PC)
                    # # PCm1
                    # de_PCm1, ue_PCm1, runtime_R = PC_R(C,n=n,alpha=alpha_PC,mmax=1,tag=tag_PCm1)
                    # time_total[3] += runtime_R
                    # measure[3,:,i,j,t] = measure_CPDAG(de0,ue0,de_PCm1,ue_PCm1)
                    # PC early stop
                    t0 = timeit.default_timer()
                    de_PCes, ue_PCes = PC_earlystop(C,n, alpha=alpha_PC) # adapted to polytree
                    t1 = timeit.default_timer()
                    time_total[3] += t1-t0
                    measure[3,:,i,j,t] = measure_CPDAG(de0,ue0,de_PCes,ue_PCes)
    tx1 = timeit.default_timer()
    total_sim_time = tx1-tx0
    print('inference time CL: ', time_total[0])
    print('inference time hc: ', time_total[1])
    print('inference time PC: ', time_total[2])
    # print('inference time PCm1: ', time_total[3])
    print('inference time PCes: ', time_total[3])
    print('total sim time: ', total_sim_time)
    with open('./data/polytree_sim_'+str(run_id)+'.npz','wb') as file1:
        np.savez(file1, p_ls=p_ls,ommin=ommin,rmin_ls=rmin_ls,rmax_ls=rmax_ls,
            ntrial=ntrial,nn=nn,parameters=parameters,
            ncase=ncase,measure=measure,nls=nls,time_total=time_total,
            total_sim_time=total_sim_time,run_id=run_id,
            flag_CL_only=flag_CL_only)
    if not flag_CL_only:
        import os
        os.remove("./data/X_"+str(tag_hc)+".csv")
        os.remove("./data/A_cpdag_"+str(tag_hc)+".csv")
        os.remove("./data/runtime_"+str(tag_hc)+".txt")
        os.remove("./data/A_cpdag_"+str(tag_PC)+".csv")
        os.remove("./data/C_"+str(tag_PC)+".csv")
        os.remove("./data/runtime_"+str(tag_PC)+".txt")
        # os.remove("./data/A_cpdag_"+str(tag_PCm1)+".csv")
        # os.remove("./data/C_"+str(tag_PCm1)+".csv")
        # os.remove("./data/runtime_"+str(tag_PCm1)+".txt")
else:
    loadall_npz('./data/polytree_sim_'+str(run_id)+'.npz')
    print('run id:', run_id)
    # print('p, din max:', p_ls[0], parameters[0][1])
    print('number of inference:', ncase*nn*ntrial)
    print('number of SEM:', ncase*ntrial)
    print('inference time CL: ', time_total[0])
    print('inference time hc: ', time_total[1])
    print('inference time PC: ', time_total[2])
    # print('inference time PCm1: ', time_total[3])
    print('inference time PCes: ', time_total[3])
    print('total sim time: ', total_sim_time)
    print('ncase:', ncase)
    print('nn:', nn)
    print('ntrial:', ntrial)

#     ninfer = ncase*nn*ntrial
#     print('inference time CL: ', round(time_total[0]/ninfer,2))
#     print('inference time hc: ', round(time_total[1]/ninfer,2))
#     print('inference time PC: ', round(time_total[2]/ninfer,2))
#     print('inference time PCes: ', round(time_total[3]/ninfer,2))


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
        p,din_max,ommin,rmin,rmax = args
        if rmin<=0.20: # select plot
        # if True:
            line_label = 'p'+str(int(p))+' dinm'+str(int(round(din_max)))+ \
                    ' rmin'+str(round(rmin,2))+' rmax'+str(round(rmax,1))
            ns = nls[i,:]

            # # exact recovery
            # fraction = np.mean(FN[:,0,i,:,:]==0,axis=-1)
            # plt.figure(fig_exact.number)
            # line = plt.plot(ns, fraction[0,:], '.-', linewidth=line_w, label='polytree:'+line_label)
            # if flag_plot_comparison:
            #     plt.plot(ns, fraction[1,:], '.--', color=line[0].get_color())
            # plt.figure(fig_exact_log.number)
            # line = plt.plot(ns/np.log(p), fraction[0,:], '.-', label='polytree:'+line_label)
            # if flag_plot_comparison:
            #     plt.plot(ns/np.log(p), fraction[1,:], '.--', color=line[0].get_color())

            # plot measure
            # measure = zeros((2,7,ncase,nn,ntrial))
            m = np.squeeze(measure[:,i_m,i,:,:])
            m_label = measure_label[i_m]
            m_label_short = measure_label_short[i_m]
            fraction = np.mean(m,axis=-1)
            f_sd = np.std(m,axis=-1) / sd_sem * 1.96
            mA = 12
            mB = 6
            # print('cpdag sd:', np.median(f_sd*sd_sem))
            if not flag_plot_nlogp:
                plt.figure(fig_edge.number)
                if flag_plot_sd:
                    line = plt.errorbar(ns, fraction[0,:], f_sd[0,:],
                            marker='.', markersize=mA, linestyle='-',label='polytree:'+line_label)
                    caps = line[1]
                    for cap in caps:
                        cap.set_markeredgewidth(0.5)
                    if flag_plot_comparison:
                        (_, caps, _) = plt.errorbar(ns, fraction[1,:], f_sd[1,:],
                            marker='^', markersize=mB, linestyle='--', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                        (_, caps, _) = plt.errorbar(ns, fraction[2,:], f_sd[2,:],
                            marker='s', markersize=mB, linestyle='--', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                        (_, caps, _) = plt.errorbar(ns, fraction[3,:], f_sd[3,:],
                            marker='s', markersize=mB, linestyle='-.', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                else:
                    line = plt.plot(ns, fraction[0,:], marker='.', markersize=mA, linestyle='-',label='polytree:'+line_label)
                    if flag_plot_comparison:
                        plt.plot(ns, fraction[1,:], marker='^', markersize=mB, linestyle='--', color=line[0].get_color())
                        plt.plot(ns, fraction[2,:], marker='s', markersize=mB, linestyle='-', color=line[0].get_color())
                        plt.plot(ns, fraction[3,:], marker='s', markersize=mB, linestyle='-.', color=line[0].get_color())
            else:
                plt.figure(fig_edge_nlogp.number)
                if flag_plot_sd:
                    line = plt.errorbar(ns/log(p), fraction[0,:], f_sd[0,:],
                            marker='.', markersize=mA, linestyle='-',label='polytree:'+line_label, capsize=capsize)
                    caps = line[1]
                    for cap in caps:
                        cap.set_markeredgewidth(0.5)
                    if flag_plot_comparison:
                        (_, caps, _) = plt.errorbar(ns/log(p), fraction[1,:], f_sd[1,:],
                            marker='^', markersize=mB, linestyle='--', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                        (_, caps, _) = plt.errorbar(ns/log(p), fraction[2,:], f_sd[2,:],
                            marker='s',markersize=mB,linestyle='--', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                        (_, caps, _) = plt.errorbar(ns/log(p), fraction[3,:], f_sd[3,:],
                            marker='s',markersize=mB,linestyle='-.', color=line[0].get_color(), capsize=capsize)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                else:
                    line = plt.plot(ns/log(p), fraction[0,:], marker='.', markersize=mA, linestyle='-',label='polytree:'+line_label)
                    if flag_plot_comparison:
                        plt.plot(ns/log(p), fraction[1,:], marker='^', markersize=mB, linestyle='--', color=line[0].get_color())
                        plt.plot(ns/log(p), fraction[2,:], marker='s', markersize=mB, linestyle='--', color=line[0].get_color())
                        plt.plot(ns/log(p), fraction[3,:], marker='s', markersize=mB, linestyle='-.', color=line[0].get_color())
    if not flag_plot_nlogp and not flag_no_label:
        plt.figure(fig_edge.number)
        plt.xlabel(r'$n$')
    elif not flag_no_label:
        plt.figure(fig_edge_nlogp.number)
        plt.xlabel(r'$n/\log(p)$')
    # plt.ylabel(m_label)
    plt.ylim([0,1])
    # plt.legend()
    if not flag_no_label: plt.title(m_label)
    # plt.title('polytree:solid, hc:dashed')
    if not flag_plot_nlogp:
        plt.savefig('./figure/polytree_sim_'+m_label_short+'_'+str(run_id), dpi=300)
    else:
        plt.savefig('./figure/polytree_sim_'+m_label_short+'_log_'+str(run_id), dpi=300)
plt.close()
