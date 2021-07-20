import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint, uniform, multivariate_normal
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import (log, exp, sqrt, zeros, ones, eye, linspace, arange,
                    dot, outer, sign,diag)
from sympy.combinatorics.prufer import Prufer
from kruskal import inf_tree_m2
import networkx as nx
from infer_polytree import (rand_polytree, gen_X_SEM,
        gen_SEM_polytree, corr_X, inf_ploytree, plot_polytree,
        CPDAG_polytree, hc_R,
        indegree, measure_CPDAG)

# needs to be defined locally not imported
def loadall_npz(file):
    with np.load(file) as data:
        for var in data:
            globals()[var] = data[var]


import random
# random.seed(1)
# np.random.seed(1)
import timeit
# draw random tree at each trial

flag_load_data = True
run_id = 5
# pre-computed data:
# run_id 5: p=100,din_max=10, ntrial=100, alpha = 0.1
# run_id 6: p=100,din_max=40, ntrial=100, alpha = 0.1
alpha = 0.1
tag = np.random.randint(1000,9999) # avoid file access collision for multiple sessions
flag_CL_only = False

if not flag_load_data:
    tx0 = timeit.default_timer()
    p_ls = [100]
    ommin = 0.1
    rmin_ls = [0.05, 0.1, 0.15, 0.3]
    rmax_ls = [0.8]
    assert max(rmax_ls)**2 <= 1-ommin
    ntrial = 100 # set of random sem parameters and X samples
    ns = [50,100,200,400,600,800,1000]
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
    measure = zeros((2,7,ncase,nn,ntrial)) # edge set difference, 0:CL, 1:hc;
    # missing, extra, wrong direction, fdr-sk, fdr-cpdag, jaccard-sk, jaccard_cpdag
    nls = zeros((ncase,nn))
    print('run id:', run_id)
    print('number of inference:', ncase*nn*ntrial)
    print('number of SEM:', ncase*ntrial)
    time_total = zeros(2)
    for (i,args) in enumerate(parameters):
        p,din_max,ommin,rmin,rmax = args
        for t in range(ntrial):
            ntry = 0
            while 1:
                T0 = rand_polytree(p, din_max=din_max)
                ntry += 1
                if np.max(indegree(T0))==din_max:
                    # print('n try', ntry)
                    break
            de0,ue0 = CPDAG_polytree(T0)
            T0_sem = gen_SEM_polytree(T0,ommin,rmin,rmax)
            # ns = [int(max(10,round(x))) for x in linspace(log(p)*8, log(p)*180, nn)]  #for small p
            # ns = [int(max(10,round(x))) for x in linspace(log(p)*150, log(p)*500, nn)]   #for large p
            nls[i,:] = ns
            for (j,n) in enumerate(ns):
                X = gen_X_SEM(T0,T0_sem[0],T0_sem[1],n)
                C = corr_X(X)
                # CL
                t0 = timeit.default_timer()
                de,ue = inf_ploytree(C,n, alpha=alpha)
                t1 = timeit.default_timer()
                time_total[0] += t1-t0
                measure[0,:,i,j,t] = measure_CPDAG(de0,ue0,de,ue)
                if not flag_CL_only:
                    # hc
                    de_hc, ue_hc, runtime_R = hc_R(X, tag=tag)
                    time_total[1] += runtime_R
                    measure[1,:,i,j,t] = measure_CPDAG(de0,ue0,de_hc,ue_hc)
    tx1 = timeit.default_timer()
    total_sim_time = tx1-tx0
    print('inference time CL: ', time_total[0])
    print('inference time hc: ', time_total[1])
    print('total sim time: ', total_sim_time)
    with open('./data/polytree_sim_'+str(run_id)+'.npz','wb') as file1:
        np.savez(file1, p_ls=p_ls,ommin=ommin,rmin_ls=rmin_ls,rmax_ls=rmax_ls,
            ntrial=ntrial,nn=nn,parameters=parameters,
            ncase=ncase,measure=measure,nls=nls,time_total=time_total,
            total_sim_time=total_sim_time,run_id=run_id,
            flag_CL_only=flag_CL_only)
    if not flag_CL_only:
        import os
        os.remove("./data/X_"+str(tag)+".csv")
        os.remove("./data/A_cpdag_"+str(tag)+".csv")
        os.remove("./data/runtime_"+str(tag)+".txt")
else:
    loadall_npz('./data/polytree_sim_'+str(run_id)+'.npz')
    print('run id:', run_id)
    print('number of inference:', ncase*nn*ntrial)
    print('number of SEM:', ncase*ntrial)
    print('inference time CL: ', time_total[0])
    print('inference time hc: ', time_total[1])
    print('total sim time: ', total_sim_time)
    print('ncase:', ncase)
    print('nn:', nn)
    print('ntrial:', ntrial)




flag_plot_hc = True
if flag_CL_only:
    flag_plot_hc = False
id_plot_measure = [3,4,5,6]
measure_label_short = ["miss", "extra","wrong-d","fdr-sk", "fdr-cpdag", "jac-sk","jac-cpdag"]
measure_label = ["missing", "extra","wrong direction",
    "FDR skeleton", "FDR CPDAG", "Jaccard index skeleton","Jaccard index CPDAG"]
flag_plot_sd = False
sd_sem = sqrt(ntrial) # plot standard error of the mean
# sd_sem = 1 # plot standard deviation
line_w = 1
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
            # if flag_plot_hc:
            #     plt.plot(ns, fraction[1,:], '.--', color=line[0].get_color())
            # plt.figure(fig_exact_log.number)
            # line = plt.plot(ns/np.log(p), fraction[0,:], '.-', label='polytree:'+line_label)
            # if flag_plot_hc:
            #     plt.plot(ns/np.log(p), fraction[1,:], '.--', color=line[0].get_color())

            # plot measure
            # measure = zeros((2,7,ncase,nn,ntrial))
            m = np.squeeze(measure[:,i_m,i,:,:])
            m_label = measure_label[i_m]
            m_label_short = measure_label_short[i_m]
            fraction = np.mean(m,axis=-1)
            f_sd = np.std(m,axis=-1) / sd_sem
            # print('cpdag sd:', np.median(f_sd*sd_sem))
            if not flag_plot_nlogp:
                plt.figure(fig_edge.number)
                if flag_plot_sd:
                    line = plt.errorbar(ns, fraction[0,:], f_sd[0,:],
                            marker='.', linestyle='-',label='polytree:'+line_label)
                    caps = line[1]
                    for cap in caps:
                        cap.set_markeredgewidth(0.5)
                    if flag_plot_hc:
                        plt.errorbar(ns, fraction[1,:], f_sd[1,:],
                            marker='.',linestyle='--', color=line[0].get_color())
                        (_, caps, _) = plt.errorbar(ns, fraction[1,:], f_sd[1,:],
                            marker='.',linestyle='--', color=line[0].get_color(), capsize=0)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                else:
                    line = plt.plot(ns, fraction[0,:], marker='.', linestyle='-',label='polytree:'+line_label)
                    if flag_plot_hc:
                        plt.plot(ns, fraction[1,:], marker='.', linestyle='--', color=line[0].get_color())
            else:
                plt.figure(fig_edge_nlogp.number)
                if flag_plot_sd:
                    line = plt.errorbar(ns/log(p), fraction[0,:], f_sd[0,:],
                            marker='.', linestyle='-',label='polytree:'+line_label, capsize=0)
                    caps = line[1]
                    for cap in caps:
                        cap.set_markeredgewidth(0.5)
                    if flag_plot_hc:
                        (_, caps, _) = plt.errorbar(ns/log(p), fraction[1,:], f_sd[1,:],
                            marker='.',linestyle='--', color=line[0].get_color(), capsize=0)
                        for cap in caps:
                            cap.set_markeredgewidth(0.5)
                else:
                    line = plt.plot(ns/log(p), fraction[0,:], marker='.', linestyle='-',label='polytree:'+line_label)
                    if flag_plot_hc:
                        plt.plot(ns/log(p), fraction[1,:], marker='.', linestyle='--', color=line[0].get_color())
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
        plt.savefig('./figure/polytree_sim_'+m_label_short+'_'+str(run_id), dpi=300)
    else:
        plt.savefig('./figure/polytree_sim_'+m_label_short+'_log_'+str(run_id), dpi=300)
plt.close()
