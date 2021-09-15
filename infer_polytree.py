import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint, uniform, multivariate_normal, binomial
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import (log, exp, sqrt, zeros, ones, eye, linspace, arange,
                    dot, outer, sign,diag)
from sympy.combinatorics.prufer import Prufer
from kruskal import inf_tree_m2
import networkx as nx
import timeit

# For a directed edge e = [i,j], the direction is from j to i

def is_pos_def(C):
    return np.all(eigvalsh(C) > 0)

def corr_X(X):
    # calculate sample correlation matrix
    n,p = X.shape
    C = dot(X.T, X)/n
    dC = np.diag(1/sqrt(np.diag(C)))
    C = dot(dot(dC, C), dC)
    return C

def C2R(C):
    d = np.diag(C)
    d2 = 1/np.sqrt(d)
    R = np.dot(np.dot(np.diag(d2), C), np.diag(d2))
    return R

def rand_tree(p, dmax=1) :
    # return a list of edges
    # dmax is the lower bound of the maximum degree
    seq = zeros(p-2,dtype=int)
    if dmax>1:
        i_max = randint(p)
        seq[0:dmax-1] = i_max
    seq[dmax-1:(p-2)] = randint(p, size=p-2-dmax+1)
    seq =np.random.permutation(seq)
    T0 = Prufer.to_tree(seq)
    return T0

def indegree(T):
    p = p_of_tree(T)
    d = zeros(p,dtype=int)
    for e in T:
        d[e[0]] += 1
    return d

def tree_polytree(T0, din_max=1):
    # return a list of directed edges
    # randomly assign direction
    # din_max is the intended max in-degree (attained at a node, but may not be maximum if din_max is small)
    # din_min is set to 1 unless for din_max=p-1 (star)
    # when din_max=1, use the rooted tree
    if din_max==1:
        return tree_rooted(T0)
    else:
        import random
        p = p_of_tree(T0)
        d = zeros(p,dtype=int)
        for e in T0:
            d[e[0]] += 1
            d[e[1]] += 1
        id_ls = np.arange(p)
        id_ls = id_ls[d>=din_max]
        assert len(id_ls)>0, 'din-max attainable'
        id_max = id_ls[np.random.randint(len(id_ls))]
        T1 = []
        ue = T0.copy() # undirected edges
        random.shuffle(ue)
        din_max_remain = din_max
        i_remove = []
        # orient edges around id_max
        for (i,e) in enumerate(ue):
            if e[0]==id_max or e[1]==id_max:
                if e[0]==id_max:
                    j = e[1]
                else:
                    j = e[0]
                if din_max_remain > 0:
                    T1.append(np.array([id_max, j]))
                    din_max_remain -= 1
                else:
                    T1.append(np.array([j,id_max]))
                i_remove.append(i)
        for i in sorted(i_remove, reverse=True):
            del ue[i]
        # set din_min=1 unless din_max=p-1 (star)
        if len(ue) > 0:
            random.shuffle(ue)
            e = ue[0]
            id_min = e[0]
            T1.append(np.array([e[0],e[1]]))
            for e in ue[1:]:
                if e[0]==id_min or e[1]==id_min:
                    if e[0]==id_min:
                        j = e[1]
                    else:
                        j = e[0]
                    T1.append(np.array([j, id_min]))
                else:
                    T1.append(np.array(e)[(np.random.permutation(2)).astype(int)])
        return T1


def tree_rooted(T0,iroot=-1):
    p = p_of_tree(T0)
    if iroot == -1:
        iroot = randint(p)
    tf_node_visit = zeros(p,dtype=bool)
    tf_edge_direct = zeros(p,dtype=bool)
    T = []
    tf_node_visit[iroot] = True
    for t in range(p):
        flag_update = False
        for i,e in enumerate(T0):
            if not tf_edge_direct[i] and (
                tf_node_visit[e[0]] or tf_node_visit[e[1]]):
                if tf_node_visit[e[0]]:
                    T.append([e[1],e[0]])
                    tf_node_visit[e[1]] = True
                else:
                    T.append([e[0],e[1]])
                    tf_node_visit[e[0]] = True
                tf_edge_direct[i] = True
                flag_update = True
        if not flag_update:
            break
    assert len(T) == p-1
    return T


def p_of_tree(T):
    p = 0
    for e in T:
        p = max([p,e[0],e[1]])
    return p+1

def CPDAG_partial_v(de,ue,tf_vnode):
    # partially oriented (of the vnode) and a list of vnode
    # return two lists of directed edges and undirected eges
    p = len(tf_vnode)
    for t in range(p):
        i_remove = []
        for i,e in enumerate(ue):
            if tf_vnode[e[1]]:
                tf_vnode[e[0]] = True
                i_remove.append(i)
            elif tf_vnode[e[0]]:
                tf_vnode[e[1]] = True
                i_remove.append(i)
                ue[i] = [e[1], e[0]]
        if not i_remove:
            break
        else:
            for i in i_remove:
                de.append(ue[i])
            for i in sorted(i_remove, reverse=True):
                del ue[i]
    return de, ue

def CPDAG_polytree(T):
    # return two lists of directed edges and undirected eges
    p = p_of_tree(T)
    din = zeros(p)
    for e in T:
        din[e[0]] += 1
    tf_vnode = zeros(p,dtype=bool) # v-node and decendent
    tf_vnode[din>=2] = True
    de = []
    ue = T.copy()
    i_remove = []
    for i,e in enumerate(ue):
        if tf_vnode[e[0]]:
            i_remove.append(i)
    for i in i_remove:
        de.append(ue[i])
    for i in sorted(i_remove, reverse=True):
        del ue[i]
    de, ue = CPDAG_partial_v(de,ue,tf_vnode)
    return de, ue

def rand_polytree(p, din_max=1):
    # return a list of directed edges e=(i,j), i<-j
    # din_max is the max in-degree (attained)
    T0 = rand_tree(p,dmax=din_max)
    return tree_polytree(T0, din_max=din_max)

def rand_DAG(p,s_edge):
    A = binomial(1,s_edge,size=(p,p))
    # lower triagular, node index are topological orders
    A[np.triu_indices(p,k=-1)] = 0
    G0 = []
    for j in range(p):
        for i in range(j+1,p):
            if A[i,j] == 1:
                G0.append([i,j])
    return A,G0

def ncomponent_A(A):
    # A is symmetric adjaceny matrix
    p,_ = A.shape
    d = np.sum(A,axis=0)
    L = diag(d) - A
    eig_L = eigvalsh(L)
    return np.sum(np.abs(eig_L)<1e-6)

def rand_DAG_nedge(p,n_edge):
    na = int(p*(p-1)/2)
    a = zeros(na)
    a[np.random.choice(range(na),n_edge,replace=False)] = 1
    A = zeros((p,p))
    A[np.tril_indices(p,k=-1)] = a
    # lower triagular, node index are topological orders
    G0 = []
    for j in range(p):
        for i in range(j+1,p):
            if A[i,j] == 1:
                G0.append([i,j])
    return A,G0

def is_connected(A):
    # A is undirected adjacency matrix
    p,_ = A.shape
    A += eye(p)
    Ap = np.linalg.matrix_power(A,p-1)
    return all(Ap[0,:]>0)

def rand_DAG_nedge_connected(p,n_edge):
    na = int(p*(p-1)/2)
    flag_connected = False
    ntry = 0
    while not flag_connected:
        a = zeros(na)
        a[np.random.choice(range(na),n_edge,replace=False)] = 1
        A = zeros((p,p))
        A[np.tril_indices(p,k=-1)] = a
        # lower triagular, node index are topological orders
        ntry += 1
        print(ntry)
        if is_connected(A+A.T):
            flag_connected = True
    G0 = []
    for j in range(p):
        for i in range(j+1,p):
            if A[i,j] == 1:
                G0.append([i,j])
    # print('ntry for connected:', ntry)
    return A,G0

def rand_DAG_nedge_giant_component(p,n_edge):
    A,G = rand_DAG_nedge(p,int(round(n_edge)))
    # A,G = rand_DAG(p,n_edge/(p*(p-1)/2)) # s_edge
    G1 = nx.from_numpy_matrix(A+A.T)
    largest_cc = list(max(nx.connected_components(G1), key=len))
    Ac = np.copy(A)
    Ac = Ac[largest_cc,:]
    Ac = Ac[:,largest_cc]
    def ismember(i,x):
        matched = False
        for j in x:
            if j==i:
                matched = True
        return matched
    # Gc = [e for e in G if ismember(e[0],largest_cc) and ismember(e[1],largest_cc)]
    # relabe G
    p,_ = Ac.shape
    Gnew = []
    for j in range(p):
        for i in range(j+1,p):
            if Ac[i,j] == 1:
                Gnew.append([i,j])
    return Ac,Gnew

def rand_DAG_nedge_giant_component_target(p_target,p_use,ne_use):
    flag_matched = False
    ntry = 0
    while not flag_matched:
        ntry += 1
        A,G = rand_DAG_nedge_giant_component(p_use,ne_use)
        p,_ = A.shape
        if p == p_target:
            flag_matched = True
            break
    print('ntry:', ntry)
    return A,G

def gen_SEM_DAG(p,G0, positive_weight=False):
    n_edge = len(G0)
    b = uniform(low=0.1,high=1.0, size=n_edge)
    if not positive_weight:
        b = b * 2*(binomial(1,0.5,size=n_edge)-0.5)
    omega = ones(p)
    B = zeros((p,p))
    for i,e in enumerate(G0):
        B[e[1],e[0]] = b[i]
    # standardize b and omega to have unit var for X_j
    C = np.linalg.inv(eye(p)-B)
    C = C.T @ diag(omega) @ C
    d = sqrt(diag(C))
    B = diag(d) @ B @ diag(1/d)
    omega = omega/d**2
    for i,e in enumerate(G0):
        b[i] = B[e[1],e[0]]
    return b,omega

def CPDAG_DAG(p,G0,tag=-1):
    if tag<0:
        tag = np.random.randint(1000,9999)
    A = edge2dA(p,G0,[])
    A_cpdag = A_DAG_CPDAG(A,tag=tag) # using R function
    de,ue = dA2edge(A_cpdag)
    return de,ue

def gen_X_tree(T,p,n):
    # generate normal samples from tree
    r0 = 0.1
    r1 = 0.5  #0.1,0.5
    s0 = 1
    s1 = 1
    flag_pd = False
    while not flag_pd:
        P = zeros((p,p))
        P[eye(p)>0] = 1/uniform(s0, s1, size=p)**2
        for e in T:
            i,j = e
            rij = uniform(r0,r1)*sign(randn())
            # rij = uniform(r0,r1)
            P[i,j] = -rij*sqrt(P[i,i]*P[j,j])
            P[j,i] = P[i,j]
        if is_pos_def(P):
            flag_pd = True
    X = multivariate_normal(zeros(p), inv(P), size=n)
    return X

def gen_SEM_polytree(T,ommin,rmin,rmax):
    # return b(ordered by T edges), diag(Om), standardized s.t. C is corr matrix
    # C = (I-B^T)^{-1} Om (I-B)^{-1}
    import random
    p = p_of_tree(T)
    din = zeros(p)
    for e in T:
        i,j = e
        din[i] += 1
    id_rmax_all = [i for i in range(p) if din[i]>0
                and (rmin**2*(din[i]-1)+rmax**2<=1-ommin)] # nodes that can attain rmax
    id_rmax = random.choice(id_rmax_all)
    id_rmin_all = [i for i in range(p) if din[i]>0 and i!= id_rmax]
    id_rmin = random.choice(id_rmin_all)
    # sequentially sample each incoming edge
    # r_e**2-rmin**2 can be sampled using the order statistics of unifom distribution
    # over 1-ommin-din*rmin**2
    # the first edge of id_rmin/max is fixed to attain rmin,rmax
    order_e = list(range(p-1)) # random order to choose b for edges
    random.shuffle(order_e)
    din0 = np.copy(din) # save original in-degree
    v = 1-ommin-din0*rmin**2 # amount of Delta rij**2 left
    b = zeros(p-1)
    for k in order_e:
        e = T[k]
        i,j = e
        if (i==id_rmax and din[i]==din0[i]) or (i==id_rmin and din[i]==din0[i]):
            if i==id_rmax:
                b[k] = rmax * sign(randn())
                v[i] -= rmax**2 - rmin**2
                din[i] -= 1
            if i==id_rmin:
                b[k] = rmin * sign(randn())
                din[i] -= 1
        else:
            b[k] = min(rmax, sqrt(rmin**2 + np.random.beta(1,din[i])*v[i])) * sign(randn())
            v[i] -= b[k]**2 - rmin**2
            din[i] -= 1
    omega = v + ommin
    # assert (omega>=ommin).all(), 'omega min'
    if not (omega+1e-9>=ommin).all():
        print('ommin error')
        print(id_rmin,id_rmax)
        print(rmin,rmax)
        print(v)
        plot_polytree(T, 'debug_polytree')
        # assert (omega>=ommin).all()
    if not(np.min(np.abs(b))==rmin):
        print('rmin error')
        print(id_rmin,id_rmax)
        print(rmin,rmax)
        print(v)
        print(b)
        print(T)
        plot_polytree(T, 'debug_polytree')
        assert 0
    if not(np.max(np.abs(b))==rmax):
        print('rmax error')
        print(id_rmin,id_rmax)
        print(rmin,rmax)
        print(v)
        print(b)
        print(T)
        plot_polytree(T, 'debug_polytree')
        # assert (omega>=ommin).all()
    # assert np.min(np.abs(b))==rmin, 'rmin'
    # assert np.max(np.abs(b))==rmax, 'rmax'
    return b,omega

def gen_X_SEM(T,b,omega,n):
    # generate normal samples from polytree SEM
    # X is n by p
    p = len(omega)
    B = zeros((p,p))
    for i,e in enumerate(T):
        B[e[1],e[0]] = b[i]

    # C is not needed, for debug
    C = np.linalg.inv(eye(p)-B)
    C = C.T @ diag(omega) @ C
    # C = np.linalg.solve(eye(p)-B.T, diag(sqrt(omega)))
    # C = C @ C.T
    assert np.allclose(diag(C), ones(p)), 'correlation scaling' # check wright formula

    X = randn(p,n)
    X = np.linalg.solve(eye(p)-B.T, diag(sqrt(omega)) @ X)
    X = X.T
    return X


# def inf_tree(C):
#     p,_ = C.shape
#     C = -C**2
#     C[eye(p)>0] = 0
#     G = nx.from_numpy_matrix(C)
#     T_nx = nx.minimum_spanning_tree(G)
#     T = sorted(T_nx.edges())
#     return T


def v_input(C,rth):
    # orient v-structure for each pair
    p,_ = C.shape
    tf_iv = zeros(p,dtype=bool)
    for i in range(p):
        for j in range(i,p):
            if abs(C[i,j]) < rth:
                tf_iv[i] = True
                tf_iv[j] = True
    iv = np.arange(p)
    iv = iv[tf_iv]
    iv = iv.tolist()
    return iv


def find_vnode(T,C,rth):
    p,_ = C.shape
    de = []
    ue = T.copy()
    deg = zeros(p)
    for e in T:
        deg[e[0]] += 1
        deg[e[1]] += 1
    tf_vnode = zeros(p, dtype=bool)
    for i in range(p):
        if deg[i] >= 2:
            i_connect = []
            for e in T:
                if e[0] == i:
                    i_connect.append(e[1])
                if e[1] == i:
                    i_connect.append(e[0])
            i_connect = np.sort(np.array(i_connect))
            C_connect = C[:,i_connect]
            C_connect = C_connect[i_connect,:]
            i_iv = v_input(C_connect,rth)
            # print('node', i, ', parent:',i_connect[i_iv])
            if i_iv:
                tf_vnode[i] = True
                iv = i_connect[i_iv]
                i_remove = []
                for k,e in enumerate(ue):
                    if e[0]==i or e[1]==i:
                        if e[0]==i:
                            j = e[1]
                        else:
                            j = e[0]
                        if j in iv:
                            de.append(np.array([i,j]))
                            i_remove.append(k)
                for k in sorted(i_remove, reverse=True):
                    del ue[k]
    return de, ue, tf_vnode



def inf_polytree(C,n,alpha=0.05):
    # return CPDAG
    # t = r/sqrt((1-r^2)/(n-2)), df=n-2
    from scipy.stats import t as tdist
    tc = tdist.ppf(1-alpha/2, df=n-2)
    rth = np.sqrt(1 - 1/(1+tc**2/(n-2)))
    T = inf_tree_m2(C)
    p,_ = C.shape
    de, ue, tf_vnode = find_vnode(T,C,rth=rth)
    de, ue = CPDAG_partial_v(de,ue,tf_vnode)
    return de, ue


def diff_tree(T0,T):
    T0 = [np.sort(e) for e in T0]
    T = [np.sort(e) for e in T]
    d = 0
    for e in T0:
        flag_match=False
        for f in T:
            if (e[0]==f[0]) and (e[1]==f[1]):
                flag_match=True
                break
        if not flag_match:
            d += 1
    return d


def diff_polytree(de1,ue1,de2,ue2):
    d = 0
    for e in de1:
        flag_match=False
        for f in de2:
            if (e[0]==f[0]) and (e[1]==f[1]):
                flag_match=True
                break
        if not flag_match:
            d += 1
    for e in ue1:
        flag_match=False
        for f in ue2:
            m1 = (e[0]==f[0]) and (e[1]==f[1])
            m2 = (e[0]==f[1]) and (e[1]==f[0])
            if m1 or m2:
                flag_match=True
                break
        if not flag_match:
            d += 1
    return d

def measure_CPDAG(de1,ue1,de2,ue2):
    # compare difference of two cpdag's
    # missing, extra, wrong direction
    nedge1 = len(de1)+len(ue1)
    nedge2 = len(de2)+len(ue2)
    missing = 0
    extra = 0
    wrongd = 0
    match = 0
    for e in de1:
        flag_match = False
        for f in de2:
            if e[0]==f[0] and e[1]==f[1]:
                flag_match = True
                match += 1
                break
            elif e[0]==f[1] and e[1]==f[0]:
                flag_match = True
                wrongd += 1
                break
        if not flag_match:
            for f in ue2:
                if (e[0]==f[0] and e[1]==f[1]) or (e[0]==f[1] and e[1]==f[0]):
                    flag_match = True
                    wrongd += 1
                    break
        if not flag_match:
            missing += 1
    for e in ue1:
        flag_match = False
        for f in ue2:
            if (e[0]==f[0] and e[1]==f[1]) or (e[0]==f[1] and e[1]==f[0]):
                flag_match=True
                match += 1
                break
        if not flag_match:
            for f in de2:
                if (e[0]==f[0] and e[1]==f[1]) or (e[0]==f[1] and e[1]==f[0]):
                    flag_match=True
                    wrongd += 1
                    break
        if not flag_match:
            missing += 1
    # assert missing + wrongd + match == nedge1
    extra = nedge2 - (wrongd + match)
    fdr_sk = extra / nedge2
    fdr_cpdag = (extra+wrongd) / nedge2
    jaccard_sk = (match+wrongd) / (nedge2+nedge1 - (match+wrongd))
    jaccard_cpdag = (match) / (nedge2+nedge1 - match)
    return missing, extra, wrongd, fdr_sk, fdr_cpdag, jaccard_sk, jaccard_cpdag

def dA2edge(A):
    # convert dgraph adj matrix to de and ue
    p,_ = A.shape
    de = []
    ue = []
    for i in range(p):
        for j in range(i+1,p):
            if A[i,j] == 1 and A[j,i] == 1:
                ue.append([i,j])
            elif A[i,j] == 1:
                de.append([i,j])
            elif A[j,i] == 1:
                de.append([j,i])
    return de, ue

def edge2dA(p,de,ue):
    # convert de and ue to dgraph adj matrix
    A = zeros((p,p))
    for e in de:
        i,j = e
        A[i,j] = 1
    for e in ue:
        i,j = e
        A[i,j] = 1
        A[j,i] = 1
    return A

def hc_R(X,tag=1000):
    # alpha=0.05
    import csv
    import subprocess
    import sys
    np.savetxt("./data/X_"+str(tag)+".csv", X, delimiter=",")
    try:
        subprocess.check_call("Rscript hc.R "+str(tag), shell=True)
    except:
        assert 0
    with open("./data/A_cpdag_"+str(tag)+".csv") as csvfile:
        data = list(csv.reader(csvfile))
    A0 = np.array(data)
    A = A0[1:,:]
    A = A[:,1:]
    A = A.astype(int) # adj matrix
    de, ue = dA2edge(A)
    with open("./data/runtime_"+str(tag)+".txt") as txtfile:
        lines = txtfile.readlines()
        runtime = float(lines[0])
    return de, ue, runtime

def PC_R(C,n, alpha=0.1,mmax=-1,tag=1000):
    # calling R script to run PC algorithm
    import csv
    import subprocess
    import sys
    np.savetxt("./data/C_"+str(tag)+".csv", C, delimiter=",")
    try:
        subprocess.check_call("Rscript PC.R "+str(tag)+" "+str(n)
            +" "+str(alpha)+" "+str(mmax), shell=True)
    except:
        assert 0
    with open("./data/A_cpdag_"+str(tag)+".csv") as csvfile:
        data = list(csv.reader(csvfile))
    A0 = np.array(data)
    A = A0[1:,:]
    A = A[:,1:]
    A = A.astype(int) # adj matrix
    de, ue = dA2edge(A)
    with open("./data/runtime_"+str(tag)+".txt") as txtfile:
        lines = txtfile.readlines()
        runtime = float(lines[0])
    return de, ue, runtime

def plot_polytree(T, filename):
    from networkx.drawing.nx_pydot import graphviz_layout
    fig = plt.figure()
    G = nx.DiGraph()
    for e in T:
        G.add_edge(e[1],e[0],weight=1)
    fig.clf()
    # pos = nx.spring_layout(G,iterations=200)
    pos = graphviz_layout(G, prog="twopi")
    nx.draw(G,pos,with_labels=True)
    fig.savefig('./figure/'+filename+'.png',dpi=200)
    plt.close()

def plot_CPDAG(de,ue,filename,p=0,node_label=[],pos=[],fig_size=(8,8)):
    node_size = 400
    font_size = 5
    from networkx.drawing.nx_pydot import graphviz_layout
    if p == 0:
        p = len(de)+len(ue)+1
    if len(node_label)==0:
        node_label = list(range(p))
    G = nx.DiGraph()
    for i in range(p):
        G.add_node(i)
    for e in de+ue:
        G.add_edge(e[1],e[0],weight=1)
    # pos = nx.spring_layout(G,iterations=200)
    de_rev = [[e[1],e[0]] for e in de]
    # pos = graphviz_layout(G, prog="twopi")
    if not pos:
        pos = graphviz_layout(G, prog="dot")
        # pos = graphviz_layout(G, prog="twopi")
    label_dict = {i:lab for i,lab in enumerate(node_label) if i in pos}
    fig = plt.figure(figsize=fig_size)
    fig.clf()
    nx.draw_networkx_edges(G, pos, edgelist=de_rev, arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=ue, edge_color='k', arrows=False,
        width=2)
    nx.draw_networkx_nodes(G, pos, node_size=node_size,node_color='w',edgecolors='k')
    nx.draw_networkx_labels(G, pos,labels=label_dict, font_size=font_size)
    fig.savefig('./figure/'+filename+'.png',dpi=400)
    plt.close()
    return pos

def plot_compare_CPDAG(de, ue, de1, ue1, filename,p=0,node_label=[],pos=[],fig_size=(8,8)):
    # use de,ue as the true graph
    node_size = 400
    font_size = 5
    from networkx.drawing.nx_pydot import graphviz_layout
    if p == 0:
        p = len(de)+len(ue)+1
    if len(node_label)==0:
        node_label = list(range(p))
    # G pos
    if not pos:
        G = nx.DiGraph()
        for i in range(p):
            G.add_node(i)
        for e in de+ue:
            G.add_edge(e[1],e[0],weight=1)
        pos = graphviz_layout(G, prog="dot")
    de_missing = [] # de need to be rev
    ue_missing = []
    de1_match = []
    de1_extra = []
    de1_wrongd = []  # in G1's direction
    ue1_match = []
    ue1_extra = []
    ue1_wrongd = []
    for e in de:
        flag_match = False
        for f in de1+ue1:
            if (e[0]==f[0] and e[1]==f[1]) or (e[0]==f[1] and e[1]==f[0]):
                flag_match = True
                break
        if not flag_match:
            de_missing.append([e[1],e[0]])
    for e in ue:
        flag_match = False
        for f in de1+ue1:
            if (e[0]==f[0] and e[1]==f[1]) or (e[0]==f[1] and e[1]==f[0]):
                flag_match = True
                break
        if not flag_match:
            ue_missing.append(e)
    # G1
    for e in de1:
        flag_match = False
        for f in de:
            if e[0]==f[0] and e[1]==f[1]:
                flag_match = True
                de1_match.append([e[1],e[0]])
                break
            elif e[0]==f[1] and e[1]==f[0]:
                flag_match = True
                de1_wrongd.append([e[1],e[0]])
                break
        if not flag_match:
            for f in ue:
                if (e[0]==f[0] and e[1]==f[1]) or (e[0]==f[1] and e[1]==f[0]):
                    flag_match = True
                    de1_wrongd.append([e[1],e[0]])
                    break
        if not flag_match:
            de1_extra.append([e[1],e[0]])
    for e in ue1:
        flag_match = False
        for f in ue:
            if (e[0]==f[0] and e[1]==f[1]) or (e[0]==f[1] and e[1]==f[0]):
                flag_match=True
                ue1_match.append(e)
                break
        if not flag_match:
            for f in de:
                if (e[0]==f[0] and e[1]==f[1]) or (e[0]==f[1] and e[1]==f[0]):
                    flag_match=True
                    ue1_wrongd.append(e)
                    break
        if not flag_match:
            ue1_extra.append(e)
    # G1 graph
    G1 = nx.DiGraph()
    for i in range(p):
        G1.add_node(i)
    all_edge = de_missing + ue_missing + de1 + ue1
    for e in all_edge:
        G1.add_edge(e[1],e[0],weight=1)
    # pos = graphviz_layout(G1, prog="dot")
    label_dict = {i:lab for i,lab in enumerate(node_label) if i in pos}
    fig = plt.figure(figsize=fig_size)
    fig.clf()
    if de_missing:
        nx.draw_networkx_edges(G1, pos, edgelist=de_missing, arrows=True, edge_color='r')
    if ue_missing:
        nx.draw_networkx_edges(G1, pos, edgelist=ue_missing, arrows=False, width=2, edge_color='r')
    if de1_match:
        nx.draw_networkx_edges(G1, pos, edgelist=de1_match, arrows=True, edge_color='k')
    if de1_extra:
        nx.draw_networkx_edges(G1, pos, edgelist=de1_extra, arrows=True, edge_color='b')
    if de1_wrongd:
        nx.draw_networkx_edges(G1, pos, edgelist=de1_wrongd, arrows=True, edge_color='g')
    if ue1_match:
        nx.draw_networkx_edges(G1, pos, edgelist=ue1_match, arrows=False, width=2, edge_color='k')
    if ue1_extra:
        nx.draw_networkx_edges(G1, pos, edgelist=ue1_extra, arrows=False, width=2, edge_color='b')
    if ue1_wrongd:
        nx.draw_networkx_edges(G1, pos, edgelist=ue1_wrongd, arrows=False, width=2, edge_color='g')
    nx.draw_networkx_nodes(G1, pos, node_size=node_size,node_color='w',edgecolors='k')
    nx.draw_networkx_labels(G1, pos,labels=label_dict, font_size=font_size)
    fig.savefig('./figure/'+filename+'.png',dpi=400)
    plt.close()
    return pos


def A_DAG_CPDAG(A,tag=-1):
    # use bnlearn package in R
    import csv
    import subprocess
    import sys
    if tag<0:
        tag = np.random.randint(1000,9999)
    np.savetxt("./data/A_"+str(tag)+".csv", A, delimiter=",")
    try:
        subprocess.check_call("Rscript A_DAG_CPDAG.R "+str(tag), shell=True)
    except:
        assert 0
    with open("./data/A_cpdag_"+str(tag)+".csv") as csvfile:
        data = list(csv.reader(csvfile))
    A0 = np.array(data)
    A_CPDAG = A0[1:,:]
    A_CPDAG = A_CPDAG[:,1:]
    A_CPDAG = A_CPDAG.astype(int) # adj matrix
    import os
    os.remove("./data/A_"+str(tag)+".csv")
    os.remove("./data/A_cpdag_"+str(tag)+".csv")
    return A_CPDAG
