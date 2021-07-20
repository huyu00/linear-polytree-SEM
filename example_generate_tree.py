import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint, uniform, multivariate_normal
from numpy.linalg import eigvals, eigvalsh, inv
from numpy import (log, exp, sqrt, zeros, ones, eye, linspace, arange,
                    dot, outer, sign,diag)
import networkx as nx
import timeit

from infer_polytree import rand_tree, rand_polytree, tree_rooted

# # example undirected tree
# p = 8
# dmax = 7 # lower bound of max degree
# fig = plt.figure()
# for t in range(5):
#     T0 = rand_tree(p,dmax=dmax)
#     G = nx.Graph()
#     G.add_edges_from(T0)
#     fig.clf()
#     nx.draw(G,with_labels=True)
#     fig.savefig('./figure/undirected_tree_'+str(t)+'.png',dpi=200)


# # example directed tree
# p = 8
# din_max = 4 # attained
# from networkx.drawing.nx_pydot import graphviz_layout
# fig = plt.figure()
# for t in range(3):
#     T1 = rand_polytree(p,din_max=din_max)
#     G = nx.DiGraph()
#     for e in T1:
#         G.add_edge(e[1],e[0],weight=1)
#     fig.clf()
#     # pos = nx.spring_layout(G,iterations=200)
#     pos = graphviz_layout(G, prog="twopi")
#     nx.draw(G,pos,with_labels=True)
#     fig.savefig('./figure/directed_tree_'+str(t)+'.png',dpi=200)


# # example rooted tree, din_max=1
# p = 10
# from networkx.drawing.nx_pydot import graphviz_layout
# fig = plt.figure()
# for t in range(3):
#     T0 = rand_tree(p, dmax=3)
#     T1 = tree_rooted(T0)
#     G = nx.DiGraph()
#     for e in T1:
#         G.add_edge(e[1],e[0],weight=1)
#     fig.clf()
#     # pos = nx.spring_layout(G,iterations=200)
#     pos = graphviz_layout(G, prog="twopi")
#     nx.draw(G,pos,with_labels=True)
    # fig.savefig('./figure/rooted_tree_'+str(t)+'.png',dpi=200)
