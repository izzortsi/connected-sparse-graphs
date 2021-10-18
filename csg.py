# %%
from graph_tool.all import Graph
import graph_tool.all as gt
import numpy as np
import numpy.random as npr

# from numpy.linalg import norm
from matplotlib import cm
import matplotlib.colors as mplc
import os, sys
from gi.repository import Gtk, Gdk, GdkPixbuf, GObject, GLib
from SAN.plot_functions import *
from sklearn.neighbors import NearestNeighbors


# %%


class ConnectedSparseGraph(Graph):
    def __init__(self, M, d=np.linalg.norm, **kwargs):

        self.M = M
        self.d = d
        self.n = len(M)
        # self.D = self.compute_distances()

        super().__init__(**kwargs)

    def compute_distances(self):

        M = self.M
        d = self.d

        n = self.n
        D = np.empty((n, n))

        for i in range(n):
            for j in range(n):
                x = M[i]
                y = M[j]
                D[i, j] = d(x - y)
        return D


# %%


# %%
# nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(M)

# adj1 = nbrs.kneighbors_graph(M, n_neighbors=1).toarray()
# adj2 = nbrs.kneighbors_graph(M, n_neighbors=2).toarray()
# (adj2 - adj1)
# %%


def calc_matrix_seqs(s: ConnectedSparseGraph, nn_algorithm):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm=nn_algorithm).fit(s.M)
    adj_0 = nbrs.kneighbors_graph(s.M, n_neighbors=1).toarray()
    adj_seq = [adj_0]
    add_seq = []
    i = 2
    while i <= s.n:

        adj = nbrs.kneighbors_graph(s.M, n_neighbors=i).toarray()
        to_add = adj - adj_seq[-1]

        adj_seq.append(adj)
        add_seq.append(to_add)

        i += 1

    return add_seq, adj_seq


# %%
def ith_graph(s: ConnectedSparseGraph, i, nn_algorithm):

    add_seq, adj_seq = calc_matrix_seqs(s, nn_algorithm)

    adj_matrix = np.triu(sum(add_seq[:i]))

    s.add_edge_list(np.transpose(adj_matrix.nonzero()))

    return add_seq, adj_seq, adj_matrix


# %%


def build_csg(s: ConnectedSparseGraph, nn_algorithm="brute"):

    i = 0
    s.clear_edges()

    add_seq, adj_seq = calc_matrix_seqs(s, nn_algorithm)
    adj_matrix = np.triu(add_seq[i])
    s.add_edge_list(np.transpose(adj_matrix.nonzero()))

    comp, hist = gt.label_components(s)

    while len(hist) > 1:

        i += 1

        adj_matrix = np.triu(add_seq[i])
        s.add_edge_list(np.transpose(adj_matrix.nonzero()))

        comp, hist = gt.label_components(s)


# %%
M = npr.random((500, 2))
csg = ConnectedSparseGraph(M, directed=False)

# %%
build_csg(csg)
# %%

gt.graph_draw(csg, vertex_size=5)
