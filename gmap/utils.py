import copy

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from numba import jit


def plot_matrix(A, labels=False, all_ticks=False, extent=None, aspect=None):
    plt.imshow(A, cmap="autumn_r", extent=extent, aspect=aspect)

    if labels:
        for (j, i), label in np.ndenumerate(A):
            plt.text(i, j, label, ha='center', va='center')

    if all_ticks:
        plt.yticks(np.arange(len(A)))
        plt.xticks(np.arange(len(A)))


def info_graph(A):
    G = nx.from_numpy_array(A)
    print("Average number of edges : ", sum(sum(A)) / len(A))
    # print("Sigma Coefficient (sw if >1) : ", nx.sigma(G))
    # print("Omega Coefficient (sw if 0) : ", nx.omega(G))
    plt.figure()
    plot_matrix(A)
    plt.figure()
    plt.plot(nx.degree_histogram(nx.from_numpy_array(A)))
    plt.xlabel("Degree")
    plt.ylabel("Frequency")


# Implementation of copying an array to use numba
@jit
def copyto_numba(a, b):
    N = len(a)
    for i in range(N):
        b[i] = a[i]


@jit
def copyto_numba_array(A, B):
    N = len(A)
    for i in range(N):
        for j in range(N):
            B[i, j] = A[i, j]


# Swap two node's label in a network. It does not change the network but it's matrix representation
@jit
def swap(A_new, buf, i, j):
    copyto_numba(A_new[i, :], buf)
    # np.copyto(buf, A_new[i,:])
    A_new[i, :] = A_new[j, :]
    A_new[j, :] = buf

    copyto_numba(A_new[:, i], buf)
    # np.copyto(buf, A_new[:,i])
    A_new[:, i] = A_new[:, j]
    A_new[:, j] = buf
    return A_new

# Reorder a list of matrix along a list of labels
def reorder_list(order, l):
    return np.array([l[i] for i in order])

# Reorder a list of matrix along a list of labels
def reorder(order, A):
    A_new = A.take(order, axis=0)
    return A_new.take(order, axis=1)

# Shuffle the label of a network. The network remains the same but not its incidency matrix.
def shuffle(A):
    A_new = np.copy(A)
    arr = np.arange(len(A_new))
    np.random.shuffle(arr)
    A_new = reorder(arr, A_new)
    return A_new









