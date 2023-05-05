# -*- coding: utf-8 -*-
from __future__ import print_function
import math
import random
from collections import defaultdict
from simanneal import Annealer
import numpy as np
import copy

from matplotlib import pyplot as plt
from numba import jit


def plot_matrix(A, labels=False, all_ticks=False, extent=None, aspect=None):
    plt.imshow(A, cmap="autumn_r", extent=extent, aspect=aspect)

    if labels:
        for (j, i), label in np.ndenumerate(A):
            plt.text(i, j, label, ha='center', va='center')

    if all_ticks:
        plt.yticks(np.arange(len(A)))
        plt.xticks(np.arange(len(A)))


def create_communities(N, C):
    A = np.zeros((N, N))
    for i in range(C):
        A[(N // C) * i:(N // C) * (i + 1), (N // C) * i:(N // C) * (i + 1)] = 1
    return A.astype(np.int64)


def create_distance_dependant(N, func, circular=True):
    x, y = np.meshgrid(range(N), range(N))
    if circular:
        A = np.minimum(np.mod(x - y, N), np.mod(y - x, N))
    else:
        A = np.abs(x - y)
    return np.vectorize(func)(A)


def uni_proba(x): return np.random.uniform(0, 1) < x


def minus_x_square(N, exp):
    return lambda x: uni_proba(((N - x) / N) ** exp)


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


# Create a reordered matrix along new labels
def reorder(partitions, A):
    order_ = np.ndarray.flatten(partitions)
    A_ord_line = np.zeros_like(A)
    for i in range(len(order_)):
        A_ord_line[i] = A[order_[i]]
    A_ord_col = np.zeros_like(A)
    for i in range(len(order_)):
        A_ord_col[:, i] = A_ord_line[:, order_[i]]
    return A_ord_col


# Reorder a list of matrix along a list of labels
def reorder_list(order, l):
    return np.array([l[i] for i in order])


# Shuffle the label of a network. The etwork remains the same but not its incidency matrix.
def shuffle(A):
    A_new = np.copy(A)
    arr = np.arange(len(A_new))
    np.random.shuffle(arr)
    A_new = reorder(arr, A_new)
    return A_new


class ProblemN(Annealer):
    """Test annealer with a travelling salesman problem.
    """

    # pass extra data (the distance matrix) into the constructor

    def __init__(self, distance_matrix, n_total, core, max_fanI=0, max_fanO=0):
        self.A = 1 * (distance_matrix > 0)
        self.Mask = 1 - 1 * create_communities(n_total, core)
        fanIO = [np.sum(self.A * self.Mask, axis=0), np.sum(self.A * self.Mask, axis=1)]

        self.max_fanI = max_fanI
        self.max_fanO = max_fanO

        # To avoid of to reallocate space.
        self.A_ik = np.empty(n_total)
        self.A_jk = np.empty(n_total)
        self.A_ki = np.empty(n_total)
        self.A_kj = np.empty(n_total)

        super(ProblemN, self).__init__({'order': np.arange(n_total), 'fanIO': fanIO})  # important!

    def update_fan(self, i, j, axis):
        self.A_ik = np.take(self.A, self.state['order'][i], axis=1 - axis)[self.state['order']]
        self.A_jk = np.take(self.A, self.state['order'][j], axis=1 - axis)[self.state['order']]

        self.A_ki = np.take(self.A, self.state['order'][i], axis=axis)[self.state['order']]
        self.A_kj = np.take(self.A, self.state['order'][j], axis=axis)[self.state['order']]

        M_i = self.Mask[:, i]
        M_j = self.Mask[:, j]

        self.state['fanIO'][axis] += (self.A_ki - self.A_kj) * (M_i - M_j)
        self.state['fanIO'][axis][i] = self.A_ik @ M_i
        self.state['fanIO'][axis][j] = self.A_jk @ M_j

        # # Check
        # B = reorder(self.state['order'], self.A)
        # fanI = np.sum(B * self.Mask, axis=0)
        # if np.any((1 - (fanI - self.state['fanI']))==0):
        #     print(fanI-self.state['fanI'])

    def move(self):
        initial = self.energy()
        i = random.randint(0, len(self.state['order']) - 1)
        j = random.randint(0, len(self.state['order']) - 1)

        self.state['order'][i], self.state['order'][j] = self.state['order'][j], self.state['order'][i]
        self.update_fan(i, j, 0)  # update fan-in
        self.update_fan(i, j, 1)  # update fan-out

        return self.energy() - initial

    def energy(self):
        exceeded_FI = np.sum(self.state['fanIO'][0] - self.max_fanI)
        exceeded_FO = np.sum(self.state['fanIO'][1] - self.max_fanO)
        return exceeded_FI + exceeded_FO


n = 2000
# a = create_distance_dependant(n, minus_x_square(n, 3), circular=True)
a = 1 * (np.random.rand(n, n) > 0.5)
sa = ProblemN(a, n, 2)
sa.copy_strategy = 'deepcopy'
param = sa.auto(minutes=1)
print(param)
sa.set_schedule(param)
order, energy = sa.anneal()
print()
print("ok")
plot_matrix(reorder(order['order'], a))
plt.show()

