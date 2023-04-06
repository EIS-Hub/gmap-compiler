from numba import jit
from gmap.mapping import Hardware
from gmap.matrix_generator import create_communities
import numpy as np


@jit
def cost_update_mask_jit(A, Mask, i, j):
    if i == j: return 0
    A_i_, A_j_, A__i, A__j, A_ij, A_ji, A_ii, A_jj = \
        A[i, :] > 0, A[j, :] > 0, A[:, i] > 0, A[:, j] > 0, A[i, j] > 0, A[j, i] > 0, A[i, i] > 0, A[j, j] > 0
    update = (((A_j_ - A_i_) * Mask[i, :]).sum()
              + ((A_i_ - A_j_) * Mask[j, :]).sum()
              + ((A__j - A__i) * Mask[:, i]).sum()
              + ((A__i - A__j) * Mask[:, j]).sum()
              + ((A_ii + A_jj - 2 * A_ji) * Mask[i, i])
              + ((A_jj + A_ii - 2 * A_ij) * Mask[j, j])
              + ((A_ij + A_ji - 2 * A_jj) * Mask[i, j])
              + ((A_ji + A_ij - 2 * A_ii) * Mask[j, i]))

    return update


class Hardware_multicore(Hardware):
    def __init__(self, n_total, core):
        super(Hardware_multicore, self).__init__(n_total)  # important !
        self.Mask = 1 - 1 * create_communities(n_total, core)

    def update_cost(self, connectivity_matrix, a, b):
        return cost_update_mask_jit(connectivity_matrix, self.Mask, a, b)

    def cost(self, connectivity_matrix):
        return np.sum((connectivity_matrix > 0) * self.Mask)


class Hardware_generic(Hardware):
    def __init__(self, n_neurons_core, n_core, n_fanI=0, n_fanO=0):
        self.n_total = n_neurons_core * n_core
        self.n_neurons_core = n_neurons_core
        self.n_core = n_core
        self.n_fanI = n_fanI
        self.n_fanO = n_fanO
        super(Hardware_generic, self).__init__(self.n_total)  # important !
        self.Mask = 1 - 1 * create_communities(self.n_total, self.n_core)

    def violated_fan(self, connectivity_matrix, axis, max_fan):
        ext_fan = self.Mask * (connectivity_matrix>0)
        sum_fan = np.sum(ext_fan, axis=axis)
        return np.sum((sum_fan > max_fan) * sum_fan)

    def cost(self, connectivity_matrix):
        violated_FO = self.violated_fan(connectivity_matrix, 0, self.n_fanO)  # Sum along axis 0 (vertically)
        violated_FI = self.violated_fan(connectivity_matrix, 1, self.n_fanI)  # Sum along axis 1 (horizontally)
        return violated_FI + violated_FO
