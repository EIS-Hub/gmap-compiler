from numba import jit
from gmap.mapping import Hardware
from gmap.matrix_generator import create_communities


@jit
def cost_update_mask_jit(A, Mask, i, j):
    if i == j: return 0
    A_i_,        A_j_,       A__i,       A__j,       A_ij,       A_ji,       A_ii,       A_jj = \
    A[i, :] > 0, A[j, :]> 0, A[:, i]> 0, A[:, j]> 0, A[i, j]> 0, A[j, i]> 0, A[i, i]> 0, A[j, j]> 0
    update = (  ((A_j_ - A_i_) * Mask[i, :]).sum()
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
        self.Mask = 1 - 1*create_communities(n_total, core)

    def update_cost(self, connectivity_matrix, a, b):
        return cost_update_mask_jit(connectivity_matrix, self.Mask, a, b)

    def cost(self, connectivity_matrix):
        return (connectivity_matrix>0 * self.Mask).sum()

