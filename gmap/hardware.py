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
        ext_fan = self.Mask * (connectivity_matrix > 0)
        sum_fan = np.sum(ext_fan, axis=axis)
        return np.sum( (sum_fan-max_fan) * (sum_fan > max_fan) )

    def cost(self, connectivity_matrix):
        violated_FI = self.violated_fan(connectivity_matrix, 0, self.n_fanI)  # Sum along axis 0 (vertically = fan-in)
        violated_FO = self.violated_fan(connectivity_matrix, 1,
                                        self.n_fanO)  # Sum along axis 1 (horizontally = fan-out)
        return violated_FI + violated_FO


class Hardware_DYNAPS(Hardware):
    def __init__(self, N, F, C, K, M=None, alpha=None):
        # Either one describe DYNAPS with M or alpha.
        assert M is None or alpha is None, "Error : Both M and alpha are not None"
        assert M is not None or alpha is not None, "Error : Both M and alpha are None"
        if alpha is not None:
            M = int(np.sqrt((F * np.log2(alpha * N)) / (alpha * np.log2(alpha * C))))

        # Check conditions
        assert F > M, "Error : Condition F > M not respected"
        assert F / M <= N / C, "Error : Condition F/M <= N/C not respected"
        assert M <= C, "Error : Condition M <= C not respected"

        self.N = N
        self.F = F
        self.M = M
        self.C = C
        self.K = K
        super(Hardware_DYNAPS, self).__init__(self.N)  # important !

        self.mat_FI = [[1.0 if j == i // self.C else 0.0 for j in range(self.N // self.C)] for i in range(self.N)]

    def violated_mem_sender(self, connectivity_matrix):
        # Compute the fan in of each neuron to each cluster
        n_FI = np.dot(connectivity_matrix>0, self.mat_FI)

        # Each neuron can communicate with at most M neurons within a cluster
        violated_M = np.sum((n_FI - self.M) * (n_FI > self.M))

        # Each neuron can communicate with at most F/M cluster
        n_FI_cluster = np.count_nonzero(n_FI, axis=1) # compute the number of cluster a neuron is communicating with.
        max_FI_cluster = self.F // self.M
        violated_F = np.sum((n_FI_cluster - max_FI_cluster) * (n_FI_cluster > max_FI_cluster))

        return violated_M + violated_F

    def violated_mem_receiver(self, connectivity_matrix):
        ''' Each cluster can have at most K combination of fan-in, the K tags '''
        violated_K = 0
        for i in range(self.N//self.C) :
            # compute the number of different combination of neurons, the number of required tags.
            A = (connectivity_matrix>0)[i*self.C:(i+1)*self.C]
            nonzero_rows = A[np.any(A != 0, axis=1)]
            unique_nonzero_rows = np.unique(nonzero_rows, axis=0)
            max_K = unique_nonzero_rows.shape[0]

            # Compare it with the actual max number of tags.
            violated_K += (max_K-self.K)*(max_K > self.K)

        return violated_K


    def cost(self, connectivity_matrix):
        return self.violated_mem_sender(connectivity_matrix) + self.violated_mem_receiver(connectivity_matrix)
