from numba import jit
from gmap.mapping import Hardware
from gmap.matrix_generator import create_communities
import numpy as np
from gmap.params_etimation import *
from gmap.utils import *


class Hardware_multicore(Hardware):
    def __init__(self, n_total, core):
        super(Hardware_multicore, self).__init__(n_total)  # important !
        self.Mask = 1 - 1 * create_communities(n_total, core)

    def cost(self, state):
        connectivity_matrix = reorder(state.order, state.connectivity_matrix)
        return np.sum(connectivity_matrix * self.Mask)


class Hardware_generic_slow(Hardware):
    def __init__(self, n_neurons_core, n_core, max_fanI=0, max_fanO=0):
        self.n_total = n_neurons_core * n_core
        self.n_neurons_core = n_neurons_core
        self.n_core = n_core
        self.max_fanI = max_fanI
        self.max_fanO = max_fanO
        super(Hardware_generic_slow, self).__init__(self.n_total)  # important !
        self.Mask = 1 - 1 * create_communities(self.n_total, self.n_core)

    def violated_fan(self, connectivity_matrix, axis, max_fan):
        ext_fan = self.Mask * connectivity_matrix
        sum_fan = np.sum(ext_fan, axis=axis)
        return np.sum((sum_fan - max_fan) * (sum_fan > max_fan))

    def cost(self, state):
        connectivity_matrix = reorder(state.order, state.connectivity_matrix)
        violated_FI = self.violated_fan(connectivity_matrix, 0, self.max_fanI)  # Sum along axis 0 (vertically = fan-in)
        violated_FO = self.violated_fan(connectivity_matrix, 1,
                                        self.max_fanO)  # Sum along axis 1 (horizontally = fan-out)
        return violated_FI + violated_FO


class Hardware_generic(Hardware):
    def __init__(self, n_neurons_core, n_core, max_fanI=0, max_fanO=0):
        self.n_total = n_neurons_core * n_core
        self.n_neurons_core = n_neurons_core
        self.n_core = n_core
        self.max_fanI = max_fanI
        self.max_fanO = max_fanO

        self.Mask = 1 - 1 * create_communities(self.n_total, self.n_core)

        # To avoid of to reallocate space.
        self.A_ik = np.empty(self.n_total)
        self.A_jk = np.empty(self.n_total)
        self.A_ki = np.empty(self.n_total)
        self.A_kj = np.empty(self.n_total)

        super(Hardware_generic, self).__init__(self.n_total)  # important !

    def get_temperature(self, connectivity_matrix):
        p = np.sum(connectivity_matrix) / (self.n_total ** 2)
        n = int(self.n_total * self.n_core - 1 / self.n_core)

        dE_fanI = expected_cost_difference(n, p, self.max_fanI)
        dE_fanO = expected_cost_difference(n, p, self.max_fanO)
        dE_min = expected_cost_difference_low(n, p)

        T_max = -(dE_fanI + dE_fanO) / math.log(0.98)
        T_min = - dE_min / math.log(1 - stats.norm.cdf(6))

        return T_min, T_max

    def update_fan(self, state, i, j, axis):
        A = state.connectivity_matrix
        self.A_ik = np.take(A, state.order[i], axis=1 - axis)[state.order]
        self.A_jk = np.take(A, state.order[j], axis=1 - axis)[state.order]

        self.A_ki = np.take(A, state.order[i], axis=axis)[state.order]
        self.A_kj = np.take(A, state.order[j], axis=axis)[state.order]

        M_i = self.Mask[:, i]
        M_j = self.Mask[:, j]

        state.cost_tracker[axis] += (self.A_ki - self.A_kj) * (M_i - M_j)
        state.cost_tracker[axis][i] = np.dot(self.A_ik, M_i)
        state.cost_tracker[axis][j] = np.dot(self.A_jk, M_j)

    def update_cost_tracker(self, state, i, j):
        self.update_fan(state, i, j, 0)  # update fan-in
        self.update_fan(state, i, j, 1)  # update fan-out

    def init_cost_tracker(self, connectivity_matrix):
        fanI = np.sum(connectivity_matrix * self.Mask, axis=0)  # Sum along axis 0 (vertically = fan-in)
        fanO = np.sum(connectivity_matrix * self.Mask, axis=1)  # Sum along axis 1 (horizontally = fan-out)
        return [fanI, fanO]

    def cost(self, state):
        if state.cost_tracker is None:
            state.cost_tracker = self.init_cost_tracker(state.connectivity_matrix)

        exceeded_FI = np.dot((state.cost_tracker[0] - self.max_fanI), (state.cost_tracker[0] > self.max_fanI))
        exceeded_FO = np.dot((state.cost_tracker[1] - self.max_fanI), (state.cost_tracker[1] > self.max_fanO))
        return exceeded_FI + exceeded_FO


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
        n_FI = np.dot(connectivity_matrix, self.mat_FI)

        # Each neuron can communicate with at most M neurons within a cluster
        violated_M = np.sum((n_FI - self.M) * (n_FI > self.M))

        # Each neuron can communicate with at most F/M cluster
        n_FI_cluster = np.count_nonzero(n_FI, axis=1)  # compute the number of cluster a neuron is communicating with.
        max_FI_cluster = self.F // self.M
        violated_F = np.sum((n_FI_cluster - max_FI_cluster) * (n_FI_cluster > max_FI_cluster))

        return violated_M + violated_F

    def violated_mem_receiver(self, connectivity_matrix):
        ''' Each cluster can have at most K combination of fan-in, the K tags '''
        violated_K = 0
        for i in range(self.N // self.C):
            # compute the number of different combination of neurons, the number of required tags.
            A = connectivity_matrix[i * self.C:(i + 1) * self.C]
            nonzero_rows = A[np.any(A != 0, axis=1)]
            unique_nonzero_rows = np.unique(nonzero_rows, axis=0)
            max_K = unique_nonzero_rows.shape[0]

            # Compare it with the actual max number of tags.
            violated_K += (max_K - self.K) * (max_K > self.K)

        return violated_K

    def cost(self, state):
        connectivity_matrix = reorder(state.order, state.connectivity_matrix)
        return self.violated_mem_sender(connectivity_matrix) + self.violated_mem_receiver(connectivity_matrix)
