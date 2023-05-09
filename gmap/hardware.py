from gmap.compiler import Hardware
from gmap.matrix.utils import reorder, expected_cost_difference_line
from gmap.matrix.generator import create_multicore_mask
import numpy as np
import math
import scipy.stats as stats


class Multicore(Hardware):
    def __init__(self, n_neurons_core, n_core, max_fanI=0, max_fanO=0):
        self.n_total = n_neurons_core * n_core
        self.n_neurons_core = n_neurons_core
        self.n_core = n_core
        self.max_fanI = max_fanI
        self.max_fanO = max_fanO

        self.Mask = 1 - create_multicore_mask(self.n_total, self.n_core)

        # To avoid of to reallocate space.
        self.A_ik = np.empty(self.n_total)
        self.A_jk = np.empty(self.n_total)
        self.A_ki = np.empty(self.n_total)
        self.A_kj = np.empty(self.n_total)

        super(Multicore, self).__init__(self.n_total)  # important !

    def get_temperature(self, connectivity_matrix):

        # Compute the parameters to properly compute the expected dE
        p = np.sum(connectivity_matrix) / (self.n_total ** 2)
        n = int(self.n_total * self.n_core - 1 / self.n_core) # The maximal fan

        # Define how the cost is computed
        cost_line = lambda f: lambda x: x - f if x > f else 0

        # Compute dE
        dE_fanI = expected_cost_difference_line(n, p, cost_line(self.max_fanI))
        dE_fanO = expected_cost_difference_line(n, p, cost_line(self.max_fanO))

        # Compute T_max and T_min
        T_max = -(dE_fanI + dE_fanO) / math.log(0.98)
        T_min = -(dE_fanI + dE_fanO) / math.log(1 - stats.norm.cdf(6))

        return T_min, T_max

    def update_fan(self, mapping, i, j, axis):
        """
        Update efficiently mapping.cost_tracker, i.e. the inter-core fan-in (resp. fan-out) if axis=0 (resp. axis=1).
        Does it in O(n), which is way better than if one has to reorder the matrix in O(n**2)
        """

        A = mapping.connectivity_matrix
        self.A_ik = np.take(A, mapping.order[i], axis=1 - axis)[mapping.order]
        self.A_jk = np.take(A, mapping.order[j], axis=1 - axis)[mapping.order]

        self.A_ki = np.take(A, mapping.order[i], axis=axis)[mapping.order]
        self.A_kj = np.take(A, mapping.order[j], axis=axis)[mapping.order]

        M_i = self.Mask[:, i]
        M_j = self.Mask[:, j]


        # Update the fan of all the neurons. O(n)
        mapping.cost_tracker[axis] += (self.A_ki - self.A_kj) * (M_i - M_j)

        # Update the fan of the 2 swapped neurons. 2*O(n)
        mapping.cost_tracker[axis][i] = np.dot(self.A_ik, M_i)
        mapping.cost_tracker[axis][j] = np.dot(self.A_jk, M_j)

    def update_cost_tracker(self, mapping, i, j):
        self.update_fan(mapping, i, j, axis = 0)  # update fan-in
        self.update_fan(mapping, i, j, axis = 1)  # update fan-out

    def init_cost_tracker(self, mapping):
        """Compute the inter-core fan of every neuron. Complexity of O(n**2) but only done once."""
        fanI = np.sum(mapping.weight_matrix>0 * self.Mask, axis=0)  # Sum along axis 0 (vertically = fan-in)
        fanO = np.sum(mapping.weight_matrix>0 * self.Mask, axis=1)  # Sum along axis 1 (horizontally = fan-out)
        mapping.cost_tracker = [fanI, fanO]

    def cost(self, mapping):
        # If it is the first time that the cost is computed, then initialize the cost_tracker
        if mapping.cost_tracker is None:
            self.init_cost_tracker(mapping)

        # Compute the exceeded fan-in and fan-out.
        exceeded_fanI = np.dot((mapping.cost_tracker[0] - self.max_fanI), (mapping.cost_tracker[0] > self.max_fanI))
        exceeded_fanO = np.dot((mapping.cost_tracker[1] - self.max_fanI), (mapping.cost_tracker[1] > self.max_fanO))

        # This is the variable to minimize
        return exceeded_fanI + exceeded_fanO


class DYNAPSE(Hardware):
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
        super(DYNAPSE, self).__init__(self.N)  # important !

        # A matrix that allows to efficiently compute the fan-in of each neuron to each cluster. Declared only once.
        self.mat_FI = [[1.0 if j == i // self.C else 0.0 for j in range(self.N // self.C)] for i in range(self.N)]

    def violated_mem_sender(self, connectivity_matrix):
        """
        Control the two conditions :
         - Each neuron can communicate with at most M neurons within a cluster.
         - Each neuron can communicate with at most F/M cluster

        Complexity of O(N**2) but can be reduced to O(N*(N//C)) with the use the cost_tracker
        """

        # Compute the fan-in of each neuron to each cluster
        n_FI = np.dot(connectivity_matrix, self.mat_FI)

        # Each neuron can communicate with at most M neurons within a cluster
        violated_M = np.sum((n_FI - self.M) * (n_FI > self.M))

        # Each neuron can communicate with at most F/M cluster
        n_FI_cluster = np.count_nonzero(n_FI, axis=1)  # compute the number of cluster a neuron is communicating with.
        max_FI_cluster = self.F // self.M
        violated_F = np.sum((n_FI_cluster - max_FI_cluster) * (n_FI_cluster > max_FI_cluster))

        return violated_M + violated_F

    def violated_mem_receiver(self, connectivity_matrix):
        """
        Each cluster can have at most K combination of fan-in, the K tags

        Complexity of O((N//C)*N*log(N)) because of use of np.unique that has a complexity of O(N*log(N))
        Could be reduced to O((N//C)*N) with the use of Hash-table and cost_trackers
        """

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

    def cost(self, mapping):
        # Non optimized way of computing the constraints violated. Complexity of O(N**2).
        actual_connectivity_matrix = reorder(mapping.order, mapping.connectivity_matrix)
        return self.violated_mem_sender(actual_connectivity_matrix) + self.violated_mem_receiver(actual_connectivity_matrix)
