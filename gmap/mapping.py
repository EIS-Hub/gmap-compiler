from random import random

from numba import jit
from simanneal import Annealer
import numpy as np

from gmap.matrix_generator import create_communities
from gmap.utils import swap


@jit
def cost_update_mask_jit(A, Mask, i, j):
    if i == j: return 0
    update = (((A[j, :] - A[i, :]) * Mask[i, :]).sum()
              + ((A[i, :] - A[j, :]) * Mask[j, :]).sum()
              + ((A[:, j] - A[:, i]) * Mask[:, i]).sum()
              + ((A[:, i] - A[:, j]) * Mask[:, j]).sum()
              + ((A[i, i] + A[j, j] - 2 * A[j, i]) * Mask[i, i])
              + ((A[j, j] + A[i, i] - 2 * A[i, j]) * Mask[j, j])
              + ((A[i, j] + A[j, i] - 2 * A[j, j]) * Mask[i, j])
              + ((A[j, i] + A[i, j] - 2 * A[i, i]) * Mask[j, i]))

    return update


class Hardware_Annealer(Annealer):
    def __init__(self, connectivity_matrix):
        super(Hardware_Annealer, self).__init__(connectivity_matrix)

    def move(self):
        self.update()

    def energy(self):
        self.cost()

    def update(self):
        pass

    def cost(self):
        pass

    def solve(self):
        self.anneal()


class Hardware(Hardware_Annealer):
    def __init__(self, connectivity_matrix):
        self.connectivity_matrix = connectivity_matrix
        super(Hardware, self).__init__(connectivity_matrix)

    def update(self):
        pass

    def cost(self):
        pass

    def solve(self):
        super.solve()



class Hardware_multicore(Hardware):
    def __init__(self, connectivity_matrix, core):
        self.Mask = 1 - create_communities(len(connectivity_matrix), core)
        self.buf = np.arange(len(connectivity_matrix))
        super(Hardware, self).__init__(connectivity_matrix)  # important!

    def move(self):
        a = random.randint(0, len(self.connectivity_matrix) - 1)
        b = random.randint(0, len(self.connectivity_matrix) - 1)
        ret = cost_update_mask_jit(self.connectivity_matrix, self.Mask, a, b)
        self.state = swap(self.connectivity_matrix, self.buf, a, b)
        return ret

    def energy(self):
        return (self.connectivity_matrix * self.Mask).sum()


