import random

from numba import jit

import numpy as np

from gmap.matrix_generator import create_communities
from gmap.utils import *
from gmap.matrix_generator import create_communities
from gmap.Hardware import Hardware

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



class Hardware_multicore(Hardware):
    def __init__(self, con_, core):
        super(Hardware_multicore, self).__init__(con_)  # important!
        self.buf = np.arange(len(self.con))
        self.Mask = 1-create_communities(len(self.con), core)

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

    def update_cost(self):
        a = random.randint(0, len(self.con) - 1)
        b = random.randint(0, len(self.con) - 1)
        ret = cost_update_mask_jit(self.con, self.Mask, a, b)
        self.con = swap(self.con, self.buf, a, b)
        return ret

    def cost(self):
        return (self.con * self.Mask).sum()


