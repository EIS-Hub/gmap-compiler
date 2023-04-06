import random
from simanneal import Annealer
from gmap.utils import *


class Hardware_Annealer(Annealer):
    def __init__(self, state, update_move, cost, debug):
        self.update_move = update_move
        self.cost = cost
        super(Hardware_Annealer, self).__init__(state)  # important!
        if not debug: self.update = lambda *args, **kwargs: None

    def move(self):
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state, dE = self.update_move(self.state, a, b)
        self.user_exit = self.best_energy is not None and self.best_energy <= 0
        return dE

    def energy(self):
        return self.cost(self.state)


class Hardware:
    def __init__(self, n_total):
        self.n_total = n_total
        self.buf = np.empty(n_total)  # pre-allocated buffer to avoid re-alloc at each swap.
        pass

    def cost(self, connectivity_matrix):
        pass

    def update_cost(self, connectivity_matrix, a, b):
        return None

    def update_move(self, connectivity_matrix, a, b):
        dE = self.update_cost(connectivity_matrix, a, b)
        if dE is None:
            E_ini = self.cost(connectivity_matrix)
            connectivity_matrix = swap(connectivity_matrix, self.buf, a, b)
            dE = self.cost(connectivity_matrix) - E_ini
        else:
            connectivity_matrix = swap(connectivity_matrix, self.buf, a, b)
        return connectivity_matrix, dE

    def mapping(self, connectivity_matrix, minutes=0.5, debug=False):
        # pre-check
        if len(connectivity_matrix) > self.n_total:
            return connectivity_matrix, float('inf')

        # pre-treatment
        pad = self.n_total - len(connectivity_matrix)
        connectivity_matrix = np.pad(connectivity_matrix, [(0, pad), (0, pad)], mode='constant')

        # solving the mapping
        sa = Hardware_Annealer(connectivity_matrix, self.update_move, self.cost, debug=debug)

        # Searching for the good params
        if debug :print("Searching for the good optimization params...")
        param = sa.auto(minutes=minutes)
        param['updates'] = 10
        sa.set_schedule(param)

        # Solving the mapping
        if debug : print("Solving the mapping...")
        mapping, violated_constrains = sa.anneal()

        if debug :
            if violated_constrains == 0: print("Mappable !")
            else: print("Not Mappable, violated : ", violated_constrains)

        return mapping, violated_constrains
