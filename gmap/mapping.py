import copy
import random

import numpy as np
from simanneal import Annealer
from gmap.utils import *
import time


class Hardware_Annealer(Annealer):
    def __init__(self, state, update_move, cost, get_temperature, debug):
        # Redefine function
        self.update_move = update_move
        self.cost = cost
        self.get_temperature = get_temperature

        super(Hardware_Annealer, self).__init__(state)  # important!

        # Mute update function
        if not debug: self.update = lambda *args, **kwargs: None

    def move(self):
        i = random.randint(0, len(self.state.order) - 1)
        j = random.randint(0, len(self.state.order) - 1)
        dE = self.update_move(self.state, i, j)
        self.user_exit = self.best_energy is not None and self.best_energy <= 0
        return dE

    def new_auto(self, minutes):
        T_min, T_max = self.get_temperature(self.state.connectivity_matrix)
        if (T_min, T_max) == (None, None):
            return self.auto(minutes)

        count = 500
        self.set_schedule({'tmax': T_max, 'tmin': T_min, 'steps': count , 'updates': 0})
        time_init = time.time()
        self.anneal()
        steps = int(count * minutes * 60 / (time.time() - time_init))
        return {'tmax': T_max, 'tmin': T_min, 'steps': steps, 'updates': 10}


    def energy(self):
        return self.cost(self.state)

class State:
    def __init__(self, connectivity_matrix, cost_tracker = None):
        self.order = np.arange(len(connectivity_matrix))
        self.connectivity_matrix = connectivity_matrix
        self.cost_tracker = None

    def copy(self):
        new = State(self.connectivity_matrix)
        new.order = copy.deepcopy(self.order)
        new.cost_tracker = copy.deepcopy(self.cost_tracker)
        return new


class Hardware:
    def __init__(self, n_total):
        self.n_total = n_total

    def cost(self, state):
        pass

    def init_cost_tracker(self, connectivity_matrix):
        return None

    def update_cost_tracker(self, state, i, j):
        pass

    def update_move(self, state, i, j):
        E_ini = self.cost(state)
        state.order[i], state.order[j] = state.order[j], state.order[i]
        self.update_cost_tracker(state, i, j)
        return self.cost(state) - E_ini

    def auto(self, connectivity_matrix, minutes = 1):
        return None

    def get_temperature(self, state):
        return None, None

    def mapping(self, connectivity_matrix, minutes=0.5, debug=False, params=None):
        # pre-check
        if len(connectivity_matrix) > self.n_total:
            return connectivity_matrix, float('inf')

        # pre-treatment
        pad = self.n_total - len(connectivity_matrix)
        connectivity_matrix = np.pad(1*(connectivity_matrix>0), [(0, pad), (0, pad)], mode='constant')


        # Initiating the solver
        initial_state = State(connectivity_matrix, self.init_cost_tracker(connectivity_matrix))
        sa = Hardware_Annealer(initial_state, self.update_move, self.cost, self.get_temperature, debug=debug)
        sa.copy_strategy = 'method'
        if debug: print("Initial cost", self.cost(initial_state))

        if params is None:
            if debug: print("Searching for the good optimization params...")
            # Searching for the good params
            params = sa.new_auto(minutes=minutes)
            params['updates'] = 10

        if debug : print("PARAMS : ", params)
        sa.set_schedule(params)


        # Solving the mapping
        if debug: print("Solving the mapping...")
        state, violated_constrains = sa.anneal()

        if debug:
            if violated_constrains == 0:
                print("Mappable !")
            else:
                print("Not Mappable, violated : ", violated_constrains)

        return {'order': state.order, 'matrix': reorder(state.order, connectivity_matrix), 'violated_constrains': violated_constrains}
