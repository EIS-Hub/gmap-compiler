import copy
import random
from simanneal import Annealer
from gmap.matrix.utils import *
import time
import math


class Hardware_Annealer(Annealer):
    """ Hardware annealer is an annealer designed specifically to anneal hardware.

     Parameters:
        mapping (Mapping) : The network to anneal
        update_move(mapping, i, j) (callable): The function to call at each swap of neurons i and j and that returns dE
        cost(mapping) (callable): The function that returns the energy of a mapping, i.e. the number of violated constraints
    """

    def __init__(self, mapping, update_move, cost, get_temperature, debug):
        self.update_move = update_move
        self.cost = cost
        self.get_temperature = get_temperature

        super(Hardware_Annealer, self).__init__(mapping)  # important!

        # Mute update function
        if not debug: self.update = lambda *args, **kwargs: None

    def move(self):
        """Moving is swapping two arbitrary selected neurons"""
        i = random.randint(0, len(self.state.order) - 1)
        j = random.randint(0, len(self.state.order) - 1)

        # Computing the difference of cost for that move
        dE = self.update_move(self.state, i, j)

        # Stop annealing if the number of violated constrains is null.
        self.user_exit = self.best_energy is not None and self.best_energy <= 0
        return dE

    def update(self, step, T, E, acceptance, improvement):
        """ Inspired from the original implementation in simanneal."""

        def time_string(seconds):
            """Returns time in seconds as a string formatted HHHH:MM:SS."""
            s = int(round(seconds))  # round to nearest second
            h, s = divmod(s, 3600)  # get hours and remainder
            m, s = divmod(s, 60)  # split remainder into minutes and seconds
            return '%4i:%02i:%02i' % (h, m, s)

        elapsed = time.time() - self.start
        if step == 0:
            print(' Temperature        Energy    Accept   Improve     Elapsed   Remaining')
            print('\r%12.5f  %12.2f                      %s            ' %
                  (T, E, time_string(elapsed)))
        else:
            remain = (self.steps - step) * (elapsed / step)
            print('\r%12.5f  %12.2f  %7.2f%%  %7.2f%%  %s  %s\r' %
                  (T, E, 100.0 * acceptance, 100.0 * improvement,
                   time_string(elapsed), time_string(remain)))

    def improved_auto(self, minutes):
        """
            If the method get_temperature is implemented, then provide the right parameters
            Otherwise, it calls the original method auto().
        """

        # Check if the method get_temperature is implmented.
        T_min, T_max = self.get_temperature(self.state.connectivity_matrix)

        if (T_min, T_max) == (None, None):  # if not, call the original method auto()
            steps = int(10e6 / len(self.state.connectivity_matrix) ** 2)
            return self.auto(minutes, steps=steps)

        # If the method get_temperature is implemented,
        # then generate the adequate number of steps regarding the argument minutes
        count = 100
        elapsed = 0
        while elapsed < 1 and not self.user_exit:
            count *= 2
            self.set_schedule({'tmax': T_max, 'tmin': T_min, 'steps': count, 'updates': 0})
            time_init = time.time()
            self.anneal()
            elapsed = time.time() - time_init

        # Compute the number of steps
        steps = int(count * minutes * 60 / elapsed)
        return {'tmax': T_max, 'tmin': T_min, 'steps': steps, 'updates': 10}

    def anneal(self, greedy_ratio=0.2):
        """
        Implementation inspired from the original implementation of simanneal.
        Modif : At the end of the annealing, set the temperature to 0. Then the annealing turn to be a greedy search.

        Parameters:
        greedy_ratio (float) : The portion of the execution for which the temperature is set to 0.

        """
        step = 0
        self.start = time.time()

        # Precompute factor for exponential cooling from Tmax to Tmin
        if self.Tmin <= 0.0:
            raise Exception('Exponential cooling requires a minimum "\
                "temperature greater than zero.')
        Tfactor = -math.log(self.Tmax / self.Tmin)

        # Note initial mapping
        T = self.Tmax
        E = self.energy()
        prevmapping = self.copy_state(self.state)
        prevEnergy = E
        self.best_mapping = self.copy_state(self.state)
        self.best_energy = E
        trials, accepts, improves = 0, 0, 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        # Attempt moves to new mappings
        while step < self.steps and not self.user_exit:
            step += 1

            # For the lasts steps, the temperature is set to 0.
            if step > self.steps * (1 - greedy_ratio):
                T = 0
            else:
                T = self.Tmax * math.exp(Tfactor * step / (self.steps * (1 - greedy_ratio)))

            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and (T == 0 or math.exp(-dE / T) < random.random()):
                # Restore previous mapping
                self.state = self.copy_state(prevmapping)
                E = prevEnergy
            else:
                # Accept new mapping and compare to best mapping
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevmapping = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_mapping = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1 and (step // updateWavelength) > ((step - 1) // updateWavelength):
                self.update(
                    step, T, E, accepts / trials, improves / trials)
                trials, accepts, improves = 0, 0, 0

        self.state = self.copy_state(self.best_mapping)
        if self.save_state_on_exit:
            self.save_state()

        # Return best mapping and energy
        return self.best_mapping, self.best_energy

    def energy(self):
        return self.cost(self.state)


class Mapping:
    """
    mapping contains the actual data useful for a mapping

    Parameters:
    order (np.array): The order to apply to connectivity_matrix and to weight_matrix to get the mapping
    connectivity_matrix (2D np.array): The original unordered connectivity matrix of the network
    weight_matrix (2D np.array): The original unordered weight matrix of the network
    cost_tracker : Any object that help to track the actual cost of the actual mapping

    """

    def __init__(self, connectivity_matrix, weight_matrix, cost_tracker=None):
        self.order = np.arange(len(connectivity_matrix))
        self.connectivity_matrix = connectivity_matrix
        self.weight_matrix = weight_matrix
        self.cost_tracker = cost_tracker

    def copy(self):
        """
        Only the order and the cost_tracker has to be copied. They depend on the mapping.
        The connectivity_matrix and th  weight_matrix are the original unordered ones.
        """
        new = Mapping(self.connectivity_matrix, self.weight_matrix)
        new.order = copy.deepcopy(self.order)
        new.cost_tracker = copy.deepcopy(self.cost_tracker)
        return new


class Hardware:
    """
    Class to define a hardware regarding its constrains.
    """

    def __init__(self, n_total):
        """
        Parameters:
            n_total: the total number of neurons in the Hardware.
        """
        self.n_total = n_total

    """ Methods to Override """
    def cost(self, mapping):
        """
            Mandatory to implement.
            Compute the number of violated constrains regarding a mapping.

            Parameters:
            mapping (Mapping): The actual tried mapping to evaluate

            Returns:
            float: The number of violated constraints.
        """
        pass

    def init_cost_tracker(self, mapping):
        """
            This method is not mandatory to implement.
            It is not necessary for this function to be optimized since called only once if implemented.

            Initialize the mapping.cost_tracker that can help to compute faster the cost.

            Parameters:
            mapping (Mapping): The initial mapping to evaluate.

            Returns:
            None but the mapping object should be updated.
        """
        return None

    def update_cost_tracker(self, mapping, i, j):
        """
            This method is not mandatory to implement.
            It is necessary for this function to be optimized since called a lots of time.

            Update the mapping.cost_tracker regarding the swap of the neurons i and j.

            Parameters:
            mapping (Mapping): The mapping to update

            Returns:
            None but the mapping.cost_tracker object should be updated.
        """
        pass

    def get_temperature(self, mapping):
        """
            This method is not mandatory to implement.

            Compute the adequate T_min and T_max to anneal. Their value should be such that :
            Expectation[e**(-dE / T) | dE < )] = p
            - For T_max : p = 0.98
            - For T_mis : p = 9.865877004244794e-10 (six sigma)

            Parameters:
            mapping (Mapping): The mapping to update

            Returns:
            T_min, T_max (float, float) : The two adequate temperatures
        """
        return None, None


    """ Methods not to override. """
    def update_move(self, mapping, i, j):
        """
        Compute the difference in cost between a mapping and a mutation
        of this mapping where the neurons i and j are swapped.

        Parameters:
        mapping (Mapping) : The mapping to be compared with its mutation.
        i and j (int) : The index of the neurons that are swapped

        Returns:
        float: The difference of cost.
        The mapping is mutated.
        """

        # Cost of the un-mutated mapping.
        E_ini = self.cost(mapping)

        # Swap the mapping
        mapping.order[i], mapping.order[j] = mapping.order[j], mapping.order[i]

        # Update the mapping accordingly.
        self.update_cost_tracker(mapping, i, j)

        # Compute the difference of cost.
        return self.cost(mapping) - E_ini


    def map(self, weight_matrix, minutes=1, debug=False, params=None, greedy_ratio=0.2):
        """
        Map a network defined by its weight matrix onto this Hardware.

        Parameters:
        weight_matrix (2D np.array) : The network to map
        minutes (float) : The limit time for computation.
        debug (bool) : If it has to print the logs
        params (dict({'tmax': T_max, 'tmin': T_min, 'steps': steps, 'updates': updates})) : Parameters to give for the annealer
        greedy_ratio (float) : The ratio of the search that has to be a greedy search

        Returns:
        with :
        order (1D np.array) : The order of neurons for the mapping
        mapped_matrix (2D np.array) : The reordered matrix
        violated_constrains (float) : The number of violated constrains for that mapping
        """

        # Pre-check the size of the network that has to be smaller than the hardware
        if len(weight_matrix) > self.n_total:
            return weight_matrix, float('inf')

        # Pre-treatment. Zero-pad the network matrix.
        pad = self.n_total - len(weight_matrix)
        weight_matrix = np.pad(weight_matrix, [(0, pad), (0, pad)], mode='constant')
        connectivity_matrix = 1 * (weight_matrix > 0)

        # Initiating the solver
        initial_mapping = Mapping(connectivity_matrix, weight_matrix)
        sa = Hardware_Annealer(initial_mapping, self.update_move, self.cost, self.get_temperature, debug=debug)
        sa.copy_strategy = 'method'
        if debug: print("Initial cost : ", self.cost(initial_mapping))

        if params is None:
            if debug: print("Searching for the good optimization parameters...")
            # Searching for the good params
            params = sa.improved_auto(minutes=minutes)
            params['updates'] = 10

        if debug: print("Parameters for the annealing :", params)
        sa.set_schedule(params)

        # Solving the mapping
        if debug: print("Solving the mapping...")
        mapping, violated_constrains = sa.anneal(greedy_ratio=greedy_ratio)

        if debug:
            if violated_constrains == 0:
                print("Mappable !")
            else:
                print("Not Mappable, violated : ", violated_constrains)

        return mapping.order, reorder(mapping.order, weight_matrix), violated_constrains
