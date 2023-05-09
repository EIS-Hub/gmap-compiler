from gmap.matrix.generator import create_WS
from gmap.matrix.utils import shuffle, plot_matrix
from matplotlib import pyplot as plt
from gmap.hardware import Multicore, DYNAPSE
import numpy as np

# Defining the network to map
size_net = 220

# Creating a Network
connectivity_matrix = create_WS(N=size_net, k_avg=25, p_drop= 0.5)  # create Watts–Strogatz connectivity matrix
weight_matrix = connectivity_matrix * np.random.random((size_net, size_net))
plot_matrix(weight_matrix)
plt.show()

weight_matrix = shuffle(weight_matrix)
plot_matrix(weight_matrix)
plt.show()


# Mapping onto a Multicore
# Defining the hardware
size_hw = 256
n_core = 4

# Initializing the Multicore object
hw_multicore = Multicore(n_neurons_core = size_hw//n_core, n_core = n_core, max_fanI=15, max_fanO=15)

# Mapping the network onto the Hardware.
order, mapped_matrix, violated_constrains = hw_multicore.map(weight_matrix, debug=True, minutes=1)

# Plotting the results
plot_matrix(mapped_matrix)
plt.show()


# Mapping onto a DYNAPSE
# Defining the hardware
hw_dynapse = DYNAPSE(N = size_hw, F = 20, C =size_hw // n_core, K = 16, alpha = 1)

# Mapping the network onto the Hardware.
order, mapped_matrix, violated_constrains = hw_dynapse.map(weight_matrix, debug=True, minutes=1)

# Plotting the results
plot_matrix(mapped_matrix)
plt.show()

