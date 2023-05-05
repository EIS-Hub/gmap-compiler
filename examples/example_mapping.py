from gmap.matrix_generator import *
from gmap.utils import *
from gmap.hardware import *
from gmap.mapping import *


# Defining the problem
size_net = 500
size_hw = 1000

n_core = 4
hw_type = "DYNAPS"

if hw_type == "DYNAPS" : # Can only afford small network.
    size_net = 50
    size_hw = 64

# Creating a Network
connectivity_matrix = create_WS(N=size_net, k_avg=int(size_net/2), p_drop= 0.5)  # create Watts–Strogatz connectivity matrix
weights = np.random.rand(size_net, size_net)


# Creating the Hardware
if hw_type == "multicore" : hw = Hardware_multicore(core=n_core)
if hw_type == "generic" : hw = Hardware_generic(n_neurons_core = size_hw // n_core, n_core=n_core, max_fanI=5, max_fanO=5)
if hw_type == "DYNAPS" : hw = Hardware_DYNAPS(N = size_hw, F = 8, C = size_hw// n_core, K = 16, alpha = 1)


connectivity_matrix = shuffle(connectivity_matrix) * weights
plot_matrix(connectivity_matrix)
plt.show()

# Mapping the network onto the Hardware.
mapping = hw.mapping(connectivity_matrix, minutes=2 , debug = True)
# mapping, violated_constrains = hw.mapping(connectivity_matrix, minutes=0.1 , debug = True)


# Plotting the results
plot_matrix(mapping['matrix'])
plt.show()
