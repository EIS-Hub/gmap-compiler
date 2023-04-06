from gmap.matrix_generator import *
from gmap.utils import *
from gmap.hardware import *

# Creating a Network
size_net = 240
connectivity_matrix = create_WS(N=size_net, k_avg=int(size_net/4))  # create Watts–Strogatz connectivity matrix
weights = np.random.rand(size_net, size_net)
connectivity_matrix = shuffle(connectivity_matrix) * weights
plot_matrix(connectivity_matrix)
plt.show()

# Creating the Hardware
size_hw = 256
n_core = 4
# hw = Hardware_multicore(core=n_core)
hw = Hardware_generic(n_neurons_core = size_hw // n_core, n_core=n_core, n_fanI=20, n_fanO=20)

# Mapping the network onto the Hardware
mapping, violated_constrains = hw.mapping(connectivity_matrix, minutes=0.1 , debug = True)
mapping, violated_constrains = hw.mapping(connectivity_matrix, minutes=1 , debug = True)

# Plotting the results
plot_matrix(mapping)
plt.show()
