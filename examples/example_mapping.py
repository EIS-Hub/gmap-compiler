from gmap.matrix_generator import *
from gmap.utils  import *
from gmap.Hardware import *

size_hw = 128
size_net = int(size_hw*4/5)
connectivity_matrix = create_WS(N = size_net, k_avg = int(size_net/4)) # create Watts–Strogatz connectivity matrix
weights = np.random.rand(size_net, size_net)
connectivity_matrix = shuffle(connectivity_matrix)*weights

hw = Hardware_multicore(size_hw, core = 4)
mapping, vioated_constrains = hw.mapping(connectivity_matrix)
plot_matrix(mapping)
plt.show()


