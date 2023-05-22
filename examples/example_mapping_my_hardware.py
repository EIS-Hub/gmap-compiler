from gmap.matrix.generator import create_multicore_mask, create_WS
from gmap.matrix.utils import plot_matrix
from gmap.mapping import shuffle
from gmap.compiler import Hardware
from matplotlib import pyplot as plt
import numpy as np
from gmap import mapping

class my_Hardware(Hardware):
    """
    Define a custom Hardware class that inherits from the base Hardware class. For this example, let's define a multicore
    Hardware with all to all connections intra core and as little as possible extra core connections
    """
    def __init__(self, n_total, core):
        # Call the base constructor
        super(my_Hardware, self).__init__(n_total)

        # Create a mask to enforce hardware constraints
        self.Mask = 1 - 1 * create_multicore_mask(n_total, core)

    def cost(self, mapping):
        """
        Compute the cost of the current mapping. The cost reflects the number of hardware constraints violated if we mapped
        a network in its mapping. The object mapping is composed of the order of the nodes, the original unordered weight
        matrix qnd connectivity matrix. mapping also has a cost_helper to compute faster the cost.
        """

        # Compute the actual connectivity matrix O(n^2).
        actual_connectivity_matrix = mapping.reordered_connectivity_matrix()

        # Compute the number of intercore connections O(n^2).
        return np.sum(actual_connectivity_matrix * self.Mask)


# Defining the network to map
size_net = 220

# Creating a Network
connectivity_matrix = create_WS(N=size_net, k_avg=50, p_drop=0.3)  # create a Watts-Strogatz connectivity matrix
weight_matrix = connectivity_matrix * np.random.random((size_net, size_net))
plot_matrix(weight_matrix)
plt.show()

weight_matrix = shuffle(weight_matrix)
plot_matrix(weight_matrix)
plt.show()


# Defining the hardware
size_hw = 256
n_core = 4
hw = my_Hardware(size_hw, n_core)

# Mapping the network onto the hardware
mapping = hw.map(weight_matrix, debug=True)

# Plotting the results
plot_matrix(mapping.reordered_connectivity_matrix())
plt.show()

