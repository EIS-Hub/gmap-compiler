# Import functions from gmap matrix module
from gmap.matrix.utils import *
from gmap.matrix.generator import *

# Set the number of nodes in the graph and the average degree
N = 256
K = 64

# Print information about the graphs created using different functions
info_graph(create_random(N, K))  # Create a random graph with N nodes and average degree K
info_graph(create_multicore_mask(N, C=8))  # Create a multicore mask with N nodes and 8 cores
info_graph(create_WS(N, K))  # Create a small-world graph with N nodes and average degree K
info_graph(create_BA(N, K))  # Create a scale-free graph with N nodes and average degree K
info_graph(create_gaussian_connect(N, K))  # Create a graph with connections distributed according to a Gaussian distribution

# Define a function that generates a probability of connection between two nodes based on their distance
def minus_x_exponent(N, exp):
    return lambda x: np.random.rand() < ((N - x) / N) ** exp

# Print information about the graph created using the distance-dependent function
info_graph(create_distance_dependant(N, minus_x_exponent(N , 4), circular=True))  # Create a circular graph with N nodes, where the probability of connection between two nodes is proportional to ((N - x) / N) ** 4, where x is the distance between the two nodes.

# Show the plots of the generated graphs
plt.show()