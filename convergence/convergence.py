from gmap.hardware import Multicore
from gmap.mapping import Mapping
from gmap.matrix.generator import create_multicore_mask
import numpy as np
from matplotlib import pyplot as plt
from gmap.mapping import shuffle
from tqdm import trange

def create_test_matrix(n, c, fan):
    """
    Creates a connectivity network that perfectly fills a c-core architecture of n neurons with a max inter-core fan of fan.

    Args:
    - n (int): The number of neurons in the network.
    - c (int): The number of cores in the network.
    - fan (int): The maximum inter-core fan of the network.

    Returns:
    - A (ndarray): A symmetric n x n matrix representing the connections between the neurons.
    """
    # Calculate the spacing between neurons to achieve the desired fan
    skip = n*(c-1)//(c*fan)

    # Create the fan mask
    fan_mask = np.array([[1 if (i + j) % skip == 0 else 0 for i in range(n)] for j in range(n)])

    # Combine the fan mask with the core mask to get the final network connectivity matrix
    return 1 * ((create_multicore_mask(n, c) + fan_mask) > 0)


def compute_result(n, filename, epoch_max, epoch_test, stat_test, compute=False):
    """
    Computes the results of the mapping of the given matrix using GMap.

    Args:
    - n (int): The number of nodes in the matrix.
    - filename (str): The name of the file where the results will be stored.
    - epoch_max (int): The maximum number of epochs to run the mapping algorithm.
    - epoch_test (int): The number of epochs to run the mapping algorithm for each iteration.
    - stat_test (int): The number of iterations to run the mapping algorithm.
    - compute (bool): If True, recompute the results; if False, load previously computed results from file.

    Returns:
    - result (ndarray): An array containing the results of the mapping algorithm.
    """

    if compute:
        # Create the test matrix and initialize the result array
        density = 1 / 2
        c = 2
        fan = density * n * (c - 1) // c
        A = create_test_matrix(n, c, fan)
        result = np.zeros(epoch_test+1)

        # Initialize the hardware and get the temperature range
        hw = Multicore(n_neurons_core=n // 4, n_core=4, max_fanI=0, max_fanO=0)
        T_min, T_max = hw.get_temperature(A)

        # Run the mapping algorithm once with no updates
        mapping = hw.map(A, params={'tmax': T_max, 'tmin': T_min, 'steps': 0, 'updates': 0})
        result[0] = mapping.cost

        # Shuffle the matrix for the subsequent iterations
        A = shuffle(A)

        # Run the mapping algorithm for the specified number of epochs and iterations
        for t in trange(stat_test):
            for i, s in enumerate(np.linspace(epoch_max//epoch_test, epoch_max, epoch_test).astype(int)):
                params = {'tmax': T_max, 'tmin': T_min, 'steps': s, 'updates': 0}
                mapping = hw.map(A, params=params)
                result[i+1] += mapping.cost

        # Average the results over the iterations
        result[1:] /= stat_test

        # Save the results to file
        np.savetxt(filename, result, delimiter=',')

    # Load the results from file
    return np.loadtxt(filename, delimiter=',')


# Set the number of epochs for testing and the maximum number of epochs to run
epoch_test = 50
epoch_max = 100000

# Set the number of statistical repetitions and whether to compute new results
stat_test = 20
compute = False

# Create a new figure to plot the results
fig = plt.figure()

# Set font size
font = {'size': 14}
plt.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2

# Create an array of x-axis values for plotting
x_axis = np.linspace(epoch_max//epoch_test, epoch_max, epoch_test).astype(int)

# Compute and plot the convergence results for different network sizes
result = compute_result(4096, 'results/convergence4096_105.csv', epoch_max, epoch_test, stat_test, compute)
plt.plot(x_axis, 100*(result[1:]-result[0])/result[1:], label = 'N = 4096')

result = compute_result(1024, 'results/convergence2048_105.csv', epoch_max, epoch_test, stat_test, compute)
plt.plot(x_axis, 100*(result[1:]-result[0])/result[1:], label = 'N = 2048')

result = compute_result(1024, 'results/convergence1024_105.csv', epoch_max, epoch_test, stat_test, compute)
plt.plot(x_axis, 100*(result[1:]-result[0])/result[1:], label = 'N = 1024')

result = compute_result(512, 'results/convergence512_105.csv', epoch_max, epoch_test, stat_test, compute)
plt.plot(x_axis, 100*(result[1:]-result[0])/result[1:], label = 'N = 512')

result = compute_result(256, 'results/convergence256_105.csv', epoch_max, epoch_test, stat_test, compute)
plt.plot(x_axis, 100*(result[1:]-result[0])/result[1:], label = 'N = 256')

# Set the plot labels
plt.xlabel('Number of steps')
plt.ylabel('Relative error wrt the optimum [%]')

# Show the plot and save it to a file
plt.plot()
plt.legend()
plt.show()
fig.savefig("results/Comparison_convergence.pdf", format = "pdf")

plt.show()
fig.savefig("results/Comparison_convergence.pdf", format = "pdf")

