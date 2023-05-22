from tqdm import trange
from gmap.mapping import shuffle
import numpy as np
from gmap.hardware import Multicore
import time
from matplotlib import pyplot as plt
from gmap.mapping import Mapping


def benchmark(filename, test, hw, sigma, mu, debug=False, compute=False):
    """
    Benchmark the hardware mapping algorithm by comparing it against the baseline and brute-force methods.

    Args:
    - filename (str): The name of the file containing the data to use for the benchmark.
    - test (int): The number of tests to perform for each value of sigma and mu.
    - hw (Hardware): The Hardware object to use for the mapping.
    - sigma (tuple): A tuple containing the minimum and maximum values of sigma to use for the benchmark.
    - mu (tuple): A tuple containing the minimum and maximum values of mu to use for the benchmark.
    - debug (bool): If True, print debug information during the benchmark.
    - compute (bool): If True, compute the results of the benchmark. Otherwise, load the results from file.

    Returns:
    - Cost_baseline (ndarray): A matrix containing the baseline costs for each value of sigma and mu.
    - Cost_bf (ndarray): A matrix containing the brute-force costs for each value of sigma and mu.
    - Cost_algo (ndarray): A matrix containing the hardware mapping costs for each value of sigma and mu.
    """
    if compute:
        # Load the brute-force costs from file
        Cost_bf = np.loadtxt('./results/Noisy_diagonal_' + filename + '_cost_bruteforce.csv', delimiter=',')

        # Initialize the baseline and algorithm costs
        Cost_baseline = np.full([sigma[1] - sigma[0] + 1, mu[1] - mu[0] + 1], np.inf)
        Cost_algo = np.full([sigma[1] - sigma[0] + 1, mu[1] - mu[0] + 1], np.inf)

        # Initialize the time variable
        tGmap = 0

        # Iterate over the tests for each value of sigma and mu
        for i in trange(test, desc="Iteration number", leave=True, disable=not debug):
            for sig in range(sigma[1] + 1 - sigma[0]):
                for m in range(mu[1] + 1 - mu[0]):
                    # Load the noisy diagonal matrix with the given sigma and mu values
                    my_net = np.loadtxt('./data/Noisy_diagonal_' + filename + '/Noisy_diagonal_' + filename + '_sigma_' + str(sig + sigma[0]) + '_mu_' + str(m + mu[0]) + '.csv', delimiter=',')

                    # Shuffle the matrix
                    my_net = shuffle(my_net)

                    # Compute the baseline cost
                    Cost_baseline[sig, m] = hw.cost(Mapping(my_net))

                    # Compute the hardware mapping cost
                    t0 = time.time()
                    mapping = hw.map(my_net, minutes=0.1, debug=debug)
                    tGmap += time.time() - t0
                    Cost_algo[sig, m] = min(Cost_algo[sig, m], mapping.cost)

        # Save the results to file
        np.savetxt('./results/Noisy_diagonal_' + filename + "_cost_baseline.csv", Cost_baseline, delimiter=',')
        np.savetxt('./results/Noisy_diagonal_' + filename + "_cost_bruteforce.csv", Cost_bf, delimiter=',')
        np.savetxt('./results/Noisy_diagonal_' + filename + "_cost_algo.csv", Cost_algo, delimiter=',')

    # Load the results from file
    Cost_baseline = np.loadtxt('./results/Noisy_diagonal_'+ filename +"_cost_baseline.csv", delimiter=',')
    Cost_bf =       np.loadtxt('./results/Noisy_diagonal_'+ filename +"_cost_bruteforce.csv", delimiter=',')
    Cost_algo =     np.loadtxt('./results/Noisy_diagonal_'+ filename +"_cost_algo.csv", delimiter=',')
    return Cost_baseline, Cost_bf, Cost_algo

def plot_comparison(Cost_baseline, Cost_bf, Cost_algo, mu, label, savefig=False):
    """
    Plot a comparison of the baseline, brute-force, and hardware mapping costs for different values of mu.

    Args:
    - Cost_baseline (ndarray): A matrix containing the baseline costs for each value of sigma and mu.
    - Cost_bf (ndarray): A matrix containing the brute-force costs for each value of sigma and mu.
    - Cost_algo (ndarray): A matrix containing the hardware mapping costs for each value of sigma and mu.
    - mu (tuple): A tuple containing the minimum and maximum values of mu used for the benchmark.
    - label (str): The label of the result that are plotted
    - savefig (bool): If True, save the plot to a file.

    Returns:
    - fig (Figure): The Figure object containing the plot.
    """
    fig = plt.figure()

    # Plot the mean costs for each method as a function of mu
    plt.plot(np.arange(mu[0], mu[1] + 1), np.mean(Cost_baseline, axis=0), 'r-', label="N = " + label + " - Baseline")
    plt.plot(np.arange(mu[0], mu[1] + 1), np.mean(Cost_algo, axis=0), 'p-', label="N = " + label + " - GMap")
    plt.plot(np.arange(mu[0], mu[1] + 1), np.mean(Cost_bf, axis=0), 'g-', label="N = " + label + " - Ground Truth")

    # Add axis labels and legend
    plt.ylabel("Number of inter-core connections")
    plt.xlabel("Average node degree")
    plt.legend()

    # Save the figure to a file if requested
    if savefig:
        fig.savefig("./results/Noisy_diagonal_" + label + "_Comparison.pdf", format="pdf")

    return fig


# Set parameters
compute = True  # Whether to recompute or rely on pre-obtained data
test = 10  # Number of statistical repetitions

# Set up Multicore hardware instances
hw = Multicore(4, 4)
# Benchmark for network size of 16 nodes
Cost_baseline_16, Cost_bf_16, Cost_algo_16 = benchmark('16', test=test, hw=hw, sigma=[2, 3], mu=[1, 6], debug=True, compute=compute)

hw = Multicore(8, 4)
# Benchmark for network size of 32 nodes
Cost_baseline_32, Cost_bf_32, Cost_algo_32 = benchmark('32', test=test, hw=hw, sigma=[3, 7], mu=[3, 7], debug=True, compute=compute)

# Benchmark for another network size of 32 nodes
Cost_baseline_32_bis, Cost_bf_32_bis, Cost_algo_32_bis = benchmark('32_bis', test=1, hw=hw, sigma=[3, 5], mu=[2, 10], debug=True, compute=compute)

# Plot the comparison of the algorithms for each network size
fig1 = plot_comparison(Cost_baseline_16, Cost_bf_16, Cost_algo_16, [1, 6], label="16", savefig=True)
fig2 = plot_comparison(Cost_baseline_32, Cost_bf_32, Cost_algo_32, [3, 7], label="32", savefig=True)
fig3 = plot_comparison(Cost_baseline_32_bis, Cost_bf_32_bis, Cost_algo_32_bis, [2, 10], label="32_bis", savefig=True)

# Display the plots
plt.show()