import networkx as nx
import numpy as np
from scipy.stats import norm
import random

def create_random(N, k_avg):
    """
    A function that generates a random directed graph with a specified average
    out-degree using the Erdős-Rényi model.

    Args:
    N (int): The number of nodes in the graph.
    k_avg (float): The desired average out-degree of the nodes.

    Returns:
    ret (array-like): The adjacency matrix of the generated graph.
    """

    # Generate a matrix of uniformly distributed random values.
    random_matrix = np.random.uniform(size=(N, N))

    # Set the elements of the adjacency matrix to 1 where the random matrix is less than the desired probability.
    adjacency_matrix = np.array(random_matrix < (k_avg / N)).astype(int)

    # Return the adjacency matrix.
    return adjacency_matrix


def create_multicore_mask(N, C):
    """
    A function that generates a mask for a network of N nodes with C cores.

    Args:
    N (int): The number of nodes in the network.
    C (int): The number of cores in the network.

    Returns:
    mask (array-like): The mask representing the network.
    """

    # Generate an empty mask matrix.
    mask = np.zeros((N, N))

    # Compute the size of each core.
    core_size = N // C

    # Set the diagonal blocks of the mask to 1, representing the intra-core connectivity.
    for i in range(C):
        mask[i * core_size:(i + 1) * core_size, i * core_size:(i + 1) * core_size] = 1

    # Return the mask.
    return mask.astype(int)



def create_WS(N, k_avg, p_drop=0.3):
    """
    A function that generates a small-world Watts-Strogatz network.

    Args:
    N (int): The number of nodes in the network.
    k_avg (int): The desired average degree of each node.
    p_drop (float): The probability of rewiring each edge.

    Returns:
    A (array-like): The adjacency matrix of the generated network.
    """

    # Generate a connected Watts-Strogatz graph using NetworkX.
    G = nx.connected_watts_strogatz_graph(N, k_avg, p_drop)

    # Convert the graph to an adjacency matrix and assign it to the upper-left block of A.
    A = nx.to_numpy_array(G)

    # Return the adjacency matrix.
    return A


def BA(N, M0, m):
    """
    Generate a Barabasi-Albert network.

    Args:
        N (int): The number of nodes in the final network.
        M0 (int): The number of nodes in the initial connected network.
        m (int): The number of edges to attach from a new node to existing nodes.

    Returns:
        A (ndarray): The adjacency matrix of the generated network.
    """

    assert M0 < N
    assert m <= M0

    # Initialize an empty adjacency matrix.
    AM = np.zeros((N, N))

    # Create a fully connected initial network with M0 nodes.
    for i in range(M0):
        for j in range(i + 1, M0):
            AM[i, j] = 1
            AM[j, i] = 1

    # Add the remaining nodes to the network.
    for c in range(M0, N):
        Allk = np.sum(AM)  # Compute the sum of all node degrees.
        ki = np.sum(AM, axis=1)  # Compute the degree of each node.

        # Compute the probability of attaching a new edge to each existing node.
        pi = np.zeros(c, dtype=np.float)
        for i in range(c):
            pi[i] = ki[i] / (float(Allk))

        # Attach m new edges to existing nodes.
        for d in range(m):
            rand01 = random.random()  # Generate a random number between 0 and 1.

            sumpi = 0.0
            for g in range(c):
                sumpi += pi[g]
                if sumpi > rand01:  # Select the node to attach the new edge to.
                    if AM[c, g] == 0 and AM[g, c] == 0:  # Check if the edge already exists.
                        AM[c, g] += 1
                        AM[g, c] += 1
                        break

    return AM



def create_BA(N, k_avg):
    """
    Generate a Barabasi-Albert network with a given average degree.

    Args:
        N (int): The number of nodes in the network.
        k_avg (float): The desired average degree of each node.

    Returns:
        A (ndarray): The adjacency matrix of the generated network.
    """

    # Calculate the number of nodes in the initial connected network.
    M = int((-1 + 2 * N - ((1 - 2 * N) ** 2 - 4 * N * k_avg) ** (1 / 2)) // 2) + 1

    # Generate the Barabasi-Albert network.
    return BA(N, M, M)

def gaussian_connect(N, sigma, mu, tol=1e-6, quantile_max=30):
    """
    Generate a binary random matrix with a Gaussian distribution.

    Args:
        N (int): The number of nodes in the network.
        sigma (float): The standard deviation of the Gaussian distribution.
        mu (float): The mean of the Gaussian distribution.
        tol (float): The tolerance for convergence of the optimization algorithm.
        quantile_max (float): The maximum quantile value used to compute the scaling factor.

    Returns:
        A (ndarray): The binary random matrix with the specified Gaussian distribution.
    """

    # Compute the scaling factor.
    alpha = 3 / quantile_max ** 2 - norm.cdf(-quantile_max) / norm.pdf(-quantile_max) * 3 / quantile_max

    # Check the conditions for the Gaussian distribution.
    assert mu ** 2 / 12 * (1 + alpha) < sigma ** 2, "Error: mu ** 2 / 12 > sigma ** 2"
    assert sigma ** 2 < N ** 2 / 12, "Error: sigma ** 2 > N ** 2 / 12"

    # Initialize the target values for mu and sigma.
    mu_target = mu
    sigma_target = sigma

    # Initialize the margin parameter.
    margin = sigma * np.sqrt(np.maximum(np.log(mu ** 2 / 2 / np.pi / sigma ** 2), 0))

    # Iterate until convergence.
    while True:
        # Compute the difference between the two cumulative distribution functions.
        cdf_diff = norm.cdf(- margin / sigma) - norm.cdf(- N / 2 / sigma)

        # Compute the updated value of mu.
        if margin:
            mu = np.sqrt(2 * np.pi) * sigma * np.exp(margin ** 2 / 2 / sigma ** 2)
        else:
            mu = mu_target / 2 / cdf_diff

        # Compute the quadratic coefficients for sigma.
        b = - mu * np.sqrt(2 / np.pi) * N / 2 * np.exp(- (N / 2) ** 2 / 2 / sigma ** 2) / mu_target
        c = 2 / 3 * margin ** 3 / mu_target - sigma_target ** 2

        # Check for convergence.
        if np.abs(2 * mu * cdf_diff + 2 * margin - mu_target) < tol and np.abs(sigma ** 2 + b * sigma + c) < tol:
            break

        # Update sigma.
        sigma = np.sqrt(- (c + b * sigma))

        # Update the margin parameter.
        if margin:
            margin = np.maximum(mu_target / 2 - mu * cdf_diff, 0)
        else:
            margin = sigma * np.sqrt(np.maximum(np.log(mu ** 2 / 2 / np.pi / sigma ** 2), 0))

    # Compute the binary random matrix.
    x, y = np.meshgrid(range(N), range(N))
    d = np.minimum(np.mod(x - y, N), np.mod(y - x, N))
    return np.random.rand(N, N) <= mu / np.sqrt(2 * np.pi) / sigma * np.exp(- d ** 2 / 2 / sigma ** 2)


def create_gaussian_connect(N, k_avg):
    """
    Creates a graph in which the probability of having an edge between any given pair of nodes is a Gaussian distribution
    with respect to the distance between them.

    Args:
    N (int): The number of nodes in the network.
    k_avg (float): The desired average degree of each node.

    Returns:
    A (ndarray): The adjacency matrix of the generated network.
    """

    # Define the maximum quantile to use when calculating alpha.
    quantile_max = 30

    # Calculate alpha using the maximum quantile value.
    alpha = 3 / quantile_max ** 2 - norm.cdf(-quantile_max) / norm.pdf(-quantile_max) * 3 / quantile_max

    # Calculate the minimum and maximum values for the standard deviation of the Gaussian distribution.
    min_sigma = (k_avg ** 2 / 12 * (1 + alpha)) ** (1 / 2)
    max_sigma = (N ** 2 / 12) ** (1 / 2)

    # Generate the Gaussian connectivity matrix using the calculated values for the standard deviation and the
    # desired average degree.
    g = gaussian_connect(N, (min_sigma + max_sigma) * (1 / 2), k_avg)

    # Return the adjacency matrix.
    return g


def create_distance_dependant(N, func, circular=True):
    """
    Creates a network with the probability of connection between two nodes depending on their distance and a given function.

    Args:
    - N (int): The number of nodes in the network.
    - func (callable): The function that will determine the probability of connection between two nodes based on their distance.
    - circular (bool): If True, the distance between nodes is taken modulo N, which results in a circular graph. Otherwise,
                       the absolute difference between the indices is used, which results in a linear graph.

    Returns:
    - A (ndarray): N x N connectivity matrix .
    """

    x, y = np.meshgrid(range(N), range(N))
    A = np.abs(x - y)
    if circular:
        # Compute the distance between nodes modulo N
        A = np.minimum(A, np.abs(A-N))

    # Apply the given function to the distance matrix to get the probability of connection
    return np.vectorize(func)(A)





