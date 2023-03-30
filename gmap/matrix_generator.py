import networkx as nx
import numpy as np
from scipy.stats import norm


# Function that create a random graph
def create_random(N, k_avg):
    ret = np.zeros((N, N))
    A = np.random.uniform(size=(N, N))
    ret[:N, :N] = np.array(1 * (A < (k_avg / N)))
    return ret


# Create a Mask representing a network of N nodes with C communities
def create_communities(N, C):
    A = np.zeros((N, N))
    for i in range(C):
        A[(N // C) * i:(N // C) * (i + 1), (N // C) * i:(N // C) * (i + 1)] = 1
    return A.astype(np.int64)


# Function that create a Small wolrd Watts Stoggart network.
def create_WS(N, k_avg, p_drop=0.5):
    A = np.zeros((N, N))
    G = nx.connected_watts_strogatz_graph(N, k_avg, p_drop)
    A[:N, :N] = nx.to_numpy_array(G)  # [K:,K:]
    return A


# Create a Barabasi-Albert network
def BA(N, M0, m):
    """
        N, number of nodes in the final network.
        M0, initial connected network of M0 nodes.
        m, Each new node is connected to m existing nodes.
    """
    assert (M0 < N)
    assert (m <= M0)

    # adjacency matrix
    AM = np.zeros((N, N))

    for i in range(0, M0):
        for j in range(i + 1, M0):
            AM[i, j] = 1
            AM[j, i] = 1

    # add 'c' node
    for c in range(M0, N):
        Allk = np.sum(AM)  # all  degree    Eki
        ki = np.sum(AM, axis=1)  # ki each degree for node i

        pi = np.zeros(c, dtype=np.float)  # probability
        for i in range(0, c):
            pi[i] = ki[i] / (Allk * 1.0)
        # print pi

        # connect m edges.
        for d in range(0, m):
            rand01 = np.random.random()  # [0,1.0)

            sumpi = 0.0
            for g in range(0, c):
                sumpi += pi[g]
                if sumpi > rand01:  # connect 'c' node with 'g' node.
                    if AM[c, g] == 0 and AM[g, c] == 0:
                        AM[c, g] += 1
                        AM[g, c] += 1
                        break

    return AM


# Build a Barabasi-Albert Network with same M0 = m and with the right average number of edges per node.
def create_BA(N, k_avg):
    M = int((-1 + 2 * N - ((1 - 2 * N) ** 2 - 4 * N * k_avg) ** (1 / 2)) // 2) + 1
    return BA(N, M, M)


def gaussian_connect(N, sigma, mu, tol=1e-6, quantile_max=30):
    alpha = 3 / quantile_max ** 2 - norm.cdf(-quantile_max) / norm.pdf(-quantile_max) * 3 / quantile_max
    assert mu ** 2 / 12 * (1 + alpha) < sigma ** 2, "Error: mu ** 2 / 12 > sigma ** 2"
    assert sigma ** 2 < N ** 2 / 12, "Error: sigma ** 2 > N ** 2 / 12"
    mu_target = mu
    sigma_target = sigma
    margin = sigma * np.sqrt(np.maximum(np.log(mu ** 2 / 2 / np.pi / sigma ** 2), 0))
    while True:
        cdf_diff = norm.cdf(- margin / sigma) - norm.cdf(- N / 2 / sigma)
        if margin:
            mu = np.sqrt(2 * np.pi) * sigma * np.exp(margin ** 2 / 2 / sigma ** 2)
        else:
            mu = mu_target / 2 / cdf_diff
        b = - mu * np.sqrt(2 / np.pi) * N / 2 * np.exp(- (N / 2) ** 2 / 2 / sigma ** 2) / mu_target
        c = 2 / 3 * margin ** 3 / mu_target - sigma_target ** 2
        if np.abs(2 * mu * cdf_diff + 2 * margin - mu_target) < tol and \
                np.abs(sigma ** 2 + b * sigma + c) < tol:
            break
        sigma = np.sqrt(- (c + b * sigma))
        if margin:
            margin = np.maximum(mu_target / 2 - mu * cdf_diff, 0)
        else:
            margin = sigma * np.sqrt(np.maximum(np.log(mu ** 2 / 2 / np.pi / sigma ** 2), 0))
    x, y = np.meshgrid(range(N), range(N))
    d = np.minimum(np.mod(x - y, N), np.mod(y - x, N))
    return np.random.rand(N, N) <= mu / np.sqrt(2 * np.pi) / sigma * np.exp(- d ** 2 / 2 / sigma ** 2)


# Create a graph in which the probability of having an edge between any given pair of nodes is a Guassian distribution with respect to the distance between them.
def create_gaussian_connect(N, k_avg):
    A = np.zeros((N, N))
    quantile_max = 30
    alpha = 3 / quantile_max ** 2 - norm.cdf(-quantile_max) / norm.pdf(-quantile_max) * 3 / quantile_max
    min_sigma = (k_avg ** 2 / 12 * (1 + alpha)) ** (1 / 2)
    max_sigma = (N ** 2 / 12) ** (1 / 2)
    g = gaussian_connect(N, (min_sigma + max_sigma) * (1 / 2), k_avg)
    A[:N, :N] = g
    return A


# Create a network with the number of connection depending of the function f along the distance
# example of use :
# A = create_distance_dependant(N, minus_x_square(N, 3) , circular = True)
def create_distance_dependant(N, func, circular=True):
    x, y = np.meshgrid(range(N), range(N))
    if circular:
        A = np.minimum(np.mod(x - y, N), np.mod(y - x, N))
    else:
        A = np.abs(x - y)
    return np.vectorize(func)(A)


def linear(x): return x


def uni_proba(x): return np.random.uniform(0, 1) < x


def minus_x_exponent(N, exp):
    return lambda x: uni_proba(((N - x) / N) ** exp)
