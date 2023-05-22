import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats


def plot_matrix(A, labels=False, all_ticks=False, extent=None, aspect=None):
    """
    A function that plots a matrix using matplotlib imshow.

    Args:
    A (array-like): The input matrix to be plotted.
    labels (bool): Whether to display labels in the matrix.
    all_ticks (bool): Whether to display all ticks on the x and y axes.
    extent (tuple): The (left, right, bottom, top) extent of the image.
    aspect (float): The aspect ratio of the axes.

    Returns:
    None
    """

    # Plot the matrix using imshow and the "autumn_r" colormap.
    plt.imshow(A, cmap="autumn_r", extent=extent, aspect=aspect)

    # If labels is True, display the value of each element in the matrix as a label.
    if labels:
        for (j, i), label in np.ndenumerate(A):
            plt.text(i, j, label, ha='center', va='center')

    # If all_ticks is True, display all ticks on the x and y axes.
    if all_ticks:
        plt.yticks(np.arange(len(A)))
        plt.xticks(np.arange(len(A)))

    # Show the plot.
    plt.show()


def info_graph(A):
    """
    A function that generates and displays basic information about a graph
    represented as an adjacency matrix.

    Args:
    A (array-like): The input adjacency matrix.

    Returns:
    None
    """

    # Convert the adjacency matrix to a graph representation using NetworkX.
    G = nx.from_numpy_array(A)

    # Compute and display the average number of edges per node.
    avg_edges = sum(sum(A)) / len(A)
    print("Average number of edges : ", avg_edges)

    # Display the adjacency matrix as a heatmap.
    plt.figure()
    plot_matrix(A)

    # Plot the degree distribution of the graph.
    plt.figure()
    degree_hist = nx.degree_histogram(G)
    plt.plot(degree_hist)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")


def expected_cost_difference_line(n, p, func_cost_line):
    """
    Computes the expectation of the difference of cost between two random lines or two random columns
    when the difference is negative.

    Args:
        n: The number of lines/columns.
        p: The probability of a connection.
        func_cost_line: The cost function of a single line/column.

    Returns:
        The expected cost difference.
    """
    esp = 0  # Initialize the expected difference of cost
    pmf_arr = stats.binom.pmf(range(n + 1), n, p)  # Compute the binomial distribution for each line/column
    for new_state_connections in range(n + 1):
        for old_state_connections in range(n + 1):
            # Compute the cost of the new and old state
            new_state_cost = func_cost_line(new_state_connections)
            old_state_cost = func_cost_line(old_state_connections)
            if new_state_cost > old_state_cost:
                # If the cost of the new state is greater than the cost of the old state, compute the probability
                # of each state and add the expected difference of cost
                prob_new_state = pmf_arr[new_state_connections]
                prob_old_state = pmf_arr[old_state_connections]
                esp += (new_state_cost - old_state_cost) * prob_new_state * prob_old_state
    return esp

