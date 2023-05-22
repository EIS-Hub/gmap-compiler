import copy
import numpy as np


class Mapping:
    """
    mapping contains the actual data useful for a mapping

    Parameters:
    order (np.array): The order to apply to connectivity_matrix and to weight_matrix to get the mapping
    connectivity_matrix (2D np.array): The original unordered connectivity matrix of the network
    weight_matrix (2D np.array): The original unordered weight matrix of the network
    cost_tracker : Any object that help to track the actual cost of the actual mapping

    """

    def __init__(self, weight_matrix, connectivity_matrix=None, cost_tracker=None, cost=0):
        if connectivity_matrix is None :
            connectivity_matrix = 1 * (weight_matrix > 0)

        self.order = np.arange(len(weight_matrix))
        self._connectivity_matrix = connectivity_matrix
        self._weight_matrix = weight_matrix
        self.cost_tracker = cost_tracker
        self.cost = cost

    def copy(self):
        """
        Only the order and the cost_tracker has to be copied. They depend on the mapping.
        The connectivity_matrix and th  weight_matrix are the original unordered ones.
        """
        new = Mapping(self._weight_matrix, self._connectivity_matrix, cost=self.cost)
        new.order = copy.deepcopy(self.order)
        new.cost_tracker = copy.deepcopy(self.cost_tracker)
        return new

    @property
    def connectivity_matrix(self):
        return self._connectivity_matrix

    @connectivity_matrix.setter
    def connectivity_matrix(self, value):
        raise AttributeError("Cannot modify connectivity_matrix")

    @property
    def weight_matrix(self):
        return self._weight_matrix

    @weight_matrix.setter
    def weight_matrix(self, value):
        raise AttributeError("Cannot modify connectivity_matrix")

    def reordered_connectivity_matrix(self):
        return reorder(self.order, self._connectivity_matrix)

    def reordered_weight_matrix(self):
        return reorder(self.order, self._weight_matrix)


def reorder(order, A):
    """
    A function that reorders the rows and columns of a matrix according
    to a given order.

    Args:
    order (array-like): The desired order of the rows and columns.
    A (array-like): The input matrix to be reordered.

    Returns:
    A_new (array-like): The reordered matrix.
    """

    # Reorder the rows of the input matrix according to the given order.
    A_new = A.take(order, axis=0)

    # Reorder the columns of the reordered matrix according to the given order.
    A_new = A_new.take(order, axis=1)

    # Return the reordered matrix.
    return A_new


def shuffle(A):
    """
    A function that shuffles the rows and columns of a matrix randomly.

    Args:
    A (array-like): The input matrix to be shuffled.

    Returns:
    A_new (array-like): The shuffled matrix.
    """

    # Create a copy of the input matrix to avoid modifying the original.
    A_new = np.copy(A)

    # Generate a random permutation of indices for the rows and columns.
    arr = np.arange(len(A_new))
    np.random.shuffle(arr)

    # Reorder the rows and columns of the matrix according to the random permutation.
    A_new = reorder(arr, A_new)

    # Return the shuffled matrix.
    return A_new
