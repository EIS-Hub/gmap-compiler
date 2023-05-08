from gmap.hardware import Hardware_generic
from gmap.matrix_generator import create_communities
import numpy as np
from gmap.utils import plot_matrix
from matplotlib import pyplot as plt
from gmap.utils import shuffle
from tqdm import trange
import math


def create_test_matrix(n, c, density):
    skip = int(1 / density)
    half_full = np.array([[1 if (i + j) % skip == 0 else 0 for i in range(n)] for j in range(n)])
    return 1 * ((create_communities(n, c) + half_full) > 0)


def compute_result(n, filename, epoch_max, epoch_test, stat_test, compute=False):
    if compute:
        A = create_test_matrix(n, 4, 1/2)
        result = np.zeros(epoch_test+1)

        hw = Hardware_generic(n_neurons_core=n // 4, n_core=4, max_fanI=0, max_fanO=0)
        T_min, T_max = hw.get_temperature(A)
        mapping = hw.mapping(A, params={'tmax': T_max, 'tmin': T_min, 'steps': 0, 'updates': 0})
        result[0] = mapping['violated_constrains']
        A = shuffle(A)

        for t in trange(stat_test):
            for i, s in enumerate(np.linspace(epoch_max//epoch_test, epoch_max, epoch_test).astype(int)):
                params = {'tmax': T_max, 'tmin': T_min, 'steps': s, 'updates': 0}
                mapping = hw.mapping(A, params=params)
                result[i+1] += mapping['violated_constrains']

        result[1:] /= stat_test
        np.savetxt(filename, result, delimiter=',')

    return np.loadtxt(filename, delimiter=',')


epoch_test = 50
epoch_max = 100000


fig = plt.figure()
# Set font size
font = {'size': 14}
plt.rc('font', **font)
plt.rcParams['lines.linewidth'] = 2
x_axis = np.linspace(epoch_max//epoch_test, epoch_max, epoch_test).astype(int)


result = compute_result(4096, filename = './convergence4096_105.csv',  epoch_max = epoch_max, epoch_test = epoch_test, stat_test = 20, compute=False)
plt.plot(x_axis, 100*(result[1:]-result[0])/result[1:], label = 'N = 4096')

result = compute_result(1024, filename = './convergence2048_105.csv',  epoch_max = epoch_max, epoch_test = epoch_test, stat_test = 20, compute=False)
plt.plot(1/(np.log(x_axis)), 100*(result[1:]-result[0])/result[1:], label = 'N = 2048')

result = compute_result(1024, filename = './convergence1024_105.csv',  epoch_max = epoch_max, epoch_test = epoch_test, stat_test = 20, compute=False)
plt.plot(1/(np.log(x_axis)), 100*(result[1:]-result[0])/result[1:], label = 'N = 1024')

result = compute_result(512, filename = './convergence512_105.csv', epoch_max = epoch_max, epoch_test = epoch_test, stat_test = 20, compute=False)
plt.plot(x_axis, 100*(result[1:]-result[0])/result[1:], label = 'N = 512')

result = compute_result(256, filename = './convergence256_105.csv', epoch_max = epoch_max, epoch_test = epoch_test, stat_test = 20, compute=False)
plt.plot(x_axis, 100*(result[1:]-result[0])/result[1:], label = 'N = 256')

plt.xlabel('Number of steps')
plt.ylabel('Relative error wrt the optimum [%]')


plt.plot()

plt.legend()
plt.show()
fig.savefig("Comparison_convergence.pdf", format = "pdf")

