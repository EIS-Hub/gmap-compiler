from tqdm import trange
from gmap.utils import shuffle
import numpy as np
from gmap.hardware import Hardware_multicore
import time
from matplotlib import pyplot as plt


def benchmark(filename, test , hw, sigma, mu, debug=False, compute = False) :
    if compute :
        Cost_bf = np.loadtxt('./results/Noisy_diagonal_'+ filename + '_cost_bruteforce.csv' , delimiter=',')
        Cost_baseline = np.full([sigma[1]-sigma[0]+1, mu[1]-mu[0]+1], np.inf)
        Cost_algo = np.full([sigma[1]-sigma[0]+1, mu[1]-mu[0]+1], np.inf)
        tGmap = 0
        for i in trange(test, desc = "Iteration number", leave= True, disable = not debug):
            for sig in range(sigma[1]+1-sigma[0]):
                for m in range(mu[1]+1-mu[0]):
                    my_net = np.loadtxt('./data/Noisy_diagonal_' + filename +'/Noisy_diagonal_' + filename + '_sigma_' + str(sig+sigma[0]) + '_mu_' + str(m+mu[0]) + '.csv', delimiter=',')
                    my_net = shuffle(my_net)
                    Cost_baseline[sig, m] = hw.cost(my_net)

                    t0 = time.time()
                    A, cost = hw.mapping(my_net, minutes=0.1)
                    tGmap += time.time() - t0

                    Cost_algo[sig, m] = min(Cost_algo[sig, m], cost)

        np.savetxt('./results/Noisy_diagonal_'+ filename +"_cost_baseline.csv", Cost_baseline, delimiter=',')
        np.savetxt('./results/Noisy_diagonal_'+ filename +"_cost_bruteforce.csv", Cost_bf, delimiter=',')
        np.savetxt('./results/Noisy_diagonal_'+ filename +"_cost_algo.csv", Cost_algo, delimiter=',')


    Cost_baseline = np.loadtxt('./results/Noisy_diagonal_'+ filename +"_cost_baseline.csv", delimiter=',')
    Cost_bf =       np.loadtxt('./results/Noisy_diagonal_'+ filename +"_cost_bruteforce.csv", delimiter=',')
    Cost_algo =     np.loadtxt('./results/Noisy_diagonal_'+ filename +"_cost_algo.csv", delimiter=',')
    return Cost_baseline, Cost_bf, Cost_algo


def plot_comparison(Cost_baseline, Cost_bf, Cost_algo, mu, label, savefig = False):
    fig = plt.figure()
    plt.plot(np.arange(mu[0], mu[1] + 1), np.mean(Cost_baseline, axis=0), 'r-', label="N = " + label + " - Baseline")
    plt.plot(np.arange(mu[0], mu[1] + 1), np.mean(Cost_algo, axis=0), 'p-', label="N = " + label + " - GMap")
    plt.plot(np.arange(mu[0], mu[1] + 1), np.mean(Cost_bf, axis=0), 'g-', label="N = " + label + " - Ground Truth")

    plt.ylabel("Number of inter-core connections")
    plt.xlabel("Average node degree")
    plt.legend()
    if savefig : fig.savefig("./results/Noisy_diagonal_"+ label + "_Comparison.pdf",format = "pdf" )
    return fig


# Parameters
compute = False # if it has to all re-compute or if it can just rely on the pre-obtained datas
test = 10 # number of statistical repetitions

hw = Hardware_multicore(16, 4)
Cost_baseline_16, Cost_bf_16, Cost_algo_16 = benchmark('16', test = 1, hw = hw, sigma = [2,3], mu = [1,6], debug = True, compute = compute)
hw = Hardware_multicore(32, 4)
Cost_baseline_32, Cost_bf_32, Cost_algo_32 = benchmark('32', test = 1, hw = hw, sigma = [3,7], mu = [3,7], debug = True, compute = compute)
Cost_baseline_32_bis, Cost_bf_32_bis, Cost_algo_32_bis = benchmark('32_bis', test = 1, hw = hw, sigma = [3,5], mu = [2,10], debug = True, compute = compute)


fig1 = plot_comparison(Cost_baseline_16, Cost_bf_16, Cost_algo_16, [1,6], label = "16", savefig=True)
fig2 = plot_comparison(Cost_baseline_32, Cost_bf_32, Cost_algo_32, [3,7], label = "32", savefig=True)
fig3 = plot_comparison(Cost_baseline_32_bis, Cost_bf_32_bis, Cost_algo_32_bis, [2,10], label = "32_bis", savefig=True)
plt.show()