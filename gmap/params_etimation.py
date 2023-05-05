import math
import scipy.stats as stats
import warnings
from numba import jit

warnings.filterwarnings("ignore")


@jit
def compute_cost(connections, f):
    return connections - f if connections > f else 0

@jit
def expected_cost_difference(n, p, f):
    esp = 0
    pmf_arr = stats.binom.pmf(range(n+1), n, p)
    for new_state_connections in range(n + 1):
        for old_state_connections in range(n + 1):
            new_state_cost = compute_cost(new_state_connections, f)
            old_state_cost = compute_cost(old_state_connections, f)
            if new_state_cost > old_state_cost:
                prob_new_state = pmf_arr[new_state_connections]
                prob_old_state = pmf_arr[old_state_connections]
                esp += (new_state_cost - old_state_cost) * prob_new_state * prob_old_state
    return esp

@jit
def expected_cost_difference_low(n, p):
    esp = 0
    pmf_arr = stats.binom.pmf(range(n+1), n, p)
    for new_state_connections in range(1, n + 1):
        old_state_connections = new_state_connections-1
        prob_new_state = pmf_arr[new_state_connections]
        prob_old_state = pmf_arr[old_state_connections]
        esp += prob_new_state * prob_old_state
    return esp



# Exemple d'utilisation
p = 1/2
n = int(3*2000/4)
f = 0

dE = expected_cost_difference(n, p, f)
result_high = -2*dE/math.log(0.98)
print(math.log(1-stats.norm.cdf(6)))
print(result_high)

dE = expected_cost_difference_low(n, p)
print(dE)
result_low = -dE/math.log(1-stats.norm.cdf(6))
print(result_low)

