import math
from scipy.special import comb
from math import comb

def compute_cost(connections, f):
    return connections - f if connections > f else 0

def expected_cost_difference(n, p, f):
    esp = 0
    for new_state_connections in range(n + 1):
        for old_state_connections in range(n + 1):
            new_state_cost = compute_cost(new_state_connections, f)
            old_state_cost = compute_cost(old_state_connections, f)
            if new_state_cost > old_state_cost:
                prob_new_state = comb(n, new_state_connections) * (p ** new_state_connections) * ((1 - p) ** (n - new_state_connections))
                prob_old_state = comb(n, old_state_connections) * (p ** old_state_connections) * ((1 - p) ** (n - old_state_connections))
                esp += (new_state_cost - old_state_cost) * prob_new_state * prob_old_state
    return esp



# Exemple d'utilisation
p = 0.5
n = int(3*256/4)
f = 95
result1 = expected_cost_difference(n, p, f)
print(result1)
result2 = -2*result1/math.log(0.98)
print(result2)