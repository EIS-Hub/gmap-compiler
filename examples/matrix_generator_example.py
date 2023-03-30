import matplotlib.pyplot as plt

from gmap.utils import *
from gmap.matrix_generator import *

N = 256
K = 64

info_graph(create_random(N, K))
info_graph(create_communities(N, C=8))
info_graph(create_WS(N, K))
info_graph(create_BA(N, K))
info_graph(create_gaussian_connect(N, K))
info_graph(create_distance_dependant(N, minus_x_exponent(N / 3, 2), circular=True))
# plt.show()
