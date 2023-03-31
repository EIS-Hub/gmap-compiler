from gmap.matrix_generator import *
from gmap.utils import *
from gmap.mapping import *

A = create_distance_dependant(64, minus_x_exponent(64 / 3, 2), True)

sa = Hardware_multicore(A, 4)
sa.set_schedule({'tmax': 50, 'tmin': 0.01, 'steps': 30000, 'updates': 10})
A_new, energy = sa.solve()
plot_matrix(A_new)
plt.show()


