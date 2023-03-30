from gmap.matrix_generator import *
from gmap.utils import *
from gmap.mapping import *

A = create_distance_dependant(64, minus_x_exponent(64 / 3, 2), True)
plt.figure()
plot_matrix(A)
plt.show()

sa = Hardware_multicore(A, 4)
sa.set_schedule({'tmax': 50, 'tmin': 0.00000001, 'steps': 3000000, 'updates': 50})
A_new, energy = sa.anneal()
plt.figure()
plot_matrix(A_new)
plt.show()


