# gmap-compiler
A versatile, easy-to-use and open-source compiler that can efficiently map any arbitrary connectivity matrix to various hardware architectures.

## Quickstart

Install it first
```bash
pip install gmap-compiler  # from pypi

pip install -e git+https://github.com/EIS-Hub/gmap-compiler.git  # latest from github
```


### Use pre-implmented Hardware
Pre-implemented Hardware are given in example to illustrate how to define the constrains of an Hardware.
See [examples/example_mapping.py](https://github.com/EIS-Hub/gmap-compiler/blob/main/examples/example_mapping.py) for examples.

#### Multicore Hardware with limited fan-in/fan-out

You can map to a generic multicore hardware defined by :
- The number of neurons per core
- The number of core
- The max fan-in
- The max fan-out

```python
from gmap.hardware import Multicore

hw = Multicore(n_neurons_core=size_hw // n_core, n_core=n_core, n_fanI=20, n_fanO=20)
mapping, vioated_constrains = hw.map(weight_matrix)
```

#### DYNAPSE Chip
Or you can map to a more complex contrained Hardware like DYNAPSE :

```python
from gmap.hardware import DYNAPSE

hw = DYNAPSE(N=size_hw, F=8, C=size_hw // n_core, K=16, alpha=1)
mapping, vioated_constrains = hw.map(weight_matrix)
```

#### Others
Let's wait the community to enrich the different hardware.


### Build your own Hardware
You can build your own Hardware and map a network on it :
```python
class My_Hardware(Hardware):
    """Implementation of a constrained hardware"""


hw = My_Hardware()
order, mapped_matrix, vioated_constrains = hw.mapping(weight_matrix)
```

For example, let's build multicore hardware where the number of intercore connections should be minimized.
Note that form this example, the complexity to compute the cost is not optimized.
See for full example.
```python
class my_Hardware(Hardware):
    """
    Define a custom Hardware class that inherits from the base Hardware class. For this example, let's define a multicore
    Hardware with all to all connections intra core and as little as possible extra core connections
    """
    def __init__(self, n_total, core):
        # Call the base constructor
        super(my_Hardware, self).__init__(n_total)

        # Create a mask to enforce hardware constraints
        self.Mask = 1 - 1 * create_multicore_mask(n_total, core)

    def cost(self, state):
        """
        Compute the cost of the current state. The cost reflects the number of hardware constraints violated if we mapped
        a network in its state. The object state is composed of the order of the nodes, the original unordered weight
        matrix qnd connectivity matrix. State also has a cost_helper to compute faster the cost.
        """

        # Compute the actual connectivity matrix.
        actual_connectivity_matrix = reorder(state.order, state.connectivity_matrix)

        # Compute the number of intercore connections.
        return np.sum(actual_connectivity_matrix * self.Mask)
```




