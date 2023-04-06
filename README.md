# gmap-compiler
A versatile, easy-to-use and open-source compiler that can efficiently map any arbitrary connectivity matrix to various hardware architectures.


## Quickstart

Install it first
```bash
pip install gmap-compiler  # from pypi

pip install -e git+https://github.com/EIS-Hub/gmap-compiler.git  # latest from github
```


You can build your own Hardware and map a network on it :
```python
class My_Hardware(Hardware):
    """Implementation of a constrained hardware"""


hw = My_Hardware()
mapping, vioated_constrains = hw.mapping(connectivity_matrix)
```

Pre-implemented Hardware are given in example to illustrate how to define the constrains of an Hardware.
See [examples/example_mapping.py](https://github.com/EIS-Hub/gmap-compiler/blob/main/examples/example_mapping.py) for examples.

You can map to a generic multicore hardware defined by :
- The number of neurons per core
- The number of core
- The max fan-in
- The max fan-out


```python
from gmap.hardware import Hardware_generic

hw = Hardware_generic(n_neurons_core = size_hw // n_core, n_core=n_core, n_fanI=20, n_fanO=20)
mapping, vioated_constrains = hw.mapping(connectivity_matrix)
```


Or you can map to a more complex contrained Hardware like DYNAPS :
```python
from gmap.hardware import Hardware_DYNAPS

hw = Hardware_DYNAPS(N = size_hw, F = 8, C = size_hw//n_core, K = 16, alpha = 1)
mapping, vioated_constrains = hw.mapping(connectivity_matrix)
```





