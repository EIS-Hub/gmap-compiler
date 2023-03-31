# gmap-compiler
A versatile, easy-to-use and open-source compiler that can efficiently map any arbitrary connectivity matrix to various hardware architectures.


## Quickstart

Install it first
```bash
pip install gmap-compiler  # from pypi

pip install -e git+https://github.com/EIS-Hub/gmap-compiler.git  # latest from github
```

See [examples/example_mapping.py](https://github.com/EIS-Hub/gmap-compiler/blob/main/examples/example_mapping.py) for examples.

```python
from gmap import Hardware
from gmap.matrix_generator import create_random


class My_Hardware(Hardware):
    """Implementation of a constrained hardware"""

hw = Hardware_multicore(size_hw, core = 4)
mapping, vioated_constrains = hw.mapping(connectivity_matrix)
```