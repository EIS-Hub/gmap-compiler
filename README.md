# gmap-compiler
A versatile, easy-to-use and open-source compiler that can efficiently map any arbitrary connectivity matrix to various hardware architectures.


## Quickstart

Install it first
```bash
pip install gmap-compiler  # from pypi

pip install -e git+https://github.com/EIS-Hub/gmap-compiler.git  # latest from github
```

See [examples/example_mapping.py](https://github.com/EIS-Hub/gmap-compiler/blob/main/examples/matrix_generator_example.py) for examples.

```python
from gmap import Hardware
from gmap.matrix_generator import create_random

class My_Hardware(Hardware):
    """Implementation of a constrained hardware"""

connectivity_matrix = create_random(N = 256, k_avg = 32)
mapping, vioated_constrains = My_Hardware.mapping(connectivity_matrix)
```