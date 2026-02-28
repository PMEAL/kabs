# `kabs`

`kabs` uses the lattice Boltzmann method to solve single phase, incompressible, creeping flow through a volumetric image of a porous material. The solution of this simulation can then be used to find the absolute permeability coefficient, $K_{abs}$, hence the name of this package.

```python
import taichi as ti
import porespy as ps
from kabs import compute_permeability, solve_flow

# Tell taichi how to run
ti.init(arch=ti.cpu)

# Generate a test image
im = ps.generators.cylinders([200, 200, 200], r=10, porosity=0.7)
im = im.astype(int)

f = "cylinders"
n = 15000
ax = "x"
solve_flow(
    im=im, 
    direction=ax, 
    n_steps=n, 
    output_prefix=f,
)
compute_permeability(f"{f}-{n}-{ax}.vtr", direction=ax)
```

![](taichi_lbm.png)

The LBM implementation used here is taken from [Taichi-LBM3D](https://github.com/yjhp1016/taichi_LBM3D) [[DOI]](z10.3390/fluids7080270) written by [Jianhui Yang](https://github.com/yjhp1016). The Taichi-LBM3D package can do phase change, multiphase flow, reactions, etc. `kabs` just pulls the single phase flow solver and wraps that in a simple function call designed just for finding the absolute permeability of an image. 
