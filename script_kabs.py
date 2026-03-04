import taichi as ti
import porespy as ps
from kabs import compute_permeability, solve_flow

# Tell taichi how to run
ti.init(arch=ti.cpu)

# Generate a test image
im = ps.generators.cylinders(
    shape=[200, 200, 200], 
    r=10, porosity=0.7,
)
im = im.astype(int)

f = "cylinders"
n = 15000
ax = "z"
solve_flow(
    im=im, 
    direction=ax, 
    n_steps=n, 
    output_prefix=f,
)
compute_permeability(f"{f}-{n}-{ax}.vtr", direction=ax)
