import taichi as ti
import porespy as ps
from kabs import compute_permeability, solve_flow

ti.init(arch=ti.cpu)
im = ps.generators.cylinders(
    shape=[100, 100, 100], 
    r=10,
    porosity=0.7,
    seed=0,
)

solver = solve_flow(
    im=im, 
    direction="x", 
    tol=1e-3,
)
res = compute_permeability(
    solver._last_vtr,
    direction="x",
)
print(f"Kabs = {res['k_lu']:.4f}")
