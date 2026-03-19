# %%
import taichi as ti
import porespy as ps
from kabs import (
    compute_permeability,
    solve_flow,
)
from kabs.utils import (
    write_flow_vtr,
    read_flow_vtr,
)
from kabs.plots import (
    render_flow,
)

# %%
ti.init(arch=ti.cpu)
im = ps.generators.cylinders(
    shape=[100, 100, 100],
    r=4,
    porosity=0.7,
    seed=0,
)

# %%
soln = solve_flow(
    im=im,
    direction="x",
    tol=1e-3,
)
res = compute_permeability(
    soln,
    direction="x",
)
print(f"Kabs = {res['k_lu']:.4f}")

# %%
write_flow_vtr("results", soln)

# %%
soln = read_flow_vtr("results.vtr", verbose=False)
fig = render_flow(soln, show=False, off_screen=True, save="flow.png")
fig.show()

# %%
