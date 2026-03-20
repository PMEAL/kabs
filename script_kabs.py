# %%
import taichi as ti
import porespy as ps
import matplotlib.pyplot as plt

import kabs

# %%
ti.init(arch=ti.cpu)
im = ps.generators.cylinders(
    shape=[100, 100, 100],
    r=4,
    porosity=0.7,
    seed=0,
)

# %%
soln = kabs.solve_flow(
    im=im,
    direction="z",
    tol=1e-3,
)
res = kabs.compute_permeability(
    soln,
    direction="z",
)
print(f"Kabs = {res['k_lu']:.4f}")

# %%
kabs.utils.write_flow_vtr("results", soln)

# %%
soln = kabs.utils.read_flow_vtr("results.vtr", verbose=False)
fig = kabs.plots.render_flow(soln, cmap=plt.cm.turbo, show=False, off_screen=False, save=None)
fig.show()

# %%
