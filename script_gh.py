# %%
import porespy as ps
import taichi as ti
import numpy as np
from kabs import (
    solve_flow,
    solve_hydraulic_conductance,
    plot_cross_section,
    add_streamlines,
)
from porespy.tools import get_edt
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)
edt = get_edt()

# Build a cylinder (pore=1, solid=0).
# Tube runs along axis 0 (the x-axis in the solver) so direction="x".
Rp = 20
R_lu = 10
L_lu = 50
W = 50
H = 50
box = np.zeros([L_lu, W, H], dtype=int)
cy, cz = int(W / 2), int(H / 2)
for i in range(L_lu):
    for j in range(W):
        for k in range(H):
            if (j - cy) ** 2 + (k - cz) ** 2 < R_lu**2:
                box[i, j, k] = 1
# Add spheres to the ends
balls = np.ones_like(box, dtype=bool)
balls[0, cy, cz] = False
balls[-1, cy, cz] = False
balls = edt(balls) < Rp
box[balls] = True
box = ps.generators.conical_capillary(shape=[60, 60, 60], r=[25, 10], axis=0)
ps.imshow(box, axis=2)

# %%

from kabs import solve_hydraulic_conductance

res = solve_hydraulic_conductance(box, pad=40, direction="x")
print(res.keys())
print(res["report_text"])

# %%
