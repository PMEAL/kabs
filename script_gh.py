import taichi as ti
import numpy as np
from kabs import solve_flow, compute_conductance, plot_cross_section, add_streamlines
from porespy.tools import get_edt
import matplotlib.pyplot as plt

ti.init(arch=ti.cpu)
edt = get_edt()

# Build a cylinder (pore=1, solid=0).
# Tube runs along axis 0 (the x-axis in the solver) so direction="x".
Rp = 28
R_lu = 15
L_lu = 100
W = 80
H = 60
box = np.zeros([L_lu, W, H], dtype=int)
cy, cz = int(W/2), int(H/2)
for i in range(L_lu):
    for j in range(W):
        for k in range(H):
            if (j - cy)**2 + (k - cz)**2 <= R_lu**2:
                box[i, j, k] = 1
balls = np.ones_like(box, dtype=bool)
balls[0, cy, cz] = False
balls[-1, cy, cz] = False
balls = edt(balls) <= Rp
box[balls] = True

solver = solve_flow(
    im=box,
    direction="x",   # cylinder axis is numpy axis 0 = solver x
    export_vtk=False,
    tol=1e-4,
)
solver.export_VTK("cylinder")

results = compute_conductance(
    "cylinder.vtr",
    direction="x",
    dx_m=1e-6,
    mu_phys=1e-3,
)
# %%
g_LBM = results['g_SI']
print(f"\ng_LBM  = {g_LBM:.4e} m³/(Pa·s)")

# Analytical Hagen-Poiseuille check.
dx_m = 1e-6
mu = 1e-3
R_exact = R_lu * dx_m
L_phys = L_lu * dx_m
g_HP = np.pi * R_exact**4 / (8 * mu * L_phys)
print(f"g_HP) = {g_HP:.4e} m³/(Pa·s)")

# %%
view = plot_cross_section("cylinder.vtr", direction='x', axis=2)
fig, ax = plt.subplots()
ax.imshow(view/box[..., 30].T)
ax = add_streamlines(
    filename="cylinder.vtr", 
    axis=2, 
    ax=ax, 
    color='white', 
    linewidth=0.5,
    density=1.5,
)
