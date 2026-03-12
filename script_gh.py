import taichi as ti
import numpy as np
from kabs import solve_flow, compute_conductance

ti.init(arch=ti.cpu)

# Build a cylinder (pore=1, solid=0).
# Tube runs along axis 0 (the x-axis in the solver) so direction="x".
R_lu = 22
L_lu = 100
W = 50
H = 50
box = np.zeros([L_lu, W, H], dtype=int)
cy, cz = int(W/2), int(H/2)
for i in range(L_lu):
    for j in range(W):
        for k in range(H):
            if (j - cy)**2 + (k - cz)**2 <= R_lu**2:
                box[i, j, k] = 1  # pore   ← i is now the first (x) index

solver = solve_flow(
    im=box,
    direction="x",   # cylinder axis is numpy axis 0 = solver x
    export_vtk=False,
    tol=1e-5,
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
