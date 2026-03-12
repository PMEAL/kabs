# %% Cross-section plot at tube midpoint ───────────────────────────────────────
import matplotlib.pyplot as plt
from kabs._compute_permeability import _parse_xml_arrays, _read_array
import re
import numpy as np

with open("cylinder.vtr", "rb") as fh:
    raw = fh.read()
marker = raw.index(b'<AppendedData encoding="raw">')
binary_start = raw.index(b"_", marker) + 1
xml_header = raw[:marker].decode("utf-8", errors="replace")
arrays = _parse_xml_arrays(xml_header)
m = re.search(r'WholeExtent="(\d+) (\d+) (\d+) (\d+) (\d+) (\d+)"', xml_header)
x0, x1, y0, y1, z0, z1 = (int(v) for v in m.groups())
nx, ny, nz = x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1
velocity = _read_array(raw, binary_start, arrays, "velocity", nx, ny, nz)
vx = velocity[..., 0]   # shape (nx, ny, nz)

# Longitudinal slice through the tube centre: vx at j=cy, showing x vs k
vx_long = vx[:, cy, :]   # shape (nx, nz): x along rows, k along columns

# Analytical Poiseuille — fully-developed profile depends only on k distance from centre
gradP = (0.01 / 3) / nx
nu_lu = 1.0 / 6.0
kk = np.arange(nz)
r2_1d = (kk - cz)**2
vx_poiseuille_1d = np.where(r2_1d <= R_lu**2, (R_lu**2 - r2_1d) / (4 * nu_lu) * gradP, 0.0)
vx_poiseuille_long = np.broadcast_to(vx_poiseuille_1d[np.newaxis, :], (nx, nz))

vmax = max(vx_long.max(), vx_poiseuille_long.max())
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(vx_long.T, origin="lower", aspect="auto", cmap="hot",
                     vmin=0, vmax=vmax, extent=[0, nx, 0, nz])
axes[0].set_xlabel("x (flow direction)")
axes[0].set_ylabel("k (radial)")
axes[0].set_title("LBM vx — longitudinal slice at j=cy")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(vx_poiseuille_long.T, origin="lower", aspect="auto", cmap="hot",
                     vmin=0, vmax=vmax, extent=[0, nx, 0, nz])
axes[1].set_xlabel("x (flow direction)")
axes[1].set_title("Poiseuille (fully-developed)")
plt.colorbar(im1, ax=axes[1])

diff = vx_long - vx_poiseuille_long
im2 = axes[2].imshow(diff.T, origin="lower", aspect="auto", cmap="bwr",
                     vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(),
                     extent=[0, nx, 0, nz])
axes[2].set_xlabel("x (flow direction)")
axes[2].set_title("LBM − Poiseuille")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("vx_longitudinal.png", dpi=150)
plt.show()
print("Longitudinal plot saved to vx_longitudinal.png")

# %% Radial profile comparison at tube midpoint ────────────────────────────────
mid = nx // 2
vx_mid = vx[mid, :, :]   # cross-section at midpoint, shape (ny, nz)

jj, kk2 = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
r = np.sqrt((jj - cy)**2 + (kk2 - cz)**2)

# Only pore voxels inside the cylinder
pore = r <= R_lu
r_pore      = r[pore].ravel()
vx_lbm      = vx_mid[pore].ravel()
vx_exact    = ((R_lu**2 - r_pore**2) / (4 * nu_lu) * gradP)

# Sort by radius for a clean line plot
order       = np.argsort(r_pore)
r_s         = r_pore[order]
vx_lbm_s   = vx_lbm[order]
vx_exact_s  = vx_exact[order]

print(f"\n{'r':>6}  {'vx_LBM':>12}  {'vx_analytical':>14}  {'ratio':>7}")
print("-" * 47)
# Print a sample at several radii
for r_target in np.linspace(0, R_lu - 1, 8):
    idx = np.argmin(np.abs(r_s - r_target))
    ratio = vx_lbm_s[idx] / vx_exact_s[idx] if vx_exact_s[idx] > 0 else float("nan")
    print(f"{r_s[idx]:6.2f}  {vx_lbm_s[idx]:12.4e}  {vx_exact_s[idx]:14.4e}  {ratio:7.4f}")

fig2, ax = plt.subplots(figsize=(7, 4))
ax.scatter(r_s, vx_lbm_s, s=4, alpha=0.5, label="LBM")
ax.plot(r_s, vx_exact_s, "r-", lw=2, label="Poiseuille (analytical)")
ax.set_xlabel("r (voxels from centre)")
ax.set_ylabel("vx (lu/ts)")
ax.set_title(f"Radial velocity profile at x={mid}  (R={R_lu})")
ax.legend()
plt.tight_layout()
plt.savefig("vx_radial_profile.png", dpi=150)
plt.show()
print("Radial profile saved to vx_radial_profile.png")

# %%