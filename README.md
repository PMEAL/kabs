# `kabs`

`kabs` computes the absolute (Darcy) permeability and effective diffusivity of a porous
material from its 3D tomographic image using the Lattice Boltzmann Method (LBM). Given a
binary voxel image of the pore space, it solves single-phase incompressible creeping flow
(for permeability) and passive scalar diffusion (for effective diffusivity), returning
results in lattice units or physical units.

![](taichi_lbm.png)

The LBM implementation is adapted from
[Taichi-LBM3D](https://github.com/yjhp1016/taichi_LBM3D)
([DOI](https://doi.org/10.3390/fluids7080270)) by Jianhui Yang.

---

## Installation

```bash
git clone https://github.com/your-org/kabs.git
cd kabs
pip install -e .
```

Key dependencies: `taichi` (GPU/CPU acceleration), `numpy`, `pyevtk`.
Optional: `porespy` (used in the examples below to generate synthetic images).

---

## Quick start

```python
import taichi as ti
import porespy as ps
from kabs import solve_flow, compute_permeability

ti.init(arch=ti.cpu)  # use ti.gpu for GPU acceleration

# Generate a synthetic test image (1 = pore, 0 = solid)
im = ps.generators.cylinders([200, 200, 200], r=10, porosity=0.7).astype(int)

# Run the LBM simulation.  Saves result to "sample.vtr" when done.
solver = solve_flow(im, direction="x", export_vtk=False)
solver.export_VTK("sample")

# Compute permeability from the saved file
results = compute_permeability("sample.vtr", direction="x")
print(f"k = {results['k_lu']:.4e} voxels²")
```

---

## Common usage patterns

### GPU acceleration

Taichi supports CUDA, Metal, and Vulkan backends.  Switch by changing `ti.init`:

```python
ti.init(arch=ti.gpu)   # picks the best available GPU backend
ti.init(arch=ti.cuda)  # CUDA explicitly
```

### Physical units

Pass the voxel size `dx_m` (in metres) to `compute_permeability` to get results in
m² and milliDarcy:

```python
results = compute_permeability(
    "sample.vtr",
    direction="x",
    dx_m=2.85e-6,   # 2.85-micron voxels, typical for micro-CT
)
print(f"k = {results['k_mD']:.2f} mD")
print(f"k = {results['k_m2']:.4e} m²")
```

### Convergence

By default `solve_flow` stops early once the velocity field has converged to within a
relative tolerance of 1e-3 (i.e. `delta|v| / |v| < 1e-3`).  The actual number of
steps run is printed and reflected in the auto-generated VTR filename.

```python
# Tighten or loosen the tolerance
solver = solve_flow(im, direction="x", tol=1e-4)  # tighter
solver = solve_flow(im, direction="x", tol=1e-2)  # faster, coarser

# Disable early stopping and always run n_steps
n = 5000
solve_flow(im, direction="x", n_steps=n, tol=None, output_prefix="sample")
compute_permeability(f"sample-{n}-x.vtr", direction="x")
```

The convergence check fires every `log_every` steps (default 500), so the true stopping
point is rounded to that interval.

### Full permeability tensor

For anisotropic materials, run all three directions:

```python
for ax in ("x", "y", "z"):
    solver = solve_flow(im, direction=ax, export_vtk=False)
    solver.export_VTK(f"sample_{ax}")

kx = compute_permeability("sample_x.vtr", direction="x", dx_m=2.85e-6)
ky = compute_permeability("sample_y.vtr", direction="y", dx_m=2.85e-6)
kz = compute_permeability("sample_z.vtr", direction="z", dx_m=2.85e-6)

print(f"Kx={kx['k_mD']:.2f}  Ky={ky['k_mD']:.2f}  Kz={kz['k_mD']:.2f}  mD")
```

### Memory-efficient sparse storage

For images with a high solid fraction, enable sparse storage so only pore voxels are
allocated in GPU memory:

```python
solver = solve_flow(im, direction="x", sparse=True)
```

---

## Effective diffusivity

`kabs` can also compute effective diffusivity D_eff using a D3Q7 BGK scalar LBM — a
simpler and faster kernel than the D3Q19 MRT used for flow.

```python
from kabs import solve_diffusion, compute_effective_diffusivity

# Run diffusion simulation (concentration-driven, analogous to pressure-driven flow)
solver = solve_diffusion(im, direction="x")

# Compute D_eff from the saved .vtr file
results = compute_effective_diffusivity("LB_Diffusion-<step>-x.vtr", direction="x")
print(f"D_eff/D_0   = {results['D_eff_norm']:.4f}")
print(f"Tortuosity  = {results['tortuosity']:.4f}")
```

### Physical D_eff

Pass the bulk diffusivity `D0_m2s` to get results in m²/s:

```python
# O₂ in air at 25 °C
results = compute_effective_diffusivity(
    "LB_Diffusion-<step>-x.vtr",
    direction="x",
    D0_m2s=2.1e-5,
)
print(f"D_eff = {results['D_eff_m2s']:.4e} m²/s")
```

### Return values

`compute_effective_diffusivity` returns a dict:

| Key               | Description                                               |
|-------------------|-----------------------------------------------------------|
| `porosity`        | Pore volume fraction (dimensionless)                      |
| `D_eff_norm`      | D_eff / D_0 (dimensionless ratio, primary output)         |
| `formation_factor`| F = D_0 / D_eff                                           |
| `tortuosity`      | τ = F / φ = D_0 / (D_eff × φ), always > 1                |
| `D_eff_m2s`       | D_eff in m²/s (`None` if `D0_m2s` not given)             |

### Diffusivity parameter D

The D3Q7 BGK solver uses a lattice diffusivity `D` (default `D=1/4`, τ_D=1.5).
This is the optimal value: steps to convergence scale as **L²/D**, so larger `D`
is faster, but accuracy degrades above τ_D ≈ 2 (D ≈ 3/8).

| D   | τ_D  | Speed vs D=1/6 | Notes                         |
|-----|------|----------------|-------------------------------|
| 1/6 | 1.17 | 1×             | matches flow solver convention |
| **1/4** | **1.50** | **~1.5×** | **default, sweet spot**   |
| 1/3 | 1.83 | ~2×            | slightly less accurate        |
| 1/2 | 2.50 | ~3×            | accuracy starts to degrade    |

---

## Return values (permeability)

`compute_permeability` returns a dict:

| Key        | Description                                      |
|------------|--------------------------------------------------|
| `porosity` | Pore volume fraction (dimensionless)             |
| `u_darcy`  | Darcy (superficial) velocity [lattice units]     |
| `u_pore`   | Mean pore-space velocity [lattice units]         |
| `k_lu`     | Permeability in lattice units (voxels²)          |
| `k_m2`     | Permeability in m²  (`None` if `dx_m` not given) |
| `k_mD`     | Permeability in milliDarcy (`None` if no `dx_m`) |

---

## How it works

1. A pressure-driven flow is imposed by fixing density (ρ_in = 1.00, ρ_out = 0.99) on
   opposite faces of the domain along the chosen axis; the other four faces are periodic.
2. The D3Q19 MRT-LBM collision operator evolves the distribution functions to steady
   state.  Solid voxels use bounce-back boundary conditions.
3. Darcy's law is applied to the converged velocity field:

   **K = u_D · μ / |∇P|**

   where u_D is the volume-averaged (Darcy) velocity and |∇P| = Δρ · c_s² / L with
   c_s² = 1/3 for D3Q19.

4. The result in lattice units is scaled to m² (or milliDarcy) using the physical
   voxel size dx_m.
