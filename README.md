![](kabs_logo.png)

[![Tests](https://github.com/PMEAL/kabs/actions/workflows/tests.yml/badge.svg)](https://github.com/PMEAL/kabs/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/PMEAL/kabs/branch/dev/graph/badge.svg)](https://codecov.io/gh/PMEAL/kabs)


`kabs` computes the absolute (Darcy) permeability of a porous material from its 3D tomographic image using the Lattice Boltzmann Method (LBM). Given a boolean voxel image of the pore space, it solves single-phase incompressible creeping flow, returning results in lattice units or physical units.

![](taichi_lbm.png)

Two LBM implementations are offered. One is adapted from
[Taichi-LBM3D](https://github.com/yjhp1016/taichi_LBM3D)
([DOI](https://doi.org/10.3390/fluids7080270)) by Jianhui Yang. The other is adapted from the [XLB package](https://github.com/Autodesk/XLB) offered by Autodesk.

---

## Installation

```bash
git clone https://github.com/PMEAL/kabs.git
cd kabs
uv sync
```

Alternatively, install with `pip install -e .`. This installs both solvers:
Taichi and XLB, with `warp-lang==1.10.0` pinned for compatibility with XLB.

Key dependencies: `taichi` (GPU/CPU acceleration), `xlb`, `numpy`, `pyevtk`.
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

# Run the LBM simulation and get a FlowResult back
result = solve_flow(im, direction="x")

# Optionally save to a VTR file for later inspection
result.export_to_vtk("sample")

# Compute permeability directly from the FlowResult
results = compute_permeability(result)
print(f"k = {results['k_lu']:.4e} voxels²")
```

---

## Common usage patterns

### Taichi acceleration

The default solver uses Taichi. Initialize Taichi before its first solve and
select the architecture appropriate to your hardware:

```python
ti.init(arch=ti.gpu)   # picks the best available GPU backend
ti.init(arch=ti.cuda)  # CUDA explicitly
ti.init(arch=ti.metal) # Apple Silicon GPU
```

### Solver backends

`solve_flow` uses the Taichi implementation by default. Select XLB through the
same API; its default JAX compute backend works on CPU and supports multi-GPU
execution on supported accelerator platforms:

```python
# Default: solve_flow(im, backend="taichi")
result = solve_flow(im, backend="xlb", compute_backend="jax")
```

The implementation-specific entry points, `solve_flow_taichi` and
`solve_flow_xlb`, remain available for advanced use. Taichi-only `storage`,
`tile_size`, and `sparse` options are not supported by XLB.

XLB's `compute_backend="warp"` targets NVIDIA CUDA; it is not an Apple Metal
backend. On macOS, use `compute_backend="jax"` (the default), which runs on
CPU with the standard JAX installation. To use Taichi Metal and XLB in the
same comparison, prefer separate notebook kernels or processes.

### Collision models

The Taichi solver supports MRT and SRT/BGK collision operators:

```python
# Backwards-compatible default
result_mrt = solve_flow(im, backend="taichi", collision_model="mrt")

# Faster distribution-space collision
result_srt = solve_flow(im, backend="taichi", collision_model="srt")
```

When `collision_model` is omitted, Taichi uses MRT and XLB uses its native
SRT/BGK operator. XLB does not implement MRT. The effective operator is stored
as `result.collision_model` and is retained by KABS VTR export/import.

SRT removes the two dense D3Q19 moment transformations. On the Apple M3 CPU
benchmark it is about 2.2x faster than MRT; the tested Metal workloads are
effectively tied because that path appears bandwidth-bound. Both models retain
the same two D3Q19 population buffers, so SRT does not materially reduce
allocated simulation memory. See [the benchmark methodology and preliminary
results](https://github.com/PMEAL/kabs/blob/dev/benchmarks/README.md).

At the default viscosity `nu=1/6` (`tau=1`), SRT agrees with the analytical
bundle-of-tubes permeability within 3% and with MRT within 1%. Unlike the
configured MRT operator, SRT bounce-back error depends on relaxation time.
Validate accuracy for the viscosity and geometry used in production rather
than assuming permeability is independent of `tau` numerically.

### Physical units

Pass the voxel size `dx_m` (in metres) to `compute_permeability` to get results in
m² and milliDarcy:

```python
result = solve_flow(im, direction="x")
results = compute_permeability(
    result,
    dx_m=2.85e-6,   # 2.85-micron voxels, typical for micro-CT
)
print(f"k = {results['k_mD']:.2f} mD")
print(f"k = {results['k_m2']:.4e} m²")
```

### Convergence

Convergence criteria are composable. Each non-`None` tolerance enables its
criterion, all enabled criteria must pass together, and two consecutive passing
checks are required. By default only the componentwise velocity-field change is
enabled with a relative tolerance of `1e-3`.

```python
# Velocity-only convergence
result = solve_flow(im, direction="x", velocity_tol=1e-3)

# Permeability stability with velocity and mass-conservation safeguards
result = solve_flow(
    im,
    direction="x",
    velocity_tol=1e-2,
    k_tol=5e-3,
    flux_tol=5e-3,
    convergence_every=500,
)

# Permeability and flux without allocating a previous-velocity field
result = solve_flow(
    im,
    velocity_tol=None,
    k_tol=5e-3,
    flux_tol=5e-3,
)

# Disable convergence monitoring and run exactly n_steps
result = solve_flow(
    im,
    n_steps=5000,
    velocity_tol=None,
    k_tol=None,
    flux_tol=None,
)
compute_permeability(result)
```

`convergence_every` controls numerical sampling independently of `log_every`.
Permeability uses the Darcy velocity over the full domain, while flux imbalance
uses axis-oriented mass flux on the inlet and outlet pressure faces. The result's
`converged`, criteria, tolerances, pass streak, and exact `n_iterations` are also
preserved by VTR export/import.

The former `tol` argument remains as a deprecated alias for `velocity_tol`.

To save the converged result to a VTR file, call `export_to_vtk` on the returned object:

```python
result = solve_flow(im, direction="x", velocity_tol=1e-4)
result.export_to_vtk("sample")  # writes sample-<step>-x.vtr
```

### Full permeability tensor

For anisotropic materials, run all three directions:

```python
results = {}
for ax in ("x", "y", "z"):
    result = solve_flow(im, direction=ax)
    results[ax] = compute_permeability(result, dx_m=2.85e-6)

print(f"Kx={results['x']['k_mD']:.2f}  Ky={results['y']['k_mD']:.2f}  Kz={results['z']['k_mD']:.2f}  mD")
```

### Loading results from a VTR file

`solve_flow` returns a `FlowResult` you can use immediately, but if you have a
previously saved `.vtr` file you can reload it with `read_flow_vtr`:

```python
from kabs import read_flow_vtr, compute_permeability

result = read_flow_vtr("sample-1000-x.vtr")
results = compute_permeability(result, dx_m=2.85e-6)
```

### Storage layouts

Choose the field layout based on image size and solid fraction:

- `dense` (the default) is fastest for small images, but its monolithic Taichi
  fields are subject to a signed 32-bit index-stride limit.
- `tiled` uses pointer-backed dense tiles and activates every tile intersecting
  the image. It avoids the monolithic stride limit, but does not reduce storage
  for a fully porous volume.
- `sparse` uses the same tiled hierarchy but activates only tiles containing at
  least one pore voxel, which is useful for mostly-solid images.

```python
result = solve_flow(im, direction="x", storage="tiled", tile_size=16)
result = solve_flow(im, direction="x", storage="sparse", tile_size=(8, 8, 16))
```

The older `sparse=True` option remains an alias for `storage="sparse"`.
Tiles at the image edge are padded internally; padded cells are treated as solid
and are never returned. Tiling removes the Taichi indexing limit, not the memory
cost: a fully porous 900³ D3Q19 simulation needs well over 100 GB for the two
distribution fields alone.

Taichi 1.7.4 does not support the pointer SNodes used by `tiled` and `sparse`
storage on Apple Metal. Use `storage="dense"` with `ti.metal`, or run a
pointer-backed layout on a supported CPU/CUDA backend.

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
2. A selectable D3Q19 MRT or SRT/BGK collision operator evolves the
   distribution functions to steady state. MRT remains the Taichi default;
   solid voxels use bounce-back boundary conditions.
3. Darcy's law is applied to the converged velocity field:

   **K = u_D · μ / |∇P|**

   where u_D is the volume-averaged (Darcy) velocity and |∇P| = Δρ · c_s² / L with
   c_s² = 1/3 for D3Q19.

4. The result in lattice units is scaled to m² (or milliDarcy) using the physical
   voxel size dx_m.

