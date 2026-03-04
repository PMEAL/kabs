@README.md

---

## Developer notes

### Source layout
- `kabs/_single_phase_solver.py` — `SinglePhaseSolver` class: D3Q19 MRT-LBM Taichi kernels (collision, streaming, BCs)
- `kabs/_solve_flow.py` — `solve_flow()`: high-level entry point; sets BCs, runs loop, handles convergence/VTK export
- `kabs/_compute_permeability.py` — `compute_permeability()`: parses VTR binary, applies Darcy's law

### Voxel convention (important — two conflicting conventions in play)
- **Public API / PoreSpy convention**: `1 = pore, 0 = solid`
- **Internal solver convention**: `1 = solid, 0 = pore` (the opposite)
- The flip happens in `_solve_flow.py`: `solid_im = (im == 0).astype(np.int8)`
- `_compute_permeability.py` uses `solid == 0` as the pore mask, consistent with the internal convention stored in the VTR

### LBM numerics
- D3Q19 MRT (Multiple Relaxation Time) collision operator
- Default viscosity `ν = 1/6`, giving `τ = 1.0` and maximum viscous damping
- Boundary conditions: fixed density (pressure) on inlet/outlet faces; periodic on the four transverse faces; bounce-back on solid voxels
- Pressure BCs: `ρ_in = 1.00`, `ρ_out = 0.99` (hardcoded — the ratio cancels in Darcy's law so only the difference matters)

### Convergence
- Checked every `log_every` steps (default 500) by comparing the total velocity field change: `delta|v| / |v|`
- Default tolerance `tol = 1e-3`; set `tol=None` to disable early stopping
- Rule of thumb for minimum steps: `~5 × r² / ν` where `r` is characteristic pore throat radius in voxels
- For the cylinders test image (`r=10`, `ν=1/6`): converges around 500–1000 steps

### VTR output
- Filename: `{prefix}-{final_step}-{axis}.vtr`  (step count reflects actual converged step, not `n_steps`)
- Flow VTR contains: `Solid` (int8), `rho` (float32), `velocity` (3-component float32)
- Parsed using only `numpy` + `struct` (no VTK/pyvista dependency)
