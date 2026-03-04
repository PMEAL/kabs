@README.md

---

## Developer notes

### Source layout
- `kabs/_single_phase_solver.py` — `SinglePhaseSolver` class: D3Q19 MRT-LBM Taichi kernels (collision, streaming, BCs)
- `kabs/_solve_flow.py` — `solve_flow()`: high-level entry point; sets BCs, runs loop, handles convergence/VTK export
- `kabs/_compute_permeability.py` — `compute_permeability()`: parses VTR binary, applies Darcy's law
- `kabs/_diffusion_solver.py` — `DiffusionSolver` class: D3Q7 BGK scalar LBM Taichi kernels (collision, streaming, BCs)
- `kabs/_solve_diffusion.py` — `solve_diffusion()`: concentration-driven diffusion; mirrors `solve_flow` API
- `kabs/_compute_effective_diffusivity.py` — `compute_effective_diffusivity()`: parses VTR binary, applies Fick's law

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
- Diffusion VTR contains: `Solid` (int8), `c` (float32), `flux` (float32, flow-direction flux)
- Both parsed using only `numpy` + `struct` (no VTK/pyvista dependency)

### Diffusion LBM numerics
- D3Q7 BGK (single relaxation time) collision operator; scalar distribution function
- Default diffusivity `D = 1/4`, giving `τ_D = 1.5` — the optimal sweet spot
- Steps to convergence scale as `L²/D` (e.g. ~40 000 for L=100, D=1/4); accuracy degrades above `τ_D ≈ 2` (D ≈ 3/8)
- Default `n_steps = 50 000`, `tol = 1e-2` — looser than flow because D_eff (mean flux) converges faster than the pointwise concentration field
- Boundary conditions: fixed concentration (`c_in=1.0`, `c_out=0.0`) on inlet/outlet; periodic on transverse faces; bounce-back (no-flux) on solid voxels
- Tortuosity: `τ = F / φ` where `F = D_0 / D_eff`; always > 1
