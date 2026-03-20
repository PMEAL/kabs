@README.md

---

## Developer notes

### Source layout
- `kabs/_single_phase_solver.py` — `SinglePhaseSolver` class: D3Q19 MRT-LBM Taichi kernels (collision, streaming, BCs)
- `kabs/_solve_flow.py` — `solve_flow()` entry point and `FlowResult` container class; sets BCs, runs loop, handles convergence/VTK export; returns a `FlowResult`
- `kabs/_compute_permeability.py` — `compute_permeability()`: accepts a `FlowResult`, applies Darcy's law
- `kabs/_compute_hydraulic_conductance.py` — `compute_hydraulic_conductance()`: accepts a `FlowResult`, computes Q/ΔP conductance
- `kabs/utils.py` — `read_flow_vtr()` / `write_flow_vtr()`: VTR file I/O (pure `numpy` + `struct`, no pyvista dependency); embeds `direction` and `nu` as an XML comment for round-trip fidelity
- `kabs/plots.py` — `plot_cross_section()`, `add_streamlines()`, `render_flow()`: visualization helpers

### Voxel convention (important — two conflicting conventions in play)
- **Public API / PoreSpy convention**: `1 = pore, 0 = solid`
- **Internal solver convention**: `1 = solid, 0 = pore` (the opposite)
- The flip happens in `_solve_flow.py`: `solid_im = (im == 0).astype(np.int8)`
- `_compute_permeability.py` uses `solid == 0` as the pore mask, consistent with the internal convention stored in the `FlowResult`

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
- Written via `result.export_to_vtk(prefix)` or automatically when `export_vtk=True` (default)
- Filename: `{prefix}-{final_step}-{axis}.vtr`  (step count reflects actual converged step, not `n_steps`)
- Flow VTR contains: `Solid` (int8), `rho` (float32), `velocity` (3-component float32)
- `direction` and `nu` embedded as an XML comment (`<!-- kabs-meta ... -->`) for round-trip fidelity
- Read back with `read_flow_vtr(path)` from `kabs.utils`; uses only `numpy` + `struct` (no pyvista dependency)

### Test style
- Tests are written as methods on a `TestClassName` class (not bare functions)
- `setup_method` handles per-test setup (temp dirs, shared fixtures)
- A `if __name__ == "__main__"` block at the bottom iterates over methods with
  `for name in sorted(dir(obj))` and runs any starting with `test_`, so tests
  can be run directly in an interactive window without pytest
  