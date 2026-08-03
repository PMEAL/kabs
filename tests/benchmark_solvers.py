"""Compare wall-clock time for Taichi vs XLB solvers.

Run with:
    python benchmark_solvers.py

Geometry: 4-cylinder bundle (same as test_solve_flow_xlb.py / test_permeability.py).
N_STEPS is kept the same for both so times are directly comparable.
"""

import time
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu)

import kabs

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
R = 10
L = 30
NY = NZ = 50
NU = 1.0 / 6.0
CENTRES = [(13, 13), (13, 37), (37, 13), (37, 37)]
N_STEPS = 4000

a, b = np.mgrid[0:NY, 0:NZ]
cross = np.zeros((NY, NZ), dtype=bool)
for ca, cb in CENTRES:
    cross |= (a - ca) ** 2 + (b - cb) ** 2 <= R**2
im = np.broadcast_to(cross[np.newaxis], (L, NY, NZ)).copy().astype(int)

print(f"Domain: {im.shape}  pore fraction: {im.mean():.3f}  n_steps: {N_STEPS}\n")

# ---------------------------------------------------------------------------
# Taichi solve (includes JIT warm-up on first call)
# ---------------------------------------------------------------------------
print("=== Taichi (MRT) ===")
t0 = time.perf_counter()
result_ti = kabs.solve_flow(im, direction="x", n_steps=N_STEPS, nu=NU,
                            log_every=N_STEPS, verbose=False, tol=None)
t_taichi = time.perf_counter() - t0

res_ti = kabs.compute_permeability(result_ti, verbose=False)
k_ti = res_ti["k_lu"]
print(f"  Time  : {t_taichi:.2f} s")
print(f"  k     : {k_ti:.4f} (LU)")

# ---------------------------------------------------------------------------
# XLB solve (includes JAX JIT warm-up on first call)
# ---------------------------------------------------------------------------
print("\n=== XLB (BGK / JAX) ===")
t0 = time.perf_counter()
result_xlb = kabs.solve_flow_xlb(im, direction="x", n_steps=N_STEPS, nu=NU,
                                  log_every=N_STEPS, verbose=False, tol=None)
t_xlb = time.perf_counter() - t0

res_xlb = kabs.compute_permeability(result_xlb, verbose=False)
k_xlb = res_xlb["k_lu"]
print(f"  Time  : {t_xlb:.2f} s")
print(f"  k     : {k_xlb:.4f} (LU)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
print(f"  Taichi : {t_taichi:.2f} s   k = {k_ti:.4f}")
print(f"  XLB    : {t_xlb:.2f} s   k = {k_xlb:.4f}")
ratio = t_taichi / t_xlb if t_xlb > 0 else float("nan")
print(f"  Taichi / XLB speedup: {ratio:.2f}x  (>1 means XLB is faster)")
print(f"  |k_ti - k_xlb| / k_ti: {abs(k_ti - k_xlb) / k_ti * 100:.2f} %")

# ---------------------------------------------------------------------------
# Optional: second run to separate JIT from steady-state throughput
# ---------------------------------------------------------------------------
print("\n=== Second run (JIT already warm) ===")

t0 = time.perf_counter()
kabs.solve_flow(im, direction="x", n_steps=N_STEPS, nu=NU,
                log_every=N_STEPS, verbose=False, tol=None)
t_taichi2 = time.perf_counter() - t0

t0 = time.perf_counter()
kabs.solve_flow_xlb(im, direction="x", n_steps=N_STEPS, nu=NU,
                     log_every=N_STEPS, verbose=False, tol=None)
t_xlb2 = time.perf_counter() - t0

ratio2 = t_taichi2 / t_xlb2 if t_xlb2 > 0 else float("nan")
print(f"  Taichi : {t_taichi2:.2f} s")
print(f"  XLB    : {t_xlb2:.2f} s")
print(f"  Taichi / XLB speedup: {ratio2:.2f}x")
