# Collision-model benchmarks

Run the reproducible benchmark from the repository root:

```console
uv run python tests/benchmark_solvers.py --arch cpu
uv run python tests/benchmark_solvers.py --arch metal
uv run python tests/benchmark_solvers.py --arch cuda --scale 2 --output cuda-scale2.json
```

The benchmark compares Taichi MRT and SRT on the same four-cylinder bundle
used by the analytical permeability tests. Each model is initialized in a
fresh Taichi runtime. One complete step compiles its kernels before timing,
and every timed sample is bracketed by `ti.sync()`.

Reported throughput excludes construction, compilation, convergence checks,
and host extraction. Time-to-convergence uses an already-compiled solver and
includes the device convergence reductions every 200 steps. JIT/setup time is
reported separately as a diagnostic; offline compiler caches can affect it.

`--scale N` multiplies every geometric dimension by `N`, so site count and the
two population buffers grow as `N³`. Scale 1 has 75,000 sites; scale 2 has
600,000 sites. CUDA testing should include scale 2 and at least one larger
case that represents the intended production workload.

## Preliminary Apple M3 results

Measured 2026-08-26 on a MacBook Air with an Apple M3 (8 CPU cores, 16 GB),
macOS 15.7.7, Python 3.12.11, Taichi 1.7.4, and KABS 0.2.0. Both cases use
`nu=1/6`, `tau=1`, three throughput samples, and `tol=1e-3`.

| Arch | Scale | Model | Total MLUPS | Pore MLUPS | Convergence time | Iterations | k (lu) | Analytical error |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| CPU | 1 | MRT | 30.940 | 15.693 | 2.587 s | 1000 | 6.42181 | +2.206% |
| CPU | 1 | SRT | 67.922 | 34.450 | 1.220 s | 1000 | 6.44775 | +2.619% |
| Metal | 1 | MRT | 174.426 | 88.469 | 2.327 s | 1000 | 6.42186 | +2.207% |
| Metal | 1 | SRT | 177.474 | 90.015 | 2.255 s | 1000 | 6.44776 | +2.619% |
| CPU | 2 | MRT | 33.990 | 17.090 | 54.261 s | 2800 | 25.21757 | +0.338% |
| CPU | 2 | SRT | 74.147 | 37.281 | 25.215 s | 2800 | 25.29887 | +0.661% |
| Metal | 2 | MRT | 193.908 | 97.497 | 35.884 s | 2800 | 25.21779 | +0.338% |
| Metal | 2 | SRT | 198.349 | 99.730 | 36.077 s | 2800 | 25.29896 | +0.661% |

| Arch | Scale | SRT throughput speedup | SRT convergence speedup |
|---|---:|---:|---:|
| CPU | 1 | 2.195x | 2.120x |
| Metal | 1 | 1.017x | 1.032x |
| CPU | 2 | 2.181x | 2.152x |
| Metal | 2 | 1.023x | 0.995x |

The CPU results show a consistent 2.1–2.2x benefit from removing the two MRT
moment transformations. Metal is effectively tied at both sizes, consistent
with a bandwidth-bound kernel on this hardware. These results do not predict
CUDA performance.

The harness intentionally compares the two Taichi collision kernels. XLB BGK
is covered by the numerical cross-backend tests, but its JAX execution and
compilation lifecycle require a separate apples-to-apples performance harness.

## Memory interpretation

Both collision models retain two D3Q19 float32 population buffers: 152 bytes
per lattice site, or 10.87 MiB at scale 1 and 86.98 MiB at scale 2. MRT adds
only two global 19x19 matrices and one 19-entry relaxation vector (2,964
bytes). SRT therefore reduces arithmetic and compiler/kernel complexity, but
does not materially reduce allocated simulation memory.

Apple Metal in Taichi 1.7.4 does not support pointer SNodes, so these Metal
benchmarks use dense storage. CPU supports dense, tiled, and sparse layouts.

## CUDA work still required

Before using these results to change the production default:

1. Run scale 2 and a production-sized case on the target NVIDIA GPU.
2. Record the GPU model, driver, CUDA, Taichi, Python, and KABS versions.
3. Capture warmed throughput and time-to-convergence with this harness.
4. Use NVIDIA profiling tools to compare registers, occupancy, achieved memory
   bandwidth, and peak allocated memory.
5. Confirm permeability remains within the numerical-validation tolerances.
