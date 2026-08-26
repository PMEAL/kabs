"""Benchmark Taichi MRT and SRT collision models on a shared workload.

Examples
--------
Run the local CPU and Apple Metal baselines::

    python tests/benchmark_solvers.py --arch cpu
    python tests/benchmark_solvers.py --arch metal

Run the same benchmark on CUDA and save machine-readable results::

    python tests/benchmark_solvers.py --arch cuda --output benchmark-cuda.json

The throughput timer excludes solver construction, JIT compilation, and host
array extraction. A warm-up step compiles every Taichi kernel before timing,
and ``ti.sync()`` brackets each sample so asynchronous GPU work is included.
Time-to-convergence is measured separately on a convergence-enabled solver
after every kernel used by that loop has been compiled.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

import kabs


RADIUS = 10
LENGTH = 30
CROSS_SECTION = 50
CENTRES = ((13, 13), (13, 37), (37, 13), (37, 37))
NU = 1.0 / 6.0
POPULATION_BYTES_PER_SITE = 2 * 19 * np.dtype(np.float32).itemsize
MRT_OPERATOR_BYTES = (2 * 19 * 19 + 19) * np.dtype(np.float32).itemsize


def bundle_image(scale: int = 1) -> np.ndarray:
    """Return the standard four-cylinder permeability test image."""
    cross_section = CROSS_SECTION * scale
    y, z = np.mgrid[0:cross_section, 0:cross_section]
    cross = np.zeros((cross_section, cross_section), dtype=bool)
    for cy, cz in CENTRES:
        cross |= (
            (y - cy * scale) ** 2 + (z - cz * scale) ** 2
            <= (RADIUS * scale) ** 2
        )
    return np.broadcast_to(
        cross[np.newaxis],
        (LENGTH * scale, cross_section, cross_section),
    ).copy()


def analytical_permeability(scale: int = 1) -> float:
    """Return the bundle-of-tubes permeability in lattice units."""
    radius = RADIUS * scale
    cross_section = CROSS_SECTION * scale
    return (
        len(CENTRES)
        * np.pi
        * radius**4
        / (8 * cross_section * cross_section)
    )


def _arch_from_name(name: str):
    return {
        "cpu": ti.cpu,
        "metal": ti.metal,
        "cuda": ti.cuda,
    }[name]


def _time_steps(solver, n_steps: int) -> float:
    ti.sync()
    start = time.perf_counter()
    for _ in range(n_steps):
        solver.step()
    ti.sync()
    return time.perf_counter() - start


def _run_to_convergence(solver, max_steps: int, tolerance: float, log_every=200):
    """Reset and run an already-compiled solver to the requested tolerance."""
    solver.init()
    has_velocity_snapshot = False
    final_step = max_steps
    final_criterion = None

    ti.sync()
    start = time.perf_counter()
    for step in range(max_steps + 1):
        solver.step()
        if step % log_every != 0:
            continue
        if has_velocity_snapshot:
            solver.reset_convergence_sums()
            solver.accumulate_convergence_sums()
            v_total, v_change = solver.get_convergence_sums()
            if v_total > 0:
                final_criterion = v_change / v_total
                if final_criterion < tolerance:
                    final_step = step
                    break
        solver.snapshot_velocity()
        has_velocity_snapshot = True
    ti.sync()
    return time.perf_counter() - start, final_step, final_criterion


def benchmark_taichi_model(
    image: np.ndarray,
    model: str,
    *,
    steps: int,
    repeats: int,
    convergence_steps: int,
    tolerance: float,
    reference_permeability: float,
) -> dict:
    """Benchmark one warmed Taichi collision model."""
    # n_steps=0 still executes one complete step in the current public loop.
    # That deliberately compiles collision, streaming, and boundary kernels.
    start = time.perf_counter()
    warmup = kabs.solve_flow(
        image,
        collision_model=model,
        n_steps=0,
        tol=None,
        log_every=1,
        verbose=False,
    )
    ti.sync()
    jit_setup_seconds = time.perf_counter() - start
    solver = warmup._solver
    timings = [_time_steps(solver, steps) for _ in range(repeats)]

    elapsed = statistics.median(timings)
    total_updates = image.size * steps
    pore_updates = int(np.count_nonzero(image)) * steps

    # This first convergence solve compiles the reduction/snapshot kernels and
    # creates a solver with the required convergence-monitor fields. It is not
    # used as the warmed performance result.
    convergence_warmup = kabs.solve_flow(
        image,
        collision_model=model,
        n_steps=400,
        tol=None,
        log_every=200,
        verbose=False,
    )
    convergence_seconds, convergence_iterations, convergence_criterion = (
        _run_to_convergence(
            convergence_warmup._solver,
            convergence_steps,
            tolerance,
        )
    )
    converged = kabs.FlowResult(
        convergence_warmup._solver,
        direction="x",
        nu=NU,
        n_iterations=convergence_iterations,
        convergence_criterion=convergence_criterion,
    )
    permeability = kabs.compute_permeability(converged, verbose=False)["k_lu"]
    reference = reference_permeability

    return {
        "model": model,
        "jit_setup_seconds": jit_setup_seconds,
        "sample_seconds": timings,
        "median_seconds": elapsed,
        "total_mlups": total_updates / elapsed / 1e6,
        "pore_mlups": pore_updates / elapsed / 1e6,
        "convergence_seconds": convergence_seconds,
        "convergence_iterations": convergence_iterations,
        "convergence_criterion": convergence_criterion,
        "permeability_lu": permeability,
        "permeability_relative_error": permeability / reference - 1.0,
    }


def _format_bytes(n_bytes: int) -> str:
    return f"{n_bytes / 1024**2:.2f} MiB"


def _print_report(report: dict) -> None:
    config = report["configuration"]
    memory = report["memory"]
    print("KABS collision benchmark")
    print(f"  platform       : {config['platform']}")
    print(f"  architecture   : {config['arch']}")
    print(f"  Taichi         : {config['taichi_version']}")
    print(f"  shape           : {tuple(config['shape'])}")
    print(f"  porosity        : {config['porosity']:.4f}")
    print(f"  timed steps     : {config['steps']} x {config['repeats']} repeats")
    print(
        "  population data : "
        f"{memory['population_bytes_per_site']} bytes/site, "
        f"{_format_bytes(memory['population_bytes_total'])} total"
    )
    print(
        "  MRT-only fields : "
        f"{memory['mrt_operator_bytes']} bytes (global, not per site)"
    )

    print()
    print(
        "model  JIT/setup  median(s)  total MLUPS  pore MLUPS  conv(s)  "
        "iterations  k(lu)      rel.error"
    )
    for result in report["results"]:
        print(
            f"{result['model']:>5}  "
            f"{result['jit_setup_seconds']:9.3f}  "
            f"{result['median_seconds']:9.4f}  "
            f"{result['total_mlups']:11.3f}  "
            f"{result['pore_mlups']:10.3f}  "
            f"{result['convergence_seconds']:7.3f}  "
            f"{result['convergence_iterations']:10d}  "
            f"{result['permeability_lu']:9.5f}  "
            f"{result['permeability_relative_error']:+9.3%}"
        )

    by_model = {item["model"]: item for item in report["results"]}
    speedup = by_model["mrt"]["median_seconds"] / by_model["srt"]["median_seconds"]
    convergence_speedup = (
        by_model["mrt"]["convergence_seconds"]
        / by_model["srt"]["convergence_seconds"]
    )
    print()
    print(f"SRT warmed-throughput speedup : {speedup:.3f}x")
    print(f"SRT time-to-convergence speedup: {convergence_speedup:.3f}x")
    print(
        "Memory note: both models retain the same two D3Q19 population "
        "buffers; portable GPU occupancy/register metrics require the CUDA run."
    )


def run_benchmark(args) -> dict:
    if args.arch == "metal" and sys.platform != "darwin":
        raise ValueError("--arch metal is only available on macOS")

    image = bundle_image(args.scale)
    reference_permeability = analytical_permeability(args.scale)
    results = []
    for model in ("mrt", "srt"):
        # Isolate each model's cold compilation cost and release permanent
        # Taichi fields before constructing the next solver.
        ti.reset()
        ti.init(arch=_arch_from_name(args.arch))
        results.append(
            benchmark_taichi_model(
                image,
                model,
                steps=args.steps,
                repeats=args.repeats,
                convergence_steps=args.convergence_steps,
                tolerance=args.tolerance,
                reference_permeability=reference_permeability,
            )
        )

    return {
        "configuration": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "taichi_version": ti.__version__,
            "kabs_version": kabs.__version__,
            "arch": args.arch,
            "shape": list(image.shape),
            "scale": args.scale,
            "porosity": float(image.mean()),
            "nu": NU,
            "tau": 3 * NU + 0.5,
            "steps": args.steps,
            "repeats": args.repeats,
            "convergence_steps": args.convergence_steps,
            "tolerance": args.tolerance,
        },
        "memory": {
            "population_bytes_per_site": POPULATION_BYTES_PER_SITE,
            "population_bytes_total": int(POPULATION_BYTES_PER_SITE * image.size),
            "mrt_operator_bytes": MRT_OPERATOR_BYTES,
            "note": (
                "Population storage is identical. MRT-only global fields are "
                "two 19x19 matrices and one 19-entry relaxation vector."
            ),
        },
        "results": results,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("cpu", "metal", "cuda"), default="cpu")
    parser.add_argument("--steps", type=_positive_int, default=500)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument(
        "--scale",
        type=_positive_int,
        default=1,
        help="multiply every bundle dimension (site count grows as scale cubed)",
    )
    parser.add_argument("--convergence-steps", type=_positive_int, default=4000)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    _print_report(report)
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
