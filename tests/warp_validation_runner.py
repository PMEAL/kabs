"""Run one isolated numerical validation case and emit a JSON summary."""

import argparse
import json

import numpy as np


_AXIS = {"x": 0, "y": 1, "z": 2}


def _cylinder_channel(direction, length=12, width=12, radius=4):
    a, b = np.mgrid[0:width, 0:width]
    cross_section = (a - width // 2) ** 2 + (b - width // 2) ** 2 <= radius**2
    if direction == "x":
        shape = (length, width, width)
        source = cross_section[np.newaxis]
    elif direction == "y":
        shape = (width, length, width)
        source = cross_section[:, np.newaxis, :]
    else:
        shape = (width, width, length)
        source = cross_section[:, :, np.newaxis]
    return np.broadcast_to(source, shape).copy().astype(np.int8)


def _open_channel():
    return np.ones((12, 12, 12), dtype=np.int8)


def _obstacle_channel():
    image = np.ones((16, 12, 12), dtype=np.int8)
    image[6:10, 4:8, 4:8] = 0
    return image


def _bundle():
    radius = 10
    width = 50
    a, b = np.mgrid[0:width, 0:width]
    cross_section = np.zeros((width, width), dtype=bool)
    for ca, cb in ((13, 13), (13, 37), (37, 13), (37, 37)):
        cross_section |= (a - ca) ** 2 + (b - cb) ** 2 <= radius**2
    return np.broadcast_to(cross_section[np.newaxis], (30, width, width)).copy().astype(np.int8)


def _solve(case, direction, backend, fixed_steps):
    from kabs import solve_flow

    if case == "channel":
        image = _cylinder_channel(direction)
        n_steps, log_every, tol = 500, 100, 1e-3
    elif case == "open":
        image = _open_channel()
        n_steps, log_every, tol = 400, 100, None
    elif case == "obstacle":
        image = _obstacle_channel()
        n_steps, log_every, tol = 800, 100, 1e-3
    else:
        image = _bundle()
        n_steps, log_every, tol = 4000, 200, 1e-3

    if fixed_steps is not None:
        n_steps = fixed_steps
        tol = None

    kwargs = {
        "direction": direction,
        "n_steps": n_steps,
        "log_every": log_every,
        "tol": tol,
        "verbose": False,
        "collision_model": "srt",
    }
    if backend == "taichi":
        import taichi as ti

        ti.init(arch=ti.cuda)
        result = solve_flow(image, backend="taichi", **kwargs)
    else:
        result = solve_flow(
            image,
            backend="xlb",
            compute_backend=backend,
            **kwargs,
        )
    return image, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("channel", "open", "obstacle", "bundle"), required=True)
    parser.add_argument("--direction", choices=("x", "y", "z"), default="x")
    parser.add_argument("--backend", choices=("warp", "jax", "taichi"), default="warp")
    parser.add_argument("--fixed-steps", type=int)
    args = parser.parse_args()

    from kabs import compute_permeability

    image, result = _solve(
        args.case,
        args.direction,
        args.backend,
        args.fixed_steps,
    )
    permeability = compute_permeability(result, verbose=False)
    axis = _AXIS[args.direction]
    pore = result.solid == 0
    pore_velocity = result.velocity[pore]
    transverse = np.delete(np.abs(pore_velocity), axis, axis=1)
    summary = {
        "case": args.case,
        "backend": args.backend,
        "direction": args.direction,
        "finite": bool(
            np.isfinite(result.rho).all() and np.isfinite(result.velocity).all()
        ),
        "k_lu": permeability["k_lu"],
        "u_darcy": permeability["u_darcy"],
        "rho_mean": float(result.rho.mean()),
        "iterations": result.n_iterations,
        "criterion": result.convergence_criterion,
        "axial_pore_velocity": float(pore_velocity[:, axis].mean()),
        "transverse_pore_velocity": float(transverse.mean()),
        "shape": image.shape,
    }
    if args.case == "open":
        inlet = [0, 0, 0]
        outlet = [0, 0, 0]
        outlet[axis] = -1
        summary["rho_in_corner"] = float(result.rho[tuple(inlet)])
        summary["rho_out_corner"] = float(result.rho[tuple(outlet)])
    if args.case == "bundle":
        summary["analytical_k_lu"] = 4 * np.pi * 10**4 / (8 * 50 * 50)

    print("KABS_VALIDATION=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
