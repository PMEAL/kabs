"""Numerical validation suite for the optional XLB/Warp CUDA backend."""

from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("KABS_TEST_WARP_CUDA") != "1",
    reason="set KABS_TEST_WARP_CUDA=1 on a CUDA-enabled Warp test runner",
)

_RUNNER = Path(__file__).with_name("warp_validation_runner.py")


@lru_cache(maxsize=None)
def _run_case(case, direction="x", backend="warp", fixed_steps=None):
    command = [
        sys.executable,
        str(_RUNNER),
        "--case",
        case,
        "--direction",
        direction,
        "--backend",
        backend,
    ]
    if fixed_steps is not None:
        command.extend(("--fixed-steps", str(fixed_steps)))
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    output = completed.stdout + completed.stderr
    if completed.returncode and "paging file is too small" in output.lower():
        # XLB 0.3.1 eagerly imports PyVista/VTK. Windows can transiently fail
        # to commit the large VTK DLL set while prior CUDA processes exit.
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("KABS_VALIDATION=")
    )
    return json.loads(result_line.removeprefix("KABS_VALIDATION="))


@pytest.mark.parametrize("direction", ("x", "y", "z"))
def test_pressure_driven_flow_in_all_directions(direction):
    result = _run_case("channel", direction)
    assert result["finite"]
    assert result["k_lu"] > 0.0
    assert result["u_darcy"] > 0.0
    assert result["axial_pore_velocity"] > 0.0
    assert result["transverse_pore_velocity"] < 0.01 * result["axial_pore_velocity"]


def test_directional_permeabilities_agree():
    results = [_run_case("channel", direction) for direction in ("x", "y", "z")]
    for result in results[1:]:
        assert result["k_lu"] == pytest.approx(results[0]["k_lu"], rel=2e-4)


@pytest.mark.parametrize("direction", ("x", "y", "z"))
def test_pressure_face_edges_and_corners(direction):
    result = _run_case("open", direction)
    assert result["finite"]
    assert result["k_lu"] > 0.0
    assert result["rho_in_corner"] == pytest.approx(1.00, abs=1e-4)
    assert result["rho_out_corner"] == pytest.approx(0.99, abs=1e-4)


def test_internal_bounce_back_obstacle_reduces_flow():
    obstacle = _run_case("obstacle")
    open_channel = _run_case("open")
    assert obstacle["finite"]
    assert obstacle["u_darcy"] > 0.0
    assert 0.0 < obstacle["k_lu"] < open_channel["k_lu"]


def test_four_cylinder_permeability_matches_analytical():
    result = _run_case("bundle")
    assert result["finite"]
    assert result["k_lu"] == pytest.approx(result["analytical_k_lu"], rel=0.07)


def test_warp_agrees_with_taichi_srt():
    warp = _run_case("bundle")
    taichi = _run_case("bundle", backend="taichi")
    assert warp["k_lu"] == pytest.approx(taichi["k_lu"], rel=0.05)
    assert warp["u_darcy"] == pytest.approx(taichi["u_darcy"], rel=0.05)
    assert warp["rho_mean"] == pytest.approx(taichi["rho_mean"], rel=0.01)


def test_warp_agrees_with_xlb_jax():
    warp = _run_case("bundle")
    jax = _run_case("bundle", backend="jax")
    assert warp["k_lu"] == pytest.approx(jax["k_lu"], rel=5e-3)
    assert warp["u_darcy"] == pytest.approx(jax["u_darcy"], rel=5e-3)
    assert warp["rho_mean"] == pytest.approx(jax["rho_mean"], rel=1e-3)
    assert warp["iterations"] == jax["iterations"]


def test_early_convergence_matches_fixed_step_result():
    converged = _run_case("bundle")
    fixed = _run_case("bundle", fixed_steps=converged["iterations"])
    assert converged["iterations"] < 4000
    assert fixed["iterations"] == converged["iterations"]
    assert fixed["criterion"] is None
    assert converged["k_lu"] == pytest.approx(fixed["k_lu"], rel=2e-3)
