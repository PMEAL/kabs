"""Subprocess checks for isolation between the Taichi and XLB runtimes."""

import json
import os
import subprocess
import sys
import textwrap

import pytest


def _run_fresh_python(source):
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_import_kabs_does_not_import_taichi():
    completed = _run_fresh_python(
        """
        import sys
        import kabs

        assert "taichi" not in sys.modules
        assert "kabs._single_phase_solver" not in sys.modules
        assert kabs.FlowResult.__module__ == "kabs._flow_common"
        """
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_public_single_phase_solver_export_remains_lazy():
    completed = _run_fresh_python(
        """
        import sys
        import kabs

        assert "taichi" not in sys.modules
        solver_class = kabs.SinglePhaseSolver
        assert solver_class.__name__ == "SinglePhaseSolver"
        assert "taichi" in sys.modules
        """
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.skipif(
    os.environ.get("KABS_TEST_WARP_CUDA") != "1",
    reason="set KABS_TEST_WARP_CUDA=1 on a CUDA-enabled Warp test runner",
)
def test_warp_solve_does_not_import_taichi():
    completed = _run_fresh_python(
        """
        import sys
        import numpy as np
        import kabs

        image = np.zeros((6, 6, 6), dtype=np.int8)
        image[:, 2:4, 2:4] = 1
        result = kabs.solve_flow(
            image,
            backend="xlb",
            compute_backend="warp",
            collision_model="srt",
            n_steps=20,
            log_every=5,
            tol=None,
            verbose=False,
        )

        assert np.isfinite(result.rho).all()
        assert np.isfinite(result.velocity).all()
        assert kabs.compute_permeability(result, verbose=False)["k_lu"] > 0.0
        assert "taichi" not in sys.modules
        assert "kabs._single_phase_solver" not in sys.modules
        """
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.skipif(
    os.environ.get("KABS_TEST_WARP_CUDA") != "1",
    reason="set KABS_TEST_WARP_CUDA=1 on a CUDA-enabled Warp test runner",
)
def test_warp_and_jax_convergence_agree_in_fresh_processes():
    source = """
        import json
        import numpy as np
        from kabs import solve_flow_xlb

        image = np.zeros((6, 6, 6), dtype=np.int8)
        image[:, 2:4, 2:4] = 1
        result = solve_flow_xlb(
            image,
            compute_backend={backend!r},
            n_steps=20,
            log_every=5,
            tol=0.2,
            verbose=False,
        )
        print("KABS_RESULT=" + json.dumps({{
            "iterations": result.n_iterations,
            "criterion": result.convergence_criterion,
        }}))
    """

    results = {}
    for backend in ("jax", "warp"):
        completed = _run_fresh_python(source.format(backend=backend))
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result_line = next(
            line
            for line in completed.stdout.splitlines()
            if line.startswith("KABS_RESULT=")
        )
        results[backend] = json.loads(result_line.removeprefix("KABS_RESULT="))

    assert results["warp"]["iterations"] == results["jax"]["iterations"]
    assert results["warp"]["criterion"] == pytest.approx(
        results["jax"]["criterion"], rel=5e-4, abs=2e-6
    )
