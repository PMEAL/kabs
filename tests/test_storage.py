"""Focused tests for dense, tiled, and sparse Taichi storage layouts."""

import subprocess
import sys

import numpy as np
import pytest

from kabs import solve_flow
from kabs._compute_permeability import compute_permeability
from kabs._single_phase_solver import SinglePhaseSolver


_QUICK_SOLVE = {
    "direction": "x",
    "n_steps": 4,
    "tol": None,
    "log_every": 10,
    "verbose": False,
}


def _channel(shape=(7, 6, 5)):
    im = np.zeros(shape, dtype=np.int8)
    im[:, 2:4, 2:4] = 1
    return im


def test_sparse_alias_and_explicit_storage_agree():
    alias = solve_flow(_channel(), sparse=True, tile_size=4, **_QUICK_SOLVE)
    explicit = solve_flow(
        _channel(), storage="sparse", tile_size=4, **_QUICK_SOLVE
    )
    assert alias._solver.storage == "sparse"
    np.testing.assert_allclose(alias.rho, explicit.rho, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        alias.velocity, explicit.velocity, rtol=1e-6, atol=1e-7
    )


def test_sparse_false_remains_dense():
    result = solve_flow(_channel(), sparse=False, **_QUICK_SOLVE)
    assert result._solver.storage == "dense"


@pytest.mark.parametrize(
    ("sparse", "storage"),
    [(True, "dense"), (True, "tiled"), (False, "sparse"), (False, "tiled")],
)
def test_contradictory_storage_options_fail(sparse, storage):
    with pytest.raises(ValueError, match="contradictory storage options"):
        solve_flow(_channel(), sparse=sparse, storage=storage, **_QUICK_SOLVE)


@pytest.mark.parametrize("storage", ["tiled", "sparse"])
def test_non_aligned_outputs_have_exact_logical_shape(storage):
    shape = (17, 19, 21)
    result = solve_flow(
        np.zeros(shape, dtype=np.int8),
        storage=storage,
        tile_size=8,
        **_QUICK_SOLVE,
    )
    assert result.rho.shape == shape
    assert result.velocity.shape == (*shape, 3)


def test_dense_tiled_and_sparse_are_numerically_equivalent():
    results = {
        storage: solve_flow(
            _channel(), storage=storage, tile_size=(4, 3, 4), **_QUICK_SOLVE
        )
        for storage in ("dense", "tiled", "sparse")
    }
    dense = results["dense"]
    k_dense = compute_permeability(dense, verbose=False)["k_lu"]
    for storage in ("tiled", "sparse"):
        result = results[storage]
        np.testing.assert_allclose(result.rho, dense.rho, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(
            result.velocity, dense.velocity, rtol=2e-5, atol=2e-6
        )
        assert compute_permeability(result, verbose=False)["k_lu"] == pytest.approx(
            k_dense, rel=2e-5, abs=2e-6
        )


def test_sparse_small_channel_matches_dense():
    im = np.zeros((12, 11, 10), dtype=np.int8)
    im[:, 5, 4:6] = 1
    dense = solve_flow(im, storage="dense", **_QUICK_SOLVE)
    sparse = solve_flow(im, storage="sparse", tile_size=4, **_QUICK_SOLVE)
    np.testing.assert_allclose(sparse.rho, dense.rho, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        sparse.velocity, dense.velocity, rtol=2e-5, atol=2e-6
    )


def test_single_phase_solver_accepts_storage_parameters():
    solver = SinglePhaseSolver(
        np.ones((3, 4, 5), dtype=np.int8), storage="tiled", tile_size=(2, 3, 4)
    )
    assert solver.storage == "tiled"
    assert solver.tile_size == (2, 3, 4)


def test_large_tiled_layout_materializes_without_activating_voxels():
    code = """
import taichi as ti
from kabs._single_phase_solver import _create_tiled_population_fields

ti.init(arch=ti.cpu)
_create_tiled_population_fields((900, 900, 900), (16, 16, 16))
ti.lang.impl.get_runtime().materialize()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
