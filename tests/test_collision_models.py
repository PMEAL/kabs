"""Focused tests for the internal Taichi collision operators."""

import numpy as np
import pytest
import taichi as ti

from kabs._single_phase_solver import SinglePhaseSolver


@pytest.fixture(autouse=True)
def _reset_taichi_after_test():
    """Prevent permanent solver fields from accumulating between tests."""
    yield
    ti.reset()
    ti.init(arch=ti.cpu)


def _new_solver(collision_model="mrt", nu=0.2):
    solver = SinglePhaseSolver(
        np.zeros((2, 2, 2), dtype=np.int8),
        collision_model=collision_model,
    )
    solver.set_viscosity(nu)
    solver.init_simulation()
    return solver


def test_mrt_remains_the_default_collision_model():
    solver = SinglePhaseSolver(np.zeros((2, 2, 2), dtype=np.int8))

    assert solver.collision_model == "mrt"
    assert solver.M is not None
    assert solver.inv_M is not None
    assert solver.S_dig is not None


def test_srt_model_is_normalized_and_omits_mrt_fields():
    solver = SinglePhaseSolver(
        np.zeros((2, 2, 2), dtype=np.int8), collision_model="SRT"
    )

    assert solver.collision_model == "srt"
    assert solver.M is None
    assert solver.inv_M is None
    assert solver.S_dig is None


@pytest.mark.parametrize("collision_model", ["bgk", "invalid", None, 1])
def test_invalid_collision_model_raises(collision_model):
    with pytest.raises(ValueError, match="collision_model"):
        SinglePhaseSolver(
            np.zeros((2, 2, 2), dtype=np.int8),
            collision_model=collision_model,
        )


def test_srt_collision_matches_distribution_space_formula():
    solver = _new_solver("srt")
    index = (0, 0, 0)

    populations = solver.F.to_numpy()
    rho = solver.rho.to_numpy()
    velocity = solver.v.to_numpy()
    directions = solver.e_f.to_numpy()
    weights = solver.w.to_numpy()

    perturbation = np.linspace(-8e-4, 8e-4, 19, dtype=np.float32)
    local_population = populations[index] + perturbation
    local_rho = local_population.sum()
    local_velocity = directions.T @ local_population / local_rho

    populations[index] = local_population
    rho[index] = local_rho
    velocity[index] = local_velocity
    solver.F.from_numpy(populations)
    solver.rho.from_numpy(rho)
    solver.v.from_numpy(velocity)

    eu = directions @ local_velocity
    equilibrium = weights * local_rho * (
        1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * local_velocity.dot(local_velocity)
    )
    expected = local_population - solver.omega * (
        local_population - equilibrium
    )

    solver.collision()
    actual = solver.f.to_numpy()[index]

    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
    assert actual.sum() == pytest.approx(local_rho, rel=2e-6, abs=2e-7)
    np.testing.assert_allclose(
        directions.T @ actual,
        local_rho * local_velocity,
        rtol=2e-6,
        atol=2e-7,
    )


@pytest.mark.parametrize("collision_model", ["mrt", "srt"])
def test_equilibrium_is_unchanged_by_collision(collision_model):
    solver = _new_solver(collision_model)
    before = solver.F.to_numpy()

    solver.collision()

    np.testing.assert_allclose(solver.f.to_numpy(), before, rtol=1e-6, atol=1e-7)


def test_macroscopic_reconstruction_does_not_copy_populations():
    solver = _new_solver("srt")
    index = (0, 0, 0)
    populations = solver.F.to_numpy()
    directions = solver.e_f.to_numpy()

    populations[index] += np.linspace(-8e-4, 8e-4, 19, dtype=np.float32)
    solver.F.from_numpy(populations)
    collision_output = solver.f.to_numpy().copy()

    expected_rho = populations[index].sum()
    expected_velocity = directions.T @ populations[index] / expected_rho

    solver.streaming3()

    np.testing.assert_array_equal(solver.f.to_numpy(), collision_output)
    assert solver.rho.to_numpy()[index] == pytest.approx(expected_rho)
    np.testing.assert_allclose(
        solver.v.to_numpy()[index], expected_velocity, rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize("storage", ["dense", "tiled", "sparse"])
def test_srt_step_runs_with_each_storage_layout(storage):
    solid = np.ones((5, 5, 5), dtype=np.int8)
    solid[:, 2, 2] = 0
    solver = SinglePhaseSolver(
        solid,
        storage=storage,
        tile_size=4,
        collision_model="srt",
    )
    solver.set_bc_rho_x0(1.0)
    solver.set_bc_rho_x1(0.99)
    solver.set_viscosity(1.0 / 6.0)
    solver.init_simulation()

    solver.step()

    assert np.isfinite(solver.get_rho()).all()
    assert np.isfinite(solver.get_velocity()).all()
