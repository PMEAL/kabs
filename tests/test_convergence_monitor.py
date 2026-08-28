"""Focused tests for device-resident Taichi convergence monitoring."""

import numpy as np
import pytest
import taichi as ti

from kabs import solve_flow_taichi
from kabs._single_phase_solver import SinglePhaseSolver


@pytest.fixture(autouse=True)
def _reset_taichi_after_test():
    """Prevent permanent test fields from accumulating in Taichi root storage."""
    yield
    ti.reset()
    ti.init(arch=ti.cpu)


def _solver(shape, storage="dense", tile_size=2):
    return SinglePhaseSolver(
        np.zeros(shape, dtype=np.int8),
        storage=storage,
        tile_size=tile_size,
        _enable_convergence_monitor=True,
    )


def _device_sums(previous, current):
    solver = _solver(current.shape[:3])
    solver.v.from_numpy(previous.astype(np.float32))
    solver.snapshot_velocity()
    solver.v.from_numpy(current.astype(np.float32))
    solver.reset_convergence_sums()
    solver.accumulate_convergence_sums()
    return solver.get_convergence_sums()


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (
            np.array([[[[1.0, -2.0, 3.0]]]], dtype=np.float32),
            np.array([[[[1.0, -2.0, 3.0]]]], dtype=np.float32),
        ),
        (
            np.ones((2, 2, 2, 3), dtype=np.float32),
            np.full((2, 2, 2, 3), 2.0, dtype=np.float32),
        ),
        (
            np.array([[[[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]]]]),
            np.array([[[[2.0, -1.0, -4.0], [-3.0, 5.0, 1.0]]]]),
        ),
    ],
)
def test_device_sums_match_numpy_formula(previous, current):
    v_total, v_change = _device_sums(previous, current)

    assert v_total == pytest.approx(np.sum(np.abs(current)), rel=1e-6)
    assert v_change == pytest.approx(np.sum(np.abs(current - previous)), rel=1e-6)


def test_zero_total_does_not_satisfy_convergence_guard():
    previous = np.ones((2, 2, 2, 3), dtype=np.float32)
    current = np.zeros_like(previous)

    v_total, v_change = _device_sums(previous, current)

    assert v_total == 0.0
    assert v_change > 0.0
    assert not (v_total > 0 and v_change / v_total < 1e-3)


@ti.kernel
def _set_test_velocity(v: ti.template(), scale: ti.f32):
    for i, j, k in v:
        v[i, j, k] = scale * ti.Vector([i - 1.0, j + 0.5, -k - 0.25])


@pytest.mark.parametrize("storage", ["dense", "tiled", "sparse"])
def test_storage_layouts_match_numpy_formula(storage):
    solver = _solver((3, 4, 5), storage=storage, tile_size=(2, 3, 4))
    solver.init_simulation()
    _set_test_velocity(solver.v, 1.0)
    previous = solver.get_velocity().copy()
    solver.snapshot_velocity()
    _set_test_velocity(solver.v, -0.5)
    current = solver.get_velocity().copy()

    solver.reset_convergence_sums()
    solver.accumulate_convergence_sums()
    v_total, v_change = solver.get_convergence_sums()

    assert v_total == pytest.approx(np.sum(np.abs(current)), rel=1e-5)
    assert v_change == pytest.approx(np.sum(np.abs(current - previous)), rel=1e-5)


def test_sparse_snapshot_shares_population_tile_activation():
    internal_solid = np.ones((8, 8, 8), dtype=np.int8)
    internal_solid[1, 1, 1] = 0
    solver = SinglePhaseSolver(
        internal_solid,
        storage="sparse",
        tile_size=4,
        _enable_convergence_monitor=True,
    )

    assert solver.v.snode.parent() == solver._population_cells
    assert solver.v_previous.snode.parent() == solver._population_cells

    solver.init_simulation()
    solver.snapshot_velocity()
    assert solver.v_previous.snode.parent() == solver.v.snode.parent()


def test_periodic_checks_do_not_extract_velocity(monkeypatch):
    calls = 0
    original = SinglePhaseSolver.get_velocity

    def counted_get_velocity(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(SinglePhaseSolver, "get_velocity", counted_get_velocity)
    image = np.ones((3, 3, 3), dtype=np.int8)

    result = solve_flow_taichi(
        image,
        n_steps=2,
        log_every=0,
        convergence_every=1,
        velocity_tol=1e-12,
        verbose=False,
    )

    assert calls == 1  # Final FlowResult extraction only.
    assert result.convergence_criterion is not None


def test_first_snapshot_is_post_step_and_terminal_check_does_not_resnapshot(
    monkeypatch,
):
    steps_completed = 0
    snapshot_steps = []
    original_step = SinglePhaseSolver.step

    def counted_step(self):
        nonlocal steps_completed
        original_step(self)
        steps_completed += 1

    def counted_snapshot(self):
        snapshot_steps.append(steps_completed)
        self.v_previous.copy_from(self.v)

    monkeypatch.setattr(SinglePhaseSolver, "step", counted_step)
    monkeypatch.setattr(SinglePhaseSolver, "snapshot_velocity", counted_snapshot)

    result = solve_flow_taichi(
        np.ones((3, 3, 3), dtype=np.int8),
        n_steps=1,
        log_every=0,
        convergence_every=1,
        velocity_tol=1e9,
        verbose=False,
    )

    assert result.n_iterations == 1
    assert snapshot_steps == [1]


def test_disabled_monitor_skips_snapshot_allocation():
    result = solve_flow_taichi(
        np.ones((3, 3, 3), dtype=np.int8),
        n_steps=0,
        log_every=10,
        velocity_tol=None,
        k_tol=None,
        flux_tol=None,
        verbose=False,
    )

    assert result.convergence_criterion is None
    assert result._solver.v_previous is None
    assert result._solver.convergence_sums is None


@pytest.mark.parametrize(
    ("storage", "direction", "axis"),
    [("dense", "x", 0), ("tiled", "y", 1), ("sparse", "z", 2)],
)
def test_combined_reducer_matches_masked_numpy(storage, direction, axis):
    shape = (3, 4, 5)
    solid = np.zeros(shape, dtype=np.int8)
    solid[1, 2, 3] = 1
    solver = SinglePhaseSolver(
        solid,
        storage=storage,
        tile_size=(2, 3, 4),
        _convergence_needs_velocity=True,
        _convergence_needs_permeability=True,
        _convergence_needs_flux=True,
        _convergence_direction=direction,
    )
    solver.init_simulation()
    rng = np.random.default_rng(42)
    previous = rng.normal(size=(*shape, 3)).astype(np.float32)
    current = rng.normal(size=(*shape, 3)).astype(np.float32)
    rho = rng.uniform(0.9, 1.1, size=shape).astype(np.float32)
    # Deliberately nonzero solid values prove that all observables are masked.
    previous[solid != 0] = 50.0
    current[solid != 0] = -50.0
    rho[solid != 0] = 20.0
    solver.v.from_numpy(previous)
    solver.snapshot_velocity()
    solver.v.from_numpy(current)
    rho_storage = np.zeros(solver.rho.shape, dtype=np.float32)
    rho_storage[: shape[0], : shape[1], : shape[2]] = rho
    solver.rho.from_numpy(rho_storage)
    solver.reset_convergence_sums()
    solver.accumulate_convergence_sums()
    actual = solver.get_convergence_sums()

    pore = solid == 0
    directional = current[..., axis]
    inlet = [slice(None)] * 3
    outlet = [slice(None)] * 3
    inlet[axis] = 0
    outlet[axis] = -1
    expected = (
        np.sum(np.abs(current[pore])),
        np.sum(np.abs(current[pore] - previous[pore])),
        np.sum(directional[pore]),
        np.sum((rho * directional * pore)[tuple(inlet)]),
        np.sum((rho * directional * pore)[tuple(outlet)]),
    )
    assert actual == pytest.approx(expected, rel=2e-5, abs=2e-6)
