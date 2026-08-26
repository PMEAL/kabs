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
        log_every=1,
        tol=None,
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
        log_every=1,
        tol=1e9,
        verbose=False,
    )

    assert result.n_iterations == 1
    assert snapshot_steps == [1]


def test_single_check_skips_unused_snapshot_allocation():
    result = solve_flow_taichi(
        np.ones((3, 3, 3), dtype=np.int8),
        n_steps=0,
        log_every=10,
        tol=None,
        verbose=False,
    )

    assert result.convergence_criterion is None
    assert result._solver.v_previous is None
