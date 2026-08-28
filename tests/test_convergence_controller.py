"""Backend-neutral convergence policy tests."""

import math

import pytest

from kabs._convergence import (
    ConvergenceConfig,
    ConvergenceController,
    ConvergenceObservables,
    TOL_UNSET,
    VELOCITY_TOL_UNSET,
    normalize_convergence_tolerances,
    validate_solver_intervals,
)


def _controller(*, velocity_tol=None, k_tol=None, flux_tol=None):
    return ConvergenceController(
        ConvergenceConfig(5, velocity_tol, k_tol, flux_tol),
        domain_shape=(10, 4, 3),
        direction="x",
        nu=1.0 / 6.0,
        rho_in=1.0,
        rho_out=0.99,
    )


def test_velocity_only_requires_two_defined_passes():
    controller = _controller(velocity_tol=0.01)
    assert not controller.update(ConvergenceObservables()).converged
    first = controller.update(
        ConvergenceObservables(velocity_total=10.0, velocity_change=0.05)
    )
    second = controller.update(
        ConvergenceObservables(velocity_total=10.0, velocity_change=0.05)
    )
    assert first.velocity_criterion == pytest.approx(0.005)
    assert first.consecutive_passes == 1
    assert second.consecutive_passes == 2
    assert second.converged


def test_permeability_only_requires_a_baseline_and_two_passes():
    controller = _controller(k_tol=0.02)
    reports = [
        controller.update(ConvergenceObservables(directional_flow=flow))
        for flow in (1.0, 1.01, 1.02)
    ]
    assert reports[0].k_criterion is None
    assert reports[1].consecutive_passes == 1
    assert reports[2].converged


def test_flux_only_can_converge_on_second_sample():
    controller = _controller(flux_tol=0.01)
    first = controller.update(
        ConvergenceObservables(inlet_mass_flux=10.0, outlet_mass_flux=9.95)
    )
    second = controller.update(
        ConvergenceObservables(inlet_mass_flux=10.0, outlet_mass_flux=9.95)
    )
    assert first.consecutive_passes == 1
    assert second.converged


@pytest.mark.parametrize("failed", ["velocity", "k", "flux"])
def test_any_enabled_failure_resets_shared_streak(failed):
    controller = _controller(velocity_tol=0.1, k_tol=0.1, flux_tol=0.1)
    baseline = ConvergenceObservables(
        velocity_total=10.0,
        velocity_change=0.1,
        directional_flow=10.0,
        inlet_mass_flux=10.0,
        outlet_mass_flux=10.0,
    )
    controller.update(baseline)
    assert controller.update(baseline).consecutive_passes == 1
    values = dict(
        velocity_total=10.0,
        velocity_change=0.1,
        directional_flow=10.0,
        inlet_mass_flux=10.0,
        outlet_mass_flux=10.0,
    )
    if failed == "velocity":
        values["velocity_change"] = 1.0
    elif failed == "k":
        values["directional_flow"] = 20.0
    else:
        values["outlet_mass_flux"] = 1.0
    assert controller.update(ConvergenceObservables(**values)).consecutive_passes == 0


def test_equal_tolerance_does_not_pass():
    controller = _controller(velocity_tol=0.1)
    report = controller.update(
        ConvergenceObservables(velocity_total=10.0, velocity_change=1.0)
    )
    assert report.velocity_criterion == pytest.approx(0.1)
    assert report.consecutive_passes == 0


@pytest.mark.parametrize("bad", [0.0, math.nan, math.inf, -math.inf, None])
def test_bad_velocity_denominator_or_observable_never_passes(bad):
    controller = _controller(velocity_tol=1.0)
    report = controller.update(
        ConvergenceObservables(velocity_total=bad, velocity_change=0.0)
    )
    assert report.velocity_criterion is None
    assert report.consecutive_passes == 0


def test_all_disabled_has_no_monitoring():
    config = ConvergenceConfig(5, None, None, None)
    assert not config.monitoring_enabled
    assert not config.needs_velocity
    assert not config.needs_permeability
    assert not config.needs_flux


def test_check_schedule_uses_completed_steps():
    controller = _controller(velocity_tol=0.1)
    assert [controller.check_due(step) for step in range(1, 11)] == [
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]


def test_omitted_tolerances_preserve_default_velocity_tolerance():
    assert normalize_convergence_tolerances(
        tol=TOL_UNSET, velocity_tol=VELOCITY_TOL_UNSET
    ) == (1e-3, None, None)


def test_legacy_tol_alias_warns_and_supports_none():
    with pytest.warns(DeprecationWarning):
        assert normalize_convergence_tolerances(tol=1e-4) == (1e-4, None, None)
    with pytest.warns(DeprecationWarning):
        assert normalize_convergence_tolerances(tol=None) == (None, None, None)


def test_legacy_and_new_velocity_tolerances_conflict():
    with pytest.raises(ValueError, match="cannot both"):
        normalize_convergence_tolerances(tol=1e-3, velocity_tol=1e-3)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("velocity_tol", 0.0, ValueError),
        ("k_tol", math.inf, ValueError),
        ("flux_tol", True, TypeError),
    ],
)
def test_invalid_tolerances(name, value, error):
    kwargs = {name: value}
    with pytest.raises(error):
        normalize_convergence_tolerances(**kwargs)


@pytest.mark.parametrize(
    "values",
    [(-1, 1, 1), (1, -1, 1), (1, 1, 0), (True, 1, 1)],
)
def test_invalid_loop_controls(values):
    with pytest.raises((TypeError, ValueError)):
        validate_solver_intervals(*values)
