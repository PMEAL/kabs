"""Backend-neutral convergence policy for pressure-driven flow solves."""

from dataclasses import dataclass
import math
from numbers import Integral, Real
import warnings

from ._flow_common import _darcy_permeability_from_flow


class _ArgumentDefault:
    """Sentinel with a user-facing representation for public signatures."""

    def __init__(self, display):
        self.display = display

    def __repr__(self):
        return repr(self.display)


TOL_UNSET = _ArgumentDefault(None)
VELOCITY_TOL_UNSET = _ArgumentDefault(1e-3)


def _optional_positive_float(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive finite number or None")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number or None")
    return value


def normalize_convergence_tolerances(
    *,
    tol=TOL_UNSET,
    velocity_tol=VELOCITY_TOL_UNSET,
    k_tol=None,
    flux_tol=None,
):
    """Resolve the deprecated ``tol`` alias and validate all tolerances."""
    tol_given = tol is not TOL_UNSET
    velocity_tol_given = velocity_tol is not VELOCITY_TOL_UNSET
    if tol_given and velocity_tol_given:
        raise ValueError("tol and velocity_tol cannot both be specified")
    if tol_given:
        warnings.warn(
            "tol is deprecated; use velocity_tol instead",
            DeprecationWarning,
            stacklevel=3,
        )
        effective_velocity_tol = tol
    elif velocity_tol_given:
        effective_velocity_tol = velocity_tol
    else:
        effective_velocity_tol = 1e-3

    return (
        _optional_positive_float("velocity_tol", effective_velocity_tol),
        _optional_positive_float("k_tol", k_tol),
        _optional_positive_float("flux_tol", flux_tol),
    )


def validate_solver_intervals(n_steps, log_every, convergence_every):
    """Validate integer loop controls and return normalized Python ints."""
    values = (
        ("n_steps", n_steps, True),
        ("log_every", log_every, True),
        ("convergence_every", convergence_every, False),
    )
    normalized = []
    for name, value, allow_zero in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        value = int(value)
        if value < 0 or (value == 0 and not allow_zero):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
        normalized.append(value)
    return tuple(normalized)


@dataclass(frozen=True)
class ConvergenceConfig:
    check_every: int
    velocity_tol: float | None
    k_tol: float | None
    flux_tol: float | None
    required_passes: int = 2

    def __post_init__(self):
        if isinstance(self.check_every, bool) or not isinstance(
            self.check_every, Integral
        ):
            raise TypeError("check_every must be an integer")
        if self.check_every <= 0:
            raise ValueError("check_every must be positive")
        if isinstance(self.required_passes, bool) or not isinstance(
            self.required_passes, Integral
        ):
            raise TypeError("required_passes must be an integer")
        if self.required_passes <= 0:
            raise ValueError("required_passes must be positive")
        for name in ("velocity_tol", "k_tol", "flux_tol"):
            object.__setattr__(
                self,
                name,
                _optional_positive_float(name, getattr(self, name)),
            )

    @property
    def monitoring_enabled(self):
        return any(
            tolerance is not None
            for tolerance in (self.velocity_tol, self.k_tol, self.flux_tol)
        )

    @property
    def needs_velocity(self):
        return self.velocity_tol is not None

    @property
    def needs_permeability(self):
        return self.k_tol is not None

    @property
    def needs_flux(self):
        return self.flux_tol is not None


@dataclass(frozen=True)
class ConvergenceObservables:
    velocity_total: float | None = None
    velocity_change: float | None = None
    directional_flow: float | None = None
    inlet_mass_flux: float | None = None
    outlet_mass_flux: float | None = None


@dataclass(frozen=True)
class ConvergenceReport:
    velocity_criterion: float | None
    k_criterion: float | None
    flux_criterion: float | None
    permeability: float | None
    consecutive_passes: int
    converged: bool


def _finite_float(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _relative_ratio(numerator, denominator):
    numerator = _finite_float(numerator)
    denominator = _finite_float(denominator)
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else None


class ConvergenceController:
    """Calculate convergence criteria and own the shared stopping policy."""

    def __init__(
        self,
        config,
        *,
        domain_shape,
        direction,
        nu,
        rho_in,
        rho_out,
    ):
        if not isinstance(config, ConvergenceConfig):
            raise TypeError("config must be a ConvergenceConfig")
        if config.check_every <= 0:
            raise ValueError("check_every must be positive")
        if config.required_passes < 1:
            raise ValueError("required_passes must be positive")
        if len(domain_shape) != 3 or any(int(size) <= 0 for size in domain_shape):
            raise ValueError("domain_shape must contain three positive dimensions")
        if direction not in ("x", "y", "z"):
            raise ValueError("direction must be 'x', 'y', or 'z'")
        for name, value in (("nu", nu), ("rho_in", rho_in), ("rho_out", rho_out)):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if float(nu) <= 0.0:
            raise ValueError("nu must be positive")
        if float(rho_in) == float(rho_out):
            raise ValueError("rho_in and rho_out must differ")

        self.config = config
        self.domain_shape = tuple(int(size) for size in domain_shape)
        self.direction = direction
        self.nu = float(nu)
        self.rho_in = float(rho_in)
        self.rho_out = float(rho_out)
        self._previous_permeability = None
        self._consecutive_passes = 0

    @property
    def monitoring_enabled(self):
        return self.config.monitoring_enabled

    @property
    def needs_velocity(self):
        return self.config.needs_velocity

    @property
    def needs_permeability(self):
        return self.config.needs_permeability

    @property
    def needs_flux(self):
        return self.config.needs_flux

    def check_due(self, completed_steps):
        return self.monitoring_enabled and completed_steps % self.config.check_every == 0

    def update(self, observables):
        if not isinstance(observables, ConvergenceObservables):
            raise TypeError("observables must be ConvergenceObservables")

        velocity_criterion = None
        if self.needs_velocity:
            total = _finite_float(observables.velocity_total)
            change = _finite_float(observables.velocity_change)
            if total is not None and change is not None:
                velocity_criterion = _relative_ratio(abs(change), abs(total))

        permeability = None
        k_criterion = None
        if self.needs_permeability:
            flow = _finite_float(observables.directional_flow)
            if flow is not None:
                permeability = _darcy_permeability_from_flow(
                    directional_flow=flow,
                    domain_shape=self.domain_shape,
                    direction=self.direction,
                    nu=self.nu,
                    rho_in=self.rho_in,
                    rho_out=self.rho_out,
                )
                permeability = _finite_float(permeability)
            if permeability is not None and self._previous_permeability is not None:
                k_criterion = _relative_ratio(
                    abs(permeability - self._previous_permeability),
                    abs(permeability),
                )
            self._previous_permeability = permeability

        flux_criterion = None
        if self.needs_flux:
            inlet = _finite_float(observables.inlet_mass_flux)
            outlet = _finite_float(observables.outlet_mass_flux)
            if inlet is not None and outlet is not None:
                flux_criterion = _relative_ratio(
                    abs(inlet - outlet),
                    0.5 * (abs(inlet) + abs(outlet)),
                )

        enabled = (
            (self.config.velocity_tol, velocity_criterion),
            (self.config.k_tol, k_criterion),
            (self.config.flux_tol, flux_criterion),
        )
        passes = [
            criterion is not None and criterion < tolerance
            for tolerance, criterion in enabled
            if tolerance is not None
        ]
        if passes and all(passes):
            self._consecutive_passes += 1
        else:
            self._consecutive_passes = 0

        converged = self._consecutive_passes >= self.config.required_passes
        return ConvergenceReport(
            velocity_criterion=velocity_criterion,
            k_criterion=k_criterion,
            flux_criterion=flux_criterion,
            permeability=permeability,
            consecutive_passes=self._consecutive_passes,
            converged=converged,
        )
