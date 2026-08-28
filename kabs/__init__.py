try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("kabs")
except PackageNotFoundError:
    import tomllib
    from pathlib import Path

    _pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(_pyproject, "rb") as _f:
        __version__ = tomllib.load(_f)["project"]["version"]
    del tomllib, Path, _pyproject, _f

from ._compute_hydraulic_conductance import (
    check_flow_development as check_flow_development,
    format_hydraulic_conductance_report as format_hydraulic_conductance_report,
    solve_hydraulic_conductance as solve_hydraulic_conductance,
)
from ._compute_permeability import compute_permeability as compute_permeability
from ._flow_common import FlowResult as FlowResult
from ._solve_flow import solve_flow as solve_flow, solve_flow_taichi as solve_flow_taichi
from ._solve_flow_xlb import solve_flow_xlb as solve_flow_xlb
from .plots import (
    add_streamlines as add_streamlines,
    plot_cross_section as plot_cross_section,
    render_flow as render_flow,
)


__all__ = [
    "FlowResult",
    "SinglePhaseSolver",
    "add_streamlines",
    "check_flow_development",
    "compute_permeability",
    "format_hydraulic_conductance_report",
    "plot_cross_section",
    "render_flow",
    "solve_flow",
    "solve_flow_taichi",
    "solve_flow_xlb",
    "solve_hydraulic_conductance",
]


def __getattr__(name):
    """Load Taichi implementation details only when explicitly requested."""
    if name == "SinglePhaseSolver":
        from ._single_phase_solver import SinglePhaseSolver

        globals()[name] = SinglePhaseSolver
        return SinglePhaseSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), "SinglePhaseSolver"})
