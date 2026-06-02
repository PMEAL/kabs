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

from ._single_phase_solver import *
from ._solve_flow import *
from ._solve_flow_xlb import *
from ._compute_permeability import *
from ._compute_hydraulic_conductance import *
from .plots import *

# Result containers (also exported via their respective solve modules' __all__)
from ._solve_flow import FlowResult
