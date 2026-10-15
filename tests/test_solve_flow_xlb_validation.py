"""Public validation for the optional XLB solver."""

import builtins
import importlib.util
import os

import numpy as np
import pytest

from kabs import solve_flow, solve_flow_xlb


_IMAGE = np.ones((3, 3, 3), dtype=np.int8)


def _block_xlb_import(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "xlb" or name.startswith("xlb."):
            raise AssertionError("XLB must not be imported during validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _hide_xlb(monkeypatch):
    """Make ``import xlb`` fail as it does when the ``xlb`` extra is not installed."""
    original_import = builtins.__import__

    def import_without_xlb(name, *args, **kwargs):
        if name == "xlb" or name.startswith("xlb."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_xlb)


@pytest.mark.parametrize("direction", ["invalid", None])
def test_invalid_direction_is_reported_without_importing_xlb(direction, monkeypatch):
    _block_xlb_import(monkeypatch)
    with pytest.raises(ValueError, match="direction"):
        solve_flow_xlb(_IMAGE, direction=direction)


@pytest.mark.parametrize("compute_backend", ["invalid", None])
def test_invalid_compute_backend_is_reported_without_importing_xlb(
    compute_backend, monkeypatch
):
    _block_xlb_import(monkeypatch)
    with pytest.raises(ValueError, match="compute_backend"):
        solve_flow_xlb(_IMAGE, compute_backend=compute_backend)


@pytest.mark.parametrize("compute_backend", ["jax", "warp"])
def test_missing_xlb_extra_is_named_in_the_error(compute_backend, monkeypatch):
    _hide_xlb(monkeypatch)
    with pytest.raises(ImportError, match=r"kabs\[xlb\]") as excinfo:
        solve_flow(_IMAGE, backend="xlb", compute_backend=compute_backend)
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


@pytest.mark.skipif(
    os.environ.get("KABS_TEST_WITHOUT_XLB") != "1",
    reason="set KABS_TEST_WITHOUT_XLB=1 in an environment installed without the xlb extra",
)
def test_install_without_the_extra_omits_the_xlb_stack():
    for module in ("xlb", "warp", "jax"):
        assert importlib.util.find_spec(module) is None, module
    with pytest.raises(ImportError, match=r"kabs\[xlb\]"):
        solve_flow(_IMAGE, backend="xlb")
