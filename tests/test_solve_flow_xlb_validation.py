"""Public validation for the optional XLB solver."""

import builtins

import numpy as np
import pytest

from kabs import solve_flow_xlb


_IMAGE = np.ones((3, 3, 3), dtype=np.int8)


def _block_xlb_import(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "xlb" or name.startswith("xlb."):
            raise AssertionError("XLB must not be imported during validation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


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
