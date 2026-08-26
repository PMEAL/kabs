"""Public API tests for collision-model selection and provenance."""

import numpy as np
import pytest
import taichi as ti

from kabs import FlowResult, solve_flow, solve_flow_taichi


_QUICK_SOLVE = {
    "direction": "x",
    "n_steps": 2,
    "tol": None,
    "log_every": 10,
    "verbose": False,
}


@pytest.fixture(autouse=True)
def _reset_taichi_after_test():
    yield
    ti.reset()
    ti.init(arch=ti.cpu)


def _channel():
    image = np.zeros((6, 5, 5), dtype=np.int8)
    image[:, 2, 2] = 1
    return image


def test_taichi_default_remains_mrt():
    result = solve_flow_taichi(_channel(), **_QUICK_SOLVE)

    assert result.collision_model == "mrt"
    assert result._solver.collision_model == "mrt"


def test_dispatcher_default_remains_taichi_mrt():
    result = solve_flow(_channel(), **_QUICK_SOLVE)

    assert result.collision_model == "mrt"


def test_public_api_selects_srt_case_insensitively():
    result = solve_flow(
        _channel(), collision_model="SRT", **_QUICK_SOLVE
    )

    assert result.collision_model == "srt"
    assert result._solver.collision_model == "srt"
    assert np.isfinite(result.rho).all()
    assert np.isfinite(result.velocity).all()


def test_flow_result_normalizes_known_collision_model():
    result = FlowResult.from_arrays(
        solid=np.zeros((2, 2, 2), dtype=np.int8),
        rho=np.ones((2, 2, 2), dtype=np.float32),
        velocity=np.zeros((2, 2, 2, 3), dtype=np.float32),
        collision_model="SRT",
    )

    assert result.collision_model == "srt"


@pytest.mark.parametrize("collision_model", ["bgk", "invalid", 1])
def test_invalid_taichi_collision_model_raises(collision_model):
    with pytest.raises(ValueError, match="collision_model"):
        solve_flow_taichi(
            _channel(), collision_model=collision_model, **_QUICK_SOLVE
        )


@pytest.mark.parametrize("collision_model", ["mrt", "MRT"])
def test_xlb_rejects_mrt_before_loading_backend(collision_model):
    with pytest.raises(ValueError, match="only supports.*srt"):
        solve_flow(
            _channel(),
            backend="xlb",
            collision_model=collision_model,
            **_QUICK_SOLVE,
        )


@pytest.mark.parametrize("collision_model", [None, "srt", "SRT"])
def test_xlb_accepts_its_default_or_explicit_srt(monkeypatch, collision_model):
    import kabs._solve_flow_xlb as xlb_module

    marker = object()
    monkeypatch.setattr(xlb_module, "solve_flow_xlb", lambda *args, **kwargs: marker)

    result = solve_flow(
        _channel(),
        backend="xlb",
        collision_model=collision_model,
        **_QUICK_SOLVE,
    )

    assert result is marker
