"""Tests for kabs/plots.py.

All tests use a synthetic FlowResult built from numpy arrays via
FlowResult.from_arrays(), so no LBM simulation is required and the suite
runs quickly.  The domain is intentionally asymmetric (10×12×14) so that
shape bugs (wrong axis, wrong transpose) are caught by the shape assertions.

plot_cross_section  — tested for all direction strings/ints and all slice axes
add_streamlines     — tested for all three slice axes using the Agg backend
render_flow         — skipped when pyvista is not installed; uses off_screen=True
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kabs._solve_flow import FlowResult
from kabs.plots import add_streamlines, plot_cross_section, render_flow


# ---------------------------------------------------------------------------
# Shared synthetic fixture
# ---------------------------------------------------------------------------

NX, NY, NZ = 10, 12, 14  # deliberately asymmetric


def _make_result(direction="x"):
    """Synthetic FlowResult: velocity increases linearly along x."""
    solid = np.zeros((NX, NY, NZ), dtype=np.int8)
    solid[0, :, :] = 1  # inlet face solid
    solid[-1, :, :] = 1  # outlet face solid

    rho = np.ones((NX, NY, NZ), dtype=np.float32)

    velocity = np.zeros((NX, NY, NZ, 3), dtype=np.float32)
    # Simple gradient: vx increases linearly along x, non-zero to avoid
    # degenerate all-zero fields (streamplot requires non-trivial vectors)
    for i in range(NX):
        velocity[i, :, :, 0] = (i + 1) * 1e-4
        velocity[i, :, :, 1] = (i + 1) * 5e-5
        velocity[i, :, :, 2] = (i + 1) * 2e-5

    return FlowResult.from_arrays(solid, rho, velocity, direction=direction, nu=1 / 6)


# ---------------------------------------------------------------------------
# plot_cross_section
# ---------------------------------------------------------------------------


class TestPlotCrossSection:
    def setup_method(self):
        self.result = _make_result(direction="x")

    # --- return type and ndim ---

    def test_returns_ndarray(self):
        out = plot_cross_section(self.result, direction="x", axis=2)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 2

    # --- shape for each slice axis ---
    # axis=0 → vel[mid_x, :, :]         → (NY, NZ)
    # axis=1 → vel[:, mid_y, :].T        → (NZ, NX)
    # axis=2 → vel[:, :, mid_z].T        → (NY, NX)

    def test_shape_axis0(self):
        out = plot_cross_section(self.result, direction="x", axis=0)
        assert out.shape == (NY, NZ)

    def test_shape_axis1(self):
        out = plot_cross_section(self.result, direction="x", axis=1)
        assert out.shape == (NZ, NX)

    def test_shape_axis2(self):
        out = plot_cross_section(self.result, direction="x", axis=2)
        assert out.shape == (NY, NX)

    # --- direction aliases ---

    def test_direction_string_x(self):
        out = plot_cross_section(self.result, direction="x", axis=2)
        assert out.shape == (NY, NX)

    def test_direction_string_y(self):
        out = plot_cross_section(self.result, direction="y", axis=2)
        assert out.shape == (NY, NX)

    def test_direction_string_z(self):
        out = plot_cross_section(self.result, direction="z", axis=2)
        assert out.shape == (NY, NX)

    def test_direction_int_0(self):
        out = plot_cross_section(self.result, direction=0, axis=2)
        assert out.shape == (NY, NX)

    def test_direction_int_1(self):
        out = plot_cross_section(self.result, direction=1, axis=2)
        assert out.shape == (NY, NX)

    def test_direction_int_2(self):
        out = plot_cross_section(self.result, direction=2, axis=2)
        assert out.shape == (NY, NX)

    def test_direction_all(self):
        out = plot_cross_section(self.result, direction="all", axis=2)
        assert out.shape == (NY, NX)

    def test_direction_none(self):
        out = plot_cross_section(self.result, direction=None, axis=2)
        assert out.shape == (NY, NX)

    # --- the midplane slice is taken at the correct index ---

    def test_axis2_midplane_values(self):
        """Slice at mid-z should equal vel[:, :, NZ//2, component].T."""
        out = plot_cross_section(self.result, direction="x", axis=2)
        expected = self.result.velocity[:, :, NZ // 2, 0].T
        np.testing.assert_array_equal(out, expected)

    def test_axis0_midplane_values(self):
        """Slice at mid-x should equal vel[NX//2, :, :, component]."""
        out = plot_cross_section(self.result, direction="x", axis=0)
        expected = self.result.velocity[NX // 2, :, :, 0]
        np.testing.assert_array_equal(out, expected)

    # --- "all" sums over components ---

    def test_direction_all_sums_components(self):
        out = plot_cross_section(self.result, direction="all", axis=2)
        v_sum = np.sum(self.result.velocity, axis=-1)
        expected = v_sum[:, :, NZ // 2].T
        np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# add_streamlines
# ---------------------------------------------------------------------------


class TestAddStreamlines:
    def setup_method(self):
        self.result = _make_result(direction="x")

    def _make_ax(self):
        fig, ax = plt.subplots()
        return ax

    def test_returns_axes(self):
        ax = self._make_ax()
        returned = add_streamlines(self.result, ax, axis=2)
        assert returned is ax
        plt.close("all")

    def test_axis0(self):
        ax = self._make_ax()
        add_streamlines(self.result, ax, axis=0)
        plt.close("all")

    def test_axis1(self):
        ax = self._make_ax()
        add_streamlines(self.result, ax, axis=1)
        plt.close("all")

    def test_axis2(self):
        ax = self._make_ax()
        add_streamlines(self.result, ax, axis=2)
        plt.close("all")

    def test_default_axis_uses_flow_direction(self):
        """When axis=None the slice axis should match the flow direction."""
        ax = self._make_ax()
        # direction="x" → axis 0 internally (yz-plane)
        add_streamlines(self.result, ax, axis=None)
        plt.close("all")

    def test_kwargs_forwarded(self):
        """Extra kwargs should not raise (e.g. color, density)."""
        ax = self._make_ax()
        add_streamlines(self.result, ax, axis=2, color="red", density=0.5)
        plt.close("all")


# ---------------------------------------------------------------------------
# render_flow  (requires pyvista)
# ---------------------------------------------------------------------------

pyvista = pytest.importorskip("pyvista", reason="pyvista not installed")


class TestRenderFlow:
    def setup_method(self):
        self.result = _make_result(direction="x")

    def test_returns_plotter(self):
        pl = render_flow(self.result, off_screen=True, show=False)
        assert isinstance(pl, pyvista.Plotter)

    def test_direction_y(self):
        result = _make_result(direction="y")
        pl = render_flow(result, off_screen=True, show=False)
        assert isinstance(pl, pyvista.Plotter)

    def test_direction_z(self):
        result = _make_result(direction="z")
        pl = render_flow(result, off_screen=True, show=False)
        assert isinstance(pl, pyvista.Plotter)

    def test_save_screenshot(self, tmp_path):
        out = tmp_path / "test_render.png"
        render_flow(self.result, off_screen=True, show=False, save=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_n_streamlines(self):
        pl = render_flow(self.result, n_streamlines=4, off_screen=True, show=False)
        assert isinstance(pl, pyvista.Plotter)


class TestRenderFlowImportError:
    """Verify a helpful ImportError is raised when pyvista is absent."""

    def test_import_error_message(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pyvista":
                raise ImportError("No module named 'pyvista'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pyvista"):
            render_flow(_make_result())


# ---------------------------------------------------------------------------
# Manual runner ("Run file in interactive window")
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    classes = [
        TestPlotCrossSection,
        TestAddStreamlines,
        TestRenderFlow,
        TestRenderFlowImportError,
    ]
    for cls in classes:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith("test_"):
                continue
            obj.setup_method()
            try:
                method = getattr(obj, name)
                import inspect

                sig = inspect.signature(method)
                if sig.parameters:
                    print(f"{cls.__name__}.{name}: skipped (needs fixtures)")
                    continue
                method()
                print(f"{cls.__name__}.{name}: passed")
            except Exception as e:
                print(f"{cls.__name__}.{name}: FAILED — {e}")
