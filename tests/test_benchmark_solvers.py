"""Fast unit tests for the collision benchmark harness."""

import importlib.util
from pathlib import Path

import argparse
import numpy as np
import pytest


_BENCHMARK_PATH = Path(__file__).with_name("benchmark_solvers.py")
_SPEC = importlib.util.spec_from_file_location("benchmark_solvers", _BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_bundle_geometry_matches_validation_workload():
    image = benchmark.bundle_image()

    assert image.shape == (30, 50, 50)
    assert image.dtype == np.bool_
    assert 0.0 < image.mean() < 1.0


def test_analytical_permeability_matches_bundle_definition():
    expected = 4 * np.pi * 10**4 / (8 * 50 * 50)

    assert benchmark.analytical_permeability() == pytest.approx(expected)


def test_scaled_bundle_preserves_geometry_and_scales_permeability():
    image = benchmark.bundle_image(scale=2)

    assert image.shape == (60, 100, 100)
    assert image.mean() == pytest.approx(benchmark.bundle_image().mean(), rel=0.02)
    assert benchmark.analytical_permeability(scale=2) == pytest.approx(
        4 * benchmark.analytical_permeability()
    )


def test_memory_accounting_distinguishes_per_site_and_global_storage():
    assert benchmark.POPULATION_BYTES_PER_SITE == 152
    assert benchmark.MRT_OPERATOR_BYTES == (2 * 19 * 19 + 19) * 4


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_cli_integer_rejects_nonpositive_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        benchmark._positive_int(value)
