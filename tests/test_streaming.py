"""Focused tests for D3Q19 streaming and bounce-back."""

import numpy as np

from kabs._single_phase_solver import (
    SinglePhaseSolver,
    _D3Q19_DIRECTIONS,
    _D3Q19_OPPOSITE,
)


def _reference_stream(populations, destination, solid):
    expected = destination.copy()
    shape = solid.shape
    for index in np.ndindex(shape):
        if solid[index] != 0:
            continue
        for direction, offset in enumerate(_D3Q19_DIRECTIONS):
            neighbour = tuple(
                (index[axis] + offset[axis]) % shape[axis]
                for axis in range(3)
            )
            if solid[neighbour] == 0:
                expected[neighbour][direction] = populations[index][direction]
            else:
                opposite = _D3Q19_OPPOSITE[direction]
                expected[index][opposite] = populations[index][direction]
    return expected


def test_streaming_matches_reference_for_interior_bounce_back_and_wrapping():
    shape = (5, 6, 7)
    solid = np.zeros(shape, dtype=np.int8)
    solid[2, 3, 3] = 1
    solid[0, 1, 5] = 1
    solid[4, 4, 0] = 1
    solver = SinglePhaseSolver(solid, collision_model="srt")
    solver.init_simulation()

    populations = np.arange(
        np.prod(shape) * 19, dtype=np.float32
    ).reshape(*shape, 19)
    populations = populations / populations.size + 0.01
    destination = np.full_like(populations, -1.0)
    solver.f.from_numpy(populations)
    solver.F.from_numpy(destination)

    expected = _reference_stream(populations, destination, solid)
    solver.streaming1()

    np.testing.assert_array_equal(solver.F.to_numpy(), expected)
