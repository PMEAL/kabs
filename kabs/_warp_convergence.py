"""Warp-native convergence measurements for the XLB backend."""

from dataclasses import dataclass

import warp as wp


@dataclass(frozen=True)
class WarpConvergenceMetrics:
    """Scalar convergence measurements copied from a Warp device."""

    velocity_total: float
    velocity_change: float


@wp.kernel
def _reduce_velocity_change(
    current: wp.array4d(dtype=wp.float32),
    previous: wp.array4d(dtype=wp.float32),
    sums: wp.array(dtype=wp.float32),
):
    i, j, k = wp.tid()
    for component in range(3):
        value = current[component, i, j, k]
        wp.atomic_add(sums, 0, wp.abs(value))
        wp.atomic_add(
            sums,
            1,
            wp.abs(value - previous[component, i, j, k]),
        )


class WarpConvergenceMonitor:
    """Measure convergence inputs while retaining state on the Warp device.

    This class is deliberately limited to measurement. The solver owns the
    convergence policy, making it possible to add permeability-stability
    metrics without coupling their calculation to stopping decisions.
    """

    def __init__(self, velocity):
        if velocity.dtype != wp.float32:
            raise TypeError("Warp convergence currently requires FP32 velocity fields")
        self._previous = wp.zeros_like(velocity)
        self._sums = wp.zeros(2, dtype=wp.float32, device=velocity.device)
        self._has_snapshot = False

    def sample(self, velocity):
        """Return scalar metrics, or ``None`` while taking the first snapshot."""
        # XLB owns the macroscopic launch, so make its output visible before
        # KABS snapshots or reduces it. This remains correct if the two
        # libraries use different Warp streams in a future release.
        wp.synchronize_device(velocity.device)
        if not self._has_snapshot:
            wp.copy(self._previous, velocity)
            self._has_snapshot = True
            return None

        self._sums.zero_()
        wp.launch(
            _reduce_velocity_change,
            dim=velocity.shape[1:],
            inputs=[velocity, self._previous, self._sums],
            device=velocity.device,
        )
        wp.synchronize_device(velocity.device)
        sums = self._sums.numpy()
        wp.copy(self._previous, velocity)
        return WarpConvergenceMetrics(
            velocity_total=float(sums[0]),
            velocity_change=float(sums[1]),
        )
