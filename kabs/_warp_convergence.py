"""Warp-native convergence measurements for the XLB backend."""

import numpy as np
import warp as wp

from ._convergence import ConvergenceObservables


@wp.kernel
def _reduce_observables(
    rho: wp.array4d(dtype=wp.float32),
    current: wp.array4d(dtype=wp.float32),
    previous: wp.array4d(dtype=wp.float32),
    pore: wp.array3d(dtype=wp.int8),
    sums: wp.array(dtype=wp.float32),
    axis: int,
    velocity_offset: int,
    flow_offset: int,
    flux_offset: int,
    has_previous: int,
):
    i, j, k = wp.tid()
    if pore[i, j, k] == 0:
        return

    if velocity_offset >= 0:
        for component in range(3):
            value = current[component, i, j, k]
            wp.atomic_add(sums, velocity_offset, wp.abs(value))
            if has_previous != 0:
                wp.atomic_add(
                    sums,
                    velocity_offset + 1,
                    wp.abs(value - previous[component, i, j, k]),
                )

    directional_velocity = current[axis, i, j, k]
    if flow_offset >= 0:
        wp.atomic_add(sums, flow_offset, directional_velocity)

    if flux_offset >= 0:
        coordinate = i
        axis_size = current.shape[1]
        if axis == 1:
            coordinate = j
            axis_size = current.shape[2]
        elif axis == 2:
            coordinate = k
            axis_size = current.shape[3]
        mass_flux = rho[0, i, j, k] * directional_velocity
        if coordinate == 0:
            wp.atomic_add(sums, flux_offset, mass_flux)
        if coordinate == axis_size - 1:
            wp.atomic_add(sums, flux_offset + 1, mass_flux)


class WarpConvergenceMonitor:
    """Reduce only enabled convergence observables on a Warp device."""

    def __init__(self, velocity, pore_mask, config, flow_axis):
        if velocity.dtype != wp.float32:
            raise TypeError("Warp convergence requires FP32 velocity fields")
        self._config = config
        self._flow_axis = int(flow_axis)
        self._previous = (
            wp.zeros_like(velocity) if config.needs_velocity else None
        )
        self._has_snapshot = False
        self._pore = wp.array(
            np.asarray(pore_mask, dtype=np.int8),
            dtype=wp.int8,
            device=velocity.device,
        )
        offset = 0
        self._velocity_offset = -1
        self._flow_offset = -1
        self._flux_offset = -1
        if config.needs_velocity:
            self._velocity_offset = offset
            offset += 2
        if config.needs_permeability:
            self._flow_offset = offset
            offset += 1
        if config.needs_flux:
            self._flux_offset = offset
            offset += 2
        self._sums = wp.zeros(offset, dtype=wp.float32, device=velocity.device)

    def sample(self, rho, velocity):
        """Synchronize and return one small host-visible observable record."""
        if (
            self._config.needs_velocity
            and not self._has_snapshot
            and not self._config.needs_permeability
            and not self._config.needs_flux
        ):
            return ConvergenceObservables()

        wp.synchronize_device(velocity.device)
        self._sums.zero_()
        wp.launch(
            _reduce_observables,
            dim=velocity.shape[1:],
            inputs=[
                rho,
                velocity,
                self._previous if self._previous is not None else velocity,
                self._pore,
                self._sums,
                self._flow_axis,
                self._velocity_offset,
                self._flow_offset,
                self._flux_offset,
                1 if self._has_snapshot else 0,
            ],
            device=velocity.device,
        )
        values = self._sums.numpy()
        return ConvergenceObservables(
            velocity_total=(
                float(values[self._velocity_offset])
                if self._velocity_offset >= 0
                else None
            ),
            velocity_change=(
                float(values[self._velocity_offset + 1])
                if self._velocity_offset >= 0 and self._has_snapshot
                else None
            ),
            directional_flow=(
                float(values[self._flow_offset])
                if self._flow_offset >= 0
                else None
            ),
            inlet_mass_flux=(
                float(values[self._flux_offset])
                if self._flux_offset >= 0
                else None
            ),
            outlet_mass_flux=(
                float(values[self._flux_offset + 1])
                if self._flux_offset >= 0
                else None
            ),
        )

    def snapshot_velocity(self, velocity):
        if self._previous is not None:
            wp.synchronize_device(velocity.device)
            wp.copy(self._previous, velocity)
            self._has_snapshot = True
