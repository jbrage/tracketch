"""Fast Marching Method (FMM) arrival-time solver."""

import numpy as np
import skfmm
from tracketch.simulation.utils import (
    convert_map_logspace_to_linspace,
    convert_map_linspace_to_logspace,
)


def arrival_time_fmm(
    r_um: np.ndarray,
    z_um: np.ndarray,
    etch_rate_map: np.ndarray,
    r_is_logscaled: bool = True,
    theta_deg: float = 0.0,
    n_uniform_multiplier: int = 3,
):
    """
    Calculate arrival time map using fast marching method.

    Parameters
    ----------
    r_um : np.ndarray
        Radial grid (may be log-scaled)
    z_um : np.ndarray
        Depth grid (linear)
    etch_rate_map : np.ndarray
        Etch rate at each (z, r) point, shape (n_z, n_r)
    r_is_logscaled : bool, optional
        Whether r grid is log-scaled (default: True)
    theta_deg : float, optional
        Track angle relative to surface normal in degrees (default: 0.0).
        theta=0: track perpendicular to surface (wavefront starts at z=0)
        theta>0: tilted track, wavefront starts along z = r * tan(theta)
    n_uniform_multiplier : int, optional
        Multiplier for number of uniform grid points (default: 3).
        Higher values better preserve variation from log grid, especially at small radii.
        Set to 1 to use same number of points as original grid.

    Returns
    -------
    arrival_time_map : np.ndarray
        Arrival time at each (z, r) point
    """
    theta_rad = np.deg2rad(theta_deg)
    if r_is_logscaled:
        r_log_um = r_um
        # Convert to uniform grid
        r_uniform_um, _, etch_rate_uniform = convert_map_logspace_to_linspace(
            r_log_um, z_um, etch_rate_map, n_uniform_multiplier=n_uniform_multiplier
        )
    else:
        # Already uniform grid, use directly
        r_uniform_um = r_um
        r_log_um = None  # Not used for uniform grids
        etch_rate_uniform = etch_rate_map

    # Set up initial condition (phi) for tilted wavefront
    # phi should be negative inside the initial front, positive outside
    # For angled tracks: surface is at z = r * tan(theta) in track frame
    phi = np.ones_like(etch_rate_uniform)  # Initialize with positive values

    if theta_deg == 0.0:
        # Perpendicular track: wavefront starts at z=0 (all r values)
        phi[0, :] = 0
    else:
        # Tilted track: wavefront starts along z = r * tan(theta)
        # Find grid points closest to the tilted surface
        tan_theta = np.tan(theta_rad)
        for j, r_val in enumerate(r_uniform_um):
            z_surface = r_val * tan_theta
            # Find the z index closest to this surface position
            if z_surface <= z_um.max():
                i = np.argmin(np.abs(z_um - z_surface))
                phi[i, j] = 0
            # Points below the surface (z < z_surface) are already etched
            # Mark them as negative (inside the front)
            below_surface = z_um < z_surface
            phi[below_surface, j] = -1

    # Grid spacing for uniform grid
    dr = float(r_uniform_um[1] - r_uniform_um[0])
    dz = float(z_um[1] - z_um[0])

    # Calculate travel time map on uniform grid
    # skfmm expects speed (higher = faster), which is what etch_rate_uniform is
    arrival_time_uniform = skfmm.travel_time(phi, etch_rate_uniform, dx=[dz, dr])

    if r_is_logscaled:
        # Convert result back to original log-scaled grid
        arrival_time_map = convert_map_linspace_to_logspace(
            r_uniform_um, z_um, arrival_time_uniform, r_log_um
        )
    else:
        # Already on uniform grid, no conversion needed
        arrival_time_map = arrival_time_uniform

    return arrival_time_map
