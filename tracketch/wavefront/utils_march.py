"""Arrival-time map dispatcher and iso-time contour extraction."""

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

from tracketch.wavefront.fmm import arrival_time_fmm
from tracketch.wavefront.dijkstra import arrival_time_dijkstra


VALID_METHODS = ["fmm", "dijkstra", "dijkstra_numba", "dijkstra_cpp"]


def get_arrival_time_map(
    r_um: np.ndarray,
    z_um: np.ndarray,
    etch_rate_map: np.ndarray,
    method: str = "dijkstra",
    r_is_logscaled: bool = True,
    theta_deg: float = 0.0,
    n_uniform_multiplier: int = 3,
    connectivity: Literal[8, 16, 32] = 8,
) -> np.ndarray:
    """Calculate the arrival-time map using the specified method.

    Parameters
    ----------
    r_um : ndarray
        Radial coordinates in um.
    z_um : ndarray
        Axial coordinates in um.
    etch_rate_map : ndarray
        Etch rates in um/hr, shape ``(n_z, n_r)``.
    method : str
        ``"dijkstra_cpp"`` (fastest), ``"dijkstra_numba"``, or ``"fmm"``
        (supports tilted wavefronts).
    r_is_logscaled : bool
        Whether the radial coordinates are log-spaced.
    theta_deg : float
        Track angle relative to the surface normal in degrees.
    n_uniform_multiplier : int
        FMM only -- multiplier for the uniform-grid resolution.
    connectivity : {8, 16, 32}
        Dijkstra only -- neighbour connectivity.

    Returns
    -------
    ndarray
        Arrival-time map in hours, same shape as *etch_rate_map*.
    """
    logger.debug("Arrival-time dispatch: method='%s'", method)
    if method == "fmm":
        return arrival_time_fmm(
            r_um,
            z_um,
            etch_rate_map,
            r_is_logscaled=r_is_logscaled,
            theta_deg=theta_deg,
            n_uniform_multiplier=n_uniform_multiplier,
        )
    elif method in ["dijkstra", "dijkstra_numba", "dijkstra_cpp"]:
        if theta_deg != 0.0:
            raise NotImplementedError(
                "Tilted wavefront (theta_deg != 0) not yet implemented for dijkstra. Use 'fmm'."
            )
        backend_map = {
            "dijkstra": "auto",
            "dijkstra_cpp": "cpp",
            "dijkstra_numba": "numba",
        }
        backend = backend_map[method]
        return arrival_time_dijkstra(
            r_um,
            z_um,
            etch_rate_map,
            r_is_logscaled=r_is_logscaled,
            connectivity=connectivity,
            backend=backend,
        )
    else:
        raise ValueError(f"Invalid method '{method}'. Choose from {VALID_METHODS}.")


def get_iso_time_contour(
    arrival_time_map: np.ndarray,
    r_um: np.ndarray,
    z_um: np.ndarray,
    etching_time_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the iso-time contour from the arrival-time map.

    Parameters
    ----------
    arrival_time_map : ndarray, shape (n_z, n_r)
        Arrival times in hours.
    r_um : ndarray
        Radial coordinates (may be log-spaced).
    z_um : ndarray
        Depth coordinates (linear).
    etching_time_h : float
        Target arrival time in hours.

    Returns
    -------
    r_coords : ndarray
    z_coords : ndarray
        Coordinates along the longest contour segment.
    """
    from skimage import measure

    contour_paths = measure.find_contours(arrival_time_map, etching_time_h)

    if len(contour_paths) == 0:
        return np.array([]), np.array([])

    # Convert from image (row, col) to physical (r, z) coordinates
    contours = []
    for contour in contour_paths:
        z_coords = np.interp(contour[:, 0], np.arange(len(z_um)), z_um)
        r_coords = np.interp(contour[:, 1], np.arange(len(r_um)), r_um)
        contours.append((r_coords, z_coords))

    longest_contour = max(contours, key=lambda c: len(c[0]))
    return longest_contour
