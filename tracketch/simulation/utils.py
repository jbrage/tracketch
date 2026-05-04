"""Simulation grid construction and coordinate-transform helpers."""

import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# defaults
MIN_R_UM = 1e-4
MAX_R_UM = 20  # max track radius
MAX_Z_UM = 40  # max track depth
N_POINTS_R = 400  # resolve the RDD sufficiently
N_POINTS_Z = 100


def convert_map_logspace_to_linspace(
    r_log: np.ndarray,
    z_linear: np.ndarray,
    data_map: np.ndarray,
    n_uniform_multiplier: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate a 2-D map from log-spaced *r* to uniform linear *r*.

    Parameters
    ----------
    r_log : ndarray
        Log-spaced radial coordinates (1-D).
    z_linear : ndarray
        Linear depth coordinates (1-D).
    data_map : ndarray, shape (n_z, n_r)
        Data on the ``(z_linear, r_log)`` grid.
    n_uniform_multiplier : int
        Resolution multiplier for the uniform grid.

    Returns
    -------
    r_uniform : ndarray
    z_linear : ndarray
    data_uniform : ndarray
    """
    n_r_uniform = len(r_log) * n_uniform_multiplier
    r_uniform = np.linspace(r_log.min(), r_log.max(), n_r_uniform)

    interp_fn = RegularGridInterpolator(
        (z_linear, r_log),
        data_map,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    R_uniform, Z_uniform = np.meshgrid(r_uniform, z_linear, indexing="xy")
    query_points = np.column_stack([Z_uniform.ravel(), R_uniform.ravel()])
    data_uniform = interp_fn(query_points).reshape(len(z_linear), n_r_uniform)

    return r_uniform, z_linear, data_uniform


def convert_map_linspace_to_logspace(
    r_uniform: np.ndarray,
    z_linear: np.ndarray,
    data_uniform: np.ndarray,
    r_log: np.ndarray,
) -> np.ndarray:
    """Interpolate a 2-D map from uniform linear *r* back to log-spaced *r*.

    Parameters
    ----------
    r_uniform : ndarray
        Uniform radial coordinates.
    z_linear : ndarray
        Depth coordinates.
    data_uniform : ndarray, shape (n_z, n_r_uniform)
        Data on the uniform grid.
    r_log : ndarray
        Target log-spaced radial coordinates.

    Returns
    -------
    ndarray, shape (n_z, n_r_log)
    """
    interp_fn = RegularGridInterpolator(
        (z_linear, r_uniform),
        data_uniform,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    R_log, Z_log = np.meshgrid(r_log, z_linear, indexing="xy")
    query_points = np.column_stack([Z_log.ravel(), R_log.ravel()])
    return interp_fn(query_points).reshape(len(z_linear), len(r_log))


def create_simulation_grid(
    rz_lims_dict: dict,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    """Create a 2-D (r, z) simulation grid.

    The radial axis is log-spaced (necessary for resolving narrow RDDs).

    Parameters
    ----------
    rz_lims_dict : dict
        Grid parameters.  Recognised keys: ``r_min_um``, ``r_max_um``,
        ``z_min_um``, ``z_max_um``, ``n_points_r``, ``n_points_z``.

    Returns
    -------
    r_grid_um : ndarray
    z_grid_um : ndarray
    r_bounds : tuple[float, float]
    z_bounds : tuple[float, float]

    Raises
    ------
    TypeError
        If a recognised key maps to a non-numeric value.
    """
    valid_keys = {
        "r_min_um",
        "r_max_um",
        "z_min_um",
        "z_max_um",
        "n_points_r",
        "n_points_z",
    }
    for key in rz_lims_dict:
        if key not in valid_keys:
            warnings.warn(
                f"Unrecognised key '{key}' in rz_lims_dict. "
                f"Valid keys: {sorted(valid_keys)}",
                stacklevel=2,
            )
        else:
            if not isinstance(rz_lims_dict[key], (int, float)):
                raise TypeError(
                    f"{key} must be a number, got {type(rz_lims_dict[key])}"
                )

    # n_points_r
    n_points_r = int(rz_lims_dict.get("n_points_r", N_POINTS_R))
    if n_points_r < 10:
        warnings.warn(
            "n_points_r should be at least 10 for meaningful results.",
            stacklevel=2,
        )

    # n_points_z
    n_points_z = int(rz_lims_dict.get("n_points_z", N_POINTS_Z))
    if n_points_z < 10:
        warnings.warn(
            "n_points_z should be at least 10 for meaningful results.",
            stacklevel=2,
        )

    # r_min
    r_min_um = rz_lims_dict.get("r_min_um", MIN_R_UM)
    if r_min_um < MIN_R_UM:
        warnings.warn(
            f"Track RDDs smaller than {MIN_R_UM * 1e3} nm may not be meaningful. "
            f"Defaulting to {MIN_R_UM} um.",
            stacklevel=2,
        )
        r_min_um = MIN_R_UM

    r_max_um = rz_lims_dict.get("r_max_um", MAX_R_UM)
    z_min_um = rz_lims_dict.get("z_min_um", 0)
    z_max_um = rz_lims_dict.get("z_max_um", MAX_Z_UM)

    r_grid_um = np.logspace(np.log10(r_min_um), np.log10(r_max_um), n_points_r)
    z_grid_um = np.linspace(z_min_um, z_max_um, n_points_z)

    return r_grid_um, z_grid_um, (r_min_um, r_max_um), (z_min_um, z_max_um)


def get_track_radius_from_contour(
    r_contour: np.ndarray,
    z_contour: np.ndarray,
    V_bulk_um_h: float,
    etch_time_h: float,
    threshold_percent: float = 5.0,
) -> float:
    """Find track radius where the iso-time contour deviates from bulk etching.

    Parameters
    ----------
    r_contour : ndarray
        Radial coordinates of iso-time contour (um).
    z_contour : ndarray
        Depth coordinates of iso-time contour (um).
    V_bulk_um_h : float
        Bulk etch rate in um/hr.
    etch_time_h : float
        Etching time in hours.
    threshold_percent : float
        Deviation threshold (percentage of bulk depth).

    Returns
    -------
    float
        Track radius in um, or ``nan`` if not found.
    """
    if len(r_contour) == 0 or len(z_contour) == 0:
        return np.nan

    z_bulk_expected = V_bulk_um_h * etch_time_h
    deviation_percent = 100 * np.abs(z_contour - z_bulk_expected) / z_bulk_expected

    mask = deviation_percent > threshold_percent
    if not np.any(mask):
        return np.nan

    # Sort by r descending to find transition from outside in
    sort_idx = np.argsort(r_contour)[::-1]
    r_sorted = r_contour[sort_idx]
    dev_sorted = deviation_percent[sort_idx]

    exceed_idx = np.where(dev_sorted > threshold_percent)[0]
    if len(exceed_idx) > 0:
        return float(r_sorted[exceed_idx[0]])

    return np.nan
