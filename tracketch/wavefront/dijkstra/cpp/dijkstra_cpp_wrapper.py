# %%
"""
C++ Dijkstra wrapper for non-uniform grids.

This module provides a Python interface to the C++ Dijkstra implementation.
To use, first compile the C++ extension:

    cd /path/to/tracketch/wavefront/dijkstra/cpp
    pip install pybind11
    python setup_dijkstra.py build_ext --inplace
"""

import numpy as np
from typing import Literal

try:
    from .dijkstra_cpp import arrival_time_dijkstra_cpp as _dijkstra_cpp

    HAS_CPP = True
except ImportError:
    HAS_CPP = False


def arrival_time_dijkstra_cpp(
    r_um: np.ndarray,
    z_um: np.ndarray,
    etch_rate_map: np.ndarray,
    r_is_logscaled: bool = True,
    connectivity: Literal[8, 16, 32] = 16,
) -> np.ndarray:
    """
    Compute arrival time using C++ Dijkstra implementation (fastest).

    Parameters
    ----------
    r_um : np.ndarray
        Radial coordinates (can be non-uniform/log-spaced)
    z_um : np.ndarray
        Depth coordinates (linear)
    etch_rate_map : np.ndarray
        Etch rate (speed) at each (z, r) point, shape (n_z, n_r)
    r_is_logscaled : bool
        Whether r grid is log-spaced (informational only)
    connectivity : {8, 16, 32}
        Number of neighbor connections per node.
        - 8: Standard (0deg, 45deg, 90deg, ...) - fastest, ~2% metrication error
        - 16: Knight's move added (~1% error)
        - 32: Extended knight's moves (~0.5% error)

    Returns
    -------
    arrival_time_map : np.ndarray
        Arrival time at each grid point, shape (n_z, n_r)

    Raises
    ------
    ImportError
        If the C++ extension is not compiled/available.
    """
    if not HAS_CPP:
        raise ImportError(
            "C++ Dijkstra extension not available. "
            "Compile with: cd tracketch/wavefront/dijkstra/cpp && python setup_dijkstra.py build_ext --inplace"
        )

    return np.ascontiguousarray(
        _dijkstra_cpp(
            np.ascontiguousarray(r_um, dtype=np.float64),
            np.ascontiguousarray(z_um, dtype=np.float64),
            np.ascontiguousarray(etch_rate_map, dtype=np.float64),
            int(connectivity),
        )
    )
