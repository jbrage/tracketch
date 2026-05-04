"""
Dijkstra's algorithm for arrival time computation on non-uniform grids.

This module provides Dijkstra implementations optimized for non-uniform
(e.g., log-spaced) grids where FMM is not applicable.

Available backends:
- dijkstra_numba: Numba JIT-compiled (always available)
- dijkstra_cpp: C++ implementation (fastest, requires compilation)

The main function `arrival_time_dijkstra` auto-selects the fastest available backend.
"""

import logging

import numpy as np
from typing import Literal

from .dijkstra_numba import (
    arrival_time_dijkstra_fast,
)

logger = logging.getLogger(__name__)

# Check if C++ backend is available (must check the wrapper's flag,
# since the wrapper Python file always imports but the .so may not exist)
from .cpp.dijkstra_cpp_wrapper import HAS_CPP

if HAS_CPP:
    from .cpp.dijkstra_cpp_wrapper import arrival_time_dijkstra_cpp


def arrival_time_dijkstra(
    r_um: np.ndarray,
    z_um: np.ndarray,
    etch_rate_map: np.ndarray,
    r_is_logscaled: bool = True,
    connectivity: Literal[8, 16, 32] = 8,
    backend: Literal["auto", "cpp", "numba"] = "auto",
) -> np.ndarray:
    """
    Compute arrival times using Dijkstra's algorithm.

    Auto-selects the fastest available backend (C++ > Numba).

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
        - 8: Standard (fastest, ~2% metrication error)
        - 16: Knight's moves added (~1% error)
        - 32: Extended knight's moves (~0.5% error)
    backend : {"auto", "cpp", "numba"}
        Which backend to use. "auto" selects C++ if available, else Numba.

    Returns
    -------
    arrival_time_map : np.ndarray
        Arrival time at each grid point, shape (n_z, n_r)
    """
    # Select backend
    use_cpp = (backend == "cpp") or (backend == "auto" and HAS_CPP)
    logger.debug(
        "Dijkstra backend: requested='%s', selected='%s'",
        backend,
        "cpp" if use_cpp else "numba",
    )

    if use_cpp and not HAS_CPP:
        raise ImportError(
            "C++ backend requested but not available. "
            "Compile with: cd tracketch/wavefront/dijkstra/cpp && python setup_dijkstra.py build_ext --inplace"
        )

    if use_cpp:
        return arrival_time_dijkstra_cpp(
            r_um,
            z_um,
            etch_rate_map,
            r_is_logscaled=r_is_logscaled,
            connectivity=connectivity,
        )
    else:
        return arrival_time_dijkstra_fast(
            r_um,
            z_um,
            etch_rate_map,
            r_is_logscaled=r_is_logscaled,
            connectivity=connectivity,
        )


__all__ = [
    "arrival_time_dijkstra",
    "arrival_time_dijkstra_fast",
    "HAS_CPP",
]

if HAS_CPP:
    __all__.append("arrival_time_dijkstra_cpp")
