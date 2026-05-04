"""
Arrival time computation methods for track etching simulation.

This package provides multiple algorithms for computing arrival times on 2D
non-uniform (e.g., log-spaced radial, linear depth) grids:

- FMM (Fast Marching Method): For uniform grids
- Dijkstra: For non-uniform grids, with Numba or C++ backends

Main algorithms
===============

FMM (wavefront.fmm)
-------------------
Fast Marching Method for smooth speed maps. Requires uniform grid
(or interpolation from log-spaced grid).

Dijkstra (wavefront.dijkstra)
------------------------------
Dijkstra's algorithm optimized for non-uniform grids using 8-connected
neighborhood. Available backends:

- dijkstra_numba (default): Numba JIT-compiled, ~5-20x speedup
- dijkstra_cpp (optional): C++ with pybind11, ~50-100x speedup

Examples
========

Using Dijkstra with log-spaced grid (typical)::

    from tracketch.wavefront.dijkstra import arrival_time_dijkstra_fast

    arrival_map = arrival_time_dijkstra_fast(
        r_log_um, z_um, etch_rate_map
    )

Using FMM with interpolation to uniform grid::

    from tracketch.wavefront.fmm import arrival_time_fmm

    arrival_map = arrival_time_fmm(
        r_log_um, z_um, etch_rate_map, r_is_logscaled=True
    )

Using C++ Dijkstra (if compiled)::

    from tracketch.wavefront.dijkstra import arrival_time_dijkstra_cpp

    arrival_map = arrival_time_dijkstra_cpp(
        r_log_um, z_um, etch_rate_map
    )
"""

from .dijkstra import (
    arrival_time_dijkstra_fast,
    arrival_time_dijkstra,
)
from .fmm import arrival_time_fmm

__all__ = [
    "arrival_time_dijkstra_fast",
    "arrival_time_dijkstra",
    "arrival_time_fmm",
]
