"""
Fast Marching Method (FMM) for arrival time computation on uniform grids.

This module provides FMM implementation using scikit-fmm for computing
arrival times on uniform or log-spaced grids (via interpolation).
"""

from .fmm import arrival_time_fmm

__all__ = ["arrival_time_fmm"]
