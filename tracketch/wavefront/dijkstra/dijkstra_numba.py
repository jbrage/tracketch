# %%
"""
Fast Dijkstra implementation for non-uniform grids using Numba JIT.

This module provides a Numba-accelerated Dijkstra algorithm for computing
arrival times on 2D non-uniform (e.g., log-spaced) grids.

Supports 8, 16, or 32-connected neighborhoods to reduce metrication error.
"""

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra
from typing import Literal


def arrival_time_dijkstra_fast(
    r_um: np.ndarray,
    z_um: np.ndarray,
    etch_rate_map: np.ndarray,
    r_is_logscaled: bool = True,
    connectivity: Literal[8, 16, 32] = 8,
) -> np.ndarray:
    """
    Compute arrival time using Dijkstra's algorithm with Numba-accelerated graph construction.

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
    """
    n_z, n_r = etch_rate_map.shape
    n_nodes = n_z * n_r

    # Build graph edges with Numba JIT
    row, col, data = _build_graph_numba(
        np.ascontiguousarray(r_um, dtype=np.float64),
        np.ascontiguousarray(z_um, dtype=np.float64),
        np.ascontiguousarray(etch_rate_map, dtype=np.float64),
        connectivity,
    )

    # Create sparse matrix in COO format, then convert to CSR
    graph_csr = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes)).tocsr()

    # Multi-source Dijkstra: start from all nodes at z=0
    i_z_start = np.argmin(np.abs(z_um - 0))
    start_indices = np.arange(i_z_start * n_r, (i_z_start + 1) * n_r)

    # Single call with multiple sources - much faster than looping
    dist_matrix = scipy_dijkstra(
        graph_csr, directed=False, indices=start_indices, return_predecessors=False
    )

    # dist_matrix shape: (n_starts, n_nodes) - take minimum across all starts
    min_dist = np.min(dist_matrix, axis=0)

    # Reshape to 2D
    arrival_time_map = min_dist.reshape(n_z, n_r)
    arrival_time_map[i_z_start, :] = 0.0

    return arrival_time_map


def _get_neighbor_offsets(connectivity: int) -> tuple:
    """Get neighbor offsets for given connectivity."""
    # 8-connected: immediate neighbors (king's moves)
    dz_8 = [-1, 1, 0, 0, -1, -1, 1, 1]
    dr_8 = [0, 0, -1, 1, -1, 1, -1, 1]

    if connectivity == 8:
        return np.array(dz_8, dtype=np.int64), np.array(dr_8, dtype=np.int64)

    # 16-connected: add knight's moves (2,1) pattern
    dz_16 = dz_8 + [-2, -2, 2, 2, -1, 1, -1, 1]
    dr_16 = dr_8 + [-1, 1, -1, 1, -2, -2, 2, 2]

    if connectivity == 16:
        return np.array(dz_16, dtype=np.int64), np.array(dr_16, dtype=np.int64)

    # 32-connected: add extended knight's moves (3,1), (3,2) patterns
    dz_32 = dz_16 + [
        -3,
        -3,
        3,
        3,
        -1,
        1,
        -1,
        1,  # (3,1)
        -3,
        -3,
        3,
        3,
        -2,
        2,
        -2,
        2,  # (3,2)
    ]
    dr_32 = dr_16 + [
        -1,
        1,
        -1,
        1,
        -3,
        -3,
        3,
        3,  # (3,1)
        -2,
        2,
        -2,
        2,
        -3,
        -3,
        3,
        3,  # (3,2)
    ]

    return np.array(dz_32, dtype=np.int64), np.array(dr_32, dtype=np.int64)


@njit(cache=True)
def _build_graph_numba(
    r_um: np.ndarray,
    z_um: np.ndarray,
    etch_rate_map: np.ndarray,
    connectivity: int = 8,
) -> tuple:
    """Build graph edges using Numba JIT."""
    n_z, n_r = etch_rate_map.shape

    # Build neighbor offsets based on connectivity
    if connectivity == 8:
        dz_offsets = np.array([-1, 1, 0, 0, -1, -1, 1, 1], dtype=np.int64)
        dr_offsets = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int64)
    elif connectivity == 16:
        dz_offsets = np.array(
            [-1, 1, 0, 0, -1, -1, 1, 1, -2, -2, 2, 2, -1, 1, -1, 1], dtype=np.int64
        )
        dr_offsets = np.array(
            [0, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -2, -2, 2, 2], dtype=np.int64
        )
    else:  # 32
        dz_offsets = np.array(
            [
                -1,
                1,
                0,
                0,
                -1,
                -1,
                1,
                1,
                -2,
                -2,
                2,
                2,
                -1,
                1,
                -1,
                1,
                -3,
                -3,
                3,
                3,
                -1,
                1,
                -1,
                1,
                -3,
                -3,
                3,
                3,
                -2,
                2,
                -2,
                2,
            ],
            dtype=np.int64,
        )
        dr_offsets = np.array(
            [
                0,
                0,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -2,
                -2,
                2,
                2,
                -1,
                1,
                -1,
                1,
                -3,
                -3,
                3,
                3,
                -2,
                2,
                -2,
                2,
                -3,
                -3,
                3,
                3,
            ],
            dtype=np.int64,
        )

    n_neighbors = len(dz_offsets)

    # Pre-allocate maximum possible edges
    max_edges = n_z * n_r * n_neighbors
    rows = np.empty(max_edges, dtype=np.int64)
    cols = np.empty(max_edges, dtype=np.int64)
    weights = np.empty(max_edges, dtype=np.float64)

    edge_count = 0

    for i_z in range(n_z):
        for i_r in range(n_r):
            z_curr = z_um[i_z]
            r_curr = r_um[i_r]
            speed_curr = etch_rate_map[i_z, i_r]

            if speed_curr <= 0:
                continue

            node_idx = i_z * n_r + i_r

            for k in range(n_neighbors):
                dz = dz_offsets[k]
                dr = dr_offsets[k]
                i_z_nb = i_z + dz
                i_r_nb = i_r + dr

                if 0 <= i_z_nb < n_z and 0 <= i_r_nb < n_r:
                    speed_nb = etch_rate_map[i_z_nb, i_r_nb]
                    if speed_nb <= 0:
                        continue

                    z_nb = z_um[i_z_nb]
                    r_nb = r_um[i_r_nb]
                    dist = np.sqrt((z_nb - z_curr) ** 2 + (r_nb - r_curr) ** 2)

                    speed_avg = 2 * speed_curr * speed_nb / (speed_curr + speed_nb)
                    travel_time = dist / speed_avg

                    neighbor_idx = i_z_nb * n_r + i_r_nb

                    rows[edge_count] = node_idx
                    cols[edge_count] = neighbor_idx
                    weights[edge_count] = travel_time
                    edge_count += 1

    return rows[:edge_count], cols[:edge_count], weights[:edge_count]


# Alias for drop-in replacement
arrival_time_dijkstra = arrival_time_dijkstra_fast
