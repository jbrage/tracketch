"""Arrival time algorithm tests.

Verifies FMM and Dijkstra on a trivial uniform-speed grid: correct output
shape, finite values, zero arrival time at the surface, and increasing
arrival time with depth. If the C++ backend is compiled, also checks that
Numba and C++ Dijkstra agree within 5% relative tolerance.
"""

import numpy as np
import pytest
from tracketch.wavefront import arrival_time_fmm, arrival_time_dijkstra_fast
from tracketch.wavefront.dijkstra import HAS_CPP, arrival_time_dijkstra


def test_fmm_uniform_speed():
    r_um = np.linspace(0.01, 5.0, 50)
    z_um = np.linspace(0.0, 10.0, 100)
    speed = np.ones((len(z_um), len(r_um)))

    t = arrival_time_fmm(r_um, z_um, speed, r_is_logscaled=False)

    assert t.shape == (len(z_um), len(r_um))
    assert np.isfinite(t).all()
    # At z=0, arrival time should be ~0
    assert t[0, :].max() < 0.1
    # Deeper z should have larger arrival times
    assert t[-1, :].mean() > t[0, :].mean()


def test_dijkstra_uniform_speed():
    r_um = np.linspace(0.01, 5.0, 50)
    z_um = np.linspace(0.0, 10.0, 100)
    speed = np.ones((len(z_um), len(r_um)))

    t = arrival_time_dijkstra_fast(r_um, z_um, speed, r_is_logscaled=False)

    assert t.shape == (len(z_um), len(r_um))
    assert np.isfinite(t).all()
    assert t[0, :].max() < 0.1
    assert t[-1, :].mean() > t[0, :].mean()


@pytest.mark.skipif(not HAS_CPP, reason="C++ Dijkstra backend not compiled")
def test_dijkstra_cpp_matches_numba():
    """C++ and Numba Dijkstra should agree within 5% on a uniform grid."""
    r_um = np.linspace(0.01, 5.0, 50)
    z_um = np.linspace(0.0, 10.0, 100)
    speed = np.ones((len(z_um), len(r_um)))

    t_numba = arrival_time_dijkstra(
        r_um, z_um, speed, r_is_logscaled=False, backend="numba"
    )
    t_cpp = arrival_time_dijkstra(
        r_um, z_um, speed, r_is_logscaled=False, backend="cpp"
    )

    # Skip surface row (near-zero values cause large relative errors)
    assert t_cpp[1:] == pytest.approx(t_numba[1:], rel=0.05)
