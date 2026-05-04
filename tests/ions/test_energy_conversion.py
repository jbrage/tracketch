"""
Energy unit conversion tests.

Verifies that MeV <-> MeV/u round-trips are exact, and that the mass-number
relationship (E_MeV = E_MeV/u * A) holds for known particles.
"""

import pytest
from tracketch.physics import convert_MeV_to_MeV_u, convert_MeV_u_to_MeV


@pytest.mark.parametrize("particle", ["1H", "4He", "7Li", "12C"])
def test_MeV_to_MeV_u_round_trip(particle):
    """MeV -> MeV/u -> MeV should be identity."""
    E_MeV_u = 100.0
    E_MeV = convert_MeV_u_to_MeV(E_MeV_u, particle)
    E_MeV_u_back = convert_MeV_to_MeV_u(E_MeV, particle)
    assert E_MeV_u_back == pytest.approx(E_MeV_u, rel=1e-10)


@pytest.mark.parametrize(
    "particle,A",
    [
        ("1H", 1),
        ("4He", 4),
        ("12C", 12),
    ],
)
def test_MeV_u_scales_with_mass_number(particle, A):
    """E_MeV should be approximately E_MeV/u * A."""
    E_MeV_u = 100.0
    E_MeV = convert_MeV_u_to_MeV(E_MeV_u, particle)
    assert E_MeV == pytest.approx(E_MeV_u * A, rel=0.01)
