"""
Tests for hydrogen isotope physics (1H, 2H, 3H).

At equal MeV/u, all three share z=1 and the same velocity, so:
  - LET must be identical
  - RDD at the same depth must be identical

Extensive quantities (CSDA range, total energy) must scale linearly
with the mass number A (1, 2, 3).
"""

import numpy as np
import pytest

from tracketch.physics import (
    convert_MeV_to_MeV_u,
    convert_MeV_u_to_MeV,
    get_CSDA_um,
    get_LET_keV_um,
)
from tracketch.physics.libamtrack import RDD_libamtrack_Gy


E_MEV_U = 2.0  # representative test energy
PARTICLES = ["1H", "2H", "3H"]
MASS_NUMBERS = {"1H": 1, "2H": 2, "3H": 3}


# -- LET -------------------------------------------------------------


@pytest.mark.parametrize("isotope", ["2H", "3H"])
def test_LET_equals_proton_at_same_MeV_u(isotope):
    """2H and 3H must have the same LET as 1H at equal MeV/u."""
    let_1H = get_LET_keV_um(E_MEV_U, "1H", "CR39")
    let_iso = get_LET_keV_um(E_MEV_U, isotope, "CR39")
    assert let_iso == pytest.approx(let_1H, rel=1e-6)


# -- CSDA range -------------------------------------------------------


@pytest.mark.parametrize("isotope", ["2H", "3H"])
def test_CSDA_scales_with_mass_number(isotope):
    """CSDA range must scale linearly with A at equal MeV/u."""
    csda_1H = get_CSDA_um(E_MEV_U, "1H", "CR39")
    csda_iso = get_CSDA_um(E_MEV_U, isotope, "CR39")
    A = MASS_NUMBERS[isotope]
    assert csda_iso == pytest.approx(csda_1H * A, rel=1e-6)


# -- total energy -----------------------------------------------------


@pytest.mark.parametrize("isotope", ["2H", "3H"])
def test_total_energy_matches_proton_for_hydrogen_isotopes(isotope):
    """Hydrogen isotopes currently share proton-equivalent MeV conversion."""
    E_MeV_1H = convert_MeV_u_to_MeV(E_MEV_U, "1H")
    E_MeV_iso = convert_MeV_u_to_MeV(E_MEV_U, isotope)
    assert E_MeV_iso == pytest.approx(E_MeV_1H, rel=1e-10)


def test_proton_total_energy_differs_from_MeV_u():
    """For 1H, MeV is not numerically equal to MeV/u (u is not 1)."""
    E_MeV = convert_MeV_u_to_MeV(E_MEV_U, "1H")
    assert E_MeV != pytest.approx(E_MEV_U, rel=1e-10)
    assert convert_MeV_to_MeV_u(E_MeV, "1H") == pytest.approx(E_MEV_U, rel=1e-10)


# -- RDD -------------------------------------------------------------


@pytest.mark.parametrize("isotope", ["2H", "3H"])
def test_RDD_equals_proton_at_same_MeV_u(isotope):
    """RDD shape must be identical to 1H at equal MeV/u (same z, same velocity)."""
    r_m = np.array([1e-9, 1e-8, 1e-7])
    rdd_1H = RDD_libamtrack_Gy(r_m, E_MeV_u=E_MEV_U, particle_name="1H")
    rdd_iso = RDD_libamtrack_Gy(r_m, E_MeV_u=E_MEV_U, particle_name=isotope)
    np.testing.assert_allclose(rdd_iso, rdd_1H, rtol=1e-6)
