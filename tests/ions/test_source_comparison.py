"""Direct source comparison tests: SRIM vs libamtrack.

Compares LET and CSDA in both configured materials (water, CR39)
for a representative ion/energy grid.
"""

import pytest

from tracketch.physics import get_CSDA_um, get_LET_keV_um

PARTICLES = ["1H", "4He", "12C"]
ENERGIES_MEV_U = [5.0, 20.0, 50.0, 100.0, 200.0]
MATERIALS = ["water", "CR39"]

LET_REL_TOL = {"water": 0.10, "CR39": 0.10}
CSDA_REL_TOL = {"water": 0.10, "CR39": 0.13}


def _relative_difference(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("particle", PARTICLES)
@pytest.mark.parametrize("energy_MeV_u", ENERGIES_MEV_U)
def test_LET_srim_vs_libamtrack(
    material: str,
    particle: str,
    energy_MeV_u: float,
) -> None:
    """LET from SRIM and libamtrack should agree within tolerance."""
    let_srim = float(get_LET_keV_um(energy_MeV_u, particle, material, source="SRIM"))
    let_libam = float(
        get_LET_keV_um(energy_MeV_u, particle, material, source="libamtrack")
    )
    rel = _relative_difference(let_srim, let_libam)
    assert rel <= LET_REL_TOL[material], (
        f"LET mismatch for {particle} in {material} at {energy_MeV_u} MeV/u: "
        f"SRIM={let_srim:.6g}, libamtrack={let_libam:.6g}, rel={rel:.4%}"
    )


@pytest.mark.parametrize("material", MATERIALS)
@pytest.mark.parametrize("particle", PARTICLES)
@pytest.mark.parametrize("energy_MeV_u", ENERGIES_MEV_U)
def test_CSDA_srim_vs_libamtrack(
    material: str,
    particle: str,
    energy_MeV_u: float,
) -> None:
    """CSDA from SRIM and libamtrack should agree within tolerance."""
    csda_srim = float(get_CSDA_um(energy_MeV_u, particle, material, source="SRIM"))
    csda_libam = float(
        get_CSDA_um(energy_MeV_u, particle, material, source="libamtrack")
    )
    rel = _relative_difference(csda_srim, csda_libam)
    assert rel <= CSDA_REL_TOL[material], (
        f"CSDA mismatch for {particle} in {material} at {energy_MeV_u} MeV/u: "
        f"SRIM={csda_srim:.6g}, libamtrack={csda_libam:.6g}, rel={rel:.4%}"
    )
