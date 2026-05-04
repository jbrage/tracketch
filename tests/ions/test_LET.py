"""
LET and CSDA range tests for the ions module.

Sanity checks:
- LET values for known particles/energies fall within expected physical ranges.

Cross-validation (SRIM vs libamtrack):
- SRIM computes LET and CSDA in CR39 (density 1.31 g/cm3).
- libamtrack computes LET and CSDA in water (density 1.0 g/cm3).
- Since stopping power scales approximately with density for similar materials,
  SRIM CR39 values are density-scaled to water and compared against libamtrack.
- Agreement is expected within ~10% (limited by material composition differences).
"""

import pytest
from tracketch.physics import get_LET_keV_um, get_CSDA_um

RHO_CR39 = 1.31
RHO_WATER = 1.0


@pytest.mark.parametrize(
    "particle,energy_MeV_u",
    [
        ("1H", 2.0),
        ("1H", 200.0),
        ("4He", 2.0),
        ("4He", 200.0),
        ("12C", 2.0),
        ("12C", 200.0),
    ],
)
def test_LET_SRIM_CR39_scaled_to_water_vs_libamtrack(particle, energy_MeV_u):
    """SRIM LET in CR39, density-scaled to water, should match libamtrack water within 10%."""
    LET_srim_cr39 = get_LET_keV_um(energy_MeV_u, particle, "CR39", source="SRIM")
    LET_libam_water = get_LET_keV_um(
        energy_MeV_u, particle, "water", source="libamtrack"
    )

    LET_scaled_to_water = LET_srim_cr39 * (RHO_WATER / RHO_CR39)
    assert LET_scaled_to_water == pytest.approx(LET_libam_water, rel=0.10)


@pytest.mark.parametrize(
    "particle,energy_MeV_u",
    [
        ("1H", 5.0),
        ("1H", 200.0),
        ("4He", 5.0),
        ("4He", 200.0),
        ("12C", 10.0),
        ("12C", 200.0),
    ],
)
def test_CSDA_SRIM_CR39_scaled_to_water_vs_libamtrack(particle, energy_MeV_u):
    """SRIM CSDA in CR39, density-scaled to water, should match libamtrack water within 10%."""
    CSDA_srim_cr39 = get_CSDA_um(energy_MeV_u, particle, "CR39", source="SRIM")
    CSDA_libam_water = get_CSDA_um(energy_MeV_u, particle, "water", source="libamtrack")

    CSDA_scaled_to_water = CSDA_srim_cr39 * (RHO_CR39 / RHO_WATER)
    assert CSDA_scaled_to_water == pytest.approx(CSDA_libam_water, rel=0.10)
