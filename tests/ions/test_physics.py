"""
Physics consistency tests for LET and CSDA range calculations.

Tests fundamental physical relationships that must hold:
- Bethe-Bloch Z^2 scaling: at the same MeV/u, heavier ions (higher Z) have higher LET.
- Bragg peak: LET increases as a particle slows down (lower energy -> higher LET).
- CSDA monotonicity: range increases monotonically with energy.
- CSDA Z-ordering: at the same MeV/u, heavier ions have shorter range.
"""

from tracketch.physics import get_LET_keV_um, get_CSDA_um

MATERIAL = "CR39"
SOURCE = "SRIM"


def test_LET_increases_with_Z_at_same_velocity():
    """At 100 MeV/u: LET(12C) > LET(4He) > LET(1H), following Z^2 scaling."""
    E = 100.0
    LET_H = get_LET_keV_um(E, "1H", MATERIAL, source=SOURCE)
    LET_He = get_LET_keV_um(E, "4He", MATERIAL, source=SOURCE)
    LET_C = get_LET_keV_um(E, "12C", MATERIAL, source=SOURCE)

    assert LET_C > LET_He > LET_H


def test_LET_increases_as_particle_slows_down():
    """Bragg peak: LET at 10 MeV/u > LET at 100 MeV/u for the same ion."""
    for particle in ["1H", "4He", "12C"]:
        LET_low = get_LET_keV_um(10.0, particle, MATERIAL, source=SOURCE)
        LET_high = get_LET_keV_um(100.0, particle, MATERIAL, source=SOURCE)
        assert LET_low > LET_high, (
            f"{particle}: LET should increase as energy decreases"
        )


def test_CSDA_increases_with_energy():
    """More energy -> longer range. Must be monotonic."""
    energies = [5.0, 20.0, 50.0, 100.0, 200.0]
    for particle in ["1H", "4He", "12C"]:
        ranges = [get_CSDA_um(E, particle, MATERIAL, source=SOURCE) for E in energies]
        assert all(r2 > r1 for r1, r2 in zip(ranges, ranges[1:])), (
            f"{particle}: CSDA range must increase with energy"
        )


def test_CSDA_decreases_with_Z_at_same_velocity():
    """At 100 MeV/u: range(1H) > range(4He) > range(12C)."""
    E = 100.0
    R_H = get_CSDA_um(E, "1H", MATERIAL, source=SOURCE)
    R_He = get_CSDA_um(E, "4He", MATERIAL, source=SOURCE)
    R_C = get_CSDA_um(E, "12C", MATERIAL, source=SOURCE)

    assert R_H > R_He > R_C
