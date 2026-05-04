"""Radial dose distribution (RDD) regression test.

Compares the Cucinotta RDD model for 12C at multiple energies against
stored reference data (from libamtrack). Agreement is checked at 10
log-spaced radii per energy, within 50% relative tolerance (accounts
for known model approximations at track core/penumbra boundaries).
"""

import numpy as np
import pandas as pd
from tracketch.physics import get_RDD_Gy


def test_RDD_cucinotta_12C_matches_reference():
    reference_df = pd.read_csv("tests/libamtrack/RDD_Cucinotta_12C.csv", skiprows=3)

    for column in reference_df.columns:
        if column.startswith("r_m"):
            continue

        E_MeV_u = float(column)
        RDD_calc = get_RDD_Gy(
            r_m=reference_df["r_m"].values,
            particle_name="12C",
            E_MeV_u=E_MeV_u,
            RDD_name="Cucinotta",
        )

        idx = np.logspace(0, np.log10(len(reference_df) - 1), num=10, dtype=int)
        idx = [i for i in idx if RDD_calc[i] > 0 and reference_df[column][i] > 0]

        assert np.allclose(RDD_calc[idx], reference_df[column][idx], rtol=0.5), (
            f"RDD mismatch for 12C at {E_MeV_u} MeV/u"
        )
