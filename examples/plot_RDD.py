# %%
"""
Radial dose distributions (RDDs) for different ions.

Plots the Cucinotta RDD for protons, helium, and carbon ions at selected
energies. The RDD describes how dose falls off radially from the ion track.
"""

import numpy as np
import matplotlib.pyplot as plt
from tracketch import get_RDD_Gy

r_m = np.logspace(-9, -3, 200)

ions = [
    ("1H", 100.0),
    ("4He", 150.0),
    ("12C", 270.0),
]

fig, ax = plt.subplots()
for particle, E_MeV_u in ions:
    dose = get_RDD_Gy(r_m, E_MeV_u, particle, RDD_name="Cucinotta")
    ax.loglog(r_m * 1e6, dose, label=f"{particle} @ {E_MeV_u} MeV/u")

ax.set_xlabel("Radius / um")
ax.set_ylabel("Dose / Gy")
ax.legend()
ax.set_title("Radial dose distribution (Cucinotta)")
plt.tight_layout()
plt.show()
