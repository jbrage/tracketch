# %%
"""
LET and CSDA range for different ions in CR39.

Plots linear energy transfer (LET) and continuous slowing-down approximation
(CSDA) range as a function of kinetic energy for hydrogen isotopes, helium,
and carbon.

Hydrogen isotopes (``1H``, ``2H``, ``3H``) share the same LET curve because
LET depends only on charge and velocity (MeV/u), not on mass number.  Their
CSDA ranges differ by a factor equal to the mass number *A* -- a deuteron
travels twice as far as a proton at the same MeV/u.
"""

import numpy as np
import matplotlib.pyplot as plt
from tracketch import get_LET_keV_um, get_CSDA_um

energies = np.logspace(-1, 2.5, 50)
particles = ["1H", "2H", "3H", "4He", "12C"]

fig, (ax_let, ax_csda) = plt.subplots(1, 2, figsize=(10, 4))

for particle in particles:
    LET = [get_LET_keV_um(E, particle, "CR39", source="SRIM") for E in energies]
    CSDA = [get_CSDA_um(E, particle, "CR39", source="SRIM") for E in energies]

    ax_let.loglog(energies, LET, label=particle)
    ax_csda.loglog(energies, CSDA, label=particle)

ax_let.set_xlabel("Energy / (MeV/u)")
ax_let.set_ylabel("LET / (keV/um)")
ax_let.set_title("Linear energy transfer in CR39")
ax_let.legend()

ax_csda.set_xlabel("Energy / (MeV/u)")
ax_csda.set_ylabel("CSDA range / um")
ax_csda.set_title("CSDA range in CR39")
ax_csda.legend()

plt.tight_layout()
plt.show()
