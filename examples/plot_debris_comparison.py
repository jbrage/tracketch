# %%
"""
Track shapes with and without debris damping.

Simulates a 7Li ion at low kinetic energy; a regime where
the narrow, deep pit geometry makes debris-transport effects visible.
Contours are shown for three etching times; each time gets a pair of
curves: with debris (solid) and without (dashed).
"""

import matplotlib.pyplot as plt
from tracketch import TrackSimulator, load_etchrate_model

kinetic_energy_MeV_u = 1.0  # low energy to accentuate debris effect
particle_name = "12C"

# Load the calibrated model (includes debris parameters from calibration).
sim = TrackSimulator(
    particle_name=particle_name,
    start_energy_MeV_u=kinetic_energy_MeV_u,
    etch_model=load_etchrate_model("Doerschel_etching"),
    r_max_um=1,
)

etch_times_h = [0.5]

fig, ax = plt.subplots()
for idx, t_h in enumerate(etch_times_h):
    color = plt.cm.viridis(idx / len(etch_times_h))

    # --- with debris (solid) ---
    r, z = sim.get_iso_time_contour(t_h)
    ax.plot(r, z, color=color, linestyle="-", label=f"{t_h} h  (debris)")
    ax.plot(-r, z, color=color, linestyle="-")

    # --- without debris (dashed) ---
    r0, z0 = sim.get_iso_time_contour_nodebris(t_h)
    ax.plot(r0, z0, color=color, linestyle="--", label=f"{t_h} h  (no debris)")
    ax.plot(-r0, z0, color=color, linestyle="--")

ax.grid(True, which="both", ls=":", color="gray", alpha=0.5)
ax.set_xlabel("r / µm")
ax.set_ylabel("z / µm")
ax.set_title(f"{particle_name} @ {kinetic_energy_MeV_u} MeV/u\ndebris damping effect")
ax.invert_yaxis()
ax.legend(title="Etch time", bbox_to_anchor=(1.05, 1), loc="upper left")
# ax.set_aspect("equal")
plt.tight_layout()
plt.show()
