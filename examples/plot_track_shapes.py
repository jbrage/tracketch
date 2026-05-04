# %%
"""
Track shapes for different ions and etching times.

Simulates the full pipeline: dose map -> etch rate -> arrival time -> track
contour. Shows how track shape evolves with etching time for a 270 MeV/u
carbon ion in CR39.

If the loaded etch model carries anchor velocity uncertainties, a 1-sigma
shaded band is added around each contour.
"""

import matplotlib.pyplot as plt
from tracketch import TrackSimulator


sim = TrackSimulator(
    particle_name="12C",
    start_energy_MeV_u=270.0,
    r_max_um=5,
)

# calculate the track contours for different etching times
etch_times_h = [0.5, 1.0, 2.0, 3.0]

fig, ax = plt.subplots()
for idx, t_h in enumerate(etch_times_h):
    color = plt.cm.viridis(idx / len(etch_times_h))

    unc_result = sim.get_iso_time_contour_with_uncertainty(t_h)
    if unc_result is not None:
        (r, z), z_band, r_lo, r_hi = unc_result
        ax.fill_betweenx(z_band, r_lo, r_hi, alpha=0.25, color=color)
        ax.fill_betweenx(z_band, -r_hi, -r_lo, alpha=0.25, color=color)
    else:
        r, z = sim.get_iso_time_contour(etching_time_h=t_h)

    ax.plot(r, z, label=f"{t_h}", color=color)
    ax.plot(-r, z, color=color)  # mirror for full track profile

ax.set_xlabel("r / um")
ax.set_ylabel("z / um")
ax.set_title("12C @ 270 MeV/u: track shape evolution")
ax.invert_yaxis()
ax.legend(title="Etch time / h")
ax.set_aspect("equal")
