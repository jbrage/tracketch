# %%
"""
Simulated track shapes vs. Dorschel experimental data.

For each of four ions (H, He, Li, C) at known beam energies,
we simulate the track contour at the experimental etch times and overlay
the measured (r, z) points from Dorschel et al. (2003).

This reproduces the comparison in Fig. 7 of the publication.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from pathlib import Path

from tracketch import TrackSimulator, load_etchrate_model, convert_MeV_to_MeV_u

# --- Load experimental reference data ----------------------------------------
# Dorschel et al. (2003) measured 2D track-shape profiles for four ions.
# Columns: r_um, z_um, r_err_um, z_err_um, time_h, particle_name, Energy_MeV
_DATA_CSV = (
    Path(__file__).resolve().parent.parent
    / "calibration"
    / "data"
    / "doerschel-data.csv"
)
ref_df = pd.read_csv(_DATA_CSV).dropna(subset=["r_um", "z_um"])

# Map each particle to its beam energy
energy_MeV_dict = ref_df.groupby("particle_name")["Energy_MeV"].first().to_dict()

# Global etch-rate model calibrated to all four ions
etch_model = load_etchrate_model("Doerschel_etching")

# Particles in the Dorschel dataset, with per-panel axis limits
particles = {
    #  name : (r_max, z_max)  -- axis limits in um
    "1H": (11, 34),
    "4He": (11, 34),
    "7Li": (2.5, 20),
    "12C": (2.5, 20),
}

# --- Build one TrackSimulator per ion ----------------------------------------
simulators = {}
for name, (rlim, zlim) in particles.items():
    energy_MeV = energy_MeV_dict[name]
    energy_MeV_u = convert_MeV_to_MeV_u(particle_name=name, Energy_MeV=energy_MeV)

    # Grid must span at least the experimental data range
    particle_df = ref_df[ref_df.particle_name == name]
    r_max = max(rlim, 3.0 * float(np.nanmax(particle_df["r_um"])) + 2.0)

    sim = TrackSimulator(
        particle_name=name,
        start_energy_MeV_u=energy_MeV_u,
        etch_model=etch_model,
        rz_lims_dict={
            "r_max_um": r_max,
            "z_max_um": zlim,
            "n_points_r": 300,
            "n_points_z": 150,
        },
    )
    simulators[name] = sim

# --- Plot: 2x2 grid, one panel per ion --------------------------------------
pretty = {"1H": "$^1$H", "4He": "$^4$He", "7Li": "$^7$Li", "12C": "$^{12}$C"}

fig, axes = plt.subplots(2, 2, figsize=(8, 7))

for idx, (name, (rlim, zlim)) in enumerate(particles.items()):
    ax = axes.flat[idx]
    sim = simulators[name]
    particle_df = ref_df[ref_df.particle_name == name]
    energy_MeV = energy_MeV_dict[name]

    # Plot arrival-time map as background
    r_grid = sim._r_grid_um
    z_grid = sim._z_grid_um
    R, Z = np.meshgrid(r_grid, z_grid)
    ax.pcolormesh(R, Z, sim.arrival_time_map, norm=LogNorm(), rasterized=True)

    # One contour + data overlay per etch time
    for i, (t_h, grp) in enumerate(particle_df.groupby("time_h")):
        # Experimental data points with error bars
        ax.errorbar(
            grp["r_um"],
            grp["z_um"],
            xerr=grp["r_err_um"],
            yerr=grp["z_err_um"],
            fmt="o",
            color="tab:red",
            ms=4,
            elinewidth=1,
            label="Dorschel et al." if i == 0 else None,
            zorder=10,
        )

        # Simulated track contour (with uncertainty band if available)
        (r_c, z_c), z_band, r_lo, r_hi = sim.get_iso_time_contour_with_uncertainty(t_h)

        ax.fill_betweenx(z_band, r_lo, r_hi, alpha=0.3, color="white", zorder=4)
        ax.plot(r_c, z_c, "w", lw=2, label="Simulation" if i == 0 else None, zorder=5)

        # Etch-time annotation
        ax.text(
            0.95,
            0.05 + 0.08 * i,
            f"{t_h} h",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="white",
            fontweight="bold",
        )

    ax.set_xlim(0, rlim)
    ax.set_ylim(0, zlim)
    ax.invert_yaxis()
    ax.set_xlabel("r / um")
    ax.set_ylabel("z / um")
    ax.set_title(f"{pretty[name]}, {energy_MeV} MeV")

axes[0, 1].legend(loc="upper right", fontsize="small")
fig.tight_layout()
plt.show()

# %%
