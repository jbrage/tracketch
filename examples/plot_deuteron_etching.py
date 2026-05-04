# %%
"""
Plotting track radii as a function of removed (etched) depth.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tracketch import TrackSimulator
import tqdm


particle = "2H"
etch_times_h = np.linspace(0.1, 20, 100)
energies_MeV_u = [0.1, 0.5, 1.0, 2.0, 5.0]

result_df = pd.DataFrame()
# iterate through combinations and calculate track radius for each
for energy_MeV_u in tqdm.tqdm(energies_MeV_u, total=len(energies_MeV_u)):
    # create simulator
    sim = TrackSimulator(
        particle_name=particle,
        start_energy_MeV_u=energy_MeV_u,
    )

    for etching_time_h in etch_times_h:
        # get results for this etching time
        new_row = pd.DataFrame(
            {
                "Particle": [particle],
                "Energy (MeV/u)": [energy_MeV_u],
                "Etch time / h": [etching_time_h],
                "Removed layer / um": [etching_time_h * sim.etch_model.V_bulk_um_h],
                "Energy / MeV": [sim.start_energy_MeV],
                "CSDA / um": [sim.CSDA_range_um],
                "Track radius / um": [
                    sim.get_track_radius_um(etch_time_h=etching_time_h)
                ],
                "Track length / um": [
                    sim.get_track_length_um(etch_time_h=etching_time_h)
                ],
            }
        )
        result_df = pd.concat([result_df, new_row], ignore_index=True)

# %%
fig, ax = plt.subplots()
sns.lineplot(
    data=result_df,
    x="Removed layer / um",
    y="Track radius / um",
    hue="Energy (MeV/u)",
    ax=ax,
)
ax.grid(alpha=0.5)
ax.set_title(f"Track radius vs removed layer for {particle} ions")
ax.set_xlabel("Removed layer / um")
ax.set_ylabel("Track radius / um")
ax.legend(title="Energy (MeV/u)")
plt.show()
