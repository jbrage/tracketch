# %%
"""
Plotting track radii as a function of ion energy.

This example computes the track radius in CR-39 for a grid of ion energies
and two etching times, using :class:`~tracketch.TrackSimulator`.  Results
are collected in a :class:`pandas.DataFrame` and visualised with
``seaborn.lineplot``.
"""

import numpy as np
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tracketch import TrackSimulator
import tqdm


etch_times_h = [5, 10]
particles = ["1H", "2H", "3H"]
energies_MeV_u = np.logspace(-2, 2, num=300)

# Create all unique combinations
combinations = list(itertools.product(particles, energies_MeV_u))
combinations_df = pd.DataFrame(combinations, columns=["particle_name", "energy_MeV_u"])

result_df = pd.DataFrame(
    columns=[
        "particle_name",
        "energy_MeV_u",
        "energy_MeV",
        "CSDA_um",
        "etch_time_h",
        "radius_um",
    ]
)

# iterate through combinations and calculate track radius for each
for idx, row in tqdm.tqdm(combinations_df.iterrows(), total=len(combinations_df)):
    particle = row["particle_name"]
    energy_MeV_u = row["energy_MeV_u"]

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

fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, figsize=(10, 10))
fig.suptitle("Track radius in CR-39 for hydrogen isotopes", y=0.92)
fig.subplots_adjust(wspace=0.05)

(ax1, ax2, ax3, ax4) = axes.flatten()

sns.lineplot(
    data=result_df,
    x="CSDA / um",
    y="Track radius / um",
    hue="Particle",
    style="Etch time / h",
    ax=ax1,
)

sns.lineplot(
    data=result_df,
    x="Energy (MeV/u)",
    y="Track radius / um",
    hue="Particle",
    style="Etch time / h",
    ax=ax2,
)

sns.lineplot(
    data=result_df,
    x="CSDA / um",
    y="Track length / um",
    hue="Particle",
    style="Etch time / h",
    ax=ax3,
)
sns.lineplot(
    data=result_df,
    x="Energy (MeV/u)",
    y="Track length / um",
    hue="Particle",
    style="Etch time / h",
    ax=ax4,
)

for ax in axes.flatten():
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

plt.show()

# %%
