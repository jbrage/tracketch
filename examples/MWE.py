# %%
from tracketch import TrackSimulator

# create a simulator for 100 MeV/u carbon ions
sim = TrackSimulator(
    particle_name="12C",
    start_energy_MeV_u=100.0,
)

# calculate etch front arriaval times
fig, ax = sim.plot_map(name="arrival")

# plot and get the track contour for 1 hour of etching
r_um, z_um = sim.plot_iso_time_contour(ax=ax, etching_time_h=1.0)
