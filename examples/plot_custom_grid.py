# %%
"""
Customising the simulation grid.

By default, :class:`~tracketch.TrackSimulator` uses a grid that covers
r in [1e-4, 20] um and z in [0, 40] um.  For ions with very long or very
short tracks, or when you need higher/lower resolution, you can override
individual grid parameters as plain keyword arguments:

* ``r_max_um``   -- maximum radial extent of the grid (um)
* ``z_max_um``   -- maximum depth (um); should exceed the CSDA range
* ``r_min_um``   -- inner radial boundary (um)
* ``n_points_r`` -- radial resolution (more points -> slower but finer RDD)
* ``n_points_z`` -- depth resolution

This example runs the same 10 MeV/u carbon ion three times:
  1. Default grid
  2. Wider, deeper grid (suitable for high-energy heavy ions)
  3. Compact low-resolution grid (fast scan / debugging)

and overlays the three iso-time contours to show the effect.
"""

import matplotlib.pyplot as plt
from tracketch import TrackSimulator

particle = "12C"
energy_MeV_u = 10.0
etch_time_h = 5.0

configs = [
    dict(label="default", color="C0"),
    dict(label="wide grid", color="C1", r_max_um=40, z_max_um=80),
    dict(label="coarse grid", color="C2", n_points_r=100, n_points_z=50),
]

fig, ax = plt.subplots(figsize=(5, 6))

for cfg in configs:
    label = cfg.pop("label")
    color = cfg.pop("color")

    sim = TrackSimulator(
        particle_name=particle,
        start_energy_MeV_u=energy_MeV_u,
        **cfg,
    )

    r, z = sim.get_iso_time_contour(etch_time_h)
    ax.plot(r, z, label=label, color=color)

    # Mark the bulk-etch depth for the default run only
    if label == "default":
        z_bulk = sim.etch_model.V_bulk_um_h * etch_time_h
        ax.axhline(
            z_bulk,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"bulk surface ({z_bulk:.1f} um)",
        )

ax.invert_yaxis()
ax.set_xlabel("r / um")
ax.set_ylabel("z / um")
ax.set_title(f"{particle} @ {energy_MeV_u} MeV/u, t = {etch_time_h} h")
ax.legend()
plt.tight_layout()
plt.show()

# %%
print(f"CSDA range: {sim.CSDA_range_um:.1f} um")
print(
    f"Grid r: [{sim._r_lims_um[0]:.1e}, {sim._r_lims_um[1]:.0f}] um "
    f"({len(sim._r_grid_um)} points)"
)
print(
    f"Grid z: [{sim._z_lims_um[0]:.0f}, {sim._z_lims_um[1]:.0f}] um "
    f"({len(sim._z_grid_um)} points)"
)
