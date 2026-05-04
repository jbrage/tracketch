# %%
"""
Arrival time maps using different marching backends.

Compares FMM and Dijkstra (Numba) on a simple uniform-speed grid.
Both should produce similar results; Dijkstra supports non-uniform grids
natively while FMM requires interpolation to a uniform grid.
"""

import numpy as np
import matplotlib.pyplot as plt
from tracketch.wavefront import arrival_time_fmm, arrival_time_dijkstra_fast

# Simple grid with uniform etch rate
r_um = np.linspace(0.01, 5.0, 80)
z_um = np.linspace(0.0, 10.0, 150)
speed = np.ones((len(z_um), len(r_um)))

t_fmm = arrival_time_fmm(r_um, z_um, speed, r_is_logscaled=False)
t_dij = arrival_time_dijkstra_fast(r_um, z_um, speed, r_is_logscaled=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, t_map, title in [(ax1, t_fmm, "FMM"), (ax2, t_dij, "Dijkstra (Numba)")]:
    im = ax.pcolormesh(r_um, z_um, t_map, shading="auto")
    ax.set_xlabel("r / um")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Arrival time / hr")

ax1.set_ylabel("z / um")
ax1.invert_yaxis()
plt.tight_layout()
plt.show()
