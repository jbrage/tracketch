"""Plotting utilities for :class:`~tracketch.simulation.simulator.TrackSimulator`."""

from typing import Any, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if TYPE_CHECKING:
    pass


def plot_map(
    etcher: "tracketch.simulation.simulator.TrackSimulator",
    name: str = "dose",
    grayscale_mode: bool = False,
    plot_contours: bool = True,
    annotate_figure: bool = True,
) -> tuple:
    """Plot the dose, etch-rate, or arrival-time map.

    Parameters
    ----------
    etcher : tracketch.simulation.simulator.TrackSimulator
        Simulator instance.
    name : str
        ``'dose'``, ``'etch'``, or ``'arrival'``.
    grayscale_mode : bool
        Use greyscale colour map.
    plot_contours : bool
        Overlay contour lines.
    annotate_figure : bool
        Add CSDA-range annotation.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    fontsize = 12

    if grayscale_mode:
        kwargs: dict[str, Any] = {"cmap": "Greys"}
    else:
        kwargs: dict[str, Any] = {"cmap": "viridis"}

    if name == "dose":
        data_map = etcher.dose_map
        label = "Dose"
        unit = "Gy"
        kwargs["norm"] = LogNorm()

    elif name == "etch":
        data_map = etcher.etch_rate_map
        label = "Etch-rate"
        unit = "um/hr"
        kwargs["norm"] = LogNorm()

    elif name == "arrival":
        data_map = etcher.arrival_time_map
        label = "Arrival time"
        unit = "hr"
        kwargs["norm"] = LogNorm()

    else:
        raise ValueError("name must be 'dose', 'etch', or 'arrival'")

    fig, ax = plt.subplots()

    if etcher._logscale_r:
        ax.set_xscale("log")
    else:
        ax.set_xscale("linear")

    # Build bin edges for correct rendering with potentially non-uniform grids
    r_vals = etcher._r_grid_um
    z_vals = etcher._z_grid_um

    R, Z = np.meshgrid(r_vals, z_vals)

    quad = ax.pcolormesh(R, Z, data_map, shading="auto", **kwargs)
    ax.set_xlabel("r / um", fontsize=fontsize)
    ax.set_ylabel("z / um", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.set_xlim(xmin=min(etcher._r_grid_um), xmax=max(etcher._r_grid_um))
    cbar = fig.colorbar(quad, ax=ax, label=f"{label} / {unit}")
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(f"{label} / {unit}", fontsize=fontsize)

    # Add CSDA range line if within the plot range
    if annotate_figure:
        if name == "dose":
            color = "black"
        else:
            color = "white"

        if etcher.CSDA_range_um < max(etcher._z_grid_um):
            ax.axhline(etcher.CSDA_range_um, color=color, linestyle="--")
            ax.text(
                max(etcher._r_grid_um) * 0.99,
                etcher.CSDA_range_um,
                f"{etcher.stopping_power_source_name} CSDA range: {etcher.CSDA_range_um:0.1f} um",
                color=color,
                fontsize=fontsize - 2,
                va="top",
                ha="right",
            )

    # Add contour lines with log-scaled levels
    if name == "dose":
        n_levels = 7
        levels = np.logspace(
            np.log10(np.nanmin(data_map)), np.log10(np.nanmax(data_map)), n_levels
        )
    else:
        levels = 3

    if plot_contours:
        ax.contour(
            etcher._r_grid_um,
            etcher._z_grid_um,
            data_map,
            colors="r",
            linestyles="dotted",
            levels=levels,
        )

    ax.set_title(
        f"{label} map for {etcher.particle_name} at {etcher.start_energy_MeV_u:0.1f} MeV/u",
        fontsize=fontsize + 1,
    )
    ax.invert_yaxis()
    return fig, ax


def plot_LET_energy_profiles(
    etcher: "tracketch.simulation.simulator.TrackSimulator",
) -> tuple:
    """Plot kinetic-energy and LET depth profiles.

    Parameters
    ----------
    etcher : tracketch.simulation.simulator.TrackSimulator
        Simulator instance.

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes
    """
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6, 8))
    ax_energy = axes[0]
    ax_LET = axes[1]

    for ax in axes:
        if etcher.CSDA_range_um < max(etcher._z_grid_um):
            ax.axvline(
                etcher.CSDA_range_um,
                color="r",
                linestyle="--",
            )
        ax.grid(alpha=0.3)

    if etcher.CSDA_range_um < max(etcher._z_grid_um):
        ax_energy.text(
            etcher.CSDA_range_um + 0.2,
            (max(etcher.energy_profile_MeV_u) + min(etcher.energy_profile_MeV_u)) / 2,
            f"CSDA range: {etcher.CSDA_range_um:0.3g} um",
            color="r",
            rotation=90,
            va="center",
            ha="left",
        )

    # plot the kinetic energy as a function of depth
    ax_energy.plot(
        etcher._z_grid_um,
        etcher.energy_profile_MeV_u,
        label="Kinetic (with straggling)",
    )
    ax_energy.plot(
        etcher._z_grid_um,
        etcher.energy_profile_MeV_u_pristine,
        label="Without long. straggling",
    )
    ax_energy.set_ylabel("Energy / (MeV/u)")
    ax_energy.legend()

    # LET as a function of depth
    ax_LET.plot(
        etcher._z_grid_um,
        etcher.LET_profile_keV_um,
        label=f" {etcher.material_name} LET (with straggling)",
    )
    ax_LET.plot(
        etcher._z_grid_um,
        etcher.LET_profile_keV_um_pristine,
        label=f" {etcher.material_name} LET",
    )
    ax_LET.set_ylabel("LET / (keV/um)")
    ax_LET.legend()

    return fig, axes
