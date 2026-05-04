import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tracketch.simulation.simulator import TrackSimulator


def _get_contour_without_debris(
    sim: TrackSimulator,
    etching_time_h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporarily disable debris damping, recalculate, and extract contour."""
    etch_model = sim.etch_model
    saved_alpha = etch_model.debris_alpha
    saved_beta = etch_model.debris_beta

    etch_model.update_debris_params(alpha=None, beta=1.0)
    sim.recalculate_from_current_etch_model()
    r, z = sim.get_iso_time_contour(etching_time_h)

    # Restore
    etch_model.update_debris_params(alpha=saved_alpha, beta=saved_beta)
    sim.recalculate_from_current_etch_model()
    return r, z


def plot_track_shapes(
    models_dict: dict[str, dict],
    xscale: str = "linear",
) -> list:
    """Plot arrival-time maps with data overlays for track-shape datasets."""
    figures = []
    for _, data_dict in models_dict.items():
        sim = data_dict["simulator"]
        particle_df = data_dict["experiment_data"]

        # Ensure arrival-time map reflects current etch-model state
        sim.recalculate_from_current_etch_model()

        fig, ax = sim.plot_map("arrival", plot_contours=False)
        for etch_time_h, particle_data_df in particle_df.groupby("time_h"):
            ax.errorbar(
                x=particle_data_df["r_um"],
                xerr=particle_data_df["r_err_um"],
                y=particle_data_df["z_um"],
                yerr=particle_data_df["z_err_um"],
                c="red",
                fmt="o",
            )
            sim.plot_iso_time_contour(ax, etching_time_h=etch_time_h)

        ax.set_xscale(xscale)
        if xscale == "linear":
            ax.set_xlim(xmin=0)
        figures.append(fig)
    return figures


def plot_track_shapes_debris_comparison(
    models_dict: dict[str, dict],
    xscale: str = "linear",
) -> list:
    """Plot iso-time contours with and without debris damping.

    For each particle/energy a line plot is produced showing, per etch
    time, the experimental data (red markers), the model contour with
    debris damping (solid blue), and the contour without debris (dashed
    grey).  This makes the effect of the debris correction visible.

    Parameters
    ----------
    models_dict : dict
        The ``track_shape`` sub-dictionary from the calibration data.
    xscale : str
        Axis scale for the r-axis (default: ``"linear"``).

    Returns
    -------
    list[Figure]
        One figure per particle/energy entry.
    """
    figures: list[Figure] = []

    for key, data_dict in models_dict.items():
        sim = data_dict["simulator"]
        particle_df = data_dict["experiment_data"]
        has_debris = sim.etch_model.debris_damping_enabled

        fig, ax = plt.subplots()
        ax.set_title(f"{key}")
        ax.set_xlabel("r / um")
        ax.set_ylabel("z / um")
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

        for etch_time_h, pdf in particle_df.groupby("time_h"):
            etch_time_h = float(etch_time_h)

            # --- experimental data ---
            ax.errorbar(
                x=pdf["r_um"],
                xerr=pdf["r_err_um"],
                y=pdf["z_um"],
                yerr=pdf["z_err_um"],
                color="red",
                fmt="o",
                markersize=4,
                label=f"Data {etch_time_h:.1f} h",
            )

            # --- contour WITHOUT debris ---
            r_no, z_no = _get_contour_without_debris(sim, etch_time_h)
            if r_no.size > 0:
                ax.plot(
                    r_no,
                    z_no,
                    color="grey",
                    linestyle="--",
                    linewidth=1.5,
                    label=f"No debris {etch_time_h:.1f} h",
                )

            # --- contour WITH debris (current model) ---
            r_d, z_d = sim.get_iso_time_contour(etch_time_h)
            if r_d.size > 0:
                ax.plot(
                    r_d,
                    z_d,
                    color="tab:blue",
                    linestyle="-",
                    linewidth=2,
                    label=f"With debris {etch_time_h:.1f} h",
                )

        ax.set_xscale(xscale)
        if xscale == "linear":
            ax.set_xlim(xmin=0)
        # deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)

        if has_debris:
            alpha = sim.etch_model.debris_alpha
            beta = sim.etch_model.debris_beta
            ax.text(
                0.98,
                0.02,
                f"alpha={alpha:.3g}, beta={beta:.3g}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="tab:blue",
            )

        figures.append(fig)
    return figures


def plot_track_length(
    models_dict: dict[str, dict],
) -> dict[str, Figure]:
    """Plot track-length results grouped by ion, overlaying all energies in one figure."""
    grouped_models: dict[str, list[tuple[str, dict]]] = {}
    for key, data_dict in models_dict.items():
        ion = data_dict["simulator"].particle_name
        grouped_models.setdefault(ion, []).append((key, data_dict))

    figures_by_ion: dict[str, Figure] = {}

    for ion, ion_entries in grouped_models.items():
        fig, ax = plt.subplots()
        ax.grid(alpha=0.3)

        for i, (_, data_dict) in enumerate(ion_entries):
            sim = data_dict["simulator"]
            particle_df = data_dict["experiment_data"]

            energy_MeV = float(particle_df["Energy_MeV"].iloc[0])
            color = f"C{i % 10}"

            # min_time_h = float(particle_df["time_h"].min())
            max_time_h = float(particle_df["time_h"].max())
            eval_time_h = np.linspace(0, max_time_h * 1.1, 100)
            pred_length_um = sim.get_track_length_um(etch_time_h=eval_time_h)

            ax.plot(
                eval_time_h,
                pred_length_um,
                "-",
                color=color,
                label=f"Model: {energy_MeV:g} MeV",
            )
            ax.errorbar(
                x=particle_df["time_h"],
                y=particle_df["length_um"],
                yerr=particle_df["length_um_std"],
                color=color,
                fmt="o",
                label=f"Data: {energy_MeV:g} MeV",
            )

        # energy_MeV_u = convert_MeV_to_MeV_u(
        #     particle_name=ion,
        #     Energy_MeV=float(ion_entries[0][1]["experiment_data"]["Energy_MeV"].iloc[0]),
        # )
        ax.set_xlabel("Etching time / hr")
        ax.set_ylabel("Track length / um")
        ax.set_title(f"{ion} track length")
        ax.legend()

        figures_by_ion[ion] = fig

    return figures_by_ion
