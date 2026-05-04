# %%
"""
Hhelper to compare SRIM Bragg variants against calibration data.

Runs the same calibration workflow for each requested Bragg correction and
stores detailed and summary CSV outputs.
"""

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import threading

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from calibration.calibration_data import create_minimisation_data
from calibration.lib_optimiser import optimize_etch_model
from calibration.plotting import plot_track_shapes
from tracketch.etching.etch_rate_model import EtchRateModel
from tracketch.etching.etch_rate_model_io import default_etch_rate_model
from tracketch.physics.SRIM import set_bragg_correction

# ---------------------------------------------------------------------------
# Automatic run settings (edit here if needed)
# ---------------------------------------------------------------------------
RUN_MODE = "fast"
RUN_MODE = "full"  # "fast" or "full"
N_JOBS = 3

if RUN_MODE == "fast":
    BRAGG_VARIANTS = [0, 3, 5]
    PARTICLES = ["1H", "4He", "7Li", "12C"]
    SEEDS = [11]
    POPSIZE = 30
    MAXITER = 50
    TOL = 1e-4
    OUTPUT_DIR_NAME = "bragg_benchmark_fast"
elif RUN_MODE == "full":
    BRAGG_VARIANTS = [0, 3, 5]
    PARTICLES = ["1H", "4He", "7Li", "12C"]
    SEEDS = [11, 22]
    POPSIZE = 50
    MAXITER = 50
    TOL = 1e-6
    OUTPUT_DIR_NAME = "bragg_benchmark"
else:
    raise ValueError(f"RUN_MODE must be 'fast' or 'full', got {RUN_MODE!r}")

RUN_PARALLEL = True
OBJECTIVE = "shape"  # "both", "shape", "length"
AGGREGATION = "equal_point"  # "equal_ion_energy", "equal_point"
RESIDUAL_SCALE = "log"  # "linear", "log"
LOSS = "soft_l1"  # "l2", "soft_l1", "huber"
LOSS_SCALE = 0.1
MAX_VELOCITY_UM_H = 100.0

SMOOTHNESS_WEIGHT = 0.001

# Match run_Vd_calibration.py anchor buckets exactly.
# d_Gy = concat([
#   logspace(2, 4, endpoint=False, num=1),
#   logspace(4, 5, endpoint=False, num=2),
#   logspace(5, 8, endpoint=False, num=8),
#   logspace(8, 9, endpoint=True,  num=2),
# ])
ANCHOR_COUNT_USER_OPTIONS = [13]
INIT_LOG_INCREMENT_NOISE_STD = 0.08

# Run debris damping in separate configurations.
# Allowed values: "without" and "fit".
DEBRIS_DAMPING_RUN_MODES = ["without", "fit"]

# Threshold used to flag a "huge" anchor-count effect in summary tables.
ANCHOR_EFFECT_HUGE_RATIO_THRESHOLD = 1.25

# Broader global search settings.
# This reduces sensitivity to starting V(d) guesses that may favor one variant.
MUTATION = (0.5, 1.4)
RECOMBINATION = 0.9
POLISH_METHOD = "Powell"

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / OUTPUT_DIR_NAME


def _log(msg: str) -> None:
    print(msg, flush=True)


def _worker_tag(
    variant: int,
    seed: int,
    fit_debris_damping: bool,
    anchor_count_user: int,
) -> str:
    debris_mode = "fit" if fit_debris_damping else "without"
    return (
        f"[pid={os.getpid()} thread={threading.current_thread().name} "
        f"bragg={variant}% seed={seed} debris={debris_mode} anchors={anchor_count_user}]"
    )


def _save_track_shape_pngs(models_dict: dict, run_dir: Path) -> list[str]:
    figures = plot_track_shapes(models_dict["track_shape"], xscale="linear")
    png_paths: list[str] = []
    keys = list(models_dict["track_shape"].keys())

    for i, key in enumerate(keys):
        safe_key = key.replace("/", "_").replace(" ", "_")
        out_path = run_dir / f"track_shape_{safe_key}.png"
        figures[i].savefig(out_path, dpi=300)
        png_paths.append(str(out_path))
        figures[i].clf()

    return png_paths


def _build_user_anchor_dose_grid(anchor_count_user: int) -> np.ndarray:
    """Build the same anchor-dose grid as run_Vd_calibration.py."""
    user_doses = np.concatenate(
        [
            np.logspace(2, 4, endpoint=False, num=1),
            np.logspace(4, 5, endpoint=False, num=2),
            np.logspace(5, 8, endpoint=False, num=8),
            np.logspace(8, 9, endpoint=True, num=2),
        ]
    )
    user_doses = np.unique(user_doses)
    user_doses = np.sort(user_doses)

    if anchor_count_user != len(user_doses):
        raise ValueError(
            "anchor_count_user must match run_Vd_calibration anchor count "
            f"({len(user_doses)}), got {anchor_count_user}"
        )

    return user_doses


def _build_seeded_etch_model(seed: int, anchor_count_user: int) -> EtchRateModel:
    """Create a broader and seed-dependent initial V(d) model."""
    base_model = default_etch_rate_model()

    # Use piecewise spacing with highest anchor density in 1e5..1e7 Gy.
    user_doses = _build_user_anchor_dose_grid(anchor_count_user)
    user_velocities = np.asarray(base_model.eval(user_doses), dtype=float)

    model = EtchRateModel(
        anchor_doses_Gy=user_doses,
        anchor_velocities_um_h=user_velocities,
        V_bulk_um_h=base_model.V_bulk_um_h,
        V_max_um_h=MAX_VELOCITY_UM_H,
        name=f"benchmark_seed_{seed}",
    )

    rng = np.random.default_rng(seed)
    log_increments = model.get_log_velocity_increments()
    noise = rng.normal(0.0, INIT_LOG_INCREMENT_NOISE_STD, size=log_increments.shape)
    noise[0] = 0.0  # keep low-dose anchor tied to bulk value
    seeded_increments = np.maximum(log_increments + noise, 0.0)

    # Keep seeded starts valid under the configured V_max cap.
    max_total_log_gain = max(
        0.0,
        np.log10(MAX_VELOCITY_UM_H) - np.log10(float(model.V_bulk_um_h)),
    )
    total_log_gain = float(np.sum(seeded_increments))
    if total_log_gain > max_total_log_gain and total_log_gain > 0.0:
        seeded_increments *= max_total_log_gain / total_log_gain

    model.update_from_log_velocity_increments(seeded_increments)

    return model


def _save_vd_overview_png(vd_curves: list[dict], out_path: Path) -> None:
    if not vd_curves:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    rows: list[dict] = []
    for curve in sorted(
        vd_curves,
        key=lambda c: (
            c["bragg_correction"],
            c["fit_debris_damping"],
            c["anchor_count_user"],
            c["seed"],
        ),
    ):
        bragg = int(curve["bragg_correction"])
        seed = int(curve["seed"])
        fit_debris_damping = bool(curve["fit_debris_damping"])
        anchor_count_user = int(curve["anchor_count_user"])
        debris_mode = "fit" if fit_debris_damping else "without"
        doses = np.asarray(curve["doses_Gy"], dtype=float)
        velocities = np.asarray(curve["velocities_um_h"], dtype=float)
        for dose, velocity in zip(doses, velocities):
            rows.append(
                {
                    "bragg_correction": f"Bragg {bragg}%",
                    "run": f"B{bragg}|{debris_mode}|A{anchor_count_user}|S{seed}",
                    "dose_Gy": dose,
                    "velocity_um_h": velocity,
                }
            )

    plot_df = pd.DataFrame(rows)

    if sns is not None:
        sns.lineplot(
            data=plot_df,
            x="dose_Gy",
            y="velocity_um_h",
            hue="bragg_correction",
            style="run",
            linewidth=2.0,
            ax=ax,
        )
        sns.scatterplot(
            data=plot_df,
            x="dose_Gy",
            y="velocity_um_h",
            hue="bragg_correction",
            style="run",
            s=26,
            edgecolor="white",
            linewidth=0.4,
            legend=False,
            ax=ax,
        )
    else:
        bragg_values = sorted(plot_df["bragg_correction"].unique())
        run_values = sorted(plot_df["run"].unique())
        color_map = plt.cm.get_cmap("tab10", max(len(bragg_values), 1))
        linestyle_cycle = ["-", "--", "-.", ":"]
        run_to_linestyle = {
            run: linestyle_cycle[i % len(linestyle_cycle)]
            for i, run in enumerate(run_values)
        }

        for i, bragg in enumerate(bragg_values):
            for run in run_values:
                sub_df = plot_df[
                    (plot_df["bragg_correction"] == bragg) & (plot_df["run"] == run)
                ].sort_values("dose_Gy")
                if sub_df.empty:
                    continue
                ax.plot(
                    sub_df["dose_Gy"],
                    sub_df["velocity_um_h"],
                    color=color_map(i),
                    linestyle=run_to_linestyle[run],
                    linewidth=1.8,
                    label=f"{bragg} | {run}",
                )
                ax.scatter(
                    sub_df["dose_Gy"],
                    sub_df["velocity_um_h"],
                    color=[color_map(i)],
                    s=20,
                    edgecolor="white",
                    linewidth=0.4,
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dose / Gy")
    ax.set_ylabel("Etch rate V(d) / (um/h)")
    ax.set_title("Optimized V(d) curves with anchors by Bragg variant and seed")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2, title="Run")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_one(
    variant: int,
    seed: int,
    fit_debris_damping: bool,
    anchor_count_user: int,
    particles: list[str],
    run_dir: Path,
) -> dict:
    tag = _worker_tag(
        variant,
        seed,
        fit_debris_damping=fit_debris_damping,
        anchor_count_user=anchor_count_user,
    )
    _log(f"{tag} Starting run")
    set_bragg_correction(variant)

    etch_model = _build_seeded_etch_model(seed, anchor_count_user=anchor_count_user)
    _log(
        f"{tag} Initial model anchors: {len(etch_model.anchor_doses_Gy)} "
        f"(dose range {etch_model.anchor_doses_Gy.min():.2e}..{etch_model.anchor_doses_Gy.max():.2e} Gy)"
    )
    models_dict = create_minimisation_data(
        particle_names=particles,
        etch_model=etch_model,
    )

    result = optimize_etch_model(
        etch_model,
        models_dict,
        objective=OBJECTIVE,
        aggregation=AGGREGATION,
        residual_scale=RESIDUAL_SCALE,
        loss=LOSS,
        loss_scale=LOSS_SCALE,
        popsize=POPSIZE,
        maxiter=MAXITER,
        tol=TOL,
        mutation=MUTATION,
        recombination=RECOMBINATION,
        max_velocity_um_h=MAX_VELOCITY_UM_H,
        seed=seed,
        polish_method=POLISH_METHOD,
        smoothness_weight=SMOOTHNESS_WEIGHT,
        fit_debris_damping=fit_debris_damping,
    )

    png_paths = _save_track_shape_pngs(models_dict, run_dir)
    _log(f"{tag} Saved {len(png_paths)} track-shape PNG(s) in {run_dir}")
    _log(f"{tag} Final cost: {float(result['fun']):.6e}")

    return {
        "bragg_correction": variant,
        "seed": seed,
        "fit_debris_damping": fit_debris_damping,
        "debris_mode": "fit" if fit_debris_damping else "without",
        "anchor_count_user": anchor_count_user,
        "objective": OBJECTIVE,
        "final_cost": float(result["fun"]),
        "initial_cost": float(result["initial_cost"]),
        "improved": bool(result["improved"]),
        "elapsed_seconds": float(result["elapsed_seconds"]),
        "debris_alpha": float(result.get("debris_alpha", np.nan)),
        "debris_beta": float(result.get("debris_beta", np.nan)),
        "run_dir": str(run_dir),
        "n_track_shape_png": len(png_paths),
        "track_shape_png_example": png_paths[0] if png_paths else "",
        "_vd_curve": {
            "bragg_correction": variant,
            "seed": seed,
            "fit_debris_damping": fit_debris_damping,
            "anchor_count_user": anchor_count_user,
            "doses_Gy": etch_model.anchor_doses_Gy.copy(),
            "velocities_um_h": etch_model.anchor_velocities_um_h.copy(),
        },
    }


def _build_anchor_effect_table(
    detailed_df: pd.DataFrame,
    huge_ratio_threshold: float,
) -> pd.DataFrame:
    anchor_stats = (
        detailed_df.groupby(
            ["bragg_correction", "debris_mode", "anchor_count_user"],
            as_index=False,
        )
        .agg(
            final_cost_median=("final_cost", "median"),
            final_cost_mean=("final_cost", "mean"),
            n_runs=("final_cost", "size"),
        )
        .sort_values(["bragg_correction", "debris_mode", "anchor_count_user"])
    )

    if anchor_stats.empty:
        return anchor_stats

    span = (
        anchor_stats.groupby(["bragg_correction", "debris_mode"], as_index=False)
        .agg(
            anchor_count_min=("anchor_count_user", "min"),
            anchor_count_max=("anchor_count_user", "max"),
            anchor_median_cost_min=("final_cost_median", "min"),
            anchor_median_cost_max=("final_cost_median", "max"),
            n_anchor_settings=("anchor_count_user", "nunique"),
        )
        .sort_values(["bragg_correction", "debris_mode"])
    )
    span["anchor_effect_ratio_max_to_min"] = (
        span["anchor_median_cost_max"] / span["anchor_median_cost_min"]
    )
    span["anchor_effect_percent"] = 100.0 * (
        span["anchor_effect_ratio_max_to_min"] - 1.0
    )
    span["anchor_effect_is_huge"] = (
        span["anchor_effect_ratio_max_to_min"] >= huge_ratio_threshold
    )
    return span


def main() -> None:
    results_dir = OUTPUT_DIR.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    vd_curves: list[dict] = []
    tasks: list[tuple[int, int, bool, int, list[str], Path]] = []

    invalid_modes = sorted(set(DEBRIS_DAMPING_RUN_MODES) - {"without", "fit"})
    if invalid_modes:
        raise ValueError(
            "DEBRIS_DAMPING_RUN_MODES can only contain 'without' and 'fit'; "
            f"got invalid values: {invalid_modes}"
        )
    if not ANCHOR_COUNT_USER_OPTIONS:
        raise ValueError("ANCHOR_COUNT_USER_OPTIONS must contain at least one value")
    if any(c < 2 for c in ANCHOR_COUNT_USER_OPTIONS):
        raise ValueError("All ANCHOR_COUNT_USER_OPTIONS values must be >= 2")

    fit_debris_damping_options = [m == "fit" for m in DEBRIS_DAMPING_RUN_MODES]

    for variant in BRAGG_VARIANTS:
        variant_dir = results_dir / f"bragg_{variant}"
        variant_dir.mkdir(parents=True, exist_ok=True)

        for fit_debris_damping in fit_debris_damping_options:
            debris_mode = "fit" if fit_debris_damping else "without"
            debris_dir = variant_dir / f"debris_{debris_mode}"
            debris_dir.mkdir(parents=True, exist_ok=True)
            for anchor_count_user in ANCHOR_COUNT_USER_OPTIONS:
                anchor_dir = debris_dir / f"anchors_{anchor_count_user}"
                anchor_dir.mkdir(parents=True, exist_ok=True)

                for seed in SEEDS:
                    run_dir = anchor_dir / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    tasks.append(
                        (
                            variant,
                            seed,
                            fit_debris_damping,
                            anchor_count_user,
                            PARTICLES,
                            run_dir,
                        )
                    )

    if RUN_PARALLEL and len(tasks) > 1:
        n_workers = min(N_JOBS, len(tasks))
        _log(f"Starting {len(tasks)} runs in parallel with {n_workers} workers")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(
                    run_one,
                    variant,
                    seed,
                    fit_debris_damping,
                    anchor_count_user,
                    particles,
                    run_dir,
                ): (variant, seed, fit_debris_damping, anchor_count_user)
                for (
                    variant,
                    seed,
                    fit_debris_damping,
                    anchor_count_user,
                    particles,
                    run_dir,
                ) in tasks
            }
            for future in as_completed(future_to_task):
                variant, seed, fit_debris_damping, anchor_count_user = future_to_task[
                    future
                ]
                row = future.result()
                vd_curves.append(row.pop("_vd_curve"))
                rows.append(row)
                debris_mode = "fit" if fit_debris_damping else "without"
                _log(
                    "[main] Completed "
                    f"Bragg {variant}% ({debris_mode}, anchors={anchor_count_user}, seed={seed})"
                )
    else:
        for (
            variant,
            seed,
            fit_debris_damping,
            anchor_count_user,
            particles,
            run_dir,
        ) in tasks:
            debris_mode = "fit" if fit_debris_damping else "without"
            _log(
                "[main] Running "
                f"Bragg {variant}% ({debris_mode}, anchors={anchor_count_user}, seed={seed})"
            )
            row = run_one(
                variant=variant,
                seed=seed,
                fit_debris_damping=fit_debris_damping,
                anchor_count_user=anchor_count_user,
                particles=particles,
                run_dir=run_dir,
            )
            vd_curves.append(row.pop("_vd_curve"))
            rows.append(row)

    detailed_df = pd.DataFrame(rows).sort_values(
        [
            "bragg_correction",
            "debris_mode",
            "anchor_count_user",
            "seed",
        ]
    )
    detailed_path = results_dir / "bragg_benchmark_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False)

    summary_df = (
        detailed_df.groupby(
            ["bragg_correction", "debris_mode", "anchor_count_user"],
            as_index=False,
        )
        .agg(
            test_cost_both_median=("final_cost", "median"),
            test_cost_both_mean=("final_cost", "mean"),
            test_cost_both_std=("final_cost", "std"),
            n_runs=("final_cost", "size"),
            improved_runs=("improved", "sum"),
            elapsed_seconds_median=("elapsed_seconds", "median"),
        )
        .sort_values(["test_cost_both_median", "bragg_correction"])
    )

    best_idx = detailed_df.groupby(
        ["bragg_correction", "debris_mode", "anchor_count_user"]
    )["final_cost"].idxmin()
    best_runs_df = detailed_df.loc[
        best_idx,
        [
            "bragg_correction",
            "debris_mode",
            "anchor_count_user",
            "seed",
            "final_cost",
            "run_dir",
            "debris_alpha",
            "debris_beta",
        ],
    ]
    best_runs_df = best_runs_df.rename(
        columns={
            "seed": "best_seed",
            "final_cost": "best_final_cost",
            "run_dir": "best_run_dir",
            "debris_alpha": "best_debris_alpha",
            "debris_beta": "best_debris_beta",
        }
    )
    summary_df = summary_df.merge(
        best_runs_df,
        on=["bragg_correction", "debris_mode", "anchor_count_user"],
        how="left",
    )

    anchor_effect_df = _build_anchor_effect_table(
        detailed_df,
        huge_ratio_threshold=ANCHOR_EFFECT_HUGE_RATIO_THRESHOLD,
    )
    summary_df = summary_df.merge(
        anchor_effect_df,
        on=["bragg_correction", "debris_mode"],
        how="left",
    )

    summary_path = results_dir / "bragg_benchmark_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    overview_path = results_dir / "bragg_benchmark_overview.csv"
    summary_df.to_csv(overview_path, index=False)

    vd_plot_path = results_dir / "bragg_benchmark_vd_curves.png"
    _save_vd_overview_png(vd_curves, vd_plot_path)

    _log(f"\nSaved detailed results: {detailed_path}")
    _log(f"Saved summary results : {summary_path}")
    _log(f"Saved overview results: {overview_path}")
    _log(f"Saved V(d) figure     : {vd_plot_path}")
    _log("\nBest configuration by median cost:")
    _log(summary_df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
