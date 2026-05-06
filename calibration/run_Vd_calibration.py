# %%
import os

from tracketch.etching.etch_rate_model import EtchRateModel

from tracketch.etching.etch_rate_model_io import (
    save_etchrate_model,
    load_etchrate_model,
)
from calibration.calibration_data import create_minimisation_data

from calibration.plotting import (
    plot_track_shapes,
    plot_track_shapes_debris_comparison,
    plot_track_length,
)
from calibration.lib_optimiser import (
    cost_function,
    optimize_etch_model,
    estimate_parameter_uncertainties,
)
from tracketch.physics.SRIM import set_bragg_correction

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
import numpy as np


CALIBRATED_MODEL_NAME = "Doerschel_etching"


def starting_guess_v_um_h(
    anchor_dose_Gy: np.ndarray,
    max_v_um: float,
):
    """
    Interpolate anchor velocities from a previously saved optimized model.

    Loads the saved model JSON, then uses PCHIP interpolation to map its
    v(d) curve onto the requested anchor doses.  Values are clipped to
    [V_bulk, max_v_um].
    """
    saved_model = load_etchrate_model(CALIBRATED_MODEL_NAME)
    interpol = PchipInterpolator(
        np.log10(saved_model.anchor_doses_Gy),
        saved_model.anchor_velocities_um_h,
        extrapolate=True,
    )
    v_um_h = interpol(np.log10(anchor_dose_Gy))
    v_um_h = np.clip(v_um_h, saved_model.V_bulk_um_h, max_v_um * 0.5)
    # Ensure monotonicity: each value must be >= its predecessor
    v_um_h = np.maximum.accumulate(v_um_h)
    return v_um_h


select_particles = ["1H", "4He", "7Li", "12C"]

objective_mode = "shape"  # "both", "shape", or "length"
aggregation_mode = "equal_point"  # "equal_ion_energy" or "equal_point"
residual_scale_mode = "log"  # "linear" or "log"
loss_mode = "soft_l1"  # "l2", "soft_l1", or "huber"
loss_scale = 0.1  # tighter robust transition for closer fit in log-space
max_velocity_um_h = 1e2
bragg_correction_percent = 3
calibration_seed = 11
init_log_increment_noise_std = 0.08

# ---- Toggle between fast dev fit and final fit ----
final_fit = False
final_fit = True

if final_fit:
    # Most thorough fit profile: broad DE search with extended convergence.
    de_popsize = 70
    de_maxiter = 300
    de_tol = 1e-6
    de_mutation = (0.5, 1.4)
    de_recombination = 0.9
    de_polish_method = "Powell"
    de_smoothness_weight = 0.001
    fit_debris_damping = True  # 5%-capped damping is now identifiable
else:
    # Dev profile: faster iterations for quick checks
    de_popsize = 30
    de_maxiter = 50
    de_tol = 1e-5
    de_mutation = (0.5, 1.0)
    de_recombination = 0.7
    de_polish_method = "Powell"
    de_smoothness_weight = 0.0
    fit_debris_damping = True  # 5%-capped damping is now identifiable

# ---- Debris damping ----
# Not co-optimized: shape data alone cannot identify alpha/beta (optimizer
# always escapes to alpha=max, beta=max, i.e. damping disabled).  Set fixed
# values here if debris correction is desired; leave as None to disable.
# eta = 1 / (1 + (aspect_ratio / alpha)**beta)
debris_alpha_bounds_log10 = (0.1, 3)
debris_beta_bounds = (1.0, 6.0)

# ---- Anchor setup ----
# Only dose positions matter; DE finds the velocities automatically.
d_Gy = np.concatenate(
    [
        # Match benchmark anchor buckets exactly.
        np.logspace(2, 4, endpoint=False, num=2),
        np.logspace(4, 5, endpoint=False, num=3),
        np.logspace(5, 8, endpoint=False, num=9),
        np.logspace(8, 9, endpoint=True, num=2),
    ]
)
d_Gy = np.unique(d_Gy)  # Remove duplicates
d_Gy = np.sort(d_Gy)

v_um_h = starting_guess_v_um_h(d_Gy, max_velocity_um_h)

etch_model = EtchRateModel(
    anchor_doses_Gy=d_Gy,
    anchor_velocities_um_h=v_um_h,
    V_max_um_h=max_velocity_um_h,
    V_bulk_um_h=1.73,
    name="general_model",
    # debris_alpha starts as None -- fitted in stage 2
)

# Benchmark-style seeded perturbation of log-increments to diversify starts.
if init_log_increment_noise_std > 0:
    rng = np.random.default_rng(calibration_seed)
    log_increments = etch_model.get_log_velocity_increments()
    noise = rng.normal(0.0, init_log_increment_noise_std, size=log_increments.shape)
    noise[0] = 0.0  # keep low-dose anchor tied to bulk value
    seeded_increments = np.maximum(log_increments + noise, 0.0)

    max_total_log_gain = max(
        0.0,
        np.log10(max_velocity_um_h) - np.log10(float(etch_model.V_bulk_um_h)),
    )
    total_log_gain = float(np.sum(seeded_increments))
    if total_log_gain > max_total_log_gain and total_log_gain > 0.0:
        seeded_increments *= max_total_log_gain / total_log_gain

    etch_model.update_from_log_velocity_increments(seeded_increments)
    print(
        f"Applied seeded anchor perturbation (seed={calibration_seed}, "
        f"sigma={init_log_increment_noise_std})."
    )

etch_model.plot()
plt.show()

set_bragg_correction(bragg_correction_percent)
print(f"Using SRIM Bragg correction: {bragg_correction_percent}%")

models_dict = create_minimisation_data(
    particle_names=select_particles,
    etch_model=etch_model,
)

J_initial = cost_function(
    etch_model,
    models_dict,
    verbose=True,
    objective=objective_mode,
    residual_scale=residual_scale_mode,
    loss=loss_mode,
    loss_scale=loss_scale,
    aggregation=aggregation_mode,
)
print(f"Initial cost: {J_initial:.6e}")

# ---- Single-pass: co-optimize v(d) + debris damping ----
result = optimize_etch_model(
    etch_model,
    models_dict,
    residual_method="vertical",
    residual_scale=residual_scale_mode,
    loss=loss_mode,
    loss_scale=loss_scale,
    max_velocity_um_h=max_velocity_um_h,
    objective=objective_mode,
    aggregation=aggregation_mode,
    popsize=de_popsize,
    maxiter=de_maxiter,
    tol=de_tol,
    mutation=de_mutation,
    recombination=de_recombination,
    seed=calibration_seed,
    polish_method=de_polish_method,
    smoothness_weight=de_smoothness_weight,
    fit_debris_damping=fit_debris_damping,
    debris_alpha_bounds_log10=debris_alpha_bounds_log10,
    debris_beta_bounds=debris_beta_bounds,
)

# Estimate parameter uncertainties
unc = estimate_parameter_uncertainties(
    etch_model,
    models_dict,
    residual_method="vertical",
    residual_scale=residual_scale_mode,
    loss=loss_mode,
    loss_scale=loss_scale,
    objective=objective_mode,
    aggregation=aggregation_mode,
    smoothness_weight=de_smoothness_weight,
    max_velocity_um_h=max_velocity_um_h,
    fit_debris_damping=fit_debris_damping,
)
etch_model.anchor_velocity_uncertainties_um_h = unc["velocity_uncertainties_um_h"]
# Use the last (highest-dose) anchor uncertainty as a proxy for V_max
# uncertainty: when anchors saturate at V_max, this is the most relevant
# constraint on the saturation velocity.
if np.isfinite(etch_model.V_max_um_h):
    etch_model.V_max_uncertainty_um_h = float(unc["velocity_uncertainties_um_h"][-1])

# Save results
etch_model.name = CALIBRATED_MODEL_NAME
etch_model.plot()
save_etchrate_model(etch_model, CALIBRATED_MODEL_NAME)

# Recalculate all simulators with final model params before plotting
for sub_dict in models_dict.values():
    for entry in sub_dict.values():
        entry["simulator"].recalculate_from_current_etch_model()

folder_name = "figures_calibration" if final_fit else "figures_calibration_dev"
os.makedirs(folder_name, exist_ok=True)

if objective_mode in ["both", "shape"]:
    figs = plot_track_shapes(models_dict["track_shape"], xscale="linear")
    shape_keys = list(models_dict["track_shape"].keys())
    for i, particle in enumerate(shape_keys):
        figs[i].savefig(f"{folder_name}/track_shape_{particle}.png", dpi=300)

    # Debris comparison: contours with vs without damping
    debris_figs = plot_track_shapes_debris_comparison(
        models_dict["track_shape"], xscale="linear"
    )
    for i, particle in enumerate(shape_keys):
        debris_figs[i].savefig(
            f"{folder_name}/debris_comparison_{particle}.png",
            dpi=300,
        )

if objective_mode in ["both", "length"]:
    figs_by_ion = plot_track_length(models_dict["track_length"])
    for ion, fig in figs_by_ion.items():
        fig.savefig(f"{folder_name}/track_length_{ion}.png", dpi=300)

    # %%

sim = models_dict["track_length"]["7Li_11.99MeV"]["simulator"]

fig, ax = sim.plot_map("arrival")
sim.plot_iso_time_contour(etching_time_h=6, ax=ax)
