"""
Etch rate model optimization library.

Provides differential-evolution global optimization with increment-based
parameterization for monotonic etch-rate calibration.
"""

import numpy as np
from time import perf_counter
from functools import partial
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import CubicSpline
from tracketch.etching.etch_rate_model import EtchRateModel
from calibration.helpers import compute_closest_distances


def _robust_loss_values(
    residuals: np.ndarray,
    loss: str = "l2",
    f_scale: float = 1.0,
) -> np.ndarray:
    """Return per-residual robust loss values with dimensionless scaling.

    The transition scale ``f_scale`` controls where robust behavior begins,
    while the returned magnitude is normalized so tiny ``f_scale`` does not
    collapse objective/gradient magnitudes.
    """
    abs_residuals = np.abs(np.asarray(residuals, dtype=float))
    scale = float(max(f_scale, 1e-12))
    scaled = abs_residuals / scale

    if loss == "l2":
        return scaled**2
    if loss == "soft_l1":
        z = scaled**2
        return 2.0 * (np.sqrt(1.0 + z) - 1.0)
    if loss == "huber":
        out = np.empty_like(scaled)
        quad = scaled <= 1.0
        out[quad] = scaled[quad] ** 2
        out[~quad] = 2.0 * scaled[~quad] - 1.0
        return out

    raise ValueError(f"loss must be one of ['l2', 'soft_l1', 'huber'], got '{loss}'")


def _get_effective_max_velocity_um_h(
    etch_model: EtchRateModel,
    max_velocity_um_h: float | None,
) -> float:
    """Resolve effective maximum etch-rate cap from argument or model setting."""
    if max_velocity_um_h is None:
        return float(etch_model.V_max_um_h)
    return float(max_velocity_um_h)


def _cost_track_shape(
    etch_model: EtchRateModel,
    models_dict: dict,
    verbose: bool = False,
    residual_method: str = "vertical",
    residual_scale: str = "linear",
    loss: str = "l2",
    loss_scale: float = 1.0,
    aggregation: str = "equal_ion_energy",
) -> float:
    """Compute cost for 2D track-shape datasets."""
    if residual_method not in ["vertical", "closest"]:
        raise ValueError(
            f"residual_method must be 'vertical' or 'closest', got '{residual_method}'"
        )
    if residual_scale not in ["linear", "log"]:
        raise ValueError(
            f"residual_scale must be 'linear' or 'log', got '{residual_scale}'"
        )
    if loss not in ["l2", "soft_l1", "huber"]:
        raise ValueError(
            f"loss must be one of ['l2', 'soft_l1', 'huber'], got '{loss}'"
        )
    if loss_scale <= 0:
        raise ValueError(f"loss_scale must be > 0, got {loss_scale}")

    ion_energy_costs = []
    group_costs_all = []
    group_point_counts_all = []

    for particle_name, data_dict in models_dict.items():
        obj = data_dict["simulator"]
        particle_df = data_dict["experiment_data"]
        group_costs = []
        group_point_counts = []

        obj.update_etch_model_and_recalculate(etch_model)

        for etch_time_h, particle_data_df in particle_df.groupby("time_h"):
            r_pred_um, z_pred_um = obj.get_iso_time_contour(etching_time_h=etch_time_h)

            if len(r_pred_um) == 0 or len(z_pred_um) == 0:
                print(
                    f"Warning: No contour found for {particle_name} at {etch_time_h} hr"
                    " -- skipping this time step."
                )
                continue

            r_exp_um = particle_data_df["r_um"].values
            z_exp_um = particle_data_df["z_um"].values
            r_err_um = particle_data_df["r_err_um"].values
            z_err_um = particle_data_df["z_err_um"].values

            depth_weights = (z_exp_um / np.max(z_exp_um)) ** 2

            if residual_method == "vertical":
                uncertainty_weights = 1.0 / z_err_um**2
            else:
                combined_err = np.sqrt(r_err_um**2 + z_err_um**2)
                uncertainty_weights = 1.0 / combined_err**2

            weights = depth_weights * uncertainty_weights
            weights /= np.sum(weights)

            if residual_method == "vertical":
                sort_idx = np.argsort(r_pred_um)
                r_pred_sorted = r_pred_um[sort_idx]
                z_pred_sorted = z_pred_um[sort_idx]

                finite_mask = np.isfinite(r_pred_sorted) & np.isfinite(z_pred_sorted)
                r_pred_sorted = r_pred_sorted[finite_mask]
                z_pred_sorted = z_pred_sorted[finite_mask]

                if len(r_pred_sorted) < 2:
                    print(
                        f"Warning: Insufficient finite contour points for {particle_name} "
                        f"at {etch_time_h} hr -- skipping this time step."
                    )
                    continue

                # CubicSpline and np.interp require strictly increasing x values.
                # Keep first occurrence for duplicate radii after sorting.
                r_pred_unique, unique_idx = np.unique(r_pred_sorted, return_index=True)
                z_pred_unique = z_pred_sorted[unique_idx]

                if len(r_pred_unique) < 2:
                    print(
                        f"Warning: Contour radii are not sufficiently distinct for {particle_name} "
                        f"at {etch_time_h} hr -- skipping this time step."
                    )
                    continue

                if len(r_pred_unique) > 3:
                    cs = CubicSpline(r_pred_unique, z_pred_unique)
                    z_pred_at_r_exp = cs(r_exp_um)
                else:
                    z_pred_at_r_exp = np.interp(r_exp_um, r_pred_unique, z_pred_unique)

                if residual_scale == "linear":
                    residuals = np.abs(z_exp_um - z_pred_at_r_exp)
                else:
                    eps_um = 1e-9
                    z_exp_safe = np.maximum(z_exp_um, eps_um)
                    z_pred_safe = np.maximum(z_pred_at_r_exp, eps_um)
                    residuals = np.abs(np.log10(z_exp_safe) - np.log10(z_pred_safe))
            else:
                residuals = compute_closest_distances(
                    r_exp_um, z_exp_um, r_pred_um, z_pred_um
                )
                if residual_scale == "log":
                    scale_um = max(float(np.nanmedian(np.abs(z_exp_um))), 1e-9)
                    residuals = np.log10(1.0 + residuals / scale_um)

            robust_values = _robust_loss_values(
                residuals, loss=loss, f_scale=loss_scale
            )
            dataset_cost = np.sum(weights * robust_values)
            group_costs.append(float(dataset_cost))
            group_point_counts.append(len(residuals))

            if verbose:
                residual_unit = "um" if residual_scale == "linear" else "dex"
                print(
                    f"Particle: {particle_name}, Etch time: {etch_time_h} hr, "
                    f"Shape dataset cost: {dataset_cost:.6f}, Points: {len(residuals)}, "
                    f"Max residual: {np.max(residuals):.2f} {residual_unit}"
                )

        if not group_costs:
            return 1e10

        if aggregation == "equal_ion_energy":
            ion_energy_costs.append(float(np.mean(group_costs)))
        else:
            group_costs_all.extend(group_costs)
            group_point_counts_all.extend(group_point_counts)

    if aggregation == "equal_ion_energy":
        return float(np.mean(ion_energy_costs)) if ion_energy_costs else 1e10

    if not group_costs_all:
        return 1e10
    return float(np.average(group_costs_all, weights=group_point_counts_all))


def _cost_track_length(
    etch_model: EtchRateModel,
    models_dict: dict,
    verbose: bool = False,
    residual_scale: str = "linear",
    loss: str = "l2",
    loss_scale: float = 1.0,
    aggregation: str = "equal_ion_energy",
) -> float:
    """Compute cost for track-length datasets."""
    if residual_scale not in ["linear", "log"]:
        raise ValueError(
            f"residual_scale must be 'linear' or 'log', got '{residual_scale}'"
        )
    if loss not in ["l2", "soft_l1", "huber"]:
        raise ValueError(
            f"loss must be one of ['l2', 'soft_l1', 'huber'], got '{loss}'"
        )
    if loss_scale <= 0:
        raise ValueError(f"loss_scale must be > 0, got {loss_scale}")

    ion_energy_costs = []
    group_costs_all = []
    group_point_counts_all = []

    for particle_name, data_dict in models_dict.items():
        obj = data_dict["simulator"]
        particle_df = data_dict["experiment_data"]
        group_costs = []
        group_point_counts = []

        obj.update_etch_model_and_recalculate(etch_model)

        for etch_time_h, particle_data_df in particle_df.groupby("time_h"):
            length_exp_um = particle_data_df["length_um"].values
            length_err_um = particle_data_df["length_um_std"].values

            length_pred_um = obj.get_track_length_um(
                etch_time_h=float(etch_time_h), relative_to_surface=True
            )

            if not np.isfinite(length_pred_um):
                print(
                    f"Warning: No track length found for {particle_name} at {etch_time_h} hr"
                    " -- skipping this time step."
                )
                continue

            if residual_scale == "linear":
                residuals = np.abs(length_exp_um - float(length_pred_um))
            else:
                eps_um = 1e-9
                length_exp_safe = np.maximum(length_exp_um, eps_um)
                length_pred_safe = max(float(length_pred_um), eps_um)
                residuals = np.abs(
                    np.log10(length_exp_safe) - np.log10(length_pred_safe)
                )
            uncertainty_weights = 1.0 / np.maximum(length_err_um, 1e-12) ** 2
            weights = uncertainty_weights / np.sum(uncertainty_weights)

            robust_values = _robust_loss_values(
                residuals, loss=loss, f_scale=loss_scale
            )
            dataset_cost = np.sum(weights * robust_values)
            group_costs.append(float(dataset_cost))
            group_point_counts.append(len(residuals))

            if verbose:
                residual_unit = "um" if residual_scale == "linear" else "dex"
                print(
                    f"Particle: {particle_name}, Etch time: {etch_time_h} hr, "
                    f"Length dataset cost: {dataset_cost:.6f}, Points: {len(residuals)}, "
                    f"Max residual: {np.max(residuals):.2f} {residual_unit}"
                )

        if not group_costs:
            return 1e10

        if aggregation == "equal_ion_energy":
            ion_energy_costs.append(float(np.mean(group_costs)))
        else:
            group_costs_all.extend(group_costs)
            group_point_counts_all.extend(group_point_counts)

    if aggregation == "equal_ion_energy":
        return float(np.mean(ion_energy_costs)) if ion_energy_costs else 1e10

    if not group_costs_all:
        return 1e10
    return float(np.average(group_costs_all, weights=group_point_counts_all))


def cost_function(
    etch_model: EtchRateModel,
    models_dict: dict,
    verbose: bool = False,
    residual_method: str = "vertical",
    residual_scale: str = "linear",
    loss: str = "l2",
    loss_scale: float = 1.0,
    shape_weight: float = 0.5,
    length_weight: float = 0.5,
    objective: str = "both",
    aggregation: str = "equal_ion_energy",
) -> float:
    """
    Compute cost (residual sum of squares) for given etch model.

    Parameters
    ----------
    etch_model : EtchRateModel
        The etch rate model to evaluate
    models_dict : dict
        Combined dictionary with keys `track_shape` and `track_length`.
        Each value must be a particle-keyed dictionary with simulator and experiment_data.
    verbose : bool, optional
        Print detailed residual information (default: False)
    residual_method : str, optional
        How to compute residuals (default: "vertical"):
        - "vertical": Compute z-difference at each experimental r position
        - "closest": Compute minimum distance from point to contour line
    residual_scale : str, optional
        Scale used to evaluate residuals (default: "linear"):
        - "linear": absolute differences in um
        - "log": absolute differences in log10-space (dex)
    loss : str, optional
        Point-wise loss function (default: "l2"):
        - "l2": squared residuals
        - "soft_l1": smooth robust loss
        - "huber": piecewise quadratic/linear robust loss
    loss_scale : float, optional
        Scale parameter for robust losses (default: 1.0)
    shape_weight : float, optional
        Weight for track-shape objective when both datasets are provided
    length_weight : float, optional
        Weight for track-length objective when both datasets are provided

    Returns
    -------
    float
        Total squared residual cost
    """
    valid_objectives = ["both", "shape", "length"]
    if objective not in valid_objectives:
        raise ValueError(
            f"objective must be one of {valid_objectives}, got '{objective}'"
        )

    valid_aggregation = ["equal_ion_energy", "equal_point"]
    if aggregation not in valid_aggregation:
        raise ValueError(
            f"aggregation must be one of {valid_aggregation}, got '{aggregation}'"
        )

    valid_residual_scale = ["linear", "log"]
    if residual_scale not in valid_residual_scale:
        raise ValueError(
            f"residual_scale must be one of {valid_residual_scale}, got '{residual_scale}'"
        )

    valid_loss = ["l2", "soft_l1", "huber"]
    if loss not in valid_loss:
        raise ValueError(f"loss must be one of {valid_loss}, got '{loss}'")
    if loss_scale <= 0:
        raise ValueError(f"loss_scale must be > 0, got {loss_scale}")

    shape_models_dict = models_dict.get("track_shape", {})
    length_models_dict = models_dict.get("track_length", {})

    if objective == "shape" and not shape_models_dict:
        raise ValueError("'track_shape' dataset is required for objective='shape'.")
    if objective == "length" and not length_models_dict:
        raise ValueError("'track_length' dataset is required for objective='length'.")
    if objective == "both" and not shape_models_dict and not length_models_dict:
        raise ValueError(
            "At least one dataset is required for objective='both' "
            "(provide 'track_shape' and/or 'track_length')."
        )

    shape_cost = None
    length_cost = None

    if objective in ["both", "shape"] and shape_models_dict:
        shape_cost = _cost_track_shape(
            etch_model,
            shape_models_dict,
            verbose=verbose,
            residual_method=residual_method,
            residual_scale=residual_scale,
            loss=loss,
            loss_scale=loss_scale,
            aggregation=aggregation,
        )

    if objective in ["both", "length"] and length_models_dict:
        length_cost = _cost_track_length(
            etch_model,
            length_models_dict,
            verbose=verbose,
            residual_scale=residual_scale,
            loss=loss,
            loss_scale=loss_scale,
            aggregation=aggregation,
        )

    if objective == "both":
        weighted_terms = []
        if shape_cost is not None:
            weighted_terms.append((shape_weight, shape_cost))
        if length_cost is not None:
            weighted_terms.append((length_weight, length_cost))

        total_weight = float(sum(weight for weight, _ in weighted_terms))
        if total_weight <= 0:
            raise ValueError(
                "For objective='both', the sum of weights for available datasets "
                "must be > 0"
            )

        fit_cost = sum(weight * cost for weight, cost in weighted_terms) / total_weight
    elif objective == "shape":
        assert shape_cost is not None
        fit_cost = shape_cost
    else:
        assert length_cost is not None
        fit_cost = length_cost

    if verbose:
        if shape_cost is not None:
            print(f"  Shape cost: {shape_cost:.6f}")
        if length_cost is not None:
            print(f"  Length cost: {length_cost:.6f}")
        print(f"  Total cost: {fit_cost:.6f}")

    return fit_cost


def _objective_increments(
    params: np.ndarray,
    etch_model: EtchRateModel,
    models_dict: dict,
    log_v_bulk: float,
    log_v_budget: float,
    n_anchors: int,
    smoothness_weight: float = 0.0,
    residual_method: str = "vertical",
    residual_scale: str = "linear",
    loss: str = "l2",
    loss_scale: float = 1.0,
    shape_weight: float = 0.5,
    length_weight: float = 0.5,
    objective: str = "both",
    aggregation: str = "equal_ion_energy",
    fit_debris_damping: bool = False,
) -> float:
    """Objective using non-negative increment parameterization.

    Each element ``increments[i]`` is a non-negative step in log10(velocity)
    between consecutive anchors.  Monotonicity is guaranteed by construction
    (cumulative sum of non-negative values).  If the total exceeds the
    max-rate budget, all increments are proportionally rescaled.

    When ``fit_debris_damping=True`` the last two elements of *params* are
    ``[log10(debris_alpha), debris_beta]``.  They are applied to the etch
    model before cost evaluation.
    """
    try:
        params = np.asarray(params, dtype=float)

        if fit_debris_damping:
            increments = params[:n_anchors]
            log_alpha = float(params[n_anchors])
            beta = max(float(params[n_anchors + 1]), 0.1)
            etch_model.update_debris_params(10.0**log_alpha, beta)
        else:
            increments = params

        inc = np.maximum(increments, 0.0)

        # Budget projection: rescale if total exceeds allowed range
        total = float(np.sum(inc))
        if total > log_v_budget and total > 0:
            inc = inc * (log_v_budget / total)

        # Build monotonic log-velocities via cumulative sum
        log_velocities = log_v_bulk + np.cumsum(inc)
        velocities = np.power(10.0, log_velocities)

        etch_model.update_velocities(velocities)
        data_cost = cost_function(
            etch_model,
            models_dict,
            residual_method=residual_method,
            residual_scale=residual_scale,
            loss=loss,
            loss_scale=loss_scale,
            shape_weight=shape_weight,
            length_weight=length_weight,
            objective=objective,
            aggregation=aggregation,
        )
        if not np.isfinite(data_cost):
            return 1e10

        # Smoothness regularization: penalise large differences between
        # consecutive increments.  This prevents unconstrained anchors
        # (those at doses not probed by the data) from taking wild values.
        if smoothness_weight > 0 and len(inc) > 1:
            # Squared differences of consecutive increments (total-variation)
            tv = float(np.sum(np.diff(inc) ** 2))
            data_cost += smoothness_weight * tv

        return float(data_cost)
    except Exception:
        return 1e10


def _objective_direct_logv(
    params: np.ndarray,
    etch_model: EtchRateModel,
    models_dict: dict,
    log_v_bulk: float,
    log_v_max: float,
    n_free: int,
    smoothness_weight: float = 0.0,
    residual_method: str = "vertical",
    residual_scale: str = "linear",
    loss: str = "l2",
    loss_scale: float = 1.0,
    shape_weight: float = 0.5,
    length_weight: float = 0.5,
    objective: str = "both",
    aggregation: str = "equal_ion_energy",
    fit_debris_damping: bool = False,
) -> float:
    """Objective using direct log-velocity parameterization.

    The first ``n_free`` parameters are free log10(velocity) values for the
    user-defined anchors (those above ``LOW_DOSE_ANCHOR_GY``).  The
    synthetic low-dose anchor is always pinned at ``V_bulk``.  Values are
    **sorted** inside the objective to enforce monotonicity.

    When ``fit_debris_damping=True`` the last two elements of *params* are
    ``[log10(debris_alpha), debris_beta]``.
    """
    try:
        params = np.asarray(params, dtype=float)

        if fit_debris_damping:
            log_v_raw = params[:n_free]
            log_alpha = float(params[n_free])
            beta = max(float(params[n_free + 1]), 0.1)
            etch_model.update_debris_params(10.0**log_alpha, beta)
        else:
            log_v_raw = params

        # Sort to enforce monotonicity; clamp to valid range
        log_v_sorted = np.sort(log_v_raw)
        log_v_sorted = np.clip(log_v_sorted, log_v_bulk, log_v_max)

        # Prepend V_bulk for the pinned low-dose anchor
        all_log_v = np.concatenate([[log_v_bulk], log_v_sorted])
        velocities = np.power(10.0, all_log_v)

        etch_model.update_velocities(velocities)
        data_cost = cost_function(
            etch_model,
            models_dict,
            residual_method=residual_method,
            residual_scale=residual_scale,
            loss=loss,
            loss_scale=loss_scale,
            shape_weight=shape_weight,
            length_weight=length_weight,
            objective=objective,
            aggregation=aggregation,
        )
        if not np.isfinite(data_cost):
            return 1e10

        # Smoothness regularization on log-velocity increments
        if smoothness_weight > 0 and len(log_v_sorted) > 2:
            inc = np.diff(log_v_sorted)
            tv = float(np.sum(np.diff(inc) ** 2))
            data_cost += smoothness_weight * tv

        return float(data_cost)
    except Exception:
        return 1e10


def differential_evolution_optimization(
    etch_model: EtchRateModel,
    models_dict: dict,
    popsize: int = 15,
    maxiter: int = 300,
    tol: float = 1e-4,
    seed: int = 42,
    mutation: float | tuple[float, float] = (0.5, 1.4),
    recombination: float = 0.9,
    polish_method: str | None = "Powell",
    smoothness_weight: float = 0.01,
    residual_method: str = "vertical",
    residual_scale: str = "linear",
    loss: str = "l2",
    loss_scale: float = 1.0,
    max_velocity_um_h: float | None = None,
    shape_weight: float = 0.5,
    length_weight: float = 0.5,
    objective: str = "both",
    aggregation: str = "equal_ion_energy",
    fit_debris_damping: bool = False,
    debris_alpha_bounds_log10: tuple[float, float] = (0.0, 6.0),
    debris_beta_bounds: tuple[float, float] = (0.3, 4.0),
) -> dict:
    """Global optimization of V(D) and (optionally) debris damping.

    Uses a **non-negative increment** parameterization where each DE
    parameter is the step in log10(velocity) between consecutive anchors.
    Monotonicity is guaranteed by construction (cumulative sum of
    non-negative values), and the total is capped at `log10(V_max/V_bulk)`.

    When ``fit_debris_damping=True`` two extra parameters are appended:
    ``[log10(alpha), beta]``.  The upper alpha bound is large enough that
    eta ~ 1 (no damping) is always reachable, so the optimizer can never
    produce a fit worse than the debris-free solution.

    Parameters
    ----------
    etch_model : EtchRateModel
        Model to optimize (modified in place).
    models_dict : dict
        Experimental data and simulators.
    popsize : int
        Population size multiplier for DE (default: 15).
    maxiter : int
        Maximum DE generations (default: 300).
    tol : float
        Relative tolerance for convergence (default: 1e-4).
    seed : int
        Random seed (default: 42).
    mutation : float or tuple(float, float)
        DE mutation constant or dithering range (default: (0.5, 1.4)).
    recombination : float
        DE crossover probability in [0, 1] (default: 0.9).
    polish_method : str or None
        Gradient-free method for local polishing after DE.
        None disables polishing.  Default: 'Powell'.
    smoothness_weight : float
        Regularization weight penalising rough increment profiles
        (default: 0.01).
    residual_method, residual_scale, loss, loss_scale, shape_weight,
    length_weight, objective, aggregation
        Forwarded to ``cost_function``.
    max_velocity_um_h : float or None
        Hard upper cap on etch rate.  Determines bounds.
    fit_debris_damping : bool
        If True, co-optimize debris damping parameters alongside V(D).
    debris_alpha_bounds_log10 : tuple[float, float]
        Bounds on log10(alpha).  Default (0, 6) -> alpha in [1, 10^6].
    debris_beta_bounds : tuple[float, float]
        Bounds on beta.  Default (0.3, 4.0).

    Returns
    -------
    dict
        Result with keys 'x', 'fun', 'success', 'nfev', 'method',
        and optionally 'debris_alpha', 'debris_beta'.
    """
    print("=" * 60)
    print("DIFFERENTIAL EVOLUTION (direct log-velocity parameterization)")
    print(f"Residual method: {residual_method}")
    print(f"Residual scale: {residual_scale}")
    print(f"Loss: {loss} (scale={loss_scale})")
    print(f"Mutation: {mutation}")
    print(f"Recombination: {recombination}")
    if smoothness_weight > 0:
        print(f"Smoothness regularization: {smoothness_weight}")
    if fit_debris_damping:
        print("Debris damping: co-optimized")
        print(f"  log10(alpha) bounds: {debris_alpha_bounds_log10}")
        print(f"  beta bounds: {debris_beta_bounds}")
    else:
        damping_status = (
            "active (fixed)" if etch_model.debris_damping_enabled else "disabled"
        )
        print(f"Debris damping: {damping_status}")
    print("=" * 60)

    # Preserve original model state so objective-side mutations can be undone.
    original_velocities = etch_model.anchor_velocities_um_h.copy()
    original_debris_alpha = etch_model.debris_alpha
    original_debris_beta = etch_model.debris_beta

    # If not co-optimizing debris, disable it during fitting
    if not fit_debris_damping:
        saved_debris_alpha = etch_model.debris_alpha
        saved_debris_beta = etch_model.debris_beta
        etch_model.update_debris_params(alpha=None, beta=1.0)

    effective_vmax = _get_effective_max_velocity_um_h(etch_model, max_velocity_um_h)
    log_v_bulk = float(np.log10(etch_model.V_bulk_um_h))
    log_v_max = (
        float(np.log10(effective_vmax))
        if np.isfinite(effective_vmax)
        else log_v_bulk + 4.0
    )
    n_total = len(etch_model.anchor_doses_Gy)
    # The first anchor is the synthetic low-dose anchor pinned at V_bulk.
    # Only the remaining "user" anchors are free parameters.
    n_free = n_total - 1

    # Direct log-velocity parameterization: each *user* anchor gets an
    # independent log10(V) in [log_v_bulk, log_v_max].  Monotonicity is
    # enforced by sorting inside the objective.
    bounds_de: list[tuple[float, float]] = [(log_v_bulk, log_v_max)] * n_free

    if fit_debris_damping:
        bounds_de.append(debris_alpha_bounds_log10)
        bounds_de.append(debris_beta_bounds)

    if np.isfinite(effective_vmax):
        print(f"Max etch-rate cap: {effective_vmax:.6g} um/h")
    print(f"Anchor count: {n_total}  (1 pinned at V_bulk + {n_free} free)")
    print(
        f"Log-velocity range: [{log_v_bulk:.4f}, {log_v_max:.4f}] dex  "
        f"(V_bulk={etch_model.V_bulk_um_h:.4g} -> V_max={effective_vmax:.4g})"
    )
    n_params = len(bounds_de)
    print(
        f"Parameters: {n_params}  (free anchors={n_free}"
        + (", debris=2)" if fit_debris_damping else ")")
    )
    print(f"Population size: {popsize * n_params}")

    cost_kwargs = dict(
        residual_method=residual_method,
        residual_scale=residual_scale,
        loss=loss,
        loss_scale=loss_scale,
        shape_weight=shape_weight,
        length_weight=length_weight,
        objective=objective,
        aggregation=aggregation,
    )

    objective_fn = partial(
        _objective_direct_logv,
        etch_model=etch_model,
        models_dict=models_dict,
        log_v_bulk=log_v_bulk,
        log_v_max=log_v_max,
        n_free=n_free,
        smoothness_weight=smoothness_weight,
        fit_debris_damping=fit_debris_damping,
        **cost_kwargs,
    )

    # Initial log-velocities for the free (user) anchors only
    # (skip index 0 which is the pinned low-dose anchor)
    log_v_init = np.log10(etch_model.anchor_velocities_um_h[1:])
    log_v_init = np.clip(log_v_init, log_v_bulk, log_v_max)

    if fit_debris_damping:
        if etch_model.debris_damping_enabled and etch_model.debris_alpha is not None:
            init_log_alpha = float(np.log10(etch_model.debris_alpha))
        else:
            # Start at midpoint of bounds so the optimizer can sense the
            # damping effect (gradient ~ 0 at the upper bound where eta ~ 1).
            init_log_alpha = 0.5 * (
                debris_alpha_bounds_log10[0] + debris_alpha_bounds_log10[1]
            )
        init_beta = float(etch_model.debris_beta)
        init_vector = np.concatenate([log_v_init, [init_log_alpha, init_beta]])
    else:
        init_vector = log_v_init.copy()

    # Evaluate initial cost through the *same* objective that DE will use.
    # This ensures the comparison is fair (includes debris damping when
    # fit_debris_damping=True, even if the model started without it).
    initial_cost = float(objective_fn(init_vector))
    print(f"Initial cost: {initial_cost:.6e}")

    # Build initial population centered on the initial guess.
    # Graded perturbation scales ensure the population explores a useful
    # neighborhood around the initial guess rather than the full parameter
    # range (where most members would have terrible cost and contribute
    # nothing to DE mutation/crossover).
    rng = np.random.default_rng(seed)
    pop_total = popsize * n_params
    init_pop = np.empty((pop_total, n_params))

    # Perturbation tiers (fraction of population -> noise std in dex):
    #   10% tight   (sigma=0.2 dex) -- local refinement around initial guess
    #   20% medium  (sigma=0.8 dex) -- moderate exploration
    #   30% wide    (sigma=2.0 dex) -- large-scale reshaping
    #   40% uniform random       -- global diversity (latin-hypercube style)
    tier_fracs = [0.10, 0.20, 0.30, 0.40]
    tier_sigmas = [0.2, 0.8, 2.0]  # last tier is uniform random
    tier_sizes = [max(1, int(f * pop_total)) for f in tier_fracs]
    # Adjust last tier to fill remainder
    tier_sizes[-1] = pop_total - sum(tier_sizes[:-1])

    idx = 0
    for tier_i, (size, sigma) in enumerate(zip(tier_sizes[:-1], tier_sigmas)):
        for _ in range(size):
            noise = rng.normal(0, sigma, size=n_free)
            init_pop[idx, :n_free] = np.clip(log_v_init + noise, log_v_bulk, log_v_max)
            if fit_debris_damping:
                alpha_noise = rng.normal(0, sigma * 0.5)
                beta_noise = rng.normal(0, sigma * 0.3)
                init_pop[idx, n_free] = np.clip(
                    init_vector[n_free] + alpha_noise,
                    *debris_alpha_bounds_log10,
                )
                init_pop[idx, n_free + 1] = np.clip(
                    init_vector[n_free + 1] + beta_noise,
                    *debris_beta_bounds,
                )
            idx += 1

    # Last tier: uniform random for global diversity
    remaining = pop_total - idx
    init_pop[idx:, :n_free] = rng.uniform(
        log_v_bulk, log_v_max, size=(remaining, n_free)
    )
    if fit_debris_damping:
        init_pop[idx:, n_free] = rng.uniform(*debris_alpha_bounds_log10, size=remaining)
        init_pop[idx:, n_free + 1] = rng.uniform(*debris_beta_bounds, size=remaining)

    # Member 0 is always the exact initial guess
    init_pop[0] = init_vector

    if fit_debris_damping:
        # Add a short deterministic alpha sweep with fixed velocities so the
        # initial population always spans from strong damping to near-disabled
        # damping. This avoids early stagnation when random perturbations of
        # many velocity parameters dominate selection pressure.
        n_sweep = int(min(max(4, pop_total // 20), pop_total - 1))
        alpha_grid = np.linspace(
            debris_alpha_bounds_log10[0], debris_alpha_bounds_log10[1], n_sweep
        )
        fixed_beta = np.clip(init_vector[n_free + 1], *debris_beta_bounds)
        # Start at index 1 to preserve member 0 (exact initial guess)
        for j, log_alpha in enumerate(alpha_grid):
            init_pop[1 + j, :n_free] = log_v_init
            init_pop[1 + j, n_free] = log_alpha
            init_pop[1 + j, n_free + 1] = fixed_beta

    # Stagnation callback: stop early if best cost unchanged for many steps
    stagnation_limit = max(20, n_params * 3)
    _stag_state = {"best": np.inf, "count": 0}

    def _stagnation_callback(xk, convergence):
        current_best = float(objective_fn(xk))
        if abs(current_best - _stag_state["best"]) < 1e-12:
            _stag_state["count"] += 1
        else:
            _stag_state["best"] = current_best
            _stag_state["count"] = 0
        if _stag_state["count"] >= stagnation_limit:
            print(
                f"\nEarly stop: best cost unchanged for {stagnation_limit} generations."
            )
            return True  # tell DE to stop
        return False

    result = differential_evolution(
        objective_fn,
        bounds_de,
        init=init_pop,
        maxiter=maxiter,
        tol=tol,
        seed=seed,
        mutation=mutation,
        recombination=recombination,
        polish=False,
        callback=_stagnation_callback,
        disp=True,
    )

    total_nfev = int(result.nfev)
    best_params = np.asarray(result.x, dtype=float)

    if fit_debris_damping:
        best_log_v = np.sort(best_params[:n_free])
        best_log_alpha = float(best_params[n_free])
        best_beta = max(float(best_params[n_free + 1]), 0.1)
    else:
        best_log_v = np.sort(best_params)
        best_log_alpha = 0.0
        best_beta = 1.0

    best_log_v = np.clip(best_log_v, log_v_bulk, log_v_max)
    best_cost = float(result.fun)
    method_label = "differential_evolution"

    # Optional local polishing with gradient-free optimizer
    if polish_method is not None:
        print(f"\nPolishing with {polish_method} ...")
        polish_result = minimize(
            objective_fn,
            best_params,
            method=polish_method,
            bounds=list(bounds_de),
            options={"maxiter": 5000, "disp": False},
        )
        total_nfev += int(polish_result.nfev)
        polish_params = np.asarray(polish_result.x, dtype=float)
        polish_cost = float(polish_result.fun)
        if polish_cost < best_cost:
            best_params = polish_params
            if fit_debris_damping:
                best_log_v = np.sort(polish_params[:n_free])
                best_log_alpha = float(polish_params[n_free])
                best_beta = max(float(polish_params[n_free + 1]), 0.1)
            else:
                best_log_v = np.sort(polish_params)
            best_log_v = np.clip(best_log_v, log_v_bulk, log_v_max)
            best_cost = polish_cost
            method_label += f"+{polish_method}"
            print(f"Polish improved cost: {polish_cost:.6e}")
        else:
            print(f"Polish did not improve (cost {polish_cost:.6e} vs {best_cost:.6e})")

    # Accept only if at least as good as initial
    if best_cost > initial_cost:
        print("DE did not improve over initial model; keeping original.")
        best_cost = initial_cost
        etch_model.update_velocities(original_velocities)
        etch_model.update_debris_params(original_debris_alpha, original_debris_beta)
        best_log_v = np.log10(original_velocities[1:])
        if fit_debris_damping:
            if original_debris_alpha is not None:
                best_log_alpha = float(np.log10(original_debris_alpha))
            best_beta = float(original_debris_beta)
    else:
        # Prepend V_bulk for the pinned low-dose anchor
        all_log_v = np.concatenate([[log_v_bulk], best_log_v])
        velocities = np.power(10.0, all_log_v)
        etch_model.update_velocities(velocities)
        if fit_debris_damping:
            etch_model.update_debris_params(alpha=10.0**best_log_alpha, beta=best_beta)

    # Restore debris if it was disabled for fitting
    if not fit_debris_damping:
        etch_model.update_debris_params(
            alpha=saved_debris_alpha, beta=saved_debris_beta
        )

    print("\nDE complete")
    print(f"Success: {result.success}")
    print(f"Final cost: {best_cost:.6e}  (initial was {initial_cost:.6e})")
    print(f"Function evaluations: {total_nfev}")
    print(f"Final velocities: {etch_model.anchor_velocities_um_h}")
    if fit_debris_damping:
        if etch_model.debris_alpha is None:
            print(f"Debris damping: disabled (beta={etch_model.debris_beta:.4g})")
        else:
            print(
                f"Debris damping: alpha={etch_model.debris_alpha:.4g}, "
                f"beta={etch_model.debris_beta:.4g}"
            )

    result_dict: dict = {
        "x": best_log_v,
        "fun": best_cost,
        "success": result.success,
        "nfev": total_nfev,
        "method": method_label,
        "improved": bool(best_cost < initial_cost),
        "initial_cost": initial_cost,
    }
    if fit_debris_damping:
        result_dict["debris_alpha"] = etch_model.debris_alpha
        result_dict["debris_beta"] = etch_model.debris_beta
    return result_dict


def optimize_etch_model(
    etch_model: EtchRateModel,
    models_dict: dict,
    residual_method: str = "vertical",
    residual_scale: str = "log",
    loss: str = "l2",
    loss_scale: float = 1.0,
    max_velocity_um_h: float | None = None,
    shape_weight: float = 0.5,
    length_weight: float = 0.5,
    objective: str = "both",
    aggregation: str = "equal_ion_energy",
    **kwargs,
) -> dict:
    """
    High-level interface to run differential-evolution optimization.

    Parameters
    ----------
    etch_model : EtchRateModel
        Model to optimize
    models_dict : dict
        Experimental data and simulators
    residual_method : str, optional
        How to compute residuals (default: "vertical"):
        - "vertical": Compute z-difference at each experimental r position
        - "closest": Compute minimum distance from point to contour line
    residual_scale : str, optional
        Scale used to evaluate residuals (default: "log"):
        - "linear": absolute differences in um
        - "log": absolute differences in log10-space (dex)
    loss : str, optional
        Point-wise loss function (default: "l2"):
        - "l2": squared residuals
        - "soft_l1": smooth robust loss
        - "huber": piecewise quadratic/linear robust loss
    loss_scale : float, optional
        Scale parameter for robust losses (default: 1.0)
    max_velocity_um_h : float | None, optional
        Optional hard cap for etch rates during optimization.
    shape_weight : float, optional
        Weight for track-shape objective when both datasets are provided
    length_weight : float, optional
        Weight for track-length objective when both datasets are provided
    **kwargs
        Additional arguments passed to ``differential_evolution_optimization``
        (e.g. popsize, maxiter, tol, seed, mutation, recombination,
        polish_method, smoothness_weight).

    Returns
    -------
    dict
        Result dictionary from the optimization.

    Examples
    --------
    >>> result = optimize_etch_model(model, data)
    >>> result = optimize_etch_model(model, data, popsize=30, maxiter=200)
    """
    de_kwargs_allowed = {
        "popsize",
        "maxiter",
        "tol",
        "seed",
        "mutation",
        "recombination",
        "polish_method",
        "smoothness_weight",
        "fit_debris_damping",
        "debris_alpha_bounds_log10",
        "debris_beta_bounds",
    }

    start_time_s = perf_counter()

    filtered_kwargs = {
        key: value for key, value in kwargs.items() if key in de_kwargs_allowed
    }
    ignored_kwargs = sorted(set(kwargs) - de_kwargs_allowed)
    if ignored_kwargs:
        print("Ignoring unknown kwargs: " + ", ".join(ignored_kwargs))
    result = differential_evolution_optimization(
        etch_model,
        models_dict,
        residual_method=residual_method,
        residual_scale=residual_scale,
        loss=loss,
        loss_scale=loss_scale,
        max_velocity_um_h=max_velocity_um_h,
        shape_weight=shape_weight,
        length_weight=length_weight,
        objective=objective,
        aggregation=aggregation,
        **filtered_kwargs,
    )
    result["elapsed_seconds"] = float(perf_counter() - start_time_s)
    return result


# ---------------------------------------------------------------------------
# Uncertainty estimation via Hessian (curvature at the optimum)
# ---------------------------------------------------------------------------


def _count_data_points(models_dict: dict) -> int:
    """Count total experimental data points across all datasets."""
    n = 0
    for sub_dict in models_dict.values():
        for data_dict in sub_dict.values():
            n += len(data_dict["experiment_data"])
    return n


def estimate_parameter_uncertainties(
    etch_model: EtchRateModel,
    models_dict: dict,
    residual_method: str = "vertical",
    residual_scale: str = "log",
    loss: str = "soft_l1",
    loss_scale: float = 0.08,
    shape_weight: float = 0.5,
    length_weight: float = 0.5,
    objective: str = "both",
    aggregation: str = "equal_ion_energy",
    smoothness_weight: float = 0.012,
    step_size: float = 1e-2,
    max_velocity_um_h: float | None = None,
    fit_debris_damping: bool = False,
    min_relative_uncertainty_pct: float = 2.0,
) -> dict:
    """Estimate uncertainties of optimized parameters via the Hessian.

    Numerically computes the Hessian of the cost function at the current
    (optimized) parameter values using central finite differences.  The
    inverse Hessian gives a covariance matrix in log-velocity space, which
    is then propagated to velocity space via the Jacobian of the
    log-velocity -> velocity transformation.

    The covariance is scaled by ``2 * f_min / (N_data - N_params)``
    (reduced chi-squared analogue) so that uncertainties reflect both
    the curvature and the residual misfit.

    Parameters
    ----------
    etch_model : EtchRateModel
        Already-optimized model.
    models_dict : dict
        Experimental data dictionary (same as used for optimization).
    residual_method, residual_scale, loss, loss_scale, shape_weight,
    length_weight, objective, aggregation, smoothness_weight
        Must match the settings used during optimization.
    step_size : float
        Relative step for finite differences (default: 1e-2).  The step
        is applied in log10(velocity) space.  Values around 1e-2 (~2 %
        velocity change) are needed to see finite differences through the
        discrete simulation grid; 1e-4 is too small and produces a
        near-singular Hessian with near-zero uncertainties.
    max_velocity_um_h : float or None
        Same cap used during optimization.
    fit_debris_damping : bool
        Whether debris params were co-optimized.
    min_relative_uncertainty_pct : float
        Minimum uncertainty floor as a percentage of each anchor velocity
        (default: 2.0 %).  Applied after the Hessian-based estimate so that
        anchors with near-zero Hessian curvature (poorly constrained by the
        data) still carry a physically reasonable uncertainty rather than
        reporting exactly zero.

    Returns
    -------
    dict with keys:
        'velocity_uncertainties_um_h' : np.ndarray
            1-sigma uncertainties for each anchor velocity [um/hr].
        'velocity_relative_pct' : np.ndarray
            Relative uncertainties as percentages.
        'increment_covariance' : np.ndarray
            Covariance matrix in log-velocity space.
        'debris_alpha_uncertainty' : float or None
            1-sigma uncertainty on alpha (if co-optimized).
        'debris_beta_uncertainty' : float or None
            1-sigma uncertainty on beta (if co-optimized).
        'hessian' : np.ndarray
            Raw Hessian matrix.
        'condition_number' : float
            Condition number of the Hessian (large = ill-conditioned).
        'n_data' : int
            Number of experimental data points.
        'n_params' : int
            Number of fitted parameters.
        'scale_factor' : float
            Variance scale factor applied.
    """
    print("=" * 60)
    print("UNCERTAINTY ESTIMATION (Hessian at optimum)")
    print("=" * 60)

    effective_vmax = _get_effective_max_velocity_um_h(etch_model, max_velocity_um_h)
    log_v_bulk = float(np.log10(etch_model.V_bulk_um_h))
    log_v_max = (
        float(np.log10(effective_vmax))
        if np.isfinite(effective_vmax)
        else log_v_bulk + 4.0
    )
    n_total = len(etch_model.anchor_doses_Gy)
    n_free = n_total - 1  # exclude pinned low-dose anchor

    # Current optimal log-velocities for free anchors only (skip index 0)
    log_v_opt = np.log10(etch_model.anchor_velocities_um_h[1:])

    if fit_debris_damping and etch_model.debris_damping_enabled:
        opt_log_alpha = float(np.log10(etch_model.debris_alpha))
        opt_beta = float(etch_model.debris_beta)
        x0 = np.concatenate([log_v_opt, [opt_log_alpha, opt_beta]])
    else:
        x0 = log_v_opt.copy()

    n_params = len(x0)

    cost_kwargs = dict(
        residual_method=residual_method,
        residual_scale=residual_scale,
        loss=loss,
        loss_scale=loss_scale,
        shape_weight=shape_weight,
        length_weight=length_weight,
        objective=objective,
        aggregation=aggregation,
    )

    objective_fn = partial(
        _objective_direct_logv,
        etch_model=etch_model,
        models_dict=models_dict,
        log_v_bulk=log_v_bulk,
        log_v_max=log_v_max,
        n_free=n_free,
        smoothness_weight=smoothness_weight,
        fit_debris_damping=fit_debris_damping,
        **cost_kwargs,
    )

    f0 = objective_fn(x0)
    print(f"Cost at optimum: {f0:.6e}")
    print(f"Parameters: {n_params}")

    # --- Compute Hessian via central finite differences ---
    hessian = np.empty((n_params, n_params))
    for i in range(n_params):
        for j in range(i, n_params):
            hi = max(step_size * abs(x0[i]), step_size * 0.01)
            hj = max(step_size * abs(x0[j]), step_size * 0.01)

            x_pp = x0.copy()
            x_pp[i] += hi
            x_pp[j] += hj

            x_pm = x0.copy()
            x_pm[i] += hi
            x_pm[j] -= hj

            x_mp = x0.copy()
            x_mp[i] -= hi
            x_mp[j] += hj

            x_mm = x0.copy()
            x_mm[i] -= hi
            x_mm[j] -= hj

            hessian[i, j] = (
                objective_fn(x_pp)
                - objective_fn(x_pm)
                - objective_fn(x_mp)
                + objective_fn(x_mm)
            ) / (4.0 * hi * hj)
            hessian[j, i] = hessian[i, j]

    n_evals = 4 * n_params * (n_params + 1) // 2
    print(f"Hessian computed ({n_evals} function evaluations)")

    cond = float(np.linalg.cond(hessian))
    print(f"Hessian condition number: {cond:.2e}")

    # --- Invert Hessian to get covariance ---
    try:
        cov_inc = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        print("WARNING: Hessian is singular -- using pseudo-inverse")
        cov_inc = np.linalg.pinv(hessian)

    # Scale factor: 2 * f_min / (N_data - N_params)
    n_data = _count_data_points(models_dict)
    dof = max(n_data - n_params, 1)
    scale = 2.0 * f0 / dof
    cov_inc_scaled = cov_inc * scale
    print(f"Data points: {n_data}, DOF: {dof}, scale factor: {scale:.4e}")

    # --- Propagate to velocity space ---
    # Direct parameterization: v_i = 10^(x_i), so dv_i/dx_j = v_i * ln(10) * delta_ij
    # The Jacobian maps n_free optimized params to n_total velocities.
    # Index 0 (pinned low-dose anchor) has zero uncertainty.
    all_velocities = etch_model.anchor_velocities_um_h.copy()
    ln10 = np.log(10.0)
    J = np.zeros((n_total, n_params))
    for i in range(n_free):
        # param i maps to velocity i+1 (index 0 is pinned)
        J[i + 1, i] = all_velocities[i + 1] * ln10

    cov_vel = J @ cov_inc_scaled @ J.T

    # Ensure non-negative variances (numerical protection)
    var_vel = np.maximum(np.diag(cov_vel), 0.0)
    sigma_vel = np.sqrt(var_vel)

    # Apply minimum relative uncertainty floor.  Anchors not well-constrained
    # by the data produce a near-zero Hessian curvature; the resulting
    # near-zero covariance entries are physically wrong (those parameters are
    # poorly known, not perfectly known).  A floor of a few percent ensures
    # that downstream radius/depth uncertainty propagation is realistic.
    if min_relative_uncertainty_pct > 0.0:
        sigma_vel_floor = (min_relative_uncertainty_pct / 100.0) * all_velocities
        sigma_vel = np.maximum(sigma_vel, sigma_vel_floor)

    relative_pct = 100.0 * sigma_vel / np.maximum(all_velocities, 1e-30)

    print("\nVelocity uncertainties:")
    print(
        f"{'Dose [Gy]':>14s}  {'V [um/hr]':>12s}  {'sigma_V [um/hr]':>12s}  {'sigma_V [%]':>8s}"
    )
    for i in range(n_total):
        print(
            f"{etch_model.anchor_doses_Gy[i]:14.4e}  "
            f"{all_velocities[i]:12.4e}  "
            f"{sigma_vel[i]:12.4e}  "
            f"{relative_pct[i]:8.2f}"
        )

    result: dict = {
        "velocity_uncertainties_um_h": sigma_vel,
        "velocity_relative_pct": relative_pct,
        "increment_covariance": cov_inc_scaled[:n_free, :n_free],
        "hessian": hessian,
        "condition_number": cond,
        "n_data": n_data,
        "n_params": n_params,
        "scale_factor": scale,
    }

    if fit_debris_damping and n_params > n_free:
        var_log_alpha = max(cov_inc_scaled[n_free, n_free], 0.0)
        var_beta = max(cov_inc_scaled[n_free + 1, n_free + 1], 0.0)
        # Propagate log10(alpha) uncertainty to alpha
        sigma_log_alpha = np.sqrt(var_log_alpha)
        alpha = 10.0 ** x0[n_free]
        sigma_alpha = alpha * ln10 * sigma_log_alpha
        sigma_beta = np.sqrt(var_beta)
        result["debris_alpha_uncertainty"] = float(sigma_alpha)
        result["debris_beta_uncertainty"] = float(sigma_beta)
        print(f"\nDebris alpha: {alpha:.4g} +/- {sigma_alpha:.4g}")
        print(f"Debris beta:  {x0[n_free + 1]:.4g} +/- {sigma_beta:.4g}")
    else:
        result["debris_alpha_uncertainty"] = None
        result["debris_beta_uncertainty"] = None

    # Restore the model to its optimal state (FD perturbations modified it)
    etch_model.update_velocities(all_velocities)
    if fit_debris_damping and etch_model.debris_damping_enabled:
        etch_model.update_debris_params(
            10.0 ** float(x0[n_free]), float(x0[n_free + 1])
        )

    print("=" * 60)
    return result
