"""
Etch rate modeling for track etching simulations.

Implements monotonic etch rate functions V(dose) over wide dose ranges.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class EtchRateModel:
    LOW_DOSE_ANCHOR_GY: float = 1e-3

    def __init__(
        self,
        anchor_doses_Gy: np.ndarray | list[float] = np.logspace(1, 10, 6),
        V_bulk_um_h: float = 1.73,
        V_max_um_h: float | None = None,
        V_max_uncertainty_um_h: float | None = None,
        anchor_velocities_um_h: np.ndarray | list[float] | None = None,
        anchor_velocity_uncertainties_um_h: np.ndarray | list[float] | None = None,
        name: str | None = None,
        extrapolation_mode: str = "pchip",
        debris_alpha: float | None = None,
        debris_beta: float = 1.0,
    ) -> None:
        """
        Monotonically-increasing etch rate model V(Dose).

        Models etch rate over wide dose ranges (e.g., 1e-8 to 1e10 Gy).
        Automatically adds a low-dose anchor at 1e-3 Gy with V = V_bulk_um_h
        to ensure proper behavior at very low doses. For doses < 1e-3 Gy,
        the model returns V_bulk_um_h (constant extrapolation).

        Monotonicity is guaranteed by storing relative velocity increments.
        All anchor velocities must be >= V_bulk_um_h, and velocities are
        computed as cumulative sums of positive increments.

        Parameters
        ----------
        anchor_doses_Gy : array-like
            Dose anchor points in log-space [Gy]. Must be strictly increasing.
            These are the dose points where velocities are fit. Recommendation:
            6-8 points log-spaced across the main variation region (e.g., 1e2 to 1e9).
            Example: np.logspace(2, 9, 8)
            Note: A low-dose anchor at 1e-3 Gy will be automatically prepended.
        V_bulk_um_h : float, optional
            Etch rate at very low doses [um/hour] (default: 1.73)
        V_max_um_h : float | None, optional
            Optional hard physical upper bound for etch rate [um/hour].
            If provided, all anchors and evaluated values are capped at this value.
        anchor_velocities_um_h : array-like, optional
            Absolute etch rates at anchor points [um/hour].
            Must match length of anchor_doses_Gy and be monotonically increasing.
            If None, generates smooth increase from V_bulk_um_h to 5*V_bulk_um_h.
            Will be converted internally to relative increments for optimization.
            Note: A low-dose velocity V_bulk_um_h will be automatically prepended.
        name : str, optional
            Name of the model (for identification/plotting)
        anchor_velocity_uncertainties_um_h : array-like, optional
            1-sigma uncertainty for each anchor velocity [um/hour].
            Must match length of anchor_doses_Gy before low-dose filtering.

        Attributes
        ----------
        V_bulk_um_h : float
            Low-dose etch rate [um/hour]
        anchor_doses_Gy : np.ndarray
            Dose anchor points [Gy] (includes automatic 1e-3 Gy prepend)
        anchor_velocities_um_h : np.ndarray
            Absolute velocity anchor points [um/hour] (computed from increments)
        velocity_increments_um_h : np.ndarray
            Relative velocity increments [um/hour] (stored for monotonicity)
        spline : PchipInterpolator
            Interpolator for dose-etch rate relationship
        name : str | None
            Model name

        Raises
        ------
        ValueError
            If anchor_velocities_um_h is not monotonically increasing
            If anchor_velocities_um_h[0] < V_bulk_um_h
            If lengths don't match
            If anchor_doses_Gy is not strictly increasing

        Examples
        --------
        >>> doses = np.logspace(2, 9, 8)  # 1e2 to 1e9 Gy
        >>> rates = np.array([1.73, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0])
        >>> model = EtchRateModel(
        ...     V_bulk_um_h=1.73,
        ...     anchor_doses_Gy=doses, anchor_velocities_um_h=rates
        ... )
        >>> V = model.eval(1e5)  # Evaluate at 1e5 Gy
        >>> model.plot()
        """

        self.V_bulk_um_h: float = float(V_bulk_um_h)
        self.V_max_um_h: float = (
            float(V_max_um_h) if V_max_um_h is not None else float("inf")
        )
        self.V_max_uncertainty_um_h: float | None = (
            float(V_max_uncertainty_um_h)
            if V_max_uncertainty_um_h is not None
            else None
        )
        self.name: str | None = name
        self.extrapolation_mode: str = extrapolation_mode

        # Debris / diffusion-limitation damping parameters.
        # debris_alpha: characteristic aspect-ratio at which damping sets in.
        #   None means damping is disabled.
        # debris_beta: steepness of the sigmoidal transition (>0).
        self.debris_alpha: float | None = (
            float(debris_alpha) if debris_alpha is not None else None
        )
        self.debris_beta: float = float(debris_beta)

        if self.V_max_um_h < self.V_bulk_um_h:
            raise ValueError(
                "V_max_um_h must be >= V_bulk_um_h. "
                f"Got V_max_um_h={self.V_max_um_h:.6g}, V_bulk_um_h={self.V_bulk_um_h:.6g}."
            )

        valid_modes = {"pchip", "clamp_last"}
        if self.extrapolation_mode not in valid_modes:
            raise ValueError(
                f"extrapolation_mode must be one of {sorted(valid_modes)}, "
                f"got '{self.extrapolation_mode}'"
            )

        # Convert input doses to array
        user_doses = np.asarray(anchor_doses_Gy, dtype=float)
        if user_doses.size == 0:
            raise ValueError("anchor_doses_Gy must not be empty")

        if np.any(user_doses <= 0):
            raise ValueError("anchor_doses_Gy must contain only positive values")

        # Drop anchors below low-dose anchor, then prepend low-dose anchor if missing
        keep_mask = user_doses >= self.LOW_DOSE_ANCHOR_GY
        filtered_user_doses = user_doses[keep_mask]

        if filtered_user_doses.size == 0:
            self.anchor_doses_Gy = np.array([self.LOW_DOSE_ANCHOR_GY], dtype=float)
        elif np.min(filtered_user_doses) > self.LOW_DOSE_ANCHOR_GY:
            self.anchor_doses_Gy = np.concatenate(
                [[self.LOW_DOSE_ANCHOR_GY], filtered_user_doses]
            )
        else:
            self.anchor_doses_Gy = filtered_user_doses

        if anchor_velocities_um_h is None:
            # Default: smooth increase from V_bulk_um_h to 5*V_bulk_um_h
            n_anchors: int = len(self.anchor_doses_Gy)
            self.anchor_velocities_um_h: np.ndarray = np.linspace(
                self.V_bulk_um_h, self.V_bulk_um_h * 5, n_anchors
            )
        else:
            user_velocities = np.asarray(anchor_velocities_um_h, dtype=float)
            if len(user_velocities) != len(user_doses):
                raise ValueError(
                    "anchor_velocities_um_h must have the same length as "
                    "anchor_doses_Gy before low-dose filtering"
                )

            filtered_user_velocities = user_velocities[keep_mask]
            # Prepend V_bulk_um_h if we prepended a dose anchor
            if len(self.anchor_doses_Gy) == len(filtered_user_velocities) + 1:
                self.anchor_velocities_um_h = np.concatenate(
                    [[self.V_bulk_um_h], filtered_user_velocities]
                )
            else:
                self.anchor_velocities_um_h = filtered_user_velocities

        # Validate and convert to relative increments
        self._validate_and_convert_to_increments()

        # Optional uncertainties at anchors
        if anchor_velocity_uncertainties_um_h is None:
            self.anchor_velocity_uncertainties_um_h: np.ndarray | None = None
        else:
            user_unc = np.asarray(anchor_velocity_uncertainties_um_h, dtype=float)
            if len(user_unc) != len(user_doses):
                raise ValueError(
                    "anchor_velocity_uncertainties_um_h must have the same length as "
                    "anchor_doses_Gy before low-dose filtering"
                )
            filtered_unc = user_unc[keep_mask]
            # Prepend 0 uncertainty for the synthetic low-dose anchor if inserted
            if len(self.anchor_doses_Gy) == len(filtered_unc) + 1:
                self.anchor_velocity_uncertainties_um_h = np.concatenate(
                    [[0.0], filtered_unc]
                )
            else:
                self.anchor_velocity_uncertainties_um_h = filtered_unc

            if np.any(self.anchor_velocity_uncertainties_um_h < 0):
                raise ValueError("anchor_velocity_uncertainties_um_h must be >= 0")

        self._create_spline()

    def __repr__(self) -> str:
        dose_min = np.min(self.anchor_doses_Gy)
        dose_max = np.max(self.anchor_doses_Gy)
        vel_min = np.min(self.anchor_velocities_um_h)
        vel_max = np.max(self.anchor_velocities_um_h)
        debris_str = (
            f"\n\tdebris_alpha={self.debris_alpha}, debris_beta={self.debris_beta:.3g},"
            if self.debris_alpha is not None
            else "\n\tdebris_damping=disabled,"
        )
        return (
            f"EtchRateModel(\n\tV_bulk_um_h={self.V_bulk_um_h:.4f}, \n"
            f"\tV_max_um_h={self.V_max_um_h:.4g}, \n"
            f"\tn_anchors={len(self.anchor_doses_Gy)}, \n"
            f"\tdose_range=[{dose_min:.2e}, {dose_max:.2e}] Gy, \n"
            f"\tvelocity_range=[{vel_min:.3g}, {vel_max:.3g}] um/hr, \n"
            f"\textrapolation_mode='{self.extrapolation_mode}',{debris_str}\n"
            f"\tname='{self.name}')"
        )

    def summary(self) -> str:
        """Return a compact human-readable summary of model settings and ranges."""
        monotonic_ok = self.is_monotonic(num_points=512)
        return (
            "EtchRateModel Summary\n"
            f"- name: {self.name}\n"
            f"- V_bulk_um_h: {self.V_bulk_um_h:.6g}\n"
            f"- V_max_um_h: {self.V_max_um_h:.6g}\n"
            f"- n_anchors: {len(self.anchor_doses_Gy)}\n"
            f"- dose range [Gy]: {np.min(self.anchor_doses_Gy):.2e} -> {np.max(self.anchor_doses_Gy):.2e}\n"
            f"- velocity range [um/hr]: {np.min(self.anchor_velocities_um_h):.6g} -> {np.max(self.anchor_velocities_um_h):.6g}\n"
            f"- extrapolation_mode: {self.extrapolation_mode}\n"
            f"- monotonic on dense grid: {monotonic_ok}"
        )

    def update_debris_params(
        self, alpha: float | None, beta: float | None = None
    ) -> None:
        """Update debris-damping parameters.

        Parameters
        ----------
        alpha : float or None
            Characteristic aspect-ratio scale.  None disables damping.
        beta : float or None
            Transition steepness.  None keeps current value.
        """
        self.debris_alpha = float(alpha) if alpha is not None else None
        if beta is not None:
            self.debris_beta = float(beta)

    @property
    def debris_damping_enabled(self) -> bool:
        """Whether debris damping is active."""
        return self.debris_alpha is not None and self.debris_alpha > 0

    def is_monotonic(self, num_points: int = 256) -> bool:
        """Check monotonic non-decreasing behavior of v(d) on a dense log-dose grid."""
        dose_grid = np.logspace(
            np.log10(self.LOW_DOSE_ANCHOR_GY),
            np.log10(np.max(self.anchor_doses_Gy)),
            num_points,
        )
        values = np.asarray(self.eval(dose_grid), dtype=float)
        return bool(np.all(np.diff(values) >= -1e-12))

    def _validate_and_convert_to_increments(self) -> None:
        """Validate absolute velocities and convert to relative increments.

        Stores velocity_increments_um_h internally to guarantee monotonicity.
        All increments are guaranteed to be >= 0.
        Ensures all velocities >= V_bulk_um_h (physically consistent).
        """
        if len(self.anchor_doses_Gy) != len(self.anchor_velocities_um_h):
            raise ValueError(
                f"anchor_doses_Gy and anchor_velocities_um_h must have same length. "
                f"Got {len(self.anchor_doses_Gy)} and {len(self.anchor_velocities_um_h)}"
            )

        if not np.all(np.diff(self.anchor_doses_Gy) > 0):
            raise ValueError("anchor_doses_Gy must be strictly increasing")

        if not np.all(np.diff(self.anchor_velocities_um_h) >= 0):
            raise ValueError("anchor_velocities_um_h must be monotonically increasing")

        lower_tol = max(1e-12, 1e-12 * max(1.0, abs(self.V_bulk_um_h)))
        upper_tol = max(1e-12, 1e-12 * max(1.0, abs(self.V_max_um_h)))

        if np.any(self.anchor_velocities_um_h < self.V_bulk_um_h - lower_tol):
            min_vel = np.min(self.anchor_velocities_um_h)
            raise ValueError(
                f"All anchor velocities must be >= V_bulk_um_h ({self.V_bulk_um_h:.4f}). "
                f"Got minimum velocity = {min_vel:.4f}"
            )

        if np.any(self.anchor_velocities_um_h > self.V_max_um_h + upper_tol):
            max_vel = np.max(self.anchor_velocities_um_h)
            raise ValueError(
                f"All anchor velocities must be <= V_max_um_h ({self.V_max_um_h:.4f}). "
                f"Got maximum velocity = {max_vel:.4f}"
            )

        # Clip tiny boundary violations caused by floating-point roundoff
        self.anchor_velocities_um_h = np.clip(
            self.anchor_velocities_um_h,
            self.V_bulk_um_h,
            self.V_max_um_h,
        )

        # Convert absolute velocities to relative increments
        # First increment is from V_bulk_um_h to first anchor
        increments: list[float] = [
            float(self.anchor_velocities_um_h[0] - self.V_bulk_um_h)
        ]
        # Subsequent increments are differences between consecutive anchors
        for i in range(1, len(self.anchor_velocities_um_h)):
            increment = float(
                self.anchor_velocities_um_h[i] - self.anchor_velocities_um_h[i - 1]
            )
            increments.append(increment)

        self.velocity_increments_um_h: np.ndarray = np.array(increments, dtype=float)

    def _create_spline(self) -> None:
        """Create/update the PCHIP interpolation spline in log-dose space.

        Virtual padding anchors at V_bulk are prepended below the lowest
        stored anchor so that PCHIP produces a near-zero derivative at the
        low-dose boundary, ensuring smooth (C^1) convergence to V_bulk
        instead of a slope discontinuity at the cutoff.
        """
        log_doses: np.ndarray = np.log10(self.anchor_doses_Gy)

        # Pad with flat V_bulk anchors several decades below the lowest
        # stored anchor so PCHIP sees a flat region and computes near-zero
        # slope at the boundary, removing the derivative discontinuity.
        n_pad = 3
        lowest_log_dose = float(log_doses[0])
        pad_log_doses = np.linspace(lowest_log_dose - 4, lowest_log_dose - 1, n_pad)
        pad_velocities = np.full(n_pad, self.V_bulk_um_h)

        spline_log_doses = np.concatenate([pad_log_doses, log_doses])
        spline_velocities = np.concatenate(
            [pad_velocities, self.anchor_velocities_um_h]
        )

        self.spline: PchipInterpolator = PchipInterpolator(
            spline_log_doses, spline_velocities
        )

    def eval(self, dose_Gy: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate etch rate at given dose(s).

        Parameters
        ----------
        dose_Gy : float or array-like
            Dose in Gy.

        Returns
        -------
        V_um_h : float or ndarray
            Etch rate in um/hour
            Returns scalar if input is scalar, array if input is array
            Returns V_bulk_um_h for NaN inputs
            Always >= V_bulk_um_h (physically consistent)
        """
        dose_array: np.ndarray = np.atleast_1d(dose_Gy).astype(float)
        result: np.ndarray = np.zeros_like(dose_array)

        # Very low doses (< 1e-3 Gy) -> constant V_bulk_um_h
        very_low_mask: np.ndarray = dose_array < self.LOW_DOSE_ANCHOR_GY
        result[very_low_mask] = self.V_bulk_um_h

        # Normal doses (>= 1e-3 Gy) -> interpolation + high-dose extrapolation
        normal_mask: np.ndarray = dose_array >= self.LOW_DOSE_ANCHOR_GY
        if np.any(normal_mask):
            doses_normal = dose_array[normal_mask]

            if self.extrapolation_mode == "clamp_last":
                clamped_doses = np.minimum(doses_normal, np.max(self.anchor_doses_Gy))
                log_dose_normal: np.ndarray = np.log10(clamped_doses)
                result[normal_mask] = self.spline(log_dose_normal)
            else:
                anchor_max = float(np.max(self.anchor_doses_Gy))
                log_anchor_doses = np.log10(self.anchor_doses_Gy)

                result_normal = np.zeros_like(doses_normal)

                in_range_mask = doses_normal <= anchor_max
                if np.any(in_range_mask):
                    result_normal[in_range_mask] = self.spline(
                        np.log10(doses_normal[in_range_mask])
                    )

                high_mask = doses_normal > anchor_max
                if np.any(high_mask):
                    v_last = float(self.anchor_velocities_um_h[-1])
                    if len(self.anchor_velocities_um_h) >= 2:
                        dv = float(
                            self.anchor_velocities_um_h[-1]
                            - self.anchor_velocities_um_h[-2]
                        )
                        dlog = float(log_anchor_doses[-1] - log_anchor_doses[-2])
                        slope = dv / dlog if dlog > 0 else 0.0
                    else:
                        slope = 0.0

                    # Guard monotonicity: never allow decreasing high-dose extension
                    slope = max(0.0, slope)
                    result_normal[high_mask] = v_last + slope * (
                        np.log10(doses_normal[high_mask]) - float(log_anchor_doses[-1])
                    )

                result[normal_mask] = result_normal

        # Handle NaN input -> return V_bulk_um_h
        nan_mask: np.ndarray = np.isnan(dose_array)
        if np.any(nan_mask):
            result[nan_mask] = self.V_bulk_um_h

        # Ensure no value goes below V_bulk_um_h (physical constraint)
        result = np.maximum(result, self.V_bulk_um_h)
        # Optional hard cap on physically plausible etch rate
        result = np.minimum(result, self.V_max_um_h)

        # Return scalar if input was scalar
        return float(result[0]) if np.isscalar(dose_Gy) else result

    def eval_uncertainty_band(
        self,
        dose_Gy: np.ndarray,
        n_sigma: float = 1.0,
        max_sigma_log10: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return lower and upper 1-sigma etch-rate bands at the given doses.

        Converts per-anchor velocity uncertainties to log10-velocity space,
        interpolates them onto ``dose_Gy`` with PCHIP (in log-dose space), and
        applies the band symmetrically around ``eval(dose_Gy)`` in log10 space.
        Results are clipped to ``[V_bulk, V_max]``.

        Parameters
        ----------
        dose_Gy : array-like
            Dose values at which to evaluate the band.
        n_sigma : float
            Number of standard deviations for the band width (default 1).
        max_sigma_log10 : float
            Cap on interpolated sigma in log10-velocity units (default 0.5,
            i.e. ~factor-of-3 per sigma) to suppress artefacts from
            poorly-constrained high-dose anchors.

        Returns
        -------
        tuple of (lower, upper) ndarrays, or None if no anchor uncertainties
        are set on the model.
        """
        if self.anchor_velocity_uncertainties_um_h is None:
            return None
        if len(self.anchor_velocity_uncertainties_um_h) != len(
            self.anchor_velocities_um_h
        ):
            return None

        from scipy.interpolate import PchipInterpolator as _Pchip

        dose_arr = np.asarray(dose_Gy, dtype=float)
        v_anchors = np.asarray(self.anchor_velocities_um_h, dtype=float)
        sigma_v = np.asarray(self.anchor_velocity_uncertainties_um_h, dtype=float)

        sigma_log10_v = sigma_v / np.maximum(v_anchors * np.log(10.0), 1e-30)
        sigma_log10_v = np.minimum(sigma_log10_v, max_sigma_log10)

        log10_d_anchors = np.log10(np.maximum(self.anchor_doses_Gy, 1e-30))
        log10_d_plot = np.log10(np.maximum(dose_arr, 1e-30))
        sigma_interp = np.clip(
            _Pchip(log10_d_anchors, sigma_log10_v, extrapolate=True)(log10_d_plot),
            0.0,
            max_sigma_log10,
        )
        sigma_interp *= n_sigma

        mean_v = np.asarray(self.eval(dose_arr), dtype=float)
        log10_mean = np.log10(np.maximum(mean_v, 1e-30))
        lower = np.clip(
            np.power(10.0, log10_mean - sigma_interp),
            self.V_bulk_um_h,
            self.V_max_um_h,
        )
        upper = np.clip(
            np.power(10.0, log10_mean + sigma_interp),
            self.V_bulk_um_h,
            self.V_max_um_h,
        )
        return lower, upper

    def update_velocities(self, new_velocities_um_h: np.ndarray | list[float]) -> None:
        """Update anchor velocities and recreate spline.

        Accepts absolute velocities, validates them for monotonicity,
        and converts to relative increments internally.

        Useful during optimization loops.

        Parameters
        ----------
        new_velocities_um_h : array-like
            New absolute velocity values for anchor points [um/hour]

        Raises
        ------
        ValueError
            If new velocities violate monotonicity
        """
        self.anchor_velocities_um_h = np.asarray(new_velocities_um_h, dtype=float)
        self._validate_and_convert_to_increments()
        self._create_spline()

    def get_increments(self) -> np.ndarray:
        """Get relative velocity increments (for optimization).

        Returns
        -------
        increments : np.ndarray
            Relative increments [um/hour], all >= 0
        """
        return self.velocity_increments_um_h.copy()

    def get_log_velocity_increments(self) -> np.ndarray:
        """Get monotonic increments in log10(velocity) space.

        Defines:
            u_i = log10(v_i)
            delta_u_0 = u_0 - log10(V_bulk_um_h)
            delta_u_i = u_i - u_{i-1} for i >= 1

        Since anchor velocities are monotonic and >= V_bulk_um_h,
        all returned increments are guaranteed to be >= 0.

        Returns
        -------
        np.ndarray
            Non-negative increments in log10(velocity) space.
        """
        log_velocities = np.log10(np.asarray(self.anchor_velocities_um_h, dtype=float))
        base_log_velocity = np.log10(float(self.V_bulk_um_h))

        log_velocity_increments = np.empty_like(log_velocities)
        log_velocity_increments[0] = log_velocities[0] - base_log_velocity
        if len(log_velocities) > 1:
            log_velocity_increments[1:] = np.diff(log_velocities)

        return np.maximum(log_velocity_increments, 0.0)

    def get_log_doses(self) -> np.ndarray:
        """Get log10 of dose anchors (for optimization in log-space).

        Returns
        -------
        log_doses : np.ndarray
            log10 of anchor doses [dimensionless]
        """
        return np.log10(self.anchor_doses_Gy.copy())

    def update_dose_anchors(self, anchor_doses_Gy: np.ndarray | list[float]) -> None:
        """Update dose anchor positions and recreate spline.

        Validates that doses are strictly increasing.

        Parameters
        ----------
        anchor_doses_Gy : array-like
            New dose anchor points [Gy]. Must be strictly increasing.

        Raises
        ------
        ValueError
            If doses not strictly increasing or length doesn't match velocities
        """
        new_doses = np.asarray(anchor_doses_Gy, dtype=float)

        if len(new_doses) != len(self.anchor_velocities_um_h):
            raise ValueError(
                f"Length mismatch: got {len(new_doses)} doses but "
                f"expected {len(self.anchor_velocities_um_h)} (matching velocities)"
            )

        if not np.all(np.diff(new_doses) > 0):
            raise ValueError("anchor_doses_Gy must be strictly increasing")

        self.anchor_doses_Gy = new_doses
        self._create_spline()

    def update_from_increments(self, increments_um_h: np.ndarray | list[float]) -> None:
        """Update velocities from relative increments (for optimization).

        This is the recommended approach for optimization: optimize the increments
        directly to guarantee monotonicity by construction.

        Parameters
        ----------
        increments_um_h : array-like
            Relative increments [um/hour]. Must be all >= 0 and have same length
            as anchor points.

        Raises
        ------
        ValueError
            If increments are negative or length doesn't match
            If resulting velocities would be < V_bulk_um_h
        """
        increments = np.asarray(increments_um_h, dtype=float)

        if len(increments) != len(self.anchor_doses_Gy):
            raise ValueError(
                f"Length mismatch: got {len(increments)} increments but "
                f"expected {len(self.anchor_doses_Gy)} (matching anchor_doses_Gy)"
            )

        if not np.all(increments >= 0):
            raise ValueError(
                "All increments must be >= 0 to guarantee monotonicity. "
                f"Got min increment = {np.min(increments):.6f}"
            )

        # Convert increments back to absolute velocities
        velocities: list[float] = [self.V_bulk_um_h + float(increments[0])]
        for i in range(1, len(increments)):
            velocities.append(velocities[-1] + float(increments[i]))

        velocities_array = np.array(velocities, dtype=float)

        # Ensure all velocities are >= V_bulk_um_h
        if np.any(velocities_array < self.V_bulk_um_h):
            min_vel = np.min(velocities_array)
            raise ValueError(
                f"All resulting velocities must be >= V_bulk_um_h ({self.V_bulk_um_h:.4f}). "
                f"Got minimum velocity = {min_vel:.4f}"
            )

        self.anchor_velocities_um_h = velocities_array
        self.velocity_increments_um_h = increments.copy()
        self._create_spline()

    def update_from_log_velocity_increments(
        self, log_velocity_increments: np.ndarray | list[float]
    ) -> None:
        """Update anchor velocities from non-negative log10(velocity) increments.

        This parameterization is robust for wide dynamic ranges because optimization
        steps act multiplicatively on velocity.

        Parameters
        ----------
        log_velocity_increments : array-like
            Increments in log10(velocity) space. Must all be >= 0 and have the
            same length as anchor points.
        """
        increments = np.asarray(log_velocity_increments, dtype=float)

        if len(increments) != len(self.anchor_doses_Gy):
            raise ValueError(
                f"Length mismatch: got {len(increments)} log-velocity increments but "
                f"expected {len(self.anchor_doses_Gy)} (matching anchor_doses_Gy)"
            )

        if not np.all(increments >= 0):
            raise ValueError(
                "All log-velocity increments must be >= 0 to guarantee monotonicity. "
                f"Got min increment = {np.min(increments):.6f}"
            )

        base_log_velocity = np.log10(float(self.V_bulk_um_h))
        log_velocities = np.empty_like(increments)
        log_velocities[0] = base_log_velocity + float(increments[0])
        if len(increments) > 1:
            log_velocities[1:] = log_velocities[0] + np.cumsum(increments[1:])

        velocities = np.power(10.0, log_velocities)
        self.update_velocities(velocities)

    @staticmethod
    def log_increments_to_linear(
        log_increments: np.ndarray | list[float], epsilon: float = 1e-6
    ) -> np.ndarray:
        """Convert log-space increments to linear (for unconstrained optimization).

        For optimization, parameterize as: log_increments = log(increments + epsilon)
        This ensures increments are always positive without constraints.

        Parameters
        ----------
        log_increments : array-like
            Log-transformed increments (can be any real number)
        epsilon : float, optional
            Small offset to ensure positivity (default: 1e-6)

        Returns
        -------
        increments : np.ndarray
            Linear increments, all > epsilon
        """
        return np.exp(np.asarray(log_increments, dtype=float)) - epsilon

    @staticmethod
    def linear_increments_to_log(
        increments: np.ndarray | list[float], epsilon: float = 1e-6
    ) -> np.ndarray:
        """Convert linear increments to log-space (for unconstrained optimization).

        Inverse of log_increments_to_linear.

        Parameters
        ----------
        increments : array-like
            Linear increments (must all be > -epsilon)
        epsilon : float, optional
            Small offset used during forward conversion (default: 1e-6)

        Returns
        -------
        log_increments : np.ndarray
            Log-transformed increments
        """
        increments_arr = np.asarray(increments, dtype=float)
        if np.any(increments_arr <= -epsilon):
            min_inc = np.min(increments_arr)
            raise ValueError(
                f"increments must satisfy increments > -epsilon ({-epsilon:.2e}) for log transform. "
                f"Got minimum increment = {min_inc:.6g}"
            )
        return np.log(increments_arr + epsilon)

    @staticmethod
    def log_velocity_increments_to_linear(
        optimization_parameters: np.ndarray | list[float], epsilon: float = 1e-12
    ) -> np.ndarray:
        """Map unconstrained optimizer parameters to positive log-velocity increments."""
        return np.exp(np.asarray(optimization_parameters, dtype=float)) - epsilon

    @staticmethod
    def linear_log_velocity_increments_to_optimization(
        log_velocity_increments: np.ndarray | list[float], epsilon: float = 1e-12
    ) -> np.ndarray:
        """Map positive log-velocity increments to unconstrained optimizer parameters."""
        increments_arr = np.asarray(log_velocity_increments, dtype=float)
        if np.any(increments_arr <= -epsilon):
            min_inc = np.min(increments_arr)
            raise ValueError(
                "log_velocity_increments must satisfy values > -epsilon "
                f"({-epsilon:.2e}) for log transform. Got minimum = {min_inc:.6g}"
            )
        return np.log(increments_arr + epsilon)

    def plot(
        self,
        dose_range: tuple[float, float] = (1e-6, 1e13),
        n_points: int = 500,
        show_anchors: bool = True,
        show_v_bulk: bool = True,
        show_uncertainty_band: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Plot the etch rate function.

        Parameters
        ----------
        dose_range : tuple, optional
            (min_dose, max_dose) in Gy for x-axis
        n_points : int, optional
            Number of evaluation points
        show_anchors : bool, optional
            Whether to show anchor points
        show_v_bulk : bool, optional
            Whether to show V_bulk_um_h reference line
        show_uncertainty_band : bool, optional
            Whether to show uncertainty band if anchor uncertainties are available

        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        doses_plot: np.ndarray = np.logspace(
            np.log10(dose_range[0]), np.log10(dose_range[1]), n_points
        )
        rates_plot: np.ndarray = np.asarray(self.eval(doses_plot), dtype=float)

        fig, ax = plt.subplots()

        # Optional uncertainty band around the curve (from anchor uncertainties)
        if (
            show_uncertainty_band
            and self.anchor_velocity_uncertainties_um_h is not None
            and len(self.anchor_velocity_uncertainties_um_h)
            == len(self.anchor_velocities_um_h)
        ):
            sigma = np.asarray(self.anchor_velocity_uncertainties_um_h, dtype=float)
            v_anchors = np.asarray(self.anchor_velocities_um_h, dtype=float)

            # Convert sigma_v -> sigma_log10_v at each anchor via local linearisation,
            # then cap at 0.5 log10 units (~factor-of-3 per sigma) to suppress artefacts
            # from under-constrained high-dose anchor points.
            sigma_log10_v = sigma / np.maximum(v_anchors * np.log(10.0), 1e-30)
            sigma_log10_v = np.minimum(sigma_log10_v, 0.5)

            # Interpolate the per-anchor sigma_log10_v onto the fine dose grid using
            # PCHIP in log-dose space.  This avoids constructing two separate
            # EtchRateModel instances (which require monotonic velocities and force
            # np.maximum.accumulate, destroying the below-mean portion of the band).
            from scipy.interpolate import PchipInterpolator as _Pchip

            _log10_d_anchors = np.log10(np.maximum(self.anchor_doses_Gy, 1e-30))
            _log10_d_plot = np.log10(np.maximum(doses_plot, 1e-30))
            _sigma_interp = _Pchip(_log10_d_anchors, sigma_log10_v, extrapolate=True)(
                _log10_d_plot
            )
            _sigma_interp = np.clip(_sigma_interp, 0.0, 0.5)

            # Band is symmetric +/-sigma in log10-velocity space -> extends both above
            # and below the mean model line.
            log10_mean_curve = np.log10(np.maximum(rates_plot, 1e-30))
            lower_curve = np.clip(
                np.power(10.0, log10_mean_curve - _sigma_interp),
                self.V_bulk_um_h,
                self.V_max_um_h,
            )
            upper_curve = np.clip(
                np.power(10.0, log10_mean_curve + _sigma_interp),
                self.V_bulk_um_h,
                self.V_max_um_h,
            )
            ax.fill_between(
                doses_plot,
                lower_curve,
                upper_curve,
                color="k",
                alpha=0.15,
                linewidth=0,
                label="Model uncertainty",
            )

        # Main curve
        ax.plot(
            doses_plot,
            rates_plot,
            "k-",
            linewidth=2,
            label="Etch rate model",
            alpha=0.8,
        )

        # V_bulk reference line
        if show_v_bulk:
            ax.axhline(
                self.V_bulk_um_h,
                color="blue",
                linestyle=":",
                linewidth=1,
                alpha=0.5,
                label=f"V_bulk_um_h = {self.V_bulk_um_h:.2f} um/h",
            )

        # Anchor points
        if show_anchors:
            if self.anchor_velocity_uncertainties_um_h is not None and len(
                self.anchor_velocity_uncertainties_um_h
            ) == len(self.anchor_velocities_um_h):
                v_a = np.asarray(self.anchor_velocities_um_h, dtype=float)
                sig_a = np.asarray(self.anchor_velocity_uncertainties_um_h, dtype=float)
                sig_a = np.nan_to_num(sig_a, nan=0.0, posinf=0.0, neginf=0.0)
                sig_a = np.maximum(sig_a, 0.0)
                # Cap error bars in log space (same 0.5 log10 cap as the band)
                log10_v_a = np.log10(np.maximum(v_a, 1e-30))
                sig_log10_a = np.minimum(
                    sig_a / np.maximum(v_a * np.log(10.0), 1e-30), 0.5
                )
                yerr_lower = v_a - np.power(10.0, log10_v_a - sig_log10_a)
                yerr_upper = np.power(10.0, log10_v_a + sig_log10_a) - v_a
                yerr_lower = np.maximum(np.nan_to_num(yerr_lower, nan=0.0), 0.0)
                yerr_upper = np.maximum(np.nan_to_num(yerr_upper, nan=0.0), 0.0)
                ax.errorbar(
                    self.anchor_doses_Gy,
                    v_a,
                    yerr=[yerr_lower, yerr_upper],
                    fmt="o",
                    color="red",
                    ecolor="red",
                    elinewidth=1,
                    capsize=2,
                    markersize=4,
                    label="Anchor points +/-1sigma",
                    zorder=5,
                )
            else:
                ax.errorbar(
                    self.anchor_doses_Gy,
                    self.anchor_velocities_um_h,
                    yerr=None,
                    fmt="o",
                    color="red",
                    markersize=4,
                    label="Anchor points",
                    zorder=5,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Dose / Gy")
        ax.set_ylabel("Etch-rate / um/h")

        title: str = (
            f"Etch-rate Model: '{self.name}'" if self.name else "Etch-rate Model"
        )
        ax.set_title(title)

        ax.legend(loc="best")
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()

        return fig, ax

    def save_to_json(self, filepath: str) -> None:
        """
        Save the etch rate model to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the output JSON file

        Examples
        --------
        >>> model.save_to_json("my_model.json")
        """
        import json

        data = {
            "name": self.name,
            "V_bulk_um_h": self.V_bulk_um_h,
            "V_max_um_h": self.V_max_um_h,
            "V_max_uncertainty_um_h": self.V_max_uncertainty_um_h,
            "anchor_doses_Gy": self.anchor_doses_Gy.tolist(),
            "anchor_velocities_um_h": self.anchor_velocities_um_h.tolist(),
            "anchor_velocity_uncertainties_um_h": (
                self.anchor_velocity_uncertainties_um_h.tolist()
                if self.anchor_velocity_uncertainties_um_h is not None
                else None
            ),
            "extrapolation_mode": self.extrapolation_mode,
            "debris_alpha": self.debris_alpha,
            "debris_beta": self.debris_beta,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[EtchRateModel] Saved model '{self.name}' to: {filepath}")

    @classmethod
    def load_from_json(cls, filepath: str, verbose: bool = False) -> "EtchRateModel":
        """
        Load an etch rate model from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON file

        Returns
        -------
        EtchRateModel
            The loaded model

        Examples
        --------
        >>> model = EtchRateModel.load_from_json("my_model.json")
        """
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        required_keys = ["V_bulk_um_h", "anchor_doses_Gy", "anchor_velocities_um_h"]
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise ValueError(f"Invalid model JSON: missing required key(s): {missing}")

        model = cls(
            anchor_doses_Gy=data["anchor_doses_Gy"],
            anchor_velocities_um_h=data["anchor_velocities_um_h"],
            anchor_velocity_uncertainties_um_h=data.get(
                "anchor_velocity_uncertainties_um_h"
            ),
            V_bulk_um_h=data["V_bulk_um_h"],
            V_max_um_h=data.get("V_max_um_h"),
            V_max_uncertainty_um_h=data.get("V_max_uncertainty_um_h"),
            name=data.get("name"),
            extrapolation_mode=data.get("extrapolation_mode", "pchip"),
            debris_alpha=data.get("debris_alpha"),
            debris_beta=data.get("debris_beta", 1.0),
        )

        if verbose:
            print(f"[EtchRateModel] Loaded model '{data.get('name')}' from: {filepath}")
        return model
