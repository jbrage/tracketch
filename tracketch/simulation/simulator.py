"""Track etching simulator for CR-39 nuclear track detectors.

This module provides :class:`TrackSimulator`, which computes the 2-D dose
map, etch-rate map, and etchant arrival-time map for a single ion track
in CR-39, then extracts observable track geometry (radius, depth, contour).
"""

import logging
import numpy as np


logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import matplotlib.axes

from tracketch.etching.etch_rate_model import EtchRateModel
from tracketch.etching.etch_rate_model_io import load_etchrate_model
from tracketch.simulation import plots as plot_utils
from tracketch.physics import (
    get_RDD_Gy,
    get_CSDA_um,
    get_dose_map_Gy,
    get_LET_energy_profile,
    convert_MeV_u_to_MeV,
)
from tracketch.simulation.utils import (
    create_simulation_grid,
    get_track_radius_from_contour,
)
from tracketch.wavefront.utils_march import get_arrival_time_map, get_iso_time_contour

# Import HAS_CPP to set default method
from tracketch.wavefront.dijkstra import HAS_CPP
from tracketch import MATERIALS, SRIM_PARTICLES

_VALID_METHODS = ("fmm", "dijkstra", "dijkstra_numba", "dijkstra_cpp")
_VALID_CONNECTIVITY = (8, 16, 32)
_VALID_SOURCES = ("SRIM", "libamtrack")


class TrackSimulator:
    """Simulate chemical etching of a single ion track in CR-39.

    The simulation proceeds in three stages:

    1. **Dose map** -- radial dose distribution (RDD) integrated along the
       ion path, accounting for energy loss and optional range straggling.
    2. **Etch-rate map** -- the calibrated ``etch_model`` converts local dose
       to local etch velocity, with optional debris-damping correction.
    3. **Arrival-time map** -- shortest-time wavefront propagation (Dijkstra
       or Fast Marching) from the detector surface into the bulk.

    From the arrival-time map, iso-time contours, track radii, and track
    depths are extracted for any desired etching duration.

    Parameters
    ----------
    particle_name : str
        Ion species identifier in the form ``"<A><symbol>"``,
        e.g. ``"12C"``, ``"1H"``, ``"56Fe"``, ``"238U"``.
        When ``stopping_power_source="SRIM"`` (default) the particle must be
        one of ``tracketch.SRIM_PARTICLES``; switch to
        ``stopping_power_source="libamtrack"`` to use any nuclide recognised
        by libamtrack.
    start_energy_MeV_u : float
        Kinetic energy at the detector surface in MeV per nucleon.
    etch_model : tracketch.etching.etch_rate_model.EtchRateModel or None
        Calibrated dose -> etch-rate model.  If *None*, a default model is
        loaded (``"Doerschel_etching"``).
    RDD_name : str
        Radial dose distribution model name (default ``"Cucinotta"``).
    arrival_time_method_name : str
        Wavefront algorithm: ``"dijkstra"`` (default, auto-selects fastest
        available backend), ``"dijkstra_cpp"``, ``"dijkstra_numba"``,
        or ``"fmm"``.
    dijkstra_connectivity : int
        Neighbour connectivity for Dijkstra: 8, 16, or 32.
    r_min_um : float or None
        Minimum radial coordinate in um (default 1e-4).
    r_max_um : float or None
        Maximum radial coordinate in um (default 20).
    z_max_um : float or None
        Maximum depth coordinate in um (default 40).
    n_points_r : int or None
        Number of radial grid points (default 400).
    n_points_z : int or None
        Number of depth grid points (default 100).
    rz_lims_dict : dict or None
        Legacy grid specification dict.  Individual keyword arguments above
        take precedence over keys in this dict when both are supplied.
    material_name : str
        Target material -- ``"CR39"`` (default) or ``"water"``.
    stopping_power_source : str
        Stopping-power database: ``"SRIM"`` (default, tabulated data for
        common ions) or ``"libamtrack"`` (supports any nuclide, uses the
        PSTAR/ASTAR parametrisation).
    n_straggling_sigma : int
        Number of longitudinal-straggling sigma to include (0 = none).
    n_uniform_multiplier : int
        Resolution multiplier when regridding to uniform spacing for FMM.
    n_jobs : int
        Number of parallel workers for the RDD/dose-map computation.
        ``-1`` uses all available CPU cores (via :mod:`joblib`).  Parallel
        execution benefits large grids (n_points_r x n_points_z > ~50 000)
        or ``stopping_power_source="libamtrack"``.  For the default grid
        the inter-process overhead outweighs the gain; keep ``n_jobs=1``
        (sequential) unless you have increased the grid resolution.
        Default ``1``.
    log_level : int or str or None
        Set the logging level for the ``tracketch`` logger, e.g.
        ``logging.DEBUG``, ``logging.INFO``, or the string ``"DEBUG"``.
        When *None* (default) the logger is left untouched.

    Attributes
    ----------
    dose_map : ndarray, shape (n_z, n_r)
        Local dose in Gy.
    etch_rate_map : ndarray, shape (n_z, n_r)
        Local etch velocity in um/hr (with debris damping applied, if enabled).
    etch_rate_map_nodebris : ndarray, shape (n_z, n_r)
        Local etch velocity in um/hr *before* any debris damping.  Identical
        to ``etch_rate_map`` when debris damping is disabled.
    arrival_time_map : ndarray, shape (n_z, n_r)
        Shortest etchant arrival time in hours.
    CSDA_range_um : float
        Continuous-slowing-down-approximation range in um.
    energy_profile_MeV_u : ndarray
        Kinetic energy vs. depth (with straggling).
    LET_profile_keV_um : ndarray
        Linear energy transfer vs. depth (with straggling).
    etch_model : tracketch.etching.etch_rate_model.EtchRateModel
        The active dose -> etch-rate model.

    Notes
    -----
    **Track radius and depth**

    After construction, extract the observable track geometry for any etching
    duration with :meth:`get_track_radius_um` and :meth:`get_track_length_um`:

    .. code-block:: python

        sim = TrackSimulator(particle_name="12C", start_energy_MeV_u=270.0)

        radius_um = sim.get_track_radius_um(etch_time_h=3.0)
        depth_um  = sim.get_track_length_um(etch_time_h=3.0)  # pit depth below surface

    Both return ``float("nan")`` when the track is not yet open (too short
    an etching time) or when the ion stops before reaching the detector
    surface.

    **Track contour**

    For the full 2-D pit shape use :meth:`get_iso_time_contour`, which returns
    ``(r, z)`` arrays in um:

    .. code-block:: python

        r, z = sim.get_iso_time_contour(etching_time_h=3.0)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(r, z)
        ax.plot(-r, z)   # mirror for full axisymmetric view
        ax.invert_yaxis()
        ax.set_aspect("equal")

    **Uncertainties**

    Uncertainties are propagated from the calibrated v(d) anchor-velocity
    uncertainties.  These must first be estimated from a calibration dataset
    using :func:`calibration.lib_optimiser.estimate_parameter_uncertainties`
    and then stored on the etch model:

    .. code-block:: python

        from calibration.lib_optimiser import estimate_parameter_uncertainties

        result = estimate_parameter_uncertainties(
            etch_model=sim.etch_model,
            models_dict=my_calibration_data,  # same dict used during fitting
        )
        sim.etch_model.anchor_velocity_uncertainties_um_h = (
            result["anchor_velocity_uncertainties_um_h"]
        )
        sim.etch_model.V_max_uncertainty_um_h = result.get("V_max_uncertainty_um_h")

    Once the uncertainties are attached, use the ``_with_uncertainty`` variants:

    .. code-block:: python

        radius_um, radius_unc_um = sim.get_track_radius_um_with_uncertainty(
            etch_time_h=3.0,  # etching duration
            n_sigma=1.0,       # propagate +/-1 sigma of v(d) anchors
        )

        result = sim.get_track_length_um_with_uncertainty(
            etch_time_h=3.0,
            n_sigma=1.0,
        )
        if result is not None:
            depth_um, depth_unc_um = result

        print(f"radius = {radius_um:.2f} +/- {radius_unc_um:.2f} um")
        print(f"depth  = {depth_um:.2f} +/- {depth_unc_um:.2f} um")

    Both methods return ``nan`` uncertainties when a bound contour cannot be
    extracted.  :meth:`get_track_radius_um_with_uncertainty` also floors the
    uncertainty at half the minimum radial grid step, since the contour
    extraction cannot resolve sub-cell differences.

    For a full uncertainty *band* around the track contour see
    :meth:`get_iso_time_contour_with_uncertainty`.
    """

    def __init__(
        self,
        particle_name: str,  # Literal["1H", "2H", "4He", ...]
        start_energy_MeV_u: float,
        etch_model: EtchRateModel | None = None,
        RDD_name: str = "Cucinotta",
        arrival_time_method_name: str = (
            "dijkstra_cpp" if HAS_CPP else "dijkstra_numba"
        ),
        dijkstra_connectivity: int = (16 if HAS_CPP else 8),
        rz_lims_dict: dict | None = None,
        r_min_um: float | None = None,
        r_max_um: float | None = None,
        z_max_um: float | None = None,
        n_points_r: int | None = None,
        n_points_z: int | None = None,
        material_name: str = "CR39",  # Literal["CR39", "water"]
        stopping_power_source: str = "SRIM",  # Literal["SRIM", "libamtrack"]
        n_straggling_sigma: int = 1,
        n_uniform_multiplier: int = 3,
        n_jobs: int = 1,
        log_level: int | str | None = None,
        # Accept but guard unimplemented features
        normalise_to_LET: bool = False,
        logscale_r: bool = True,
        theta_deg: float = 0.0,
    ):
        # --- guards for unimplemented features --------------------------------
        if theta_deg != 0.0:
            raise NotImplementedError(
                "Angled tracks (theta_deg != 0) are not yet implemented."
            )
        if normalise_to_LET:
            raise NotImplementedError(
                "LET-normalised RDD (normalise_to_LET=True) is not yet implemented."
            )
        if not logscale_r:
            raise NotImplementedError(
                "Linear r-grid (logscale_r=False) is not yet implemented."
            )

        # --- validate ---------------------------------------------------------
        if material_name not in MATERIALS:
            raise ValueError(
                f"Unknown material '{material_name}'. Valid options: {MATERIALS}"
            )
        if stopping_power_source not in _VALID_SOURCES:
            raise ValueError(
                f"Unknown stopping_power_source '{stopping_power_source}'. "
                f"Valid options: {_VALID_SOURCES}"
            )
        if stopping_power_source == "SRIM" and particle_name not in SRIM_PARTICLES:
            raise ValueError(
                f"Particle '{particle_name}' has no SRIM data file. "
                f"SRIM particles: {SRIM_PARTICLES}. "
                f"For other ions (e.g. '56Fe') use stopping_power_source='libamtrack'."
            )
        if stopping_power_source == "libamtrack":
            import pyamtrack.libAT as _libam

            _p_no = _libam.AT_particle_no_from_particle_name_single(particle_name)
            if _p_no == -1:
                raise ValueError(
                    f"Particle '{particle_name}' is not recognised by libamtrack. "
                    f"Expected format: '<A><symbol>', e.g. '12C', '56Fe', '238U'."
                )
        if arrival_time_method_name not in _VALID_METHODS:
            raise ValueError(
                f"Invalid arrival_time_method '{arrival_time_method_name}'. "
                f"Choose from {_VALID_METHODS}."
            )
        if dijkstra_connectivity not in _VALID_CONNECTIVITY:
            raise ValueError(
                f"dijkstra_connectivity must be 8, 16, or 32, "
                f"got {dijkstra_connectivity}"
            )

        # --- store configuration ----------------------------------------------
        self.particle_name = particle_name
        self.start_energy_MeV_u = start_energy_MeV_u
        self.RDD_name = RDD_name
        self.material_name = material_name
        self.stopping_power_source_name = stopping_power_source

        if log_level is not None:
            _root = logging.getLogger("tracketch")
            # Remove any NullHandlers added by the package-level initialisation
            # before checking whether a real StreamHandler is already present.
            _root.handlers = [
                h for h in _root.handlers if not isinstance(h, logging.NullHandler)
            ]
            if not any(isinstance(h, logging.StreamHandler) for h in _root.handlers):
                _handler = logging.StreamHandler()
                _handler.setFormatter(
                    logging.Formatter("%(name)s [%(levelname)s] %(message)s")
                )
                _root.addHandler(_handler)
            _root.setLevel(log_level)

        logger.info(
            "TrackSimulator: %s @ %.3g MeV/u in %s (RDD=%s, method=%s)",
            particle_name,
            start_energy_MeV_u,
            material_name,
            RDD_name,
            arrival_time_method_name,
        )

        self._arrival_time_method = arrival_time_method_name
        self._dijkstra_connectivity = dijkstra_connectivity
        self._logscale_r = True  # always True for now
        self._n_straggling_sigma = n_straggling_sigma
        self._n_uniform_multiplier = n_uniform_multiplier
        self._n_jobs = n_jobs

        # --- etch-rate model --------------------------------------------------
        if etch_model is None:
            logger.debug(
                "No etch model provided; loading default 'Doerschel_etching' model"
            )
            self.etch_model = load_etchrate_model("Doerschel_etching")
        else:
            self.etch_model = etch_model

        # --- spatial grids ----------------------------------------------------
        # Merge legacy rz_lims_dict with the individual keyword arguments.
        # Keyword arguments take precedence.
        grid_params: dict = dict(rz_lims_dict) if rz_lims_dict else {}
        if r_min_um is not None:
            grid_params["r_min_um"] = r_min_um
        if r_max_um is not None:
            grid_params["r_max_um"] = r_max_um
        if z_max_um is not None:
            grid_params["z_max_um"] = z_max_um
        if n_points_r is not None:
            grid_params["n_points_r"] = n_points_r
        if n_points_z is not None:
            grid_params["n_points_z"] = n_points_z
        self._r_grid_um, self._z_grid_um, self._r_lims_um, self._z_lims_um = (
            create_simulation_grid(grid_params)
        )
        logger.info(
            "Grid: r=[%.4g, %.4g] um, z=[0, %.4g] um  (%d r-points x %d z-points)",
            self._r_grid_um[0],
            self._r_grid_um[-1],
            self._z_grid_um[-1],
            len(self._r_grid_um),
            len(self._z_grid_um),
        )

        # --- compute everything -----------------------------------------------
        self._calculate_physics()
        self._calculate_dose_map()
        self._calculate_etch_rate_map()
        self._apply_debris_damping()
        self._calculate_arrival_time_map()

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TrackSimulator(\n"
            f"  particle='{self.particle_name}',\n"
            f"  start_energy_MeV_u={self.start_energy_MeV_u:0.3g} MeV/u,\n"
            f"  material='{self.material_name}',\n"
            f"  RDD_model='{self.RDD_name}',\n"
            f"  stopping_power_source='{self.stopping_power_source_name}'\n"
            f")"
        )

    # ------------------------------------------------------------------
    # internal computation
    # ------------------------------------------------------------------

    def _calculate_physics(self) -> None:
        """Compute CSDA range, energy profile, and LET profile along depth."""
        logger.debug("Computing ion physics (CSDA range + LET profile)")

        self.start_energy_MeV = convert_MeV_u_to_MeV(
            self.start_energy_MeV_u, self.particle_name
        )

        self.CSDA_range_um = get_CSDA_um(
            energy_MeV_u=self.start_energy_MeV_u,
            particle_name=self.particle_name,
            material_name=self.material_name,
            source=self.stopping_power_source_name,
        )

        # Pristine profile (no range straggling)
        self.energy_profile_MeV_u_pristine, self.LET_profile_keV_um_pristine = (
            get_LET_energy_profile(
                dz_um_list=self._z_grid_um,
                energy_MeV_u=self.start_energy_MeV_u,
                particle_name=self.particle_name,
                material_name=self.material_name,
                source=self.stopping_power_source_name,
                n_straggling_sigma=self._n_straggling_sigma,
                range_straggling=False,
            )
        )

        # With longitudinal straggling
        self.energy_profile_MeV_u, self.LET_profile_keV_um = get_LET_energy_profile(
            dz_um_list=self._z_grid_um,
            energy_MeV_u=self.start_energy_MeV_u,
            particle_name=self.particle_name,
            material_name=self.material_name,
            source=self.stopping_power_source_name,
            n_straggling_sigma=self._n_straggling_sigma,
            range_straggling=True,
        )
        logger.debug("CSDA range: %.1f um", self.CSDA_range_um)

    def _calculate_dose_map(self) -> None:
        """Build the 2-D local-dose map from radial dose distributions."""
        self.dose_map = get_dose_map_Gy(
            z_um=self._z_grid_um,
            r_um=self._r_grid_um,
            energy_MeV_u_array=self.energy_profile_MeV_u,
            LET_source=self.stopping_power_source_name,
            particle_name=self.particle_name,
            RDD_name=self.RDD_name,
            material_name=self.material_name,
            normalise_to_LET=False,
            n_jobs=self._n_jobs,
        )

    def _calculate_etch_rate_map(self) -> None:
        """Convert dose map to etch-rate map via the etch model."""
        self.etch_rate_map = np.asarray(self.etch_model.eval(self.dose_map))
        # Snapshot before debris damping is applied in-place.
        self.etch_rate_map_nodebris = self.etch_rate_map.copy()

    def _compute_arrival_time_for(self, etch_rate_map: np.ndarray) -> np.ndarray:
        """Run wavefront propagation for an arbitrary etch-rate map."""
        return get_arrival_time_map(
            r_um=self._r_grid_um,
            z_um=self._z_grid_um,
            etch_rate_map=etch_rate_map,
            method=self._arrival_time_method,
            r_is_logscaled=self._logscale_r,
            theta_deg=0.0,
            n_uniform_multiplier=self._n_uniform_multiplier,
            connectivity=self._dijkstra_connectivity,  # type: ignore
        )

    def _calculate_arrival_time_map(self) -> None:
        """Compute shortest etchant arrival time via wavefront propagation."""
        logger.info(
            "Computing arrival-time map (method=%s, connectivity=%d)",
            self._arrival_time_method,
            self._dijkstra_connectivity,
        )
        self.arrival_time_map = self._compute_arrival_time_for(self.etch_rate_map)

    def _apply_debris_damping(self) -> None:
        """Apply diffusion-limited debris damping to the etch-rate map.

        Models transport limitation in deep, narrow pits.  Parameterised
        by ``etch_model.debris_alpha`` (characteristic aspect ratio) and
        ``etch_model.debris_beta`` (transition steepness).  No-op when
        damping is disabled (alpha is None or <= 0).

        The aspect ratio that governs damping magnitude is computed from
        ``r_track`` — the outermost radius where the dose exceeds a low
        threshold — which approximates the full pit radius and gives
        physically meaningful aspect ratios (order 1–10).

        Damping is *applied* only to cells where the etch rate exceeds
        ``core_factor * V_bulk`` (the narrow high-rate core at r ≈ 0).
        Because r_core << r_track, those cells all lie very close to r = 0
        and receive near-uniform suppression.  Penumbra cells are untouched.

        The original ``max(eta_core, 0.95)`` floor is reinstated to cap
        damping at 5 % maximum strength, keeping the correction mild and
        preventing the optimizer from over-compensating via v(d).
        """
        alpha = self.etch_model.debris_alpha
        beta = self.etch_model.debris_beta

        if alpha is None or alpha <= 0:
            return
        logger.debug("Applying debris damping (alpha=%.3g, beta=%.3g)", alpha, beta)

        V_bulk = float(self.etch_model.V_bulk_um_h)
        dose_threshold_Gy = 1.0  # defines full pit radius r_track
        core_factor = 5.0  # cells with v > core_factor*V_bulk are damped
        core_v_threshold = core_factor * V_bulk
        min_r_track_um = 0.01

        r_grid = self._r_grid_um
        z_grid = self._z_grid_um
        etch_map = self.etch_rate_map
        dose_map = self.dose_map

        for j in range(len(z_grid)):
            z = z_grid[j]
            if z <= 0:
                continue

            # --- pit radius from dose (gives physically meaningful AR) ---
            dose_profile = dose_map[j, :]
            track_mask = np.isfinite(dose_profile) & (dose_profile > dose_threshold_Gy)
            if not np.any(track_mask):
                continue
            track_indices = np.where(track_mask)[0]
            r_track = r_grid[track_indices[-1]]
            if r_track < min_r_track_um:
                continue

            # --- core indices: where to apply damping ---
            etch_slice = etch_map[j, :]
            core_mask = etch_slice > core_v_threshold
            if not np.any(core_mask):
                continue
            core_indices = np.where(core_mask)[0]

            # --- damping magnitude from full pit aspect ratio ---
            aspect_ratio = z / r_track
            eta_core = 1.0 / (1.0 + (aspect_ratio / alpha) ** beta)
            # Hard cap: maximum 5 % suppression anywhere in the core.
            # This keeps the correction physically mild and prevents the
            # optimizer from collapsing v(d) to compensate for over-damping.
            eta_core = max(eta_core, 0.95)

            # --- radial profile normalised to r_track ---
            # r_core << r_track so all core cells are near r=0 and receive
            # approximately the full eta_core suppression; exponent 6 gives
            # a smooth profile that concentrates damping tightly at centre.
            r_values = r_grid[core_indices]
            radial_factor = np.clip(1.0 - r_values / r_track, 0.0, 1.0) ** 6
            eta_local = 1.0 - radial_factor * (1.0 - eta_core)
            etch_map[j, core_indices] *= eta_local
            # Penumbra cells (not in core_indices) are untouched.

        np.maximum(etch_map, V_bulk, out=etch_map)

    # ------------------------------------------------------------------
    # etch-model update
    # ------------------------------------------------------------------

    def update_etch_model_and_recalculate(
        self, new_etch_model: EtchRateModel | None = None
    ) -> None:
        """Replace the etch model and recompute etch-rate & arrival-time maps.

        The dose map is *not* recomputed (it depends only on physics, not
        the etch model).

        Parameters
        ----------
        new_etch_model : tracketch.etching.etch_rate_model.EtchRateModel or None
            New model.  If *None*, recomputes using the current model
            (useful after in-place parameter changes).
        """
        if new_etch_model is not None:
            self.etch_model = new_etch_model
        self._calculate_etch_rate_map()
        self._apply_debris_damping()
        self._calculate_arrival_time_map()

    def recalculate_from_current_etch_model(self) -> None:
        """Recompute maps using the current etch model.

        Alias for ``update_etch_model_and_recalculate(None)``.
        """
        self.update_etch_model_and_recalculate()

    # ------------------------------------------------------------------
    # track geometry queries
    # ------------------------------------------------------------------

    def get_iso_time_contour(
        self, etching_time_h: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract the iso-time contour from the arrival-time map.

        Parameters
        ----------
        etching_time_h : float
            Etching duration in hours.

        Returns
        -------
        r_coords, z_coords : tuple[ndarray, ndarray]
            Coordinates along the longest contour segment.
        """
        return get_iso_time_contour(
            arrival_time_map=self.arrival_time_map,
            r_um=self._r_grid_um,
            z_um=self._z_grid_um,
            etching_time_h=etching_time_h,
        )

    def get_iso_time_contour_with_uncertainty(
        self,
        etching_time_h: float,
        n_sigma: float = 1.0,
        n_band_points: int = 200,
    ) -> (
        tuple[
            tuple[np.ndarray, np.ndarray],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
        | None
    ):
        """Iso-time contour with uncertainty band from v(d) anchor uncertainties.

        Computes arrival-time maps for v(d)+/-n_sigma*dV, extracts the three
        contours, and interpolates the upper/lower bounds onto a common z grid
        suitable for ``ax.fill_betweenx``.

        Parameters
        ----------
        etching_time_h : float
            Etching duration in hours.
        n_sigma : float
            Number of standard deviations to propagate (default 1).
        n_band_points : int
            Number of points on the common z grid for the band (default 200).

        Returns
        -------
        tuple of ``((r_central, z_central), z_band, r_lo, r_hi)`` or *None*.

        Returns *None* (and logs a warning) when the etch model carries no
        anchor velocity uncertainties.  Assign the output of
        ``estimate_parameter_uncertainties`` to
        ``etch_model.anchor_velocity_uncertainties_um_h`` first.
        """
        if self.etch_model.anchor_velocity_uncertainties_um_h is None:
            logger.warning(
                "get_iso_time_contour_with_uncertainty: "
                "etch_model.anchor_velocity_uncertainties_um_h is None -- "
                "returning None. Assign uncertainties from "
                "estimate_parameter_uncertainties first."
            )
            return None

        central = self.get_iso_time_contour(etching_time_h)

        bound_contours: list[tuple[np.ndarray, np.ndarray]] = []
        for sign in (+1.0, -1.0):
            perturbed = self._make_perturbed_etch_model(sign, n_sigma)
            etch_rate_map_pert = np.asarray(perturbed.eval(self.dose_map))
            arrival_time_map_pert = get_arrival_time_map(
                r_um=self._r_grid_um,
                z_um=self._z_grid_um,
                etch_rate_map=etch_rate_map_pert,
                method=self._arrival_time_method,
                r_is_logscaled=self._logscale_r,
                theta_deg=0.0,
                n_uniform_multiplier=self._n_uniform_multiplier,
                connectivity=self._dijkstra_connectivity,  # type: ignore
            )
            r_b, z_b = get_iso_time_contour(
                arrival_time_map=arrival_time_map_pert,
                r_um=self._r_grid_um,
                z_um=self._z_grid_um,
                etching_time_h=etching_time_h,
            )
            bound_contours.append((r_b, z_b))

        # Interpolate both bounds onto a common z grid for fill_betweenx
        all_z = np.concatenate([central[1]] + [c[1] for c in bound_contours])
        z_min = float(np.nanmin(all_z))
        z_max = float(np.nanmax(all_z))
        z_band = np.linspace(z_min, z_max, n_band_points)

        def _interp_r_on_z(
            r: np.ndarray, z: np.ndarray, z_grid: np.ndarray
        ) -> np.ndarray:
            if len(r) == 0:
                return np.full_like(z_grid, np.nan)
            order = np.argsort(z)
            return np.interp(z_grid, z[order], r[order])

        r_bound_0 = _interp_r_on_z(*bound_contours[0], z_band)
        r_bound_1 = _interp_r_on_z(*bound_contours[1], z_band)
        r_lo = np.minimum(r_bound_0, r_bound_1)
        r_hi = np.maximum(r_bound_0, r_bound_1)

        # Enforce a minimum experimental radial uncertainty band of +/- 0.15 um
        # around the central contour at every z.
        min_uncertainty_um = 0.1
        r_central_on_band = _interp_r_on_z(central[0], central[1], z_band)
        r_lo = np.minimum(r_lo, r_central_on_band - min_uncertainty_um)
        r_hi = np.maximum(r_hi, r_central_on_band + min_uncertainty_um)

        return central, z_band, r_lo, r_hi

    def get_iso_time_contour_with_normal_uncertainty(
        self,
        etching_time_h: float,
        n_sigma: float = 1.0,
        n_band_points: int = 200,
        min_uncertainty_um: float = 0.15,
    ) -> (
        tuple[
            tuple[np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray],
        ]
        | None
    ):
        """Iso-time contour with uncertainty expressed along local contour normals.

        Unlike ``get_iso_time_contour_with_uncertainty`` (which returns a
        radial ``r(z)`` band for ``fill_betweenx``), this method computes the
        uncertainty envelope perpendicular to the contour itself. This keeps
        the band visible and geometrically meaningful even where the contour
        is horizontal.

        Returns
        -------
        tuple of ``((r_c, z_c), (r_lo, z_lo), (r_hi, z_hi))`` or *None*.
        """
        if self.etch_model.anchor_velocity_uncertainties_um_h is None:
            logger.warning(
                "get_iso_time_contour_with_normal_uncertainty: "
                "etch_model.anchor_velocity_uncertainties_um_h is None -- "
                "returning None. Assign uncertainties from "
                "estimate_parameter_uncertainties first."
            )
            return None

        central = self.get_iso_time_contour(etching_time_h)
        r_c_raw, z_c_raw = central
        if len(r_c_raw) < 2:
            return None

        bound_contours: list[tuple[np.ndarray, np.ndarray]] = []
        for sign in (+1.0, -1.0):
            perturbed = self._make_perturbed_etch_model(sign, n_sigma)
            etch_rate_map_pert = np.asarray(perturbed.eval(self.dose_map))
            arrival_time_map_pert = get_arrival_time_map(
                r_um=self._r_grid_um,
                z_um=self._z_grid_um,
                etch_rate_map=etch_rate_map_pert,
                method=self._arrival_time_method,
                r_is_logscaled=self._logscale_r,
                theta_deg=0.0,
                n_uniform_multiplier=self._n_uniform_multiplier,
                connectivity=self._dijkstra_connectivity,  # type: ignore
            )
            r_b, z_b = get_iso_time_contour(
                arrival_time_map=arrival_time_map_pert,
                r_um=self._r_grid_um,
                z_um=self._z_grid_um,
                etching_time_h=etching_time_h,
            )
            if len(r_b) < 2:
                continue
            bound_contours.append((r_b, z_b))

        if len(bound_contours) != 2:
            return None

        # Resample central contour uniformly in arc length.
        ds = np.hypot(np.diff(r_c_raw), np.diff(z_c_raw))
        s = np.concatenate(([0.0], np.cumsum(ds)))
        s_max = float(s[-1])
        if s_max <= 0:
            return None
        s_new = np.linspace(0.0, s_max, n_band_points)
        r_c = np.interp(s_new, s, r_c_raw)
        z_c = np.interp(s_new, s, z_c_raw)

        # Unit normals from arc-length parameterization.
        dr_ds = np.gradient(r_c, s_new)
        dz_ds = np.gradient(z_c, s_new)
        n_x = -dz_ds
        n_y = dr_ds
        n_norm = np.hypot(n_x, n_y)
        n_norm = np.maximum(n_norm, 1e-12)
        n_x /= n_norm
        n_y /= n_norm

        def _signed_normal_distance(
            r_ref: np.ndarray,
            z_ref: np.ndarray,
            nx_ref: np.ndarray,
            ny_ref: np.ndarray,
            contour: tuple[np.ndarray, np.ndarray],
        ) -> np.ndarray:
            r_b, z_b = contour
            # Nearest-point assignment from each central sample to bound contour.
            dr = r_b[None, :] - r_ref[:, None]
            dz = z_b[None, :] - z_ref[:, None]
            d2 = dr * dr + dz * dz
            j_near = np.argmin(d2, axis=1)
            dr_near = dr[np.arange(len(r_ref)), j_near]
            dz_near = dz[np.arange(len(r_ref)), j_near]
            return dr_near * nx_ref + dz_near * ny_ref

        sdist_0 = _signed_normal_distance(r_c, z_c, n_x, n_y, bound_contours[0])
        sdist_1 = _signed_normal_distance(r_c, z_c, n_x, n_y, bound_contours[1])

        s_lo = np.minimum(sdist_0, sdist_1)
        s_hi = np.maximum(sdist_0, sdist_1)

        # Enforce experimental minimum uncertainty along the local normal.
        s_lo = np.minimum(s_lo, -float(min_uncertainty_um))
        s_hi = np.maximum(s_hi, float(min_uncertainty_um))

        r_lo = r_c + s_lo * n_x
        z_lo = z_c + s_lo * n_y
        r_hi = r_c + s_hi * n_x
        z_hi = z_c + s_hi * n_y

        # Clip to non-negative r: the symmetry axis (r=0) is the physical boundary.
        # Near the axis the inward side of the band naturally clips there.
        r_lo = np.maximum(r_lo, 0.0)
        r_hi = np.maximum(r_hi, 0.0)

        # Near the tip the contour is nearly vertical (radially inward), so the
        # outward normals are nearly horizontal.  This means z_lo ≈ z_hi ≈ z_c
        # at the axis and the z-extent of the band collapses to nearly zero there.
        # Enforce a minimum z-band width of axis_z_width_um at the tip so the
        # polygon always has a visible shaded cap around the contour terminus.
        axis_z_width_um = 0.3
        tip_idx = 0 if r_c[0] <= r_c[-1] else len(r_c) - 1
        z_tip = float(z_c[tip_idx])
        z_lo[tip_idx] = min(float(z_lo[tip_idx]), z_tip - axis_z_width_um)
        z_hi[tip_idx] = max(float(z_hi[tip_idx]), z_tip + axis_z_width_um)

        # Extend the uncertainty envelope all the way to the symmetry axis (r=0).
        # The contour runs from the surface opening (large r) to the tip (small r).
        # Detect dynamically which end is the tip and append/prepend accordingly.
        if r_c[-1] <= r_c[0]:
            # Tip at the end — append axis point
            if r_lo[-1] > 0.0 or r_hi[-1] > 0.0:
                r_lo = np.concatenate((r_lo, [0.0]))
                z_lo = np.concatenate((z_lo, [z_lo[-1]]))
                r_hi = np.concatenate((r_hi, [0.0]))
                z_hi = np.concatenate((z_hi, [z_hi[-1]]))
        else:
            # Tip at the start — prepend axis point
            if r_lo[0] > 0.0 or r_hi[0] > 0.0:
                r_lo = np.concatenate(([0.0], r_lo))
                z_lo = np.concatenate(([z_lo[0]], z_lo))
                r_hi = np.concatenate(([0.0], r_hi))
                z_hi = np.concatenate(([z_hi[0]], z_hi))

        return (r_c, z_c), (r_lo, z_lo), (r_hi, z_hi)

    def get_track_radius_um(
        self, etch_time_h: float, threshold_percent: float = 5.0
    ) -> float:
        """Track opening radius from iso-time contour deviation.

        Parameters
        ----------
        etch_time_h : float
            Etching duration in hours.
        threshold_percent : float
            Deviation threshold as percentage of bulk depth (default 5 %).

        Returns
        -------
        float
            Track radius in um, or ``nan`` if no contour is found.
        """
        r_contour, z_contour = self.get_iso_time_contour(etch_time_h)
        return get_track_radius_from_contour(
            r_contour,
            z_contour,
            self.etch_model.V_bulk_um_h,
            etch_time_h,
            threshold_percent,
        )

    def get_iso_time_contour_nodebris(
        self, etching_time_h: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Iso-time contour computed from the etch-rate map *without* debris damping.

        Useful for comparing predicted track shapes with and without the
        debris-transport correction.  The arrival-time map is computed on
        demand from ``etch_rate_map_nodebris`` each time this method is
        called (not cached).

        Parameters
        ----------
        etching_time_h : float
            Etching duration in hours.

        Returns
        -------
        r_coords, z_coords : tuple[ndarray, ndarray]
            Coordinates along the iso-time contour.
        """
        arrival_time_nodebris = self._compute_arrival_time_for(
            self.etch_rate_map_nodebris
        )
        return get_iso_time_contour(
            arrival_time_map=arrival_time_nodebris,
            r_um=self._r_grid_um,
            z_um=self._z_grid_um,
            etching_time_h=etching_time_h,
        )

    def get_track_radius_um_nodebris(
        self, etch_time_h: float, threshold_percent: float = 5.0
    ) -> float:
        """Track opening radius computed *without* debris damping.

        Convenience wrapper around :meth:`get_iso_time_contour_nodebris`.

        Parameters
        ----------
        etch_time_h : float
            Etching duration in hours.
        threshold_percent : float
            Deviation threshold as percentage of bulk depth (default 5 %).

        Returns
        -------
        float
            Track radius in um, or ``nan`` if no contour is found.
        """
        r_contour, z_contour = self.get_iso_time_contour_nodebris(etch_time_h)
        return get_track_radius_from_contour(
            r_contour,
            z_contour,
            self.etch_model.V_bulk_um_h,
            etch_time_h,
            threshold_percent,
        )

    def _make_perturbed_etch_model(self, sign: float, n_sigma: float) -> EtchRateModel:
        """Return a copy of the etch model with anchor velocities shifted by +/-n_sigma.

        Clamps the shifted velocities to [V_bulk, V_max] and re-enforces
        monotonicity via cumulative maximum so the result is always valid.

        Parameters
        ----------
        sign : float
            +1 for upper bound, -1 for lower bound.
        n_sigma : float
            Number of standard deviations to shift.
        """
        m = self.etch_model
        unc = m.anchor_velocity_uncertainties_um_h
        assert unc is not None, "caller must check anchor_velocity_uncertainties_um_h"
        # Perturb V_max if its uncertainty is known and V_max is finite.
        # This is essential for heavy ions where high-dose anchors saturate at
        # V_max -- without perturbing V_max itself, the clipping erases all signal.
        if m.V_max_uncertainty_um_h is not None and np.isfinite(m.V_max_um_h):
            v_max_pert = m.V_max_um_h + sign * n_sigma * m.V_max_uncertainty_um_h
            v_max_pert = max(v_max_pert, m.V_bulk_um_h)
        else:
            v_max_pert = m.V_max_um_h
        v_pert = m.anchor_velocities_um_h + sign * n_sigma * unc
        v_pert = np.clip(v_pert, m.V_bulk_um_h, v_max_pert)
        v_pert = np.maximum.accumulate(v_pert)
        return EtchRateModel(
            anchor_doses_Gy=m.anchor_doses_Gy,
            anchor_velocities_um_h=v_pert,
            V_bulk_um_h=m.V_bulk_um_h,
            V_max_um_h=v_max_pert,
            extrapolation_mode=m.extrapolation_mode,
            debris_alpha=m.debris_alpha,
            debris_beta=m.debris_beta,
        )

    def get_track_radius_um_with_uncertainty(
        self,
        etch_time_h: float,
        threshold_percent: float = 5.0,
        n_sigma: float = 1.0,
    ) -> tuple[float, float]:
        """Track radius with uncertainty propagated from v(d) anchor uncertainties.

        Evaluates the etch-rate and arrival-time maps for v(d)+n_sigma*dV and
        v(d)-n_sigma*dV, extracts a radius from each, and returns the central
        radius together with half the spread as the uncertainty estimate.

        Parameters
        ----------
        etch_time_h : float
            Etching duration in hours.
        threshold_percent : float
            Deviation threshold as percentage of bulk depth (default 5 %).
        n_sigma : float
            Number of standard deviations to propagate (default 1).

        Returns
        -------
        radius_um : float
            Central track radius in um.
        radius_uncertainty_um : float
            Half-spread of the upper/lower bound radii (1-sigma estimate).

        Raises
        ------
        ValueError
            If the etch model carries no anchor velocity uncertainties.
        """
        if self.etch_model.anchor_velocity_uncertainties_um_h is None:
            raise ValueError(
                "etch_model.anchor_velocity_uncertainties_um_h is None. "
                "Run estimate_parameter_uncertainties and assign the result first."
            )

        radius_central = self.get_track_radius_um(etch_time_h, threshold_percent)

        radii_bounds: list[float] = []
        for sign in (+1.0, -1.0):
            perturbed = self._make_perturbed_etch_model(sign, n_sigma)

            etch_rate_map_pert = np.asarray(perturbed.eval(self.dose_map))

            arrival_time_map_pert = get_arrival_time_map(
                r_um=self._r_grid_um,
                z_um=self._z_grid_um,
                etch_rate_map=etch_rate_map_pert,
                method=self._arrival_time_method,
                r_is_logscaled=self._logscale_r,
                theta_deg=0.0,
                n_uniform_multiplier=self._n_uniform_multiplier,
                connectivity=self._dijkstra_connectivity,  # type: ignore
            )

            r_contour, z_contour = get_iso_time_contour(
                arrival_time_map=arrival_time_map_pert,
                r_um=self._r_grid_um,
                z_um=self._z_grid_um,
                etching_time_h=etch_time_h,
            )

            r_bound = get_track_radius_from_contour(
                r_contour,
                z_contour,
                perturbed.V_bulk_um_h,
                etch_time_h,
                threshold_percent,
            )
            radii_bounds.append(r_bound)

        radius_uncertainty = 0.5 * abs(radii_bounds[0] - radii_bounds[1])
        # Floor at half the minimum radial grid step: the contour extraction
        # cannot resolve the radius to better than one grid cell regardless
        # of the perturbation size.
        dr_min = float(np.min(np.diff(self._r_grid_um)))
        min_uncertainty_um = 0.15
        radius_uncertainty = max(radius_uncertainty, 0.5 * dr_min, min_uncertainty_um)
        return radius_central, radius_uncertainty

    def get_track_length_um_with_uncertainty(
        self,
        etch_time_h: float,
        relative_to_surface: bool = True,
        n_sigma: float = 1.0,
    ) -> tuple[float, float] | None:
        """Track depth with uncertainty propagated from v(d) anchor uncertainties.

        Evaluates the arrival-time map for v(d)+n_sigma*dV and v(d)-n_sigma*dV,
        reads the depth at r = 0 for each, and returns the central depth together
        with half the spread as the uncertainty estimate.

        Parameters
        ----------
        etch_time_h : float
            Etching duration in hours.
        relative_to_surface : bool
            If *True* (default), subtract the bulk-etched surface depth.
        n_sigma : float
            Number of standard deviations to propagate (default 1).

        Returns
        -------
        tuple of ``(depth_um, depth_uncertainty_um)`` or *None*.

        Returns *None* (and logs a warning) when the etch model carries no
        anchor velocity uncertainties.
        """
        if self.etch_model.anchor_velocity_uncertainties_um_h is None:
            logger.warning(
                "get_track_length_um_with_uncertainty: "
                "etch_model.anchor_velocity_uncertainties_um_h is None -- "
                "returning None. Assign uncertainties from "
                "estimate_parameter_uncertainties first."
            )
            return None

        depth_central = self.get_track_length_um(etch_time_h, relative_to_surface)
        if not np.isfinite(depth_central):
            return float(depth_central), float("nan")

        z_coords = self._z_grid_um
        depths: list[float] = []
        for sign in (+1.0, -1.0):
            perturbed = self._make_perturbed_etch_model(sign, n_sigma)
            etch_rate_map_pert = np.asarray(perturbed.eval(self.dose_map))
            arrival_time_map_pert = get_arrival_time_map(
                r_um=self._r_grid_um,
                z_um=self._z_grid_um,
                etch_rate_map=etch_rate_map_pert,
                method=self._arrival_time_method,
                r_is_logscaled=self._logscale_r,
                theta_deg=0.0,
                n_uniform_multiplier=self._n_uniform_multiplier,
                connectivity=self._dijkstra_connectivity,  # type: ignore
            )
            arrival_at_r0 = arrival_time_map_pert[:, 0]
            if not np.any(arrival_at_r0 <= etch_time_h):
                depths.append(float("nan"))
                continue
            depth = float(np.interp(etch_time_h, arrival_at_r0, z_coords))
            if relative_to_surface:
                depth -= etch_time_h * perturbed.V_bulk_um_h
            depths.append(depth)

        finite = [d for d in depths if np.isfinite(d)]
        depth_uncertainty = (
            0.5 * abs(depths[0] - depths[1]) if len(finite) == 2 else float("nan")
        )
        if np.isfinite(depth_uncertainty):
            min_uncertainty_um = 0.6
            depth_uncertainty = max(depth_uncertainty, min_uncertainty_um)
        return float(depth_central), depth_uncertainty

    def get_track_length_um(
        self, etch_time_h: float | list, relative_to_surface: bool = True
    ) -> float | np.ndarray:
        """Track depth at r = 0 for a given etching time.

        Parameters
        ----------
        etch_time_h : float or list[float]
            Etching duration(s) in hours.
        relative_to_surface : bool
            If *True* (default), subtract the bulk-etched surface position
            so the result is the pit depth below the current surface.

        Returns
        -------
        float or ndarray
            Depth in um, or ``nan`` if the track does not reach r = 0.
        """
        etch_time_h_array = np.atleast_1d(etch_time_h)
        arrival_at_r0 = self.arrival_time_map[:, 0]
        z_coords = self._z_grid_um

        if not np.any(arrival_at_r0 <= etch_time_h_array.max()):
            return np.nan

        depth_um = np.interp(etch_time_h_array, arrival_at_r0, z_coords)

        # Check grid bounds
        if np.any(depth_um > z_coords.max()) or np.any(depth_um < 0):
            logger.warning(
                "Track depth %.1f um outside grid [0, %.1f um]. "
                "Consider increasing z_max_um.",
                float(depth_um[0]),
                z_coords.max(),
            )
            return np.nan

        if relative_to_surface:
            depth_um = depth_um - etch_time_h_array * self.etch_model.V_bulk_um_h

        return depth_um if depth_um.size > 1 else float(depth_um[0])

    def get_track_detectability(
        self,
        etch_time_h: float,
        min_radius_um: float = 0.5,
        min_depth_um: float = 0.3,
        min_cone_half_angle_deg: float = 2.0,
        threshold_percent: float = 5.0,
    ) -> dict:
        """Geometric detectability metrics for a given etching time.

        A track is considered detectable when **all three** criteria are met:

        1. ``radius_um >= min_radius_um``
        2. ``depth_um >= min_depth_um``  (pit depth below bulk surface)
        3. ``cone_half_angle_deg >= min_cone_half_angle_deg``

        Returns
        -------
        dict
            Keys: ``detected``, ``radius_um``, ``depth_um``,
            ``cone_half_angle_deg``, ``radius_ok``, ``depth_ok``, ``angle_ok``.
        """
        radius_um = self.get_track_radius_um(
            etch_time_h=etch_time_h, threshold_percent=threshold_percent
        )
        track_length_um = self.get_track_length_um(
            etch_time_h=etch_time_h, relative_to_surface=True
        )
        depth_um = float(track_length_um) if np.isfinite(track_length_um) else 0.0

        if depth_um > 0.0 and np.isfinite(radius_um) and radius_um > 0.0:
            cone_half_angle_deg = float(np.degrees(np.arctan(radius_um / depth_um)))
        else:
            cone_half_angle_deg = float("nan")

        radius_ok = np.isfinite(radius_um) and radius_um >= min_radius_um
        depth_ok = depth_um >= min_depth_um
        angle_ok = (
            np.isfinite(cone_half_angle_deg)
            and cone_half_angle_deg >= min_cone_half_angle_deg
        )

        return {
            "detected": bool(radius_ok and depth_ok and angle_ok),
            "radius_um": float(radius_um) if np.isfinite(radius_um) else float("nan"),
            "depth_um": depth_um,
            "cone_half_angle_deg": cone_half_angle_deg,
            "radius_ok": radius_ok,
            "depth_ok": depth_ok,
            "angle_ok": angle_ok,
        }

    def is_track_detected(
        self,
        etch_time_h: float,
        min_radius_um: float = 0.5,
        min_depth_um: float = 0.3,
        min_cone_half_angle_deg: float = 2.0,
        threshold_percent: float = 5.0,
    ) -> bool:
        """Return *True* if the track is detectable at this etching time.

        Convenience wrapper around :meth:`get_track_detectability`.
        """
        return self.get_track_detectability(
            etch_time_h=etch_time_h,
            min_radius_um=min_radius_um,
            min_depth_um=min_depth_um,
            min_cone_half_angle_deg=min_cone_half_angle_deg,
            threshold_percent=threshold_percent,
        )["detected"]

    # ------------------------------------------------------------------
    # plotting (delegated to plots module)
    # ------------------------------------------------------------------

    def plot_iso_time_contour(
        self, ax: matplotlib.axes.Axes, etching_time_h: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plot the iso-time contour on *ax*.

        Returns
        -------
        r_coords, z_coords : tuple[ndarray, ndarray]
        """
        r_coords, z_coords = self.get_iso_time_contour(etching_time_h)
        if r_coords.size > 0:
            ax.plot(
                r_coords,
                z_coords,
                color="white",
                linestyle="--",
                linewidth=2,
                label=f"Iso-time: {etching_time_h} hr",
            )
            ax.legend()
        return r_coords, z_coords

    def plot_track_radius(
        self,
        etch_time_h: float,
        ax: matplotlib.axes.Axes | None = None,
        show_contour: bool = True,
        threshold_percent: float = 5.0,
    ) -> tuple:
        """Plot iso-time contour with track-radius indicators.

        Returns
        -------
        fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        r_contour, z_contour = self.get_iso_time_contour(etch_time_h)
        track_radius = self.get_track_radius_um(etch_time_h, threshold_percent)

        if show_contour and len(r_contour) > 0:
            ax.plot(
                r_contour,
                z_contour,
                "b-",
                linewidth=2,
                label=f"Iso-time: {etch_time_h} hr",
            )

        V_bulk = self.etch_model.V_bulk_um_h
        z_bulk = V_bulk * etch_time_h
        ax.axhline(
            z_bulk,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Bulk reference: {z_bulk:.2f} um",
        )

        if not np.isnan(track_radius):
            ax.axvline(
                track_radius,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label=f"Track radius: {track_radius:.2f} um",
            )

        ax.set_xlabel("r / um")
        ax.set_ylabel("z / um")
        ax.set_title(f"Track radius at t = {etch_time_h} hr")
        ax.legend()
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
        return fig, ax

    def plot_map(
        self,
        name: str = "dose",
        grayscale_mode: bool = False,
        plot_contours: bool = True,
        annotate_figure: bool = True,
    ) -> tuple:
        """Plot the dose, etch-rate, or arrival-time map."""
        return plot_utils.plot_map(
            self,
            name=name,
            grayscale_mode=grayscale_mode,
            plot_contours=plot_contours,
            annotate_figure=annotate_figure,
        )

    def plot_LET_energy_profiles(self) -> tuple:
        """Plot energy and LET profiles vs. depth."""
        return plot_utils.plot_LET_energy_profiles(self)

    def calculate_RDD_Gy(self, E_MeV_u: float) -> np.ndarray:
        """Calculate the radial dose distribution at a given energy.

        Parameters
        ----------
        E_MeV_u : float
            Kinetic energy in MeV per nucleon.

        Returns
        -------
        ndarray
            Dose in Gy at each radial grid point.
        """
        return get_RDD_Gy(
            r_m=self._r_grid_um * 1e-6,
            E_MeV_u=E_MeV_u,
            particle_name=self.particle_name,
            RDD_name=self.RDD_name,
            material_name=self.material_name,
            normalise_to_LET=False,
        )
