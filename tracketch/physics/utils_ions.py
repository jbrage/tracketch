"""High-level ion-physics utilities.

Wrappers around SRIM tables and libamtrack that provide:

* LET and CSDA range lookups (multi-source),
* radial dose distributions (RDD) scaled to arbitrary materials,
* dose-map construction,
* energy-loss stepping, and
* longitudinal range-straggling convolution.
"""

import numpy as np
from functools import lru_cache
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import pyamtrack.libAT as libam

from tracketch.physics.libamtrack import RDD_libamtrack_Gy
from tracketch.utils import load_config
from tracketch.physics.SRIM.SRIM_lib import (
    SRIM_MeV_u_to_LET,
    get_SRIM_df,
)


# -- cached config ----------------------------------------------------


@lru_cache(maxsize=None)
def get_material_config():
    """Return the ``materials`` section of the material configuration (cached)."""
    return load_config("cfg_material_data.yaml")["materials"]


# -- LET --------------------------------------------------------------


def get_LET_keV_um(
    energy_MeV_u: float | np.ndarray,
    particle_name: str,
    material_name: str,
    source: str = "SRIM",
) -> float | np.ndarray:
    """Return LET in keV/um from the chosen stopping-power database.

    Parameters
    ----------
    energy_MeV_u : float or ndarray
        Kinetic energy in MeV per nucleon.
    particle_name : str
        Ion species, e.g. ``"12C"``.
    material_name : str
        Target material, e.g. ``"CR39"``.
    source : str
        ``"SRIM"`` (default) or ``"libamtrack"``.

    Returns
    -------
    float or ndarray
        LET in keV/um.

    Raises
    ------
    ValueError
        If *source* or *material_name* are not recognised.
    """
    if source == "SRIM":
        SRIM_obj = SRIM_MeV_u_to_LET(
            particle_name=particle_name, material_name=material_name
        )
        result = SRIM_obj(energy_MeV_u)
        return float(result) if np.ndim(result) == 0 else result

    elif source == "libamtrack":
        from tracketch.physics.libamtrack import get_LET_keV_um_libam

        materials = get_material_config()
        if material_name not in materials:
            raise ValueError(
                f"Material '{material_name}' not in config. "
                f"Available: {list(materials)}"
            )
        density_g_cm3 = materials[material_name]["density_g_cm3"]

        if isinstance(energy_MeV_u, (float, int)):
            return get_LET_keV_um_libam(
                particle_name=particle_name,
                E_MeV_u=energy_MeV_u,
                density_g_cm3=density_g_cm3,
            )
        return np.array(
            [
                get_LET_keV_um_libam(
                    particle_name=particle_name,
                    E_MeV_u=E,
                    density_g_cm3=density_g_cm3,
                )
                for E in energy_MeV_u
            ]
        )

    raise ValueError(f"Source '{source}' not recognised. Use 'SRIM' or 'libamtrack'.")


# -- RDD --------------------------------------------------------------


def get_RDD_Gy(
    r_m: np.ndarray,
    E_MeV_u: float,
    particle_name: str,
    RDD_name: str = "Cucinotta",
    material_name: str = "CR39",
    core_nm: float | None = None,
    normalise_to_LET: bool | None = None,
    LET_source: str = "SRIM",
    verbose: bool = False,
) -> np.ndarray:
    """Radial dose distribution scaled to *material_name*.

    The RDD is computed for water (via libamtrack) and then corrected:

    1. Radii are density-scaled to account for different electron ranges.
    2. A dose correction restores the radial integral after the range shift.
    3. A stopping-power ratio (SPR) converts dose-to-water into dose-to-
       target-material.
    4. Optionally, the profile is renormalised so that its radial integral
       equals the target-material LET.

    Parameters
    ----------
    r_m : ndarray
        Radial distances in metres.
    E_MeV_u : float
        Kinetic energy in MeV/u.
    particle_name : str
        Ion species.
    RDD_name : str
        RDD model name (default ``"Cucinotta"``).
    material_name : str
        Target material (default ``"CR39"``).
    core_nm : float or None
        Core radius override in nm.
    normalise_to_LET : bool or None
        Force LET normalisation.  ``None`` uses the model default.
    LET_source : str
        Stopping-power source for the SPR (default ``"SRIM"``).
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    ndarray
        Dose in Gy at each radius.
    """
    cfg_RDD_model = load_config("cfg_libamtrack.yaml")[RDD_name]

    # 1: density-scale the radii
    density_target = get_material_config()[material_name]["density_g_cm3"]
    density_water = 1.0
    range_correction = density_target / density_water
    density_adjusted_r_m = r_m.copy() * range_correction

    RDD_Gy = RDD_libamtrack_Gy(
        r_m=density_adjusted_r_m,
        E_MeV_u=E_MeV_u,
        particle_name=particle_name,
        RDD_name=RDD_name,
        core_nm=core_nm,
    )

    if normalise_to_LET is None:
        normalise_to_LET = cfg_RDD_model["LET_normalised"]

    # 2: dose correction for changed integration domain
    water_density_kg_m3 = 1e3
    integral_water = RDD_integral(r_m, RDD_Gy) * water_density_kg_m3
    integral_adjusted = RDD_integral(density_adjusted_r_m, RDD_Gy) * water_density_kg_m3
    if integral_water > 0:
        RDD_Gy *= integral_adjusted / integral_water

    # 3: stopping-power ratio
    LET_water = get_LET_keV_um(
        energy_MeV_u=E_MeV_u,
        particle_name=particle_name,
        material_name="water",
        source=LET_source,
    )
    LET_target = get_LET_keV_um(
        energy_MeV_u=E_MeV_u,
        particle_name=particle_name,
        material_name=material_name,
        source=LET_source,
    )
    spr = LET_target / LET_water
    RDD_Gy *= spr

    if verbose:
        print(
            f"\nTarget: {material_name} (density {density_target} g/cm^3)\n"
            f"  Range correction: {range_correction:.3f}\n"
            f"  SPR:              {spr:.3f}\n"
            f"  LET ({LET_source}):      {LET_target:.2f} keV/um"
        )

    # 4: LET normalisation
    if normalise_to_LET:
        integral = RDD_integral(r_m, RDD_Gy) * water_density_kg_m3
        norm = LET_target / integral if integral > 0 else 1.0
        RDD_Gy *= norm
        if verbose:
            print(
                f"  LET (integral):   {integral:.2f} keV/um\n"
                f"  Norm factor:      {norm:.3f}"
            )

    return RDD_Gy


def RDD_integral(
    r_m: np.ndarray, D_Gy: np.ndarray, density_g_cm3: float | None = None
) -> float:
    """Radial integral of the dose distribution.

    Computes

    .. math:: \\int 2\\pi\\, r\\, D(r)\\, \\mathrm{d}r

    and converts to keV/um (per unit density) via standard unit factors.

    Parameters
    ----------
    r_m : ndarray
        Radial positions in metres.
    D_Gy : ndarray
        Dose in Gy at each radius.
    density_g_cm3 : float or None
        If given, multiply by the density in kg/m^3.

    Returns
    -------
    float
        Integral value (keV/um if density is provided in kg/m^3 upstream).
    """
    valid = np.isfinite(D_Gy)
    integral = 2 * np.pi * np.trapezoid(D_Gy[valid] * r_m[valid], r_m[valid])

    # J/kg -> eV -> keV, m -> um
    joule_to_eV = 1 / 1.60217663e-19
    value = integral * joule_to_eV * 1e-6 / 1e3  # eV*m -> keV*um

    if density_g_cm3 is not None:
        value *= density_g_cm3 * 1e3  # g/cm^3 -> kg/m^3
    return value


# -- CSDA range -------------------------------------------------------


def get_CSDA_um(
    energy_MeV_u: float,
    particle_name: str,
    material_name: str,
    source: str = "SRIM",
) -> float:
    """Return the CSDA range in um from the chosen stopping-power database.

    Parameters
    ----------
    energy_MeV_u : float
        Kinetic energy in MeV/u.
    particle_name : str
        Ion species.
    material_name : str
        Target material.
    source : str
        ``"SRIM"`` (default) or ``"libamtrack"``.

    Returns
    -------
    float
        CSDA range in um.

    Raises
    ------
    ValueError
        If *source* or *material_name* are not recognised.
    """
    if source == "SRIM":
        from tracketch.physics.SRIM.SRIM_lib import SRIM_MeV_u_to_CSDA_um

        result = SRIM_MeV_u_to_CSDA_um(
            particle_name=particle_name, material_name=material_name
        )(energy_MeV_u)
        return float(result) if np.ndim(result) == 0 else result

    elif source == "libamtrack":
        from tracketch.physics.libamtrack import get_CSDA_range_um_libam

        materials = get_material_config()
        if material_name not in materials:
            raise ValueError(
                f"Material '{material_name}' not in config. "
                f"Available: {list(materials)}"
            )
        density = materials[material_name]["density_g_cm3"]
        if isinstance(energy_MeV_u, (float, int)):
            return get_CSDA_range_um_libam(
                particle_name=particle_name,
                E_MeV_u=energy_MeV_u,
                density_g_cm3=density,
            )
        return np.array(
            [
                get_CSDA_range_um_libam(
                    particle_name=particle_name,
                    E_MeV_u=E,
                    density_g_cm3=density,
                )
                for E in energy_MeV_u
            ]
        )

    elif source == "calculation":
        raise NotImplementedError("Analytic CSDA calculation not yet implemented.")

    raise ValueError(f"Source '{source}' not recognised. Use 'SRIM' or 'libamtrack'.")


# -- energy conversions -----------------------------------------------


def convert_MeV_to_MeV_u(Energy_MeV: float, particle_name: str) -> float:
    """Convert total kinetic energy (MeV) to energy per nucleon (MeV/u).

    Parameters
    ----------
    Energy_MeV : float
        Total kinetic energy in MeV.
    particle_name : str
        Ion species identifier.

    Returns
    -------
    float
        Energy in MeV/u.
    """
    p_no = libam.AT_particle_no_from_particle_name_single(particle_name)
    return float(libam.AT_E_MeV_u_from_E_MeV(Energy_MeV, p_no))


def convert_MeV_u_to_MeV(Energy_MeV_u: float, particle_name: str) -> float:
    """Convert energy per nucleon (MeV/u) to total kinetic energy (MeV).

    Parameters
    ----------
    Energy_MeV_u : float
        Energy in MeV per nucleon.
    particle_name : str
        Ion species identifier.

    Returns
    -------
    float
        Total kinetic energy in MeV.
    """
    p_no = libam.AT_particle_no_from_particle_name_single(particle_name)
    return float(libam.AT_E_MeV_from_E_MeV_u(Energy_MeV_u, p_no))


# -- energy loss through slabs ----------------------------------------


def get_energy_after_slab_MeV_u(
    E_initial_MeV_u: float,
    particle_name: str,
    slab_thickness_um: float,
    material_name: str = "CR39",
    source: str = "SRIM",
) -> float:
    """Residual energy after traversing a slab of given thickness.

    Steps through the slab in small increments, computing the LET at each
    step to update the energy.

    Parameters
    ----------
    E_initial_MeV_u : float
        Incident energy in MeV/u.
    particle_name : str
        Ion species.
    slab_thickness_um : float
        Slab thickness in um.
    material_name : str
        Target material (default ``"CR39"``).
    source : str
        Stopping-power source (default ``"SRIM"``).

    Returns
    -------
    float
        Residual energy in MeV/u (0 if the ion stops).
    """
    step_size_um = min(0.5, slab_thickness_um / 10)
    n_steps = max(1, int(slab_thickness_um / step_size_um))
    step_size_um = slab_thickness_um / n_steps

    for _ in range(n_steps):
        LET = get_LET_keV_um(
            energy_MeV_u=E_initial_MeV_u,
            particle_name=particle_name,
            material_name=material_name,
            source=source,
        )
        E_MeV = convert_MeV_u_to_MeV(E_initial_MeV_u, particle_name)
        E_MeV -= LET * step_size_um * 1e-3  # keV -> MeV

        if E_MeV <= 0:
            return 0.0

        E_initial_MeV_u = convert_MeV_to_MeV_u(E_MeV, particle_name)

    return max(E_initial_MeV_u, 0.0)


# -- dose map ---------------------------------------------------------


def get_dose_map_Gy(
    z_um: np.ndarray,
    r_um: np.ndarray,
    energy_MeV_u_array: np.ndarray,
    particle_name: str,
    material_name: str = "CR39",
    RDD_name: str = "Cucinotta",
    LET_source: str = "SRIM",
    core_radius_nm: float | None = None,
    normalise_to_LET: bool | None = None,
    verbose: bool = False,
    n_jobs: int = 1,
) -> np.ndarray:
    """Build a 2-D dose map by evaluating the RDD at each depth step.

    Parameters
    ----------
    z_um : ndarray, shape (n_z,)
        Depth coordinates in um.
    r_um : ndarray, shape (n_r,)
        Radial coordinates in um.
    energy_MeV_u_array : ndarray, shape (n_z,)
        Ion energy at each depth (from :func:`get_LET_energy_profile`).
    particle_name : str
        Ion species.
    material_name : str
        Target material.
    RDD_name : str
        RDD model name.
    LET_source : str
        Stopping-power source.
    core_radius_nm : float or None
        Core radius override.
    normalise_to_LET : bool or None
        Force LET normalisation.
    verbose : bool
        Print diagnostics.
    n_jobs : int
        Number of parallel workers for RDD evaluation.
        ``-1`` uses all available CPU cores (via :mod:`joblib`).
        Default ``1`` runs sequentially.

    Returns
    -------
    ndarray, shape (n_z, n_r)
        Local dose in Gy.
    """
    from joblib import Parallel, delayed

    r_m = r_um * 1e-6

    # B: deduplicate -- many z-slices can share the same energy (e.g. beyond
    # the Bragg peak where energy=0, or at the surface before significant loss)
    rounded = np.round(energy_MeV_u_array, decimals=4)
    unique_energies, inverse_idx = np.unique(rounded, return_inverse=True)
    active_mask = unique_energies > 0
    active_energies = unique_energies[active_mask]

    _kwargs = dict(
        particle_name=particle_name,
        RDD_name=RDD_name,
        material_name=material_name,
        core_nm=core_radius_nm,
        normalise_to_LET=normalise_to_LET,
        LET_source=LET_source,
        verbose=verbose,
    )

    rdd_lookup = np.zeros((len(unique_energies), len(r_m)))

    if active_mask.any():
        # C: evaluate RDD for each unique active energy in parallel
        rdds = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(get_RDD_Gy)(r_m, float(E), **_kwargs) for E in active_energies
        )
        rdd_lookup[active_mask] = np.array(rdds)

    return rdd_lookup[inverse_idx]


# -- core averaging ---------------------------------------------------


def get_mean_core_value(
    metric_map: np.ndarray,
    r_um: np.ndarray,
    core_r_threshold_um: float = 1e-1,
    min_core_r_um: float = 1e-3,
) -> np.ndarray:
    """Average a map over the track core (small-*r* region) at each depth.

    Parameters
    ----------
    metric_map : ndarray, shape (n_z, n_r)
        Dose or etch-rate map.
    r_um : ndarray, shape (n_r,)
        Radial coordinates in um.
    core_r_threshold_um : float
        Outer boundary of the core region in um.
    min_core_r_um : float
        Inner boundary of the core region in um.

    Returns
    -------
    ndarray, shape (n_z,)
        Mean value within the core at each depth.  ``nan`` where no valid
        data exists.
    """
    if min_core_r_um >= core_r_threshold_um:
        core_r_threshold_um = min_core_r_um * 2

    idx_core = np.where((r_um <= core_r_threshold_um) & (r_um >= min_core_r_um))[0]
    if len(idx_core) == 0:
        idx_core = np.array([0])

    result = np.full(metric_map.shape[0], np.nan)
    non_nan_rows = ~np.all(np.isnan(metric_map), axis=1)

    if np.any(non_nan_rows) and len(idx_core) > 0:
        core_slice = metric_map[non_nan_rows][:, idx_core]
        has_valid = ~np.all(np.isnan(core_slice), axis=1)
        if np.any(has_valid):
            core_result = np.full(len(core_slice), np.nan)
            core_result[has_valid] = np.nanmean(core_slice[has_valid], axis=1)
            result[non_nan_rows] = core_result

    return result


# -- range straggling -------------------------------------------------


def get_range_straggling_um(
    energy_MeV_u: float,
    particle_name: str,
    material_name: str,
) -> float:
    """Longitudinal range straggling from SRIM data.

    Parameters
    ----------
    energy_MeV_u : float
        Kinetic energy in MeV/u.
    particle_name : str
        Ion species.
    material_name : str
        Target material.

    Returns
    -------
    float
        Straggling in um.  Returns 0 if the particle is not available
        in SRIM (justified for heavy ions with negligible straggling).
    """
    try:
        SRIM_df = get_SRIM_df(particle_name, material_name)
    except (ValueError, FileNotFoundError):
        return 0.0

    interp_fn = interp1d(
        SRIM_df["Energy_MeV_u"],
        SRIM_df["longitudinal_straggling_um"],
        bounds_error=False,
        fill_value=(
            SRIM_df["longitudinal_straggling_um"].min(),
            SRIM_df["longitudinal_straggling_um"].max(),
        ),  # type: ignore
    )
    return float(interp_fn(energy_MeV_u))


def apply_range_straggling(
    dz_um_list: np.ndarray,
    metric: np.ndarray,
    longitudinal_straggling_um: float,
    n_sigma: float = 1,
) -> np.ndarray:
    """Broaden a depth profile by Gaussian range-straggling convolution.

    Parameters
    ----------
    dz_um_list : ndarray
        Depth grid in um (assumed uniform spacing).
    metric : ndarray
        Profile to broaden (same length as *dz_um_list*).
    longitudinal_straggling_um : float
        1-sigma straggling width in um.
    n_sigma : float
        Number of sigma to apply (0 = no straggling).

    Returns
    -------
    ndarray
        Broadened profile.
    """
    if n_sigma == 0:
        return metric

    step_size_um = dz_um_list[1] - dz_um_list[0]
    sigma_bins = longitudinal_straggling_um / step_size_um * n_sigma
    return gaussian_filter1d(metric, sigma=sigma_bins)


# -- LET / energy profiles -------------------------------------------


def get_LET_energy_profile(
    dz_um_list: np.ndarray,
    energy_MeV_u: float,
    particle_name: str,
    material_name: str,
    range_straggling: bool = True,
    n_straggling_sigma: float = 1,
    source: str = "SRIM",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute energy and LET profiles along depth.

    For ``source="SRIM"`` the profiles are computed analytically by
    inverting the CSDA-range table (vectorised, no stepping loop).
    For ``source="libamtrack"`` the original iterative stepping is used.

    Optionally applies longitudinal range straggling as a Gaussian
    convolution.

    Parameters
    ----------
    dz_um_list : ndarray
        Depth grid in um.
    energy_MeV_u : float
        Incident energy in MeV/u.
    particle_name : str
        Ion species.
    material_name : str
        Target material.
    range_straggling : bool
        Apply Gaussian straggling (default ``True``).
    n_straggling_sigma : float
        Number of straggling sigma (default 1).
    source : str
        Stopping-power database (default ``"SRIM"``).

    Returns
    -------
    energy_profile : ndarray
        Kinetic energy in MeV/u at each depth.
    LET_profile : ndarray
        LET in keV/um at each depth.
    """
    if source == "SRIM":
        from tracketch.physics.SRIM.SRIM_lib import (
            SRIM_CSDA_um_to_MeV_u,
            SRIM_MeV_u_to_CSDA_um,
            SRIM_MeV_u_to_LET,
        )

        _csda_fwd = SRIM_MeV_u_to_CSDA_um(particle_name, material_name)
        _csda_inv = SRIM_CSDA_um_to_MeV_u(particle_name, material_name)
        _let_interp = SRIM_MeV_u_to_LET(particle_name, material_name)

        R0 = float(_csda_fwd(energy_MeV_u))
        remaining = np.maximum(R0 - dz_um_list, 0.0)
        energy_arr = np.asarray(_csda_inv(remaining), dtype=float)
        LET_arr = np.asarray(_let_interp(energy_arr), dtype=float)

        # zero out depths at or beyond the CSDA stopping point
        stopped = dz_um_list >= R0
        energy_arr[stopped] = 0.0
        LET_arr[stopped] = 0.0

    else:
        # libamtrack: iterative stepping (source-agnostic fallback)
        dz_spacing = dz_um_list[1] - dz_um_list[0]

        energy_arr = np.zeros_like(dz_um_list)
        LET_arr = np.zeros_like(dz_um_list)
        current_energy = energy_MeV_u

        for i in range(len(dz_um_list)):
            LET = get_LET_keV_um(
                energy_MeV_u=current_energy,
                particle_name=particle_name,
                material_name=material_name,
                source=source,
            )
            LET_arr[i] = LET
            energy_arr[i] = current_energy

            current_energy = get_energy_after_slab_MeV_u(
                E_initial_MeV_u=current_energy,
                particle_name=particle_name,
                slab_thickness_um=dz_spacing,
                material_name=material_name,
                source=source,
            )
            if current_energy <= 0:
                break

    if range_straggling:
        straggling_um = get_range_straggling_um(
            energy_MeV_u,
            particle_name,
            material_name,
        )
        LET_arr = apply_range_straggling(
            dz_um_list,
            LET_arr,
            straggling_um,
            n_straggling_sigma,
        )
        energy_arr = apply_range_straggling(
            dz_um_list,
            energy_arr,
            straggling_um,
            n_straggling_sigma,
        )

    return energy_arr, LET_arr


# -- dose probability density -----------------------------------------


def get_f_dose_distribution(
    E_MeV_u: float,
    particle_name: str,
    RDD_name: str = "Cucinotta",
    material_name: str = "CR39",
    core_nm: float | None = None,
    LET_source: str = "SRIM",
    f_cutoff: float = 1e-15,
    d_min_Gy: float = 1e-5,
    r_min_m: float = 5e-11,
    r_max_m: float = 1e-2,
    npoints: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the local-dose probability density *f(d)* for an ion track.

    The PDF describes the fraction of the track cross-section receiving a
    local dose between *d* and *d + dd*.  Derived from the RDD by
    inverting *r(d)* and differentiating the cumulative area fraction.

    Parameters
    ----------
    E_MeV_u : float
        Kinetic energy in MeV/u.
    particle_name : str
        Ion species.
    RDD_name : str
        RDD model name.
    material_name : str
        Target material.
    core_nm : float or None
        Core radius override.
    LET_source : str
        Stopping-power source.
    f_cutoff : float
        Bins where ``f*d < f_cutoff`` are discarded.
    d_min_Gy : float
        Minimum dose threshold.
    r_min_m : float
        Inner radial sampling bound in metres.
    r_max_m : float
        Outer radial sampling bound in metres.
    npoints : int
        Number of radial sample points.

    Returns
    -------
    f : ndarray
        Normalised dose PDF in Gy^{-1}.
    midpoints_Gy : ndarray
        Dose bin midpoints in Gy.
    dd_Gy : ndarray
        Dose bin widths in Gy.

    Raises
    ------
    ValueError
        If too few valid RDD points are found.
    """
    r_m = np.logspace(np.log10(r_min_m), np.log10(r_max_m), npoints)
    d_Gy = get_RDD_Gy(
        r_m=r_m,
        E_MeV_u=E_MeV_u,
        particle_name=particle_name,
        RDD_name=RDD_name,
        material_name=material_name,
        core_nm=core_nm,
        LET_source=LET_source,
    )

    valid = np.isfinite(d_Gy) & (d_Gy >= d_min_Gy)
    r_m, d_Gy = r_m[valid], d_Gy[valid]

    if len(r_m) < 2:
        raise ValueError(
            f"Too few valid RDD points for {particle_name} at {E_MeV_u} MeV/u. "
            "Try increasing r_max_m or decreasing d_min_Gy."
        )

    r_max_valid = r_m.max()

    # Invert: build r(d) by sorting ascending in dose
    sort_idx = np.argsort(d_Gy)
    d_sorted, r_sorted = d_Gy[sort_idx], r_m[sort_idx]

    # Dense log-spaced dose grid
    d_grid = np.logspace(
        np.log10(d_sorted.min()), np.log10(d_sorted.max()), npoints * 2
    )
    r_interp = np.interp(d_grid, d_sorted, r_sorted)

    # F(d) = 1 - r(d)^2 / r_max^2  =>  f(d) = dF/dd
    F = 1.0 - r_interp**2 / r_max_valid**2
    f = np.gradient(F, d_grid)[:-1]

    left, right = d_grid[:-1], d_grid[1:]
    dd = right - left
    midpoints = (left + right) * 0.5

    use = (f * midpoints > f_cutoff) & (midpoints > d_min_Gy)
    f, midpoints, dd = f[use], midpoints[use], dd[use]

    # Normalise: integral f(d) dd = 1
    f /= np.nansum(f * dd)

    return f, midpoints, dd
