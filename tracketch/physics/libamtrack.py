# %%
"""Interface to libamtrack for RDD, LET, and CSDA calculations.

CR-39 is not defined in libamtrack.  All calculations are performed for
water (or PMMA for density-matching) and then scaled to CR-39 by the
caller.
"""

import numpy as np
import pyamtrack.libAT as libam

from tracketch.utils import load_config


# -- material helpers -------------------------------------------------


def get_material_number_and_density(
    density_g_cm3: float,
) -> tuple[int, float]:
    """Return the libamtrack material number closest to the requested density.

    Parameters
    ----------
    density_g_cm3 : float
        Target density in g/cm^3.  Must be in (0, 10].

    Returns
    -------
    material_no : int
        libamtrack material number.
    material_density_g_cm3 : float
        Actual density of the matched material.

    Raises
    ------
    ValueError
        If *density_g_cm3* is outside (0, 10].
    """
    if not 0 < density_g_cm3 <= 10:
        raise ValueError(f"density_g_cm3 must be in (0, 10], got {density_g_cm3}")

    if np.isclose(density_g_cm3, 1.0, atol=1e-1):
        material_name = "Water, Liquid"
    else:
        material_name = "PMMA"

    material_no = libam.AT_material_number_from_name(material_name)
    material_density_g_cm3 = libam.AT_density_g_cm3_from_material_no(material_no)
    return material_no, material_density_g_cm3


# -- LET --------------------------------------------------------------


def get_LET_keV_um_libam(
    particle_name: str, E_MeV_u: float, density_g_cm3: float = 1.31
) -> float:
    """Compute LET in keV/um for a given ion and energy via libamtrack.

    The stopping power is computed for the closest libamtrack material and
    then linearly scaled to *density_g_cm3*.

    Parameters
    ----------
    particle_name : str
        Ion species, e.g. ``"12C"``.
    E_MeV_u : float
        Kinetic energy in MeV per nucleon.
    density_g_cm3 : float
        Target material density (default 1.31, CR-39).

    Returns
    -------
    float
        LET in keV/um.
    """
    material_no, material_density_g_cm3 = get_material_number_and_density(
        density_g_cm3=density_g_cm3
    )
    particle_no = libam.AT_particle_no_from_particle_name_single(particle_name)

    stopping_power_keV_um = [0]
    libam.AT_Stopping_Power(
        p_stopping_power_source="PSTAR",
        p_E_MeV_u=[E_MeV_u],
        p_particle_no=[particle_no],
        p_material_no=material_no,
        p_stopping_power_keV_um=stopping_power_keV_um,
    )
    LET_keV_um = stopping_power_keV_um[0] * (density_g_cm3 / material_density_g_cm3)
    return LET_keV_um


# -- RDD --------------------------------------------------------------


def RDD_libamtrack_Gy(
    r_m: np.ndarray,
    E_MeV_u: float,
    particle_name: str,
    RDD_name: str = "Cucinotta",
    core_nm: float | None = None,
) -> np.ndarray:
    """Compute radial dose distribution in Gy at radii *r_m* (metres).

    Uses libamtrack for **water**.  Material scaling must be done by
    the caller (see :func:`tracketch.physics.get_RDD_Gy`).

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
    core_nm : float or None
        Core radius override in nm.  ``None`` uses the model default.

    Returns
    -------
    ndarray
        Dose in Gy.  Values below ``d_min_Gy`` are set to ``nan``.

    Raises
    ------
    ValueError
        If *RDD_name* is not defined in the configuration.
    """
    cfg_libam = load_config("cfg_libamtrack.yaml")
    if RDD_name not in cfg_libam:
        raise ValueError(
            f"RDD model '{RDD_name}' not defined. "
            f"Available: {[k for k in cfg_libam if k != 'stopping_power_source_no']}"
        )
    cfg_RDD_model = cfg_libam[RDD_name]

    material_no, _ = get_material_number_and_density(density_g_cm3=1.0)
    stopping_power_source_no = cfg_libam["stopping_power_source_no"]
    particle_no = libam.AT_particle_no_from_particle_name_single(particle_name)

    rdd_model_no = libam.RDDModels[cfg_RDD_model["RDD_model"]].value
    er_model_no = libam.AT_ERModels[cfg_RDD_model["ER_model"]].value

    core_m = (core_nm if core_nm is not None else cfg_RDD_model["a0_nm"]) * 1e-9
    rdd_parameters = [
        cfg_RDD_model["r_min_m"],
        core_m,
        cfg_RDD_model["d_min_Gy"],
    ]

    result_dose_Gy_tmp = [0.0] * len(r_m)
    libam.AT_D_RDD_Gy(
        p_r_m=r_m.tolist(),
        p_E_MeV_u=E_MeV_u,
        p_particle_no=particle_no,
        p_material_no=material_no,
        p_rdd_model=rdd_model_no,
        p_rdd_parameter=rdd_parameters,
        p_er_model=er_model_no,
        p_stopping_power_source_no=stopping_power_source_no,
        p_D_RDD_Gy=result_dose_Gy_tmp,
    )

    RDD_Gy = np.array(result_dose_Gy_tmp)
    RDD_Gy[RDD_Gy * 0.1 < cfg_RDD_model["d_min_Gy"]] = np.nan
    return RDD_Gy


# -- energy after slab ------------------------------------------------


def get_energy_after_slab_MeV_um_libam(
    E_initial_MeV_u: float,
    particle_name: str,
    slab_thickness_um: float,
    material_name: str,
) -> float:
    """Residual energy after traversing a slab, computed via libamtrack.

    Only works for materials defined in libamtrack (``"Liquid, Water"``
    or ``"PMMA"``).

    Parameters
    ----------
    E_initial_MeV_u : float
        Incident energy in MeV/u.
    particle_name : str
        Ion species.
    slab_thickness_um : float
        Slab thickness in um.
    material_name : str
        ``"Liquid, Water"`` or ``"PMMA"``.

    Returns
    -------
    float
        Energy after the slab in MeV/u.

    Raises
    ------
    ValueError
        If *material_name* is not supported.
    """
    if material_name not in ("Liquid, Water", "PMMA"):
        raise ValueError(
            f"material_name must be 'Liquid, Water' or 'PMMA', got '{material_name}'"
        )

    particle_no = libam.AT_particle_no_from_particle_name_single(particle_name)
    material_no = libam.AT_material_number_from_name(material_name)
    slab_thickness_m = slab_thickness_um * 1e-6

    energy_out = [0]
    libam.AT_CSDA_energy_after_slab_E_MeV_u_multi(
        p_E_initial_MeV_u=[E_initial_MeV_u],
        p_particle_no=[particle_no],
        p_material_no=material_no,
        p_slab_thickness_m=slab_thickness_m,
        p_E_final_MeV_u=energy_out,
    )
    return energy_out[0]


# -- CSDA range -------------------------------------------------------


def get_CSDA_range_um_libam(
    particle_name: str, E_MeV_u: float, density_g_cm3: float = 1.31
) -> float:
    """CSDA range in um via libamtrack, scaled to the target density.

    libamtrack returns a density-normalised range (g/cm^2), which is
    converted to um using *density_g_cm3*.

    Parameters
    ----------
    particle_name : str
        Ion species.
    E_MeV_u : float
        Kinetic energy in MeV/u.
    density_g_cm3 : float
        Target material density (default 1.31, CR-39).

    Returns
    -------
    float
        CSDA range in um.
    """
    material_no, _ = get_material_number_and_density(density_g_cm3=density_g_cm3)
    particle_no = libam.AT_particle_no_from_particle_name_single(particle_name)

    CSDA_g_cm2 = libam.AT_CSDA_range_g_cm2_single(
        p_E_final_MeV_u=0,
        p_E_initial_MeV_u=E_MeV_u,
        p_material_no=material_no,
        p_particle_no=particle_no,
    )
    return CSDA_g_cm2 / density_g_cm3 * 1e4


if __name__ == "__main__":
    # quick test
    particle_no = libam.AT_particle_no_from_particle_name_single("3H")
    print(particle_no)

    print(get_CSDA_range_um_libam("3H", 10))
    print(get_CSDA_range_um_libam("2H", 10))
    print(get_CSDA_range_um_libam("1H", 10))

    particle_no = libam.AT_particle_no_from_particle_name_single("3H")
    print(particle_no)
    print(libam.AT_E_MeV_from_E_MeV_u(p_E_MeV_u=10, p_particle_no=particle_no))
