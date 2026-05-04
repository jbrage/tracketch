"""Ion physics: RDD, LET, CSDA range, and energy-loss utilities."""

from tracketch.physics.utils_ions import (
    get_RDD_Gy,
    get_LET_keV_um,
    get_CSDA_um,
    get_dose_map_Gy,
    get_mean_core_value,
    RDD_integral,
    convert_MeV_to_MeV_u,
    convert_MeV_u_to_MeV,
    get_energy_after_slab_MeV_u,
    apply_range_straggling,
    get_range_straggling_um,
    get_LET_energy_profile,
)

__all__ = [
    "get_RDD_Gy",
    "get_LET_keV_um",
    "get_CSDA_um",
    "get_dose_map_Gy",
    "get_mean_core_value",
    "RDD_integral",
    "convert_MeV_to_MeV_u",
    "convert_MeV_u_to_MeV",
    "get_energy_after_slab_MeV_u",
    "apply_range_straggling",
    "get_range_straggling_um",
    "get_LET_energy_profile",
]
