"""SRIM output parsing, interpolation factories, and bragg-variant selection."""

from functools import lru_cache
from pathlib import Path
from typing import cast
import warnings

import pandas as pd
from scipy.interpolate import interp1d

from tracketch.utils import get_project_root, load_config


_SRIM_STOPPING_ALIAS: dict[str, str] = {"2H": "1H", "3H": "1H"}
_SRIM_ALIAS_MASS_NUMBER: dict[str, int] = {"2H": 2, "3H": 3}
_SRIM_SOURCE_MASS_NUMBER: dict[str, int] = {"1H": 1}

_active_bragg_correction: int = 0


def _interp1d_with_tuple_fill(*args, **kwargs) -> interp1d:
    """Wrap `interp1d` so tuple `fill_value` remains type-checker friendly."""
    return cast(interp1d, interp1d(*args, **kwargs))


def get_bragg_correction() -> int:
    """Return the active bragg correction percentage."""
    return _active_bragg_correction


def set_bragg_correction(pct: int) -> None:
    """Set the active bragg correction percentage and clear interpolation caches."""
    global _active_bragg_correction
    _active_bragg_correction = int(pct)
    SRIM_MeV_u_to_LET.cache_clear()
    SRIM_LET_to_MeV_u.cache_clear()
    SRIM_MeV_u_to_CSDA_um.cache_clear()
    SRIM_CSDA_um_to_MeV_u.cache_clear()


# -- interpolation factories ------------------------------------------


@lru_cache(maxsize=None)
def SRIM_MeV_u_to_LET(particle_name: str, material_name: str) -> interp1d:
    """Return an interpolator: energy (MeV/u) -> LET (keV/um)."""
    SRIM_df = get_SRIM_df(particle_name, material_name)
    return _interp1d_with_tuple_fill(
        SRIM_df.Energy_MeV_u,
        SRIM_df.LET_keV_um,
        kind="linear",
        bounds_error=False,
        fill_value=(
            float(SRIM_df.LET_keV_um.iloc[0]),
            float(SRIM_df.LET_keV_um.iloc[-1]),
        ),
    )


@lru_cache(maxsize=None)
def SRIM_LET_to_MeV_u(particle_name: str, material_name: str) -> interp1d:
    """Return an interpolator: LET (keV/um) -> energy (MeV/u)."""
    SRIM_df = get_SRIM_df(particle_name, material_name)
    return _interp1d_with_tuple_fill(
        SRIM_df.LET_keV_um,
        SRIM_df.Energy_MeV_u,
        kind="linear",
        bounds_error=False,
        fill_value=(
            float(SRIM_df.Energy_MeV_u.iloc[0]),
            float(SRIM_df.Energy_MeV_u.iloc[-1]),
        ),
    )


@lru_cache(maxsize=None)
def SRIM_MeV_u_to_CSDA_um(particle_name: str, material_name: str) -> interp1d:
    """Return an interpolator: energy (MeV/u) -> CSDA range (um)."""
    SRIM_df = get_SRIM_df(particle_name, material_name)
    return _interp1d_with_tuple_fill(
        SRIM_df.Energy_MeV_u,
        SRIM_df.CSDA_um,
        kind="linear",
        bounds_error=False,
        fill_value=(
            float(SRIM_df.CSDA_um.iloc[0]),
            float(SRIM_df.CSDA_um.iloc[-1]),
        ),
    )


@lru_cache(maxsize=None)
def SRIM_CSDA_um_to_MeV_u(particle_name: str, material_name: str) -> interp1d:
    """Return an interpolator: CSDA range (um) -> energy (MeV/u)."""
    SRIM_df = get_SRIM_df(particle_name, material_name)
    return _interp1d_with_tuple_fill(
        SRIM_df.CSDA_um,
        SRIM_df.Energy_MeV_u,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, float(SRIM_df.Energy_MeV_u.iloc[-1])),
    )


# -- config / paths ---------------------------------------------------


@lru_cache(maxsize=None)
def load_SRIM_cfg():
    """Load SRIM configuration (cached)."""
    return load_config("cfg_material_data.yaml")


def _get_variant_folder(cfg_key: str, bragg_correction: int | None = None) -> Path:
    """Resolve a raw or processed folder for a bragg correction variant."""
    pct = _active_bragg_correction if bragg_correction is None else bragg_correction
    cfg = load_SRIM_cfg()
    base = Path(cfg[cfg_key])
    if not base.is_absolute():
        base = (get_project_root() / base).resolve()
    return base / f"bragg_{pct}"


def get_SRIM_df_folder(bragg_correction: int | None = None) -> Path:
    """Return the processed CSV folder for a bragg correction variant."""
    return _get_variant_folder("SRIM_processed_path", bragg_correction)


def get_SRIM_raw_folder(bragg_correction: int | None = None) -> Path:
    """Return the raw `.dat` folder for a bragg correction variant."""
    return _get_variant_folder("SRIM_raw_path", bragg_correction)


def get_SRIM_df(particle_name: str, material_name: str | None = None) -> pd.DataFrame:
    """Load pre-processed SRIM DataFrame for the given particle."""
    srim_particle = _SRIM_STOPPING_ALIAS.get(particle_name, particle_name)
    SRIM_particle_df = pd.read_csv(
        get_SRIM_df_folder() / f"SRIM_{srim_particle}_df.csv"
    )

    if particle_name in _SRIM_STOPPING_ALIAS:
        source_name = _SRIM_STOPPING_ALIAS[particle_name]
        scale = (
            _SRIM_ALIAS_MASS_NUMBER[particle_name]
            / _SRIM_SOURCE_MASS_NUMBER[source_name]
        )
        for col in (
            "Energy_MeV",
            "CSDA_um",
            "longitudinal_straggling_um",
            "lateral_straggling_um",
        ):
            if col in SRIM_particle_df.columns:
                SRIM_particle_df[col] = SRIM_particle_df[col] * scale
        SRIM_particle_df["particle"] = particle_name

    if material_name is None:
        return SRIM_particle_df
    return SRIM_particle_df[SRIM_particle_df["material"] == material_name].reset_index(
        drop=True
    )


# -- parser helpers ----------------------------------------------------


def convert_SRIM_distance_to_um(range_unit: str, item: str) -> float:
    """Convert a SRIM distance value to micrometres."""
    factors = {"A": 1e-4, "um": 1.0, "mm": 1e3, "m": 1e6}
    return float(item) * factors[range_unit]


def convert_SRIM_output_to_df(
    particle_name: str,
    material_name: str,
    SRIM_folder: str | Path,
) -> pd.DataFrame:
    """Parse a raw SRIM `.dat` output file into a tidy DataFrame."""
    cfg = load_SRIM_cfg()
    density = cfg["materials"][material_name]["density_g_cm3"]
    expected_units = cfg["SRIM_LET_units"]
    LET_unit_conversion = density * 100

    SRIM_filename = Path(SRIM_folder) / f"{material_name}_{particle_name}.dat"

    start_data_line = 100
    rows: list[dict] = []

    with open(SRIM_filename, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if line.startswith(" Stopping Units") and expected_units not in line:
            warnings.warn(
                f"SRIM LET units are not '{expected_units}': {line.strip()}",
                stacklevel=2,
            )

        if line.endswith("Straggling\n"):
            start_data_line = idx + 2

        if idx >= start_data_line:
            if "----------------------------------------" in line:
                break
            items = [i for i in line.split(" ") if i != ""]
            energy_factors = {"eV": 1e-6, "keV": 1e-3, "MeV": 1.0, "GeV": 1e3}
            energy_MeV = float(items[0]) * energy_factors[items[1]]
            LET_keV_um = float(items[2]) * LET_unit_conversion
            CSDA_um = convert_SRIM_distance_to_um(items[5], items[4])
            long_strag = convert_SRIM_distance_to_um(items[7], items[6])
            lat_strag = convert_SRIM_distance_to_um(items[9], items[8])
            rows.append(
                {
                    "Energy_MeV": energy_MeV,
                    "LET_keV_um": LET_keV_um,
                    "CSDA_um": CSDA_um,
                    "longitudinal_straggling_um": long_strag,
                    "lateral_straggling_um": lat_strag,
                }
            )

    result_df = pd.DataFrame(rows)
    for col in result_df.columns:
        result_df[col] = pd.to_numeric(result_df[col]).round(5)
    result_df["particle"] = particle_name
    result_df["material"] = material_name
    return result_df


# -- batch generation --------------------------------------------------


def generate_SRIM_dfs(
    bragg_correction: int, SRIM_folder: str | Path | None = None
) -> None:
    """Generate pre-processed CSV DataFrames for one bragg correction variant."""
    from tracketch.physics.utils_ions import convert_MeV_to_MeV_u

    cfg = load_SRIM_cfg()
    particles = [p for p in cfg["particles"] if p not in _SRIM_STOPPING_ALIAS]
    materials = cfg["materials"]

    if SRIM_folder is None:
        raw_folder = get_SRIM_raw_folder(bragg_correction)
    else:
        raw_folder = Path(SRIM_folder)
        if not raw_folder.is_absolute():
            raw_folder = (get_project_root() / raw_folder).resolve()

    df_folder = get_SRIM_df_folder(bragg_correction)
    df_folder.mkdir(parents=True, exist_ok=True)

    print(
        f"Generating SRIM DataFrames (Bragg {bragg_correction}%):\n"
        f"  particles : {particles}\n"
        f"  materials : {list(materials)}\n"
        f"  source    : {raw_folder}\n"
        f"  output    : {df_folder}\n"
    )

    for particle_name in particles:
        output_df = pd.DataFrame()
        for material_name in materials:
            df = convert_SRIM_output_to_df(particle_name, material_name, raw_folder)
            df["Energy_MeV_u"] = (
                df["Energy_MeV"]
                .apply(lambda x: convert_MeV_to_MeV_u(x, particle_name))
                .round(5)
            )
            output_df = pd.concat([output_df, df], ignore_index=True)
        out_path = df_folder / f"SRIM_{particle_name}_df.csv"
        output_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")
