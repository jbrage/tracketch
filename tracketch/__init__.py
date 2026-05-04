"""tracketch -- simulation of ion tracks in CR-39 nuclear track detectors."""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Public constants -- enumerate valid names for user guidance
# ---------------------------------------------------------------------------

#: Materials supported by tracketch (CR39 and water).
MATERIALS: tuple[str, ...] = ("CR39", "water")

#: Particles with SRIM stopping-power data files.
#: When ``stopping_power_source="SRIM"`` (the default), ``particle_name``
#: must be one of these.  Switch to ``stopping_power_source="libamtrack"``
#: to use any nuclide recognised by libamtrack (e.g. ``"56Fe"``, ``"238U"``).
SRIM_PARTICLES: tuple[str, ...] = (
    "1H",
    "2H",
    "3H",
    "4He",
    "7Li",
    "9Be",
    "11B",
    "12C",
    "14N",
    "16O",
)

from tracketch.simulation.simulator import TrackSimulator
from tracketch.etching.etch_rate_model import EtchRateModel
from tracketch.etching.etch_rate_model_io import (
    load_etchrate_model,
    save_etchrate_model,
)
from tracketch.physics import (
    get_RDD_Gy,
    get_LET_keV_um,
    get_CSDA_um,
    convert_MeV_to_MeV_u,
    convert_MeV_u_to_MeV,
)

__all__ = [
    "TrackSimulator",
    "EtchRateModel",
    "load_etchrate_model",
    "save_etchrate_model",
    "get_RDD_Gy",
    "get_LET_keV_um",
    "get_CSDA_um",
    "convert_MeV_to_MeV_u",
    "convert_MeV_u_to_MeV",
]
