"""Etch-rate modelling: dose-to-velocity calibration curves."""

from tracketch.etching.etch_rate_model import EtchRateModel
from tracketch.etching.etch_rate_model_io import (
    load_etchrate_model,
    save_etchrate_model,
    default_etch_rate_model,
)

__all__ = [
    "EtchRateModel",
    "load_etchrate_model",
    "save_etchrate_model",
    "default_etch_rate_model",
]
