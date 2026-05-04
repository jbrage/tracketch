"""SRIM stopping-power tables and interpolation utilities."""

from tracketch.physics.SRIM.SRIM_lib import (
    get_bragg_correction,
    set_bragg_correction,
)

__all__ = [
    "get_bragg_correction",
    "set_bragg_correction",
]
