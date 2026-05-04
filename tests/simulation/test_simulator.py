"""TrackSimulator integration tests.

Verifies that pre-calibrated etch-rate models load correctly, and that
the full simulation pipeline (dose map -> etch rate -> arrival time ->
iso-time contour) produces valid output.
"""

import numpy as np
import pytest
from tracketch import TrackSimulator, load_etchrate_model


def test_load_etchrate_model_default():
    model = load_etchrate_model("Doerschel_etching")
    assert model.V_bulk_um_h > 0
    assert len(model.anchor_doses_Gy) > 0


def test_load_etchrate_model_missing_per_particle_file_raises():
    with pytest.raises(FileNotFoundError):
        load_etchrate_model("12C")


def test_simulator_creates_contour():
    model = load_etchrate_model("Doerschel_etching")
    sim = TrackSimulator(
        particle_name="12C",
        start_energy_MeV_u=270.0,
        etch_model=model,
        arrival_time_method_name="dijkstra_numba",
    )
    r, z = sim.get_iso_time_contour(etching_time_h=1.0)
    assert len(r) > 0
    assert len(z) == len(r)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(z))
