"""EtchRateModel unit tests.

Tests the core etch-rate model: anchor filtering, monotonicity of the
interpolated V(dose) curve, save/load round-trip, velocity-increment
transforms, and V_max capping.
"""

import numpy as np
import pytest

from tracketch.etching.etch_rate_model import EtchRateModel


def test_init_cuts_requested_anchors_below_low_dose_anchor() -> None:
    doses = np.array([1e-6, 1e-4, 1e2, 1e4], dtype=float)
    velocities = np.array([1.8, 2.0, 3.0, 5.0], dtype=float)

    model = EtchRateModel(
        anchor_doses_Gy=doses,
        anchor_velocities_um_h=velocities,
        V_bulk_um_h=1.73,
    )

    assert np.all(model.anchor_doses_Gy >= model.LOW_DOSE_ANCHOR_GY)
    assert np.isclose(model.anchor_doses_Gy[0], model.LOW_DOSE_ANCHOR_GY)
    assert np.allclose(model.anchor_doses_Gy, np.array([1e-3, 1e2, 1e4]))
    assert np.allclose(model.anchor_velocities_um_h, np.array([1.73, 3.0, 5.0]))


def test_eval_is_monotonic_over_wide_dose_range() -> None:
    model = EtchRateModel(extrapolation_mode="pchip")

    dose_grid = np.logspace(-3, 12, 1000)
    values = np.asarray(model.eval(dose_grid), dtype=float)

    assert np.all(np.diff(values) >= -1e-12)


def test_eval_high_dose_not_below_lower_high_dose_point() -> None:
    model = EtchRateModel(extrapolation_mode="pchip")

    assert model.eval(1e11) >= model.eval(1e8)


def test_save_and_load_round_trip_creates_path(tmp_path) -> None:
    model = EtchRateModel(name="pytest-model", extrapolation_mode="clamp_last")
    out_path = tmp_path / "etch_rate_model.json"

    model.save_to_json(str(out_path))

    assert out_path.exists()

    loaded = EtchRateModel.load_from_json(str(out_path))

    assert loaded.name == "pytest-model"
    assert loaded.extrapolation_mode == "clamp_last"
    assert loaded.V_max_um_h == model.V_max_um_h
    assert np.allclose(loaded.anchor_doses_Gy, model.anchor_doses_Gy)
    assert np.allclose(loaded.anchor_velocities_um_h, model.anchor_velocities_um_h)


def test_log_velocity_increment_transform_round_trip() -> None:
    values = np.array([0.0, 1e-8, 0.02, 0.2], dtype=float)

    encoded = EtchRateModel.linear_log_velocity_increments_to_optimization(values)
    decoded = EtchRateModel.log_velocity_increments_to_linear(encoded)

    assert np.allclose(decoded, values, atol=1e-12, rtol=1e-7)


def test_update_from_log_velocity_increments_preserves_monotonicity() -> None:
    model = EtchRateModel(
        anchor_doses_Gy=np.logspace(2, 6, 5),
        V_bulk_um_h=1.73,
    )

    log_velocity_increments = np.array([0.0, 0.05, 0.12, 0.2, 0.1, 0.08], dtype=float)
    model.update_from_log_velocity_increments(log_velocity_increments)

    assert np.all(np.diff(model.anchor_velocities_um_h) >= -1e-12)
    recovered = model.get_log_velocity_increments()
    assert np.allclose(recovered, log_velocity_increments, atol=1e-12, rtol=1e-7)


def test_vmax_cap_is_enforced_in_update() -> None:
    model = EtchRateModel(
        anchor_doses_Gy=np.array([1e2, 1e4, 1e6], dtype=float),
        anchor_velocities_um_h=np.array([2.0, 3.0, 4.0], dtype=float),
        V_bulk_um_h=1.73,
        V_max_um_h=5.0,
    )

    with pytest.raises(ValueError, match="<= V_max_um_h"):
        model.update_velocities(np.array([2.0, 6.0, 7.0, 8.0], dtype=float))


def test_eval_respects_vmax_cap_on_high_dose_extrapolation() -> None:
    model = EtchRateModel(
        anchor_doses_Gy=np.array([1e2, 1e4, 1e6], dtype=float),
        anchor_velocities_um_h=np.array([2.0, 200.0, 400.0], dtype=float),
        V_bulk_um_h=1.73,
        V_max_um_h=500.0,
        extrapolation_mode="pchip",
    )

    assert model.eval(1e30) <= 500.0 + 1e-12


def test_vmax_tiny_roundoff_overshoot_is_clipped_not_rejected() -> None:
    model = EtchRateModel(
        anchor_doses_Gy=np.array([1e2, 1e4, 1e6], dtype=float),
        anchor_velocities_um_h=np.array([2.0, 3.0, 4.0], dtype=float),
        V_bulk_um_h=1.73,
        V_max_um_h=500.0,
    )

    tiny_overshoot = 500.0 + 1e-13
    model.update_velocities(
        np.array([2.0, 200.0, tiny_overshoot, tiny_overshoot], dtype=float)
    )

    assert np.max(model.anchor_velocities_um_h) <= 500.0
