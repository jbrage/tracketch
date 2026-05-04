"""Tests for TrackSimulator core functionality.

These tests pin current behavior before refactoring. They use a fixed
12C @ 1.0 MeV/u scenario with a coarse grid for speed.
"""

import numpy as np
import pytest

from tracketch import TrackSimulator, load_etchrate_model, EtchRateModel


# ---------------------------------------------------------------------------
# Shared fixture: one simulator instance reused across read-only tests
# ---------------------------------------------------------------------------

RZ_LIMS = {
    "r_min_um": 1e-4,
    "r_max_um": 15,
    "z_min_um": 0,
    "z_max_um": 25,
    "n_points_r": 80,
    "n_points_z": 80,
}

PARTICLE = "12C"
ENERGY = 1.0  # MeV/u
ETCH_TIME_HR = 1.0


@pytest.fixture(scope="module")
def sim():
    """Create a single TrackSimulator for 12C @ 1 MeV/u (coarse grid)."""
    model = load_etchrate_model("Doerschel_etching")
    return TrackSimulator(
        particle_name=PARTICLE,
        start_energy_MeV_u=ENERGY,
        etch_model=model,
        rz_lims_dict=RZ_LIMS,
    )


# ---------------------------------------------------------------------------
# Construction & grids
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_grids_created(self, sim):
        assert sim._r_grid_um.size > 0
        assert sim._z_grid_um.size > 0

    def test_grid_shape_matches_config(self, sim):
        assert sim._r_grid_um.size == RZ_LIMS["n_points_r"]
        assert sim._z_grid_um.size == RZ_LIMS["n_points_z"]

    def test_grid_bounds(self, sim):
        assert sim._r_grid_um.min() >= RZ_LIMS["r_min_um"]
        assert sim._r_grid_um.max() == pytest.approx(RZ_LIMS["r_max_um"], rel=1e-6)
        assert sim._z_grid_um.min() >= RZ_LIMS["z_min_um"]
        assert sim._z_grid_um.max() == pytest.approx(RZ_LIMS["z_max_um"], rel=1e-6)

    def test_repr(self, sim):
        r = repr(sim)
        assert "12C" in r
        assert "TrackSimulator" in r

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Invalid arrival_time_method"):
            TrackSimulator(
                particle_name=PARTICLE,
                start_energy_MeV_u=ENERGY,
                arrival_time_method_name="invalid",
                rz_lims_dict=RZ_LIMS,
            )

    def test_invalid_connectivity_raises(self):
        with pytest.raises(ValueError, match="dijkstra_connectivity"):
            TrackSimulator(
                particle_name=PARTICLE,
                start_energy_MeV_u=ENERGY,
                dijkstra_connectivity=5,
                rz_lims_dict=RZ_LIMS,
            )


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------


class TestPhysics:
    def test_csda_range_positive(self, sim):
        assert sim.CSDA_range_um > 0

    def test_energy_profile_shape(self, sim):
        assert sim.energy_profile_MeV_u.shape == sim._z_grid_um.shape

    def test_energy_profile_decreasing(self, sim):
        """Energy should generally decrease with depth (particle loses energy)."""
        e = sim.energy_profile_MeV_u
        # First value should be close to start energy
        assert e[0] == pytest.approx(ENERGY, rel=0.1)
        # Energy at end should be less than at start
        assert e[-1] < e[0]

    def test_let_profile_shape(self, sim):
        assert sim.LET_profile_keV_um.shape == sim._z_grid_um.shape

    def test_let_profile_positive(self, sim):
        """LET values should be non-negative."""
        assert np.all(sim.LET_profile_keV_um >= 0)


# ---------------------------------------------------------------------------
# Dose map
# ---------------------------------------------------------------------------


class TestDoseMap:
    def test_shape(self, sim):
        n_z = sim._z_grid_um.size
        n_r = sim._r_grid_um.size
        assert sim.dose_map.shape == (n_z, n_r)

    def test_non_negative(self, sim):
        assert np.all(sim.dose_map[np.isfinite(sim.dose_map)] >= 0)

    def test_dose_decreases_with_radius(self, sim):
        """Dose should generally decrease with radial distance at a fixed depth."""
        mid_z = sim.dose_map.shape[0] // 2
        dose_row = sim.dose_map[mid_z, :]
        finite = np.isfinite(dose_row) & (dose_row > 0)
        if np.sum(finite) > 2:
            vals = dose_row[finite]
            # dose at small r should be larger than at large r
            assert vals[0] > vals[-1]


# ---------------------------------------------------------------------------
# Etch rate map
# ---------------------------------------------------------------------------


class TestEtchRateMap:
    def test_shape(self, sim):
        assert sim.etch_rate_map.shape == sim.dose_map.shape

    def test_at_least_bulk(self, sim):
        """Etch rate should be >= V_bulk everywhere."""
        V_bulk = sim.etch_model.V_bulk_um_h
        assert np.all(sim.etch_rate_map >= V_bulk - 1e-10)

    def test_positive(self, sim):
        assert np.all(sim.etch_rate_map > 0)


# ---------------------------------------------------------------------------
# Arrival time map
# ---------------------------------------------------------------------------


class TestArrivalTimeMap:
    def test_shape(self, sim):
        assert sim.arrival_time_map.shape == sim.dose_map.shape

    def test_non_negative(self, sim):
        finite = np.isfinite(sim.arrival_time_map)
        assert np.all(sim.arrival_time_map[finite] >= 0)

    def test_monotonic_along_z_at_r0(self, sim):
        """Arrival time at r=0 should increase with depth."""
        col = sim.arrival_time_map[:, 0]
        finite = np.isfinite(col)
        vals = col[finite]
        assert np.all(np.diff(vals) >= -1e-10)

    def test_surface_time_near_zero(self, sim):
        """Arrival time at z=0 should be close to zero."""
        surface = sim.arrival_time_map[0, :]
        assert np.nanmin(surface) < 0.1


# ---------------------------------------------------------------------------
# Iso-time contour
# ---------------------------------------------------------------------------


class TestIsoTimeContour:
    def test_returns_arrays(self, sim):
        r, z = sim.get_iso_time_contour(ETCH_TIME_HR)
        assert isinstance(r, np.ndarray)
        assert isinstance(z, np.ndarray)

    def test_non_empty(self, sim):
        r, z = sim.get_iso_time_contour(ETCH_TIME_HR)
        assert r.size > 0
        assert z.size == r.size

    def test_finite_values(self, sim):
        r, z = sim.get_iso_time_contour(ETCH_TIME_HR)
        assert np.all(np.isfinite(r))
        assert np.all(np.isfinite(z))

    def test_contour_within_grid(self, sim):
        r, z = sim.get_iso_time_contour(ETCH_TIME_HR)
        assert np.all(r >= 0)
        assert np.all(z >= 0)
        assert np.all(r <= RZ_LIMS["r_max_um"] * 1.01)
        assert np.all(z <= RZ_LIMS["z_max_um"] * 1.01)


# ---------------------------------------------------------------------------
# Track radius
# ---------------------------------------------------------------------------


class TestTrackRadius:
    def test_returns_float(self, sim):
        r = sim.get_track_radius_um(etch_time_h=ETCH_TIME_HR)
        assert isinstance(r, (float, np.floating))

    def test_positive_for_detectable_track(self, sim):
        r = sim.get_track_radius_um(etch_time_h=ETCH_TIME_HR)
        assert np.isfinite(r)
        assert r > 0

    def test_radius_within_grid(self, sim):
        r = sim.get_track_radius_um(etch_time_h=ETCH_TIME_HR)
        assert r <= RZ_LIMS["r_max_um"]

    def test_radius_increases_with_etch_time(self, sim):
        r1 = sim.get_track_radius_um(etch_time_h=0.5)
        r2 = sim.get_track_radius_um(etch_time_h=2.0)
        if np.isfinite(r1) and np.isfinite(r2):
            assert r2 >= r1


# ---------------------------------------------------------------------------
# Track length
# ---------------------------------------------------------------------------


class TestTrackLength:
    def test_returns_float(self, sim):
        length = sim.get_track_length_um(etch_time_h=ETCH_TIME_HR)
        assert isinstance(length, (float, np.floating))

    def test_positive_for_detectable_track(self, sim):
        length = sim.get_track_length_um(etch_time_h=ETCH_TIME_HR)
        assert np.isfinite(length)
        assert length > 0

    def test_relative_vs_absolute(self, sim):
        rel = sim.get_track_length_um(
            etch_time_h=ETCH_TIME_HR, relative_to_surface=True
        )
        abs_ = sim.get_track_length_um(
            etch_time_h=ETCH_TIME_HR, relative_to_surface=False
        )
        V_bulk = sim.etch_model.V_bulk_um_h
        expected_diff = V_bulk * ETCH_TIME_HR
        assert abs_ == pytest.approx(rel + expected_diff, rel=0.01)

    def test_list_input(self, sim):
        times = [0.5, 1.0, 2.0]
        lengths = sim.get_track_length_um(etch_time_h=times)
        assert isinstance(lengths, np.ndarray)
        assert lengths.shape == (3,)

    def test_length_increases_with_etch_time(self, sim):
        l1 = sim.get_track_length_um(etch_time_h=0.5)
        l2 = sim.get_track_length_um(etch_time_h=2.0)
        if np.isfinite(l1) and np.isfinite(l2):
            assert l2 >= l1


# ---------------------------------------------------------------------------
# Track detectability
# ---------------------------------------------------------------------------


class TestDetectability:
    def test_returns_dict_keys(self, sim):
        result = sim.get_track_detectability(etch_time_h=ETCH_TIME_HR)
        expected_keys = {
            "detected",
            "radius_um",
            "depth_um",
            "cone_half_angle_deg",
            "radius_ok",
            "depth_ok",
            "angle_ok",
        }
        assert set(result.keys()) == expected_keys

    def test_is_track_detected_returns_bool(self, sim):
        result = sim.is_track_detected(etch_time_h=ETCH_TIME_HR)
        assert isinstance(result, bool)

    def test_12c_1mev_detected_at_1hr(self, sim):
        """12C at 1 MeV/u should produce a visible track after 1 hour."""
        assert sim.is_track_detected(etch_time_h=ETCH_TIME_HR)


# ---------------------------------------------------------------------------
# Update etch model
# ---------------------------------------------------------------------------


class TestUpdateEtchModel:
    def test_update_changes_etch_rate_map(self):
        """Changing the etch model should change the etch rate map."""
        model1 = load_etchrate_model("Doerschel_etching")
        s = TrackSimulator(
            particle_name=PARTICLE,
            start_energy_MeV_u=ENERGY,
            etch_model=model1,
            rz_lims_dict=RZ_LIMS,
        )
        etch_map_before = s.etch_rate_map.copy()

        # Create a model with higher V_bulk
        model2 = EtchRateModel(
            anchor_doses_Gy=model1.anchor_doses_Gy,
            V_bulk_um_h=model1.V_bulk_um_h * 2,
            anchor_velocities_um_h=np.array(model1.anchor_velocities_um_h) * 2,
            name="doubled",
        )
        s.update_etch_model_and_recalculate(model2)

        assert not np.allclose(s.etch_rate_map, etch_map_before)

    def test_dose_map_unchanged_after_etch_update(self):
        """Dose map should NOT change when only etch model is updated."""
        model = load_etchrate_model("Doerschel_etching")
        s = TrackSimulator(
            particle_name=PARTICLE,
            start_energy_MeV_u=ENERGY,
            etch_model=model,
            rz_lims_dict=RZ_LIMS,
        )
        dose_before = s.dose_map.copy()

        model2 = EtchRateModel(
            anchor_doses_Gy=model.anchor_doses_Gy,
            V_bulk_um_h=model.V_bulk_um_h * 2,
            anchor_velocities_um_h=np.array(model.anchor_velocities_um_h) * 2,
            name="doubled",
        )
        s.update_etch_model_and_recalculate(model2)

        np.testing.assert_array_equal(s.dose_map, dose_before)


# ---------------------------------------------------------------------------
# Different particles (smoke tests)
# ---------------------------------------------------------------------------


class TestDifferentParticles:
    @pytest.mark.parametrize(
        "particle,energy",
        [
            ("1H", 1.0),
            ("4He", 1.0),
            ("12C", 10.0),
        ],
    )
    def test_simulator_runs(self, particle, energy):
        sim = TrackSimulator(
            particle_name=particle,
            start_energy_MeV_u=energy,
            rz_lims_dict=RZ_LIMS,
        )
        assert sim.dose_map.shape[0] > 0
        assert sim.etch_rate_map.shape[0] > 0
        assert sim.arrival_time_map.shape[0] > 0

        r, z = sim.get_iso_time_contour(1.0)
        assert r.size > 0
