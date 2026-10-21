import numpy as np
from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.physics.dynamics import corner_speed_limit
from src.simulation.simulator import Simulator
from src.optimization.racing_line import apex_bias_m
from src.optimization.trajectory import (
    _detect_corners, _greedy_corners_qss, _offset_from_z,
)
from src.weather.weather import Weather, Condition
from src.weather.wet_line import (
    spatial_grip_scale, rubber_reference_offset, WET_FILM, WET_RUBBER_PEAK,
)


def default_track():
    return Track.rounded_rectangle(width_m=400.0, height_m=200.0,
                                    corner_radii=(30.0, 20.0, 45.0, 25.0),
                                    track_width=12.0, ds=2.0)


def test_dry_and_damp_spatial_map_is_ones():
    track = default_track()
    n = np.zeros_like(track.s)
    g_dry = spatial_grip_scale(track, n, Weather(condition=Condition.DRY))
    g_damp = spatial_grip_scale(track, n, Weather(condition=Condition.DAMP))
    assert np.allclose(g_dry, 1.0)
    assert np.allclose(g_damp, 1.0)


def test_wet_map_is_deterministic():
    track = default_track().with_hill(s_center=100.0, width=140.0, height=-10.0)
    n_ref = rubber_reference_offset(track)
    wx = Weather(condition=Condition.WET)
    g1 = spatial_grip_scale(track, n_ref, wx)
    g2 = spatial_grip_scale(track, n_ref, wx)
    assert np.array_equal(g1, g2)


def test_on_rubber_corner_faster_than_edge_or_puddle():
    track = default_track().with_hill(s_center=100.0, width=140.0, height=-12.0)
    wx = Weather(condition=Condition.WET)
    n_ref = rubber_reference_offset(track)
    half = 0.5 * track.width
    n_edge = np.full_like(n_ref, 0.95 * half)
    n_center = np.zeros_like(n_ref)
    g_on = spatial_grip_scale(track, n_ref, wx)
    g_edge = spatial_grip_scale(track, n_edge, wx)
    g_mid = spatial_grip_scale(track, n_center, wx)
    assert float(np.mean(g_on)) > float(np.mean(g_mid))
    assert float(np.mean(g_on)) > float(np.mean(g_edge))
    assert float(np.max(g_on)) <= WET_RUBBER_PEAK + 1e-9
    assert float(np.min(g_edge)) >= 0.35 - 1e-12
    assert float(np.mean(g_mid)) < WET_FILM + 0.12

    i_apex = int(np.argmax(np.abs(track.curvature)))
    vehicle = Vehicle(mass=1200.0, downforce_coefficient=0.0, drag_coefficient=0.0,
                      tire_mu=1.2, top_speed=100.0)
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    k = float(track.curvature[i_apex])
    v_on = corner_speed_limit(k, vehicle, tire, u_max=float(g_on[i_apex]))
    v_edge = corner_speed_limit(k, vehicle, tire, u_max=float(g_edge[i_apex]))
    v_mid = corner_speed_limit(k, vehicle, tire, u_max=float(g_mid[i_apex]))
    assert v_on > v_edge + 0.3
    assert v_on > v_mid + 0.15


def test_wet_centerline_still_slower_than_damp_and_dry():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    dry = Simulator(track, vehicle, tire, mode="attack",
                    weather=Weather(condition=Condition.DRY)).run()
    damp = Simulator(track, vehicle, tire, mode="attack",
                     weather=Weather(condition=Condition.DAMP)).run()
    wet = Simulator(track, vehicle, tire, mode="attack",
                    weather=Weather(condition=Condition.WET)).run()
    assert dry.lap_time < damp.lap_time < wet.lap_time
    assert wet.grip_modifier < 1.0 - 1e-6
    assert "grip_s" in wet.profile
    assert np.all(wet.profile["grip_s"] < 1.0)


def test_wet_greedy_offset_differs_from_dry():
    """Greedy only (4 corners). Wet rubber/gutters should move n(s) vs dry."""
    track = default_track().with_hill(s_center=90.0, width=130.0, height=-10.0)
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    corners = _detect_corners(track)
    assert len(corners) >= 4
    n_max = max(track.width / 2.0 - 0.5, 0.05 * track.width / 2.0)
    apex = apex_bias_m(vehicle)
    wx = Weather(condition=Condition.WET)
    grip_wet = lambda line: spatial_grip_scale(track, line.get("offset"), wx)
    z_dry = _greedy_corners_qss(
        track, vehicle, tire, 1.0, n_max, 0.5, corners, apex,
    )
    z_wet = _greedy_corners_qss(
        track, vehicle, tire, 1.0, n_max, 0.5, corners, apex,
        grip_s=grip_wet,
    )
    n_dry = _offset_from_z(track, z_dry, corners, apex_shift=apex)
    n_wet = _offset_from_z(track, z_wet, corners, apex_shift=apex)
    assert float(np.mean(np.abs(n_dry - n_wet))) > 0.05
