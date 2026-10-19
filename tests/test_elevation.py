import numpy as np
from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.physics.dynamics import G, corner_speed_limit, normal_load
from src.simulation.simulator import Simulator


def default_track():
    return Track.rounded_rectangle(width_m=400.0, height_m=200.0,
                                    corner_radii=(30.0, 20.0, 45.0, 25.0),
                                    track_width=12.0, ds=2.0)


def test_flat_track_has_zero_elevation():
    track = default_track()
    assert np.allclose(track.z, 0.0)
    assert np.allclose(track.sin_theta, 0.0, atol=1e-9)
    assert np.allclose(track.cos_theta, 1.0, atol=1e-9)
    assert np.allclose(track.kappa_v, 0.0, atol=1e-9)


def test_zero_z_lap_matches_implicit_flat():
    track = default_track()
    vehicle = Vehicle()
    t0 = Simulator(track, vehicle, TireModel(mu=vehicle.tire_mu)).run().lap_time
    t1 = Simulator(track.with_z(np.zeros_like(track.s)), vehicle,
                   TireModel(mu=vehicle.tire_mu)).run().lap_time
    assert np.isclose(t0, t1, rtol=1e-12, atol=1e-9)


def test_cosine_hill_is_zero_at_the_seams():
    track = default_track().with_hill(s_center=140.0, width=180.0, height=18.0)
    assert float(np.max(track.z)) > 17.0
    # Hill sits on the first straight, away from s = 0.
    assert abs(float(track.z[0])) < 1e-6
    assert abs(float(track.z[-1])) < 1e-6


def test_corner_speed_matches_mu_g_r_cos_theta():
    vehicle = Vehicle(mass=1200.0, downforce_coefficient=0.0, drag_coefficient=0.0,
                      tire_mu=1.2, top_speed=100.0)
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    radius = 50.0
    k = 1.0 / radius
    cos_th = 0.94
    v = corner_speed_limit(k, vehicle, tire, cos_theta=cos_th)
    assert np.isclose(v, np.sqrt(tire.mu * G * radius * cos_th), rtol=1e-6)


def test_crest_unloads_and_dip_adds_grip():
    vehicle = Vehicle(mass=1200.0, downforce_coefficient=0.0, drag_coefficient=0.0,
                      tire_mu=1.2, top_speed=100.0)
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    k = 1.0 / 50.0
    v_flat = corner_speed_limit(k, vehicle, tire, kappa_v=0.0)
    v_crest = corner_speed_limit(k, vehicle, tire, kappa_v=-0.008)
    v_dip = corner_speed_limit(k, vehicle, tire, kappa_v=0.008)
    assert v_crest < v_flat - 0.3
    assert v_dip > v_flat + 0.3


def test_normal_load_uses_cos_theta():
    vehicle = Vehicle(downforce_coefficient=0.0)
    n_flat = normal_load(vehicle, v=0.0, cos_theta=1.0)
    n_slope = normal_load(vehicle, v=0.0, cos_theta=0.8)
    assert np.isclose(n_flat, vehicle.mass * G)
    assert np.isclose(n_slope, vehicle.mass * G * 0.8)


def test_hill_lap_is_slower_and_climb_is_slower_than_flat():
    flat = default_track()
    hill = flat.with_hill(s_center=140.0, width=200.0, height=20.0)
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    r_flat = Simulator(flat, vehicle, tire).run()
    r_hill = Simulator(hill, vehicle, tire).run()
    assert r_hill.lap_time > r_flat.lap_time + 0.05
    i_climb = int(np.argmax(hill.sin_theta))
    assert r_hill.profile["v"][i_climb] < r_flat.profile["v"][i_climb] - 0.2


def test_spa_has_an_eau_rouge_valley():
    track = Track.spa_francorchamps()
    assert float(np.ptp(track.z)) > 25.0
    i_low = int(np.argmin(track.z))
    s_low = float(track.s[i_low])
    # Named window for Eau Rouge on the TUM/OSM centerline.
    assert 600.0 < s_low < 1300.0
    i_high = int(np.argmax(track.z))
    assert float(track.s[i_high]) > 2000.0
