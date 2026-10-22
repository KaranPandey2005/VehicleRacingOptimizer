import numpy as np
from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.physics.dynamics import G, corner_speed_limit
from src.simulation.simulator import Simulator


def default_track():
    return Track.rounded_rectangle(width_m=400.0, height_m=200.0,
                                    corner_radii=(30.0, 20.0, 45.0, 25.0),
                                    track_width=12.0, ds=2.0)


def test_spa_and_rectangle_have_zero_bank():
    rect = default_track()
    spa = Track.spa_francorchamps()
    assert np.allclose(rect.phi, 0.0)
    assert np.allclose(spa.phi, 0.0)
    assert np.allclose(spa.sin_phi, 0.0, atol=1e-12)


def test_unbanked_corner_still_matches_mu_g_r():
    vehicle = Vehicle(mass=1200.0, downforce_coefficient=0.0, drag_coefficient=0.0,
                      tire_mu=1.2, top_speed=100.0)
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    radius = 50.0
    k = 1.0 / radius
    v = corner_speed_limit(k, vehicle, tire, sin_phi=0.0)
    assert np.isclose(v, np.sqrt(tire.mu * G * radius), rtol=1e-6)


def test_into_bank_raises_corner_speed_off_camber_cuts_it():
    vehicle = Vehicle(mass=1200.0, downforce_coefficient=0.0, drag_coefficient=0.0,
                      tire_mu=1.2, top_speed=100.0)
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    k = 1.0 / 50.0
    v_flat = corner_speed_limit(k, vehicle, tire, sin_phi=0.0)
    v_in = corner_speed_limit(k, vehicle, tire, sin_phi=0.12)
    v_off = corner_speed_limit(k, vehicle, tire, sin_phi=-0.12)
    assert v_in > v_flat + 0.4
    assert v_off < v_flat - 0.4


def test_with_bank_raises_qss_speed_in_the_window():
    track = default_track()
    i = int(np.argmax(np.abs(track.curvature)))
    sign = float(np.sign(track.curvature[i]))
    if sign == 0.0:
        sign = 1.0
    s_c = float(track.s[i])
    banked = track.with_bank(s_c, 90.0, 0.18 * sign)
    assert float(np.max(np.abs(banked.phi))) > 0.15
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    flat = Simulator(track, vehicle, tire, mode="attack").run()
    bank = Simulator(banked, vehicle, tire, mode="attack").run()
    length = float(track.length)
    d = (track.s - s_c + 0.5 * length) % length - 0.5 * length
    mask = np.abs(d) < 40.0
    assert float(np.min(bank.profile["v"][mask])) > float(np.min(flat.profile["v"][mask])) + 0.15
