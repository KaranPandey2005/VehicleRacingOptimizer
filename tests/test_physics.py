import numpy as np
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.physics.dynamics import corner_speed_limit, max_traction_accel, max_braking_decel, normal_load, G


def make_vehicle_no_aero(**overrides):
    defaults = dict(mass=1200.0, downforce_coefficient=0.0, drag_coefficient=0.0,
                     frontal_area=2.0, tire_mu=1.2, top_speed=100.0,
                     max_power=200000.0, max_traction_force=10000.0, max_brake_force=15000.0)
    defaults.update(overrides)
    return Vehicle(**defaults)


def test_corner_speed_limit_matches_classic_formula_with_no_aero():
    # With zero downforce, v_max should reduce exactly to sqrt(mu*g*r)
    vehicle = make_vehicle_no_aero()
    tire = TireModel(mu=vehicle.tire_mu)
    radius = 50.0
    curvature = 1.0 / radius
    v = corner_speed_limit(curvature, vehicle, tire)
    v_expected = np.sqrt(tire.mu * G * radius)
    assert np.isclose(v, v_expected, rtol=1e-6)


def test_corner_speed_limit_zero_curvature_returns_top_speed():
    vehicle = make_vehicle_no_aero(top_speed=77.0)
    tire = TireModel(mu=1.2)
    assert corner_speed_limit(0.0, vehicle, tire) == 77.0


def test_downforce_increases_corner_speed_limit_vs_no_downforce():
    tire = TireModel(mu=1.2)
    v_no_aero = make_vehicle_no_aero(downforce_coefficient=0.0)
    v_aero = make_vehicle_no_aero(downforce_coefficient=3.0)
    radius = 40.0
    k = 1.0 / radius
    speed_no_aero = corner_speed_limit(k, v_no_aero, tire)
    speed_aero = corner_speed_limit(k, v_aero, tire)
    assert speed_aero > speed_no_aero


def test_normal_load_increases_with_speed_when_downforce_present():
    vehicle = make_vehicle_no_aero(downforce_coefficient=2.5)
    tire = TireModel(mu=1.2)
    n_low = normal_load(vehicle, v=10.0)
    n_high = normal_load(vehicle, v=50.0)
    assert n_high > n_low
    assert np.isclose(normal_load(vehicle, v=0.0), vehicle.mass * G)


def test_max_traction_accel_is_positive_and_decreases_with_speed_when_power_limited():
    vehicle = make_vehicle_no_aero(max_power=150000.0, max_traction_force=20000.0)  # power-limited regime
    tire = TireModel(mu=1.2)
    a_low = max_traction_accel(vehicle, tire, v=10.0)
    a_high = max_traction_accel(vehicle, tire, v=40.0)
    assert a_low > 0
    assert a_high > 0
    assert a_high < a_low  # power-limited: force = P/v decreases as v increases


def test_max_traction_accel_is_grip_limited_at_low_speed_for_high_power_car():
    # Huge power, but grip should cap the force at low speed via friction circle
    vehicle = make_vehicle_no_aero(max_power=5_000_000.0, max_traction_force=1_000_000.0, tire_mu=1.0)
    tire = TireModel(mu=1.0)
    a = max_traction_accel(vehicle, tire, v=5.0)
    a_grip_limit = tire.mu * G  # a = mu*g when purely grip-limited, no aero
    assert a <= a_grip_limit + 1e-6


def test_max_braking_decel_positive():
    vehicle = make_vehicle_no_aero()
    tire = TireModel(mu=1.2)
    a_brake = max_braking_decel(vehicle, tire, v=30.0)
    assert a_brake > 0


def test_combined_slip_reduces_longitudinal_limit_in_a_corner():
    # Grip-limited car near a corner-speed limit so leftover Fx actually binds.
    vehicle = make_vehicle_no_aero(max_power=5_000_000.0, max_traction_force=1_000_000.0)
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    v = 28.0
    k = 1.0 / 80.0
    a_straight = max_traction_accel(vehicle, tire, v, curvature=0.0)
    a_corner = max_traction_accel(vehicle, tire, v, curvature=k)
    assert a_corner < a_straight - 0.05
    a_brake_straight = max_braking_decel(vehicle, tire, v, curvature=0.0)
    a_brake_corner = max_braking_decel(vehicle, tire, v, curvature=k)
    assert a_brake_corner < a_brake_straight - 0.05


def test_u_max_reduces_corner_speed():
    vehicle = make_vehicle_no_aero()
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    k = 1.0 / 50.0
    v_full = corner_speed_limit(k, vehicle, tire, u_max=1.0)
    v_safe = corner_speed_limit(k, vehicle, tire, u_max=0.75)
    assert np.isclose(v_safe, v_full * np.sqrt(0.75), rtol=1e-6)


def test_higher_mu_gives_higher_corner_speed_and_braking():
    vehicle = make_vehicle_no_aero()
    tire_low = TireModel(mu=0.8)
    tire_high = TireModel(mu=1.4)
    k = 1.0 / 50.0
    assert corner_speed_limit(k, vehicle, tire_high) > corner_speed_limit(k, vehicle, tire_low)
    assert max_braking_decel(vehicle, tire_high, v=30) > max_braking_decel(vehicle, tire_low, v=30)
