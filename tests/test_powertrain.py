import numpy as np
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.vehicle.powertrain import max_engine_drive_force, engine_torque_nm
from src.physics.dynamics import (
    G, axle_loads, max_traction_accel, max_braking_decel, corner_speed_limit,
)


def test_engine_torque_interpolates_the_curve():
    v = Vehicle(torque_curve=[[1000, 100], [5000, 200], [7000, 150]])
    assert np.isclose(engine_torque_nm(v, 1000), 100.0)
    assert np.isclose(engine_torque_nm(v, 3000), 150.0)
    assert np.isclose(engine_torque_nm(v, 7000), 150.0)


def test_drive_force_falls_at_high_speed_when_power_limited():
    vehicle = Vehicle(max_power=150000.0, max_traction_force=30000.0,
                      downforce_coefficient=0.0, drag_coefficient=0.0)
    f_low = max_engine_drive_force(vehicle, 12.0)
    f_high = max_engine_drive_force(vehicle, 45.0)
    assert f_low > 0
    assert f_high < f_low


def test_accel_shifts_load_to_the_rear():
    vehicle = Vehicle()
    n_f0, n_r0 = axle_loads(vehicle, v=20.0, a_lon=0.0)
    n_f, n_r = axle_loads(vehicle, v=20.0, a_lon=5.0)
    assert n_r > n_r0
    assert n_f < n_f0
    assert np.isclose(n_f + n_r, n_f0 + n_r0)


def test_braking_shifts_load_to_the_front():
    vehicle = Vehicle()
    n_f0, n_r0 = axle_loads(vehicle, v=20.0, a_lon=0.0)
    n_f, n_r = axle_loads(vehicle, v=20.0, a_lon=-8.0)
    assert n_f > n_f0
    assert n_r < n_r0


def test_rwd_accelerates_harder_than_fwd_all_else_equal():
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    rwd = Vehicle(drivetrain="RWD", weight_dist_front=0.55, max_power=400000.0,
                  max_traction_force=50000.0, drag_coefficient=0.0,
                  downforce_coefficient=0.0)
    fwd = Vehicle(drivetrain="FWD", weight_dist_front=0.55, max_power=400000.0,
                  max_traction_force=50000.0, drag_coefficient=0.0,
                  downforce_coefficient=0.0)
    a_rwd = max_traction_accel(rwd, tire, v=8.0)
    a_fwd = max_traction_accel(fwd, tire, v=8.0)
    assert a_rwd > a_fwd + 0.2


def test_awd_accelerates_at_least_as_hard_as_rwd_at_low_speed():
    tire = TireModel(mu=1.1, load_sensitivity=0.0)
    kwargs = dict(weight_dist_front=0.50, max_power=400000.0,
                  max_traction_force=50000.0, drag_coefficient=0.0,
                  downforce_coefficient=0.0, awd_torque_front=0.5)
    rwd = Vehicle(drivetrain="RWD", **kwargs)
    awd = Vehicle(drivetrain="AWD", **kwargs)
    assert max_traction_accel(awd, tire, v=8.0) >= max_traction_accel(rwd, tire, v=8.0)


def test_json_round_trip_keeps_gears_and_curve():
    import tempfile, os
    v = Vehicle.from_json(os.path.join(
        os.path.dirname(__file__), "..", "data", "vehicles", "subaru_brz.json",
    ))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "brz.json")
        v.to_json(path)
        v2 = Vehicle.from_json(path)
    assert v2.drivetrain == "RWD"
    assert len(v2.gear_ratios) == 6
    assert len(v2.torque_curve) >= 3
    assert np.isclose(v2.final_drive, 4.10)


def test_one_g_corner_still_matches_classic_formula_per_axle():
    vehicle = Vehicle(downforce_coefficient=0.0, drag_coefficient=0.0,
                      frontal_area=2.0, tire_mu=1.2, top_speed=100.0)
    tire = TireModel(mu=1.2, load_sensitivity=0.15)
    radius = 50.0
    v = corner_speed_limit(1.0 / radius, vehicle, tire)
    assert np.isclose(v, np.sqrt(1.2 * G * radius), rtol=1e-5)
