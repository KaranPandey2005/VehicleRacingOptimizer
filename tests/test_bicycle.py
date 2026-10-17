import numpy as np
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.vehicle.bicycle import lever_arms, lateral_axle_forces
from src.physics.dynamics import G, corner_speed_limit, max_traction_accel


def test_lever_arms_match_static_weight_split():
    vehicle = Vehicle(wheelbase=2.5, weight_dist_front=0.60)
    a, b, L = lever_arms(vehicle)
    assert np.isclose(L, 2.5)
    assert np.isclose(b / L, 0.60)
    assert np.isclose(a + b, L)


def test_bicycle_no_aero_still_matches_sqrt_mu_g_r():
    vehicle = Vehicle(downforce_coefficient=0.0, drag_coefficient=0.0,
                      tire_mu=1.2, top_speed=100.0)
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    radius = 50.0
    v = corner_speed_limit(1.0 / radius, vehicle, tire)
    assert np.isclose(v, np.sqrt(1.2 * G * radius), rtol=1e-6)


def test_aero_offset_from_weight_split_slows_the_corner():
    """Bicycle: the light axle saturates first. Point-mass would ignore Cop."""
    tire = TireModel(mu=1.2, load_sensitivity=0.0)
    k = 1.0 / 40.0
    kwargs = dict(mass=1200.0, weight_dist_front=0.50, downforce_coefficient=3.0,
                  frontal_area=2.0, drag_coefficient=0.0, tire_mu=1.2, top_speed=100.0)
    balanced = Vehicle(center_of_pressure=0.50, **kwargs)
    rear = Vehicle(center_of_pressure=0.10, **kwargs)
    assert corner_speed_limit(k, balanced, tire) > corner_speed_limit(k, rear, tire) + 0.3


def test_higher_mu_x_leaves_more_drive_force_in_a_corner():
    vehicle = Vehicle(downforce_coefficient=0.0, drag_coefficient=0.0,
                      max_power=5_000_000.0, max_traction_force=1_000_000.0,
                      drivetrain="RWD", weight_dist_front=0.50, top_speed=100.0)
    circle = TireModel(mu=1.2, mu_x=1.2, load_sensitivity=0.0)
    longish = TireModel(mu=1.2, mu_x=1.6, load_sensitivity=0.0)
    v, k = 22.0, 1.0 / 80.0
    assert max_traction_accel(vehicle, longish, v, curvature=k) > \
        max_traction_accel(vehicle, circle, v, curvature=k) + 0.05


def test_lateral_split_is_yaw_balanced_not_load_proportional():
    vehicle = Vehicle(weight_dist_front=0.40)
    fy_f, fy_r = lateral_axle_forces(vehicle, 1000.0)
    a, b, L = lever_arms(vehicle)
    assert np.isclose(fy_f, 1000.0 * b / L)
    assert np.isclose(fy_r, 1000.0 * a / L)
    assert np.isclose(a * fy_f - b * fy_r, 0.0)
