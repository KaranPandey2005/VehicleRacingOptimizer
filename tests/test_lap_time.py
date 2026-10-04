import numpy as np
from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.simulation.simulator import Simulator


def default_track():
    return Track.rounded_rectangle(width_m=400.0, height_m=200.0,
                                    corner_radii=(30.0, 20.0, 45.0, 25.0),
                                    track_width=12.0, ds=2.0)


def test_full_pipeline_produces_positive_finite_lap_time():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    sim = Simulator(track, vehicle, tire, mode="attack")
    result = sim.run()
    assert result.lap_time > 0
    assert np.isfinite(result.lap_time)


def test_speed_profile_respects_corner_speed_limit_everywhere():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    sim = Simulator(track, vehicle, tire, mode="attack")
    result = sim.run()
    v = result.profile["v"]
    v_limit = result.profile["v_corner_limit"]
    assert np.all(v <= v_limit + 1e-6)


def test_speed_never_negative():
    track = default_track()
    vehicle = Vehicle()
    sim = Simulator(track, vehicle, mode="attack")
    result = sim.run()
    assert np.all(result.profile["v"] >= 0)


def test_safe_mode_is_slower_than_attack_mode():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)

    attack = Simulator(track, vehicle, tire, mode="attack").run()
    safe = Simulator(track, vehicle, tire, mode="safe").run()

    assert safe.lap_time > attack.lap_time
    assert safe.telemetry["max_speed"] <= attack.telemetry["max_speed"] + 1e-6


def test_race_mode_between_attack_and_safe():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)

    attack = Simulator(track, vehicle, tire, mode="attack").run()
    race = Simulator(track, vehicle, tire, mode="race").run()
    safe = Simulator(track, vehicle, tire, mode="safe").run()

    assert attack.lap_time <= race.lap_time <= safe.lap_time


def test_higher_power_car_is_faster_on_same_track_all_else_equal():
    track = default_track()
    tire = TireModel(mu=1.2)

    low_power = Vehicle(name="Low Power", max_power=60_000.0, max_traction_force=4000.0,
                         downforce_coefficient=0.0, tire_mu=1.2)
    high_power = Vehicle(name="High Power", max_power=300_000.0, max_traction_force=4000.0,
                          downforce_coefficient=0.0, tire_mu=1.2)

    result_low = Simulator(track, low_power, tire, mode="attack").run()
    result_high = Simulator(track, high_power, tire, mode="attack").run()

    assert result_high.lap_time < result_low.lap_time


def test_higher_downforce_car_is_faster_through_corners():
    track = default_track()
    tire = TireModel(mu=1.3)

    no_aero = Vehicle(name="No Aero", downforce_coefficient=0.0, tire_mu=1.3, top_speed=100.0)
    high_aero = Vehicle(name="High Aero", downforce_coefficient=3.0, tire_mu=1.3, top_speed=100.0,
                         drag_coefficient=1.0)

    result_no_aero = Simulator(track, no_aero, tire, mode="attack").run()
    result_high_aero = Simulator(track, high_aero, tire, mode="attack").run()

    # Higher downforce should allow a higher minimum corner speed
    assert result_high_aero.telemetry["min_speed"] > result_no_aero.telemetry["min_speed"]


def test_lower_grip_track_condition_via_tire_modifier_is_slower():
    track = default_track()
    vehicle = Vehicle()

    dry_tire = TireModel(mu=vehicle.tire_mu, grip_modifier=1.0)
    wet_tire = TireModel(mu=vehicle.tire_mu, grip_modifier=0.65)

    dry_result = Simulator(track, vehicle, dry_tire, mode="attack").run()
    wet_result = Simulator(track, vehicle, wet_tire, mode="attack").run()

    assert wet_result.lap_time > dry_result.lap_time
