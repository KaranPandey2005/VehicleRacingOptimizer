import numpy as np
from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.simulation.simulator import Simulator
from src.vehicle.aero import RHO_AIR_SEA_LEVEL
from src.weather.weather import Weather, Condition, GRIP_MODIFIER


def default_track():
    return Track.rounded_rectangle(width_m=400.0, height_m=200.0,
                                    corner_radii=(30.0, 20.0, 45.0, 25.0),
                                    track_width=12.0, ds=2.0)


def test_default_weather_matches_legacy_dry_run():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    legacy = Simulator(track, vehicle, tire, mode="attack").run()
    dry = Simulator(track, vehicle, tire, mode="attack",
                    weather=Weather(condition=Condition.DRY)).run()
    assert dry.weather == "dry"
    assert np.isclose(dry.lap_time, legacy.lap_time)


def test_wet_is_slower_than_damp_is_slower_than_dry():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    times = {}
    for cond in (Condition.DRY, Condition.DAMP, Condition.WET):
        times[cond] = Simulator(
            track, vehicle, tire, mode="attack",
            weather=Weather(condition=cond),
        ).run().lap_time
    assert times[Condition.DRY] < times[Condition.DAMP] < times[Condition.WET]


def test_wet_lowers_minimum_corner_speed():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    dry = Simulator(track, vehicle, tire, mode="attack",
                    weather=Weather(condition=Condition.DRY)).run()
    wet = Simulator(track, vehicle, tire, mode="attack",
                    weather=Weather(condition=Condition.WET)).run()
    assert wet.telemetry["min_speed"] < dry.telemetry["min_speed"]
    assert wet.grip_modifier == GRIP_MODIFIER[Condition.WET]


def test_air_density_falls_when_air_is_warmer():
    cool = Weather(ambient_temp_c=15.0).air_density()
    hot = Weather(ambient_temp_c=35.0).air_density()
    assert np.isclose(cool, RHO_AIR_SEA_LEVEL)
    assert hot < cool


def test_hotter_air_changes_aero_sensitive_lap():
    track = default_track()
    vehicle = Vehicle(downforce_coefficient=3.0, drag_coefficient=1.0, tire_mu=1.3)
    tire = TireModel(mu=vehicle.tire_mu)
    cool = Simulator(
        track, vehicle, tire, mode="attack",
        weather=Weather(condition=Condition.DRY, ambient_temp_c=15.0),
    ).run()
    hot = Simulator(
        track, vehicle, tire, mode="attack",
        weather=Weather(condition=Condition.DRY, ambient_temp_c=40.0),
    ).run()
    assert not np.isclose(cool.lap_time, hot.lap_time)
