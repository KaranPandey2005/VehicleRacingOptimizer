import numpy as np
from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.simulation.simulator import Simulator
from src.optimization.safety import DRIVING_MODES, objective_cost


def default_track():
    return Track.rounded_rectangle(width_m=400.0, height_m=200.0,
                                    corner_radii=(30.0, 20.0, 45.0, 25.0),
                                    track_width=12.0, ds=2.0)


def test_attack_has_higher_peak_utilization_than_safe():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    attack = Simulator(track, vehicle, tire, mode="attack").run()
    safe = Simulator(track, vehicle, tire, mode="safe").run()
    assert attack.u_max == DRIVING_MODES["attack"].u_max
    assert safe.u_max == DRIVING_MODES["safe"].u_max
    assert safe.telemetry["peak_utilization"] <= safe.u_max + 0.08
    assert attack.telemetry["peak_utilization"] > safe.telemetry["peak_utilization"]
    assert safe.lap_time > attack.lap_time


def test_safe_objective_is_not_just_lap_time():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    result = Simulator(track, vehicle, tire, mode="safe").run()
    mode = DRIVING_MODES["safe"]
    j = objective_cost(result.lap_time, result.telemetry, mode)
    assert j > result.lap_time
    assert np.isclose(j, result.objective)


def test_telemetry_includes_friction_circle_channels():
    track = default_track()
    vehicle = Vehicle()
    result = Simulator(track, vehicle, mode="race").run()
    for key in ("u", "mean_utilization", "peak_utilization",
                "wall_risk", "tire_stress"):
        assert key in result.telemetry
    assert result.telemetry["wall_risk"] >= 0.0
    assert result.telemetry["tire_stress"] >= 0.0
