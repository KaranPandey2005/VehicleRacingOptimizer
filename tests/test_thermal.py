import numpy as np
from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.vehicle.thermal import ThermalModel
from src.simulation.simulator import Simulator
from src.weather.weather import Weather, Condition
from src.optimization.trajectory import _detect_corners, _qss_lap


def default_track():
    return Track.rounded_rectangle(width_m=400.0, height_m=200.0,
                                    corner_radii=(30.0, 20.0, 45.0, 25.0),
                                    track_width=12.0, ds=2.0)


def test_mu_factor_is_one_at_opt_and_lower_when_cold_or_hot():
    th = ThermalModel(t_opt_c=80.0, t_width_c=40.0, mu_floor=0.70)
    assert np.isclose(th.mu_factor(80.0), 1.0)
    assert th.mu_factor(20.0) < 0.95
    assert th.mu_factor(140.0) < 0.95
    assert th.mu_factor(20.0) >= 0.70


def test_heat_input_raises_temperature_above_start():
    th = ThermalModel()
    n = 40
    ds = np.full(n, 5.0)
    v = np.full(n, 30.0)
    u = np.full(n, 0.9)
    load = np.full(n, 7000.0)
    out = th.integrate(ds, v, u, u, load, load, t_amb_c=15.0,
                       t0_front_c=30.0, t0_rear_c=30.0, wraps=1)
    assert out["peak_t_front_c"] > 30.0
    assert out["wear_index"] > 0.0


def test_from_vehicle_reads_thermal_block():
    v = Vehicle(thermal={"t_opt_c": 95.0, "k_heat": 0.08})
    th = ThermalModel.from_vehicle(v)
    assert th.t_opt_c == 95.0
    assert th.k_heat == 0.08


def test_cold_track_is_slower_than_tires_in_the_window():
    track = default_track()
    vehicle = Vehicle()
    cold = Weather(condition=Condition.DRY, ambient_temp_c=10.0, track_temp_c=10.0)
    hot = Weather(condition=Condition.DRY, ambient_temp_c=20.0, track_temp_c=80.0)
    t_cold = Simulator(track, vehicle, mode="attack", weather=cold).run()
    t_hot = Simulator(track, vehicle, mode="attack", weather=hot).run()
    assert t_cold.lap_time > t_hot.lap_time


def test_qss_lap_cost_sees_track_temperature():
    """Greedy/CasADi cost eval is the two-pass thermal QSS, not cold μ."""
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    thermal = ThermalModel.from_vehicle(vehicle)
    corners = _detect_corners(track)
    z = np.zeros(2 * len(corners))
    _c_cold, t_cold, _ = _qss_lap(
        track, vehicle, tire, 1.0, z, corners, 0.5,
        thermal=thermal, t_amb_c=10.0, t0_c=10.0,
    )
    _c_hot, t_hot, _ = _qss_lap(
        track, vehicle, tire, 1.0, z, corners, 0.5,
        thermal=thermal, t_amb_c=20.0, t0_c=thermal.t_opt_c,
    )
    assert t_cold > t_hot


def test_attack_wears_more_than_safe():
    track = default_track()
    vehicle = Vehicle()
    attack = Simulator(track, vehicle, mode="attack").run()
    safe = Simulator(track, vehicle, mode="safe").run()
    assert attack.telemetry["wear_index"] > safe.telemetry["wear_index"]
    assert "t_front" in attack.telemetry
    assert attack.telemetry["peak_t_front_c"] > 0.0


def test_surface_peak_above_rim_on_high_u_march():
    th = ThermalModel()
    n = 80
    ds = np.full(n, 5.0)
    v = np.full(n, 40.0)
    u = np.full(n, 0.95)
    load = np.full(n, 8000.0)
    out = th.integrate(ds, v, u, u, load, load, t_amb_c=20.0,
                       t0_front_c=30.0, t0_rear_c=30.0, wraps=1)
    assert out["peak_t_front_c"] > out["peak_t_front_rim_c"]
    assert out["peak_t_rear_c"] > out["peak_t_rear_rim_c"]


def test_n_laps_one_matches_default_run():
    track = default_track()
    vehicle = Vehicle()
    a = Simulator(track, vehicle, mode="attack").run()
    b = Simulator(track, vehicle, mode="attack", n_laps=1).run()
    assert np.isclose(a.lap_time, b.lap_time)
    assert np.isclose(a.telemetry["wear_index"], b.telemetry["wear_index"])
    assert a.n_laps == 1
    assert len(a.lap_times) == 1


def test_three_lap_attack_last_slower_and_deterministic():
    track = default_track()
    vehicle = Vehicle()
    r1 = Simulator(track, vehicle, mode="attack", n_laps=3).run()
    r2 = Simulator(track, vehicle, mode="attack", n_laps=3).run()
    assert len(r1.lap_times) == 3
    assert r1.lap_times[-1] > r1.lap_times[0]
    assert r1.telemetry["wear_index"] > Simulator(
        track, vehicle, mode="attack", n_laps=1,
    ).run().telemetry["wear_index"]
    assert np.isclose(r1.lap_time, r2.lap_time)
    assert np.isclose(r1.telemetry["wear_index"], r2.telemetry["wear_index"])
    assert np.allclose(r1.lap_times, r2.lap_times)
