import numpy as np
from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.simulation.simulator import Simulator
from src.optimization.trajectory import (
    casadi_available, optimized_racing_line, _detect_corners,
)


def default_track():
    return Track.rounded_rectangle(width_m=400.0, height_m=200.0,
                                    corner_radii=(30.0, 20.0, 45.0, 25.0),
                                    track_width=12.0, ds=2.0)


def test_casadi_is_importable():
    assert casadi_available(), "Phase 5 requires casadi (see requirements.txt)"


def test_detects_all_four_rectangle_corners():
    track = default_track()
    corners = _detect_corners(track)
    assert len(corners) >= 4


def test_spa_detects_every_major_corner():
    track = Track.spa_francorchamps()
    corners = _detect_corners(track)
    assert len(corners) >= 20


def test_spa_optimized_line_is_not_slower_than_centerline():
    track = Track.spa_francorchamps()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    line = optimized_racing_line(track, vehicle, tire)
    assert line["fine_lap_time"] <= line["centerline_lap_time"] + 0.05
    if not line.get("used_fallback"):
        assert float(np.max(np.abs(line["offset"]))) > 0.2


def test_optimized_line_stays_in_bounds():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    line = optimized_racing_line(track, vehicle, tire)
    dist = np.hypot(line["x"] - track.x, line["y"] - track.y)
    assert np.all(dist <= track.width / 2.0 + 1e-6)


def test_optimized_line_not_slower_than_centerline():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    sim_c = Simulator(track, vehicle, tire, mode="attack", line_mode="centerline").run()
    sim_o = Simulator(track, vehicle, tire, mode="attack", line_mode="optimized").run()
    # 0.05 s is the same referee tolerance used inside the optimizer fallback.
    assert sim_o.lap_time <= sim_c.lap_time + 0.05


def test_optimized_mode_runs_and_is_finite():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    result = Simulator(track, vehicle, tire, mode="attack", line_mode="optimized").run()
    assert result.line_mode == "optimized"
    assert result.lap_time > 0
    assert np.isfinite(result.lap_time)
    assert np.all(result.profile["v"] >= 0)
    assert np.all(result.profile["v"] <= result.profile["v_corner_limit"] + 1e-6)


def test_optimized_line_uses_some_track_width_or_falls_back_honestly():
    track = default_track()
    vehicle = Vehicle()
    tire = TireModel(mu=vehicle.tire_mu)
    line = optimized_racing_line(track, vehicle, tire)
    offset_span = float(np.max(np.abs(line["offset"])))
    if line.get("used_fallback"):
        assert offset_span < 1e-9
    else:
        # A true racing line should leave the centerline somewhere.
        assert offset_span > 0.2
