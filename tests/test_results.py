import os
import sqlite3

from src.simulation.results import save_run
from src.simulation.simulator import SimulationResult


def _result(**kwargs):
    t = {
        "max_speed_kmh": 261.9,
        "min_speed": 63.9 / 3.6,
        "peak_lon_accel_g": 1.01,
        "peak_lon_decel_g": 2.47,
        "peak_lat_accel_g": 3.51,
        "mean_utilization": 0.38,
        "peak_utilization": 1.03,
        "wall_risk": 0.122,
        "tire_stress": 0.264,
    }
    defaults = dict(
        profile={"s": [0.0]},
        telemetry=t,
        lap_time=136.97,
        mode="attack",
        vehicle_name="FIA F3",
        track_name="Spa-Francorchamps",
        line_mode="optimized",
        used_fallback=True,
        weather="dry",
        u_max=1.0,
        objective=136.97,
    )
    defaults.update(kwargs)
    return SimulationResult(**defaults)


def test_save_run_stores_fallback_lap(tmp_path):
    db = str(tmp_path / "runs.db")
    result = _result()
    run_id = save_run(result, db_path=db)
    assert run_id == 1
    assert os.path.isfile(db)

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT lap_time, used_fallback, vehicle_name, peak_utilization FROM runs"
        ).fetchone()
    assert row[0] == result.lap_time
    assert row[1] == 1
    assert row[2] == "FIA F3"
    assert row[3] == 1.03


def test_save_run_appends_every_lap(tmp_path):
    db = str(tmp_path / "runs.db")
    save_run(_result(lap_time=10.0, used_fallback=False), db_path=db)
    save_run(_result(lap_time=11.5, used_fallback=True), db_path=db)
    with sqlite3.connect(db) as conn:
        times = [r[0] for r in conn.execute("SELECT lap_time FROM runs ORDER BY id")]
    assert times == [10.0, 11.5]


def test_default_db_is_tracked_under_data_results():
    from src.simulation.results import DEFAULT_DB
    assert DEFAULT_DB.replace("\\", "/").endswith("data/results/runs.db")
