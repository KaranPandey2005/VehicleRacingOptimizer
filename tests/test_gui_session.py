import os
from app.session import (
    SessionSpec, VEHICLE_DIR, get_track, run_session, slow_run_warning,
    vehicle_files,
)


def test_vehicle_files_include_the_three_demo_cars():
    names = [name for name, _path in vehicle_files()]
    blob = " ".join(names).lower()
    assert "brz" in blob
    assert "fit" in blob
    assert "f3" in blob


def test_gui_session_centerline_lap_is_finite():
    path = os.path.join(VEHICLE_DIR, "subaru_brz.json")
    track, result = run_session(SessionSpec(
        vehicle_path=path,
        track_key="Rounded rectangle",
        mode="attack",
        line_mode="centerline",
        weather="dry",
    ))
    assert track.name
    assert result.lap_time > 0
    assert result.line_mode == "centerline"
    assert result.weather == "dry"


def test_spa_optimized_warns():
    msg = slow_run_warning("Spa-Francorchamps", "optimized")
    assert "several minutes" in msg
    assert slow_run_warning("Rounded rectangle", "centerline") == ""


def test_get_track_caches():
    a = get_track("Rounded rectangle")
    b = get_track("Rounded rectangle")
    assert a is b
