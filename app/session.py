"""
Shared sim setup for the CLI and the Phase 8 GUI.

No Qt imports here so tests can run headless.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.track.track import Track
from src.vehicle.vehicle import Vehicle
from src.vehicle.tires import TireModel
from src.simulation.simulator import Simulator
from src.weather.weather import Weather, Condition

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VEHICLE_DIR = os.path.join(ROOT, "data", "vehicles")
OUTPUT_DIR = os.path.join(ROOT, "output")

TRACK_FACTORIES = {
    "Rounded rectangle": lambda: Track.rounded_rectangle(ds=2.0),
    "Chicane circuit": lambda: Track.chicane_circuit(ds=2.0),
    "Spa-Francorchamps": lambda: Track.spa_francorchamps(),
}

MODES = ("attack", "race", "safe")
LINE_MODES = ("centerline", "shaped", "optimized")
WEATHER = ("dry", "damp", "wet")

_track_cache = {}


def vehicle_files():
    names = sorted(f for f in os.listdir(VEHICLE_DIR) if f.endswith(".json"))
    out = []
    for fname in names:
        path = os.path.join(VEHICLE_DIR, fname)
        vehicle = Vehicle.from_json(path)
        out.append((vehicle.name, path))
    return out


def get_track(track_key: str) -> Track:
    if track_key not in TRACK_FACTORIES:
        raise ValueError(f"Unknown track '{track_key}'")
    if track_key not in _track_cache:
        _track_cache[track_key] = TRACK_FACTORIES[track_key]()
    return _track_cache[track_key]


def slow_run_warning(track_key: str, line_mode: str) -> str:
    if line_mode != "optimized":
        return ""
    if track_key == "Spa-Francorchamps":
        return "Spa + optimized can take several minutes. Centerline is much faster."
    return "Optimized line is slower to compute than centerline."


@dataclass
class SessionSpec:
    vehicle_path: str
    track_key: str
    mode: str = "attack"
    line_mode: str = "centerline"
    weather: str = "dry"


def run_session(spec: SessionSpec):
    track = get_track(spec.track_key)
    vehicle = Vehicle.from_json(spec.vehicle_path)
    tire = TireModel(mu=vehicle.tire_mu)
    weather = Weather(condition=Condition(spec.weather))
    result = Simulator(
        track, vehicle, tire,
        mode=spec.mode,
        line_mode=spec.line_mode,
        weather=weather,
    ).run()
    return track, result
