"""Append every completed SimulationResult to a local SQLite table."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_DB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "results", "runs.db")
)

_CREATE = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    vehicle_name TEXT,
    track_name TEXT,
    mode TEXT,
    line_mode TEXT,
    weather TEXT,
    used_fallback INTEGER,
    lap_time REAL,
    objective REAL,
    max_speed_kmh REAL,
    min_speed_kmh REAL,
    peak_lon_accel_g REAL,
    peak_lon_decel_g REAL,
    peak_lat_accel_g REAL,
    mean_utilization REAL,
    peak_utilization REAL,
    u_max REAL,
    wall_risk REAL,
    tire_stress REAL
)
"""

_INSERT = """
INSERT INTO runs (
    created_at, vehicle_name, track_name, mode, line_mode, weather,
    used_fallback, lap_time, objective, max_speed_kmh, min_speed_kmh,
    peak_lon_accel_g, peak_lon_decel_g, peak_lat_accel_g,
    mean_utilization, peak_utilization, u_max, wall_risk, tire_stress
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def save_run(result, db_path=None) -> int:
    """INSERT one row. No validity filter — fallback and unvalidated laps included."""
    db_path = db_path or DEFAULT_DB
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    t = result.telemetry or {}
    min_kmh = t.get("min_speed_kmh")
    if min_kmh is None:
        min_kmh = float(t.get("min_speed", 0.0)) * 3.6
    row = (
        datetime.now(timezone.utc).isoformat(),
        result.vehicle_name,
        result.track_name,
        result.mode,
        result.line_mode,
        result.weather,
        1 if result.used_fallback else 0,
        float(result.lap_time),
        float(result.objective),
        float(t.get("max_speed_kmh", 0.0)),
        float(min_kmh),
        float(t.get("peak_lon_accel_g", 0.0)),
        float(t.get("peak_lon_decel_g", 0.0)),
        float(t.get("peak_lat_accel_g", 0.0)),
        float(t.get("mean_utilization", 0.0)),
        float(t.get("peak_utilization", 0.0)),
        float(result.u_max),
        float(t.get("wall_risk", 0.0)),
        float(t.get("tire_stress", 0.0)),
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE)
        cur = conn.execute(_INSERT, row)
        conn.commit()
        return int(cur.lastrowid)
