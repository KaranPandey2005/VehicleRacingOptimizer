"""
Track representation.

A Track holds a dense, arc-length-parameterized centerline (s, x, y, heading,
curvature) plus a track width, and derives left/right boundaries by offsetting
the centerline along its normal direction by +/- width/2.

V1 scope: constant track width, 2D (no elevation yet -- elevation is a listed
future extension in the project spec, not implemented here). Surface
condition is handled by the weather module, not stored per-point yet.
"""

from __future__ import annotations
import json
import os
import numpy as np

from src.track.geometry import (
    Straight, Arc, build_centerline, rounded_rectangle_segments,
    chicane_circuit_segments, resample_uniform, centerline_from_xy,
)

_TRACKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tracks")


class Track:
    def __init__(self, s, x, y, heading, curvature, width, name="track"):
        self.s = np.asarray(s, dtype=float)
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.heading = np.asarray(heading, dtype=float)
        self.curvature = np.asarray(curvature, dtype=float)
        self.width = float(width)  # meters, constant for V1
        self.name = name

        n = len(self.s)
        assert len(self.x) == len(self.y) == len(self.heading) == len(self.curvature) == n, \
            "Track arrays must all be the same length"

        self._compute_boundaries()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_segments(cls, segments, start_pose=(0.0, 0.0, 0.0), width=10.0, ds=1.0, name="track"):
        raw = build_centerline(segments, start_pose=start_pose, ds=ds)
        data = resample_uniform(raw, ds=ds)
        return cls(data["s"], data["x"], data["y"], data["heading"], data["curvature"], width, name=name)

    @classmethod
    def rounded_rectangle(cls, width_m=500.0, height_m=250.0,
                           corner_radii=(40.0, 25.0, 60.0, 35.0),
                           track_width=12.0, ds=1.0, name="Rounded Rectangle Circuit"):
        """
        Default V1 mathematically-generated track: a closed rounded-rectangle
        loop with four independently-sized corners (so different vehicles
        will naturally prefer different lines/speeds through each one) and
        two straights of different length.
        """
        segments, start_pose = rounded_rectangle_segments(width_m, height_m, corner_radii)
        return cls.from_segments(segments, start_pose=start_pose, width=track_width, ds=ds, name=name)

    @classmethod
    def chicane_circuit(cls, width_m=500.0, height_m=250.0,
                        corner_radii=(40.0, 25.0, 60.0, 35.0),
                        chicane_radius=20.0, chicane_leg=25.0, chicane_mid=25.0,
                        chicane_lead=80.0, track_width=12.0, ds=1.0,
                        name="Chicane Circuit"):
        """
        Closed loop with eight corners: the default rounded rectangle plus a
        heading-preserving chicane inserted on the bottom straight.
        """
        segments, start_pose = chicane_circuit_segments(
            width_m, height_m, corner_radii,
            chicane_radius=chicane_radius,
            chicane_leg=chicane_leg,
            chicane_mid=chicane_mid,
            chicane_lead=chicane_lead,
        )
        return cls.from_segments(segments, start_pose=start_pose, width=track_width, ds=ds, name=name)

    @classmethod
    def from_xy(cls, x, y, width=12.0, ds=1.0, closed=True, smooth_window=51, name="track"):
        """Build a Track from a centerline polyline (imported GPS / OSM / CSV)."""
        data = centerline_from_xy(x, y, ds=ds, closed=closed, smooth_window=smooth_window)
        return cls(data["s"], data["x"], data["y"], data["heading"], data["curvature"],
                   width, name=name)

    @classmethod
    def from_centerline_csv(cls, path, width=None, ds=1.0, closed=True,
                            smooth_window=51, name="track"):
        """
        Load a TUM-style centerline CSV: x_m, y_m[, w_tr_right_m, w_tr_left_m].
        If width is None and the file has width columns, uses their mean sum.
        """
        arr = np.loadtxt(path, delimiter=",", comments="#")
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Expected at least x,y columns in {path}")
        x, y = arr[:, 0], arr[:, 1]
        if width is None:
            if arr.shape[1] >= 4:
                width = float(np.mean(arr[:, 2] + arr[:, 3]))
            else:
                width = 12.0
        return cls.from_xy(x, y, width=width, ds=ds, closed=closed,
                           smooth_window=smooth_window, name=name)

    @classmethod
    def spa_francorchamps(cls, ds=2.0, track_width=None, smooth_window=51,
                          name="Circuit de Spa-Francorchamps"):
        """
        Circuit de Spa-Francorchamps (Belgium), ~7 km.

        Centerline is the OpenStreetMap-derived, smoothed polyline from the
        TUM FTM racetrack database (2D only — no Eau Rouge elevation).
        Not a licensed FIA survey; layout is recognizable, lap times are
        illustrative.
        """
        path = os.path.join(_TRACKS_DIR, "spa.csv")
        return cls.from_centerline_csv(
            path, width=track_width, ds=ds, smooth_window=smooth_window, name=name,
        )

    # ------------------------------------------------------------------ #
    # Derived geometry
    # ------------------------------------------------------------------ #
    def _compute_boundaries(self):
        # Left-hand normal of heading: (-sin(h), cos(h))
        nx = -np.sin(self.heading)
        ny = np.cos(self.heading)
        half_w = self.width / 2.0
        self.left_x = self.x + half_w * nx
        self.left_y = self.y + half_w * ny
        self.right_x = self.x - half_w * nx
        self.right_y = self.y - half_w * ny

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def is_closed(self, tol=1.0) -> bool:
        """Check whether the centerline forms a closed loop (start ~= end)."""
        d = np.hypot(self.x[0] - self.x[-1], self.y[0] - self.y[-1])
        return d < tol

    # ------------------------------------------------------------------ #
    # Interpolated queries (useful once s no longer aligns with sample grid,
    # e.g. after resampling for the optimizer)
    # ------------------------------------------------------------------ #
    def curvature_at_s(self, s_query):
        s_wrapped = np.mod(s_query, self.length)
        return np.interp(s_wrapped, self.s, self.curvature)

    def heading_at_s(self, s_query):
        s_wrapped = np.mod(s_query, self.length)
        heading_unwrapped = np.unwrap(self.heading)
        return np.interp(s_wrapped, self.s, heading_unwrapped)

    def xy_at_s(self, s_query):
        s_wrapped = np.mod(s_query, self.length)
        x = np.interp(s_wrapped, self.s, self.x)
        y = np.interp(s_wrapped, self.s, self.y)
        return x, y

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def to_json(self, path):
        payload = {
            "name": self.name,
            "width": self.width,
            "s": self.s.tolist(),
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "heading": self.heading.tolist(),
            "curvature": self.curvature.tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            payload = json.load(f)
        return cls(payload["s"], payload["x"], payload["y"], payload["heading"],
                    payload["curvature"], payload["width"], name=payload.get("name", "track"))

    def __repr__(self):
        return (f"Track('{self.name}', length={self.length:.1f} m, "
                f"width={self.width:.1f} m, closed={self.is_closed()})")
