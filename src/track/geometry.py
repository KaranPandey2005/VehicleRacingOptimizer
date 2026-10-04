"""
Track geometry generation.

Physical / mathematical model
------------------------------
A track centerline is built as a sequence of "turtle graphics" segments:

    - Straight(length)                 -> curvature = 0
    - Arc(radius, signed_angle)        -> curvature = 1 / radius (signed)

Starting from a pose (x, y, heading), each segment is integrated analytically
(no numerical integration error) to produce the next pose. This guarantees
that a segment list whose turning angles and lengths are chosen to form a
closed shape (e.g. a rounded rectangle) closes EXACTLY, up to floating point
precision, with no accumulated numerical drift.

Curvature sign convention: positive curvature = left turn (counter-clockwise),
matching the standard math convention (heading measured CCW from +x axis).

Each segment is discretized at approximately `ds` arc-length spacing so the
final centerline is a dense point cloud suitable for physics calculations
and plotting.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Straight:
    length: float  # meters, must be > 0


@dataclass
class Arc:
    radius: float        # meters, must be > 0 (radius is always positive)
    angle: float         # signed turn angle in radians (+ = left/CCW, - = right/CW)


Segment = "Straight | Arc"


def _discretize_straight(seg: Straight, x0, y0, heading0, ds) -> Tuple[np.ndarray, ...]:
    n = max(2, int(np.ceil(seg.length / ds)) + 1)
    s_local = np.linspace(0.0, seg.length, n)
    x = x0 + s_local * np.cos(heading0)
    y = y0 + s_local * np.sin(heading0)
    heading = np.full(n, heading0)
    curvature = np.zeros(n)
    return s_local, x, y, heading, curvature


def _discretize_arc(seg: Arc, x0, y0, heading0, ds) -> Tuple[np.ndarray, ...]:
    arc_length = seg.radius * abs(seg.angle)
    n = max(2, int(np.ceil(arc_length / ds)) + 1)
    s_local = np.linspace(0.0, arc_length, n)

    turn_sign = 1.0 if seg.angle >= 0 else -1.0
    k = turn_sign / seg.radius  # signed curvature

    # Center of the arc lies along the left-normal of the current heading
    # (or right-normal if turning right), at distance = radius:
    #   left-normal direction of heading0 is (-sin(heading0), cos(heading0))
    #   center = pose + turn_sign * radius * left_normal
    nx, ny = -np.sin(heading0), np.cos(heading0)
    center_x = x0 + turn_sign * seg.radius * nx
    center_y = y0 + turn_sign * seg.radius * ny

    # Vector from center to start point, then rotate this vector by the
    # swept angle theta(s) = k * s to trace the arc.
    vx0 = x0 - center_x
    vy0 = y0 - center_y

    theta = k * s_local  # signed sweep angle
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = center_x + vx0 * cos_t - vy0 * sin_t
    y = center_y + vx0 * sin_t + vy0 * cos_t

    heading = heading0 + theta
    curvature = np.full(n, k)
    return s_local, x, y, heading, curvature


def build_centerline(segments: List, start_pose=(0.0, 0.0, 0.0), ds: float = 1.0):
    """
    Integrate a list of Straight/Arc segments into a dense centerline.

    Parameters
    ----------
    segments : list of Straight | Arc
    start_pose : (x0, y0, heading0)
    ds : target arc-length spacing between samples (meters)

    Returns
    -------
    dict with keys: s, x, y, heading, curvature  (all 1D np.ndarray, same length)
    Points are de-duplicated at segment boundaries (the end point of segment i
    equals the start point of segment i+1).
    """
    x, y, heading = start_pose
    s_accum = 0.0

    all_s, all_x, all_y, all_h, all_k = [], [], [], [], []

    for seg in segments:
        if isinstance(seg, Straight):
            s_local, xs, ys, hs, ks = _discretize_straight(seg, x, y, heading, ds)
        elif isinstance(seg, Arc):
            s_local, xs, ys, hs, ks = _discretize_arc(seg, x, y, heading, ds)
        else:
            raise TypeError(f"Unknown segment type: {type(seg)}")

        if all_s:
            # drop first point of this segment (duplicate of previous segment's last point)
            s_local, xs, ys, hs, ks = s_local[1:], xs[1:], ys[1:], hs[1:], ks[1:]

        all_s.append(s_accum + s_local)
        all_x.append(xs)
        all_y.append(ys)
        all_h.append(hs)
        all_k.append(ks)

        s_accum = all_s[-1][-1]  # cumulative arc length through end of this segment

        # advance pose to the end of this segment
        x, y, heading = xs[-1] if xs.size else x, ys[-1] if ys.size else y, hs[-1] if hs.size else heading

    return {
        "s": np.concatenate(all_s),
        "x": np.concatenate(all_x),
        "y": np.concatenate(all_y),
        "heading": np.concatenate(all_h),
        "curvature": np.concatenate(all_k),
    }


def rounded_rectangle_segments(width: float, height: float, corner_radii) -> List:
    """
    Build a closed-by-construction rounded-rectangle track as a segment list.

    corner_radii: (r_bl, r_br, r_tr, r_tl) -- radius at each rectangle corner,
    going counter-clockwise starting at bottom-left.

    Each straight edge length = edge_length - (sum of the two adjacent corner radii).
    All four corner arcs are +90 deg (left turns), which sums to +360 deg,
    guaranteeing the path closes exactly regardless of the chosen radii/lengths
    (as long as radii are geometrically consistent, i.e. each edge length > 0).
    """
    r_bl, r_br, r_tr, r_tl = corner_radii

    bottom = width - r_bl - r_br
    right = height - r_br - r_tr
    top = width - r_tr - r_tl
    left = height - r_tl - r_bl

    for name, L in [("bottom", bottom), ("right", right), ("top", top), ("left", left)]:
        if L <= 0:
            raise ValueError(
                f"Invalid track geometry: '{name}' straight length is {L:.2f} m (<=0). "
                f"Reduce corner radii or increase width/height."
            )

    segments = [
        Straight(bottom),
        Arc(r_br, np.pi / 2),
        Straight(right),
        Arc(r_tr, np.pi / 2),
        Straight(top),
        Arc(r_tl, np.pi / 2),
        Straight(left),
        Arc(r_bl, np.pi / 2),
    ]
    # Start pose is placed at (r_bl, 0) heading 0, i.e. right where the
    # bottom-left corner arc ends and the bottom straight begins.
    start_pose = (r_bl, 0.0, 0.0)
    return segments, start_pose


def chicane_circuit_segments(
    width: float,
    height: float,
    corner_radii,
    chicane_radius: float = 20.0,
    chicane_leg: float = 25.0,
    chicane_mid: float = 25.0,
    chicane_lead: float = 80.0,
):
    """
    Rounded rectangle with a symmetric left-right-right-left chicane on the
    bottom straight. Net chicane heading and lateral offset are both zero, so
    the loop still closes exactly (same construction as the plain rectangle).

    The four extra 45 deg arcs give eight corners total.
    """
    r_bl, r_br, r_tr, r_tl = corner_radii
    rect_segments, start_pose = rounded_rectangle_segments(width, height, corner_radii)
    bottom = width - r_bl - r_br

    r = chicane_radius
    if r <= 0 or chicane_leg <= 0 or chicane_mid <= 0:
        raise ValueError("Chicane radius and straight lengths must be > 0.")

    # Along-track consumption of a heading-preserving LRRL chicane (see module
    # notes: two 45 deg pairs of equal radius plus two equal legs).
    chicane_dx = np.sqrt(2.0) * (2.0 * r + chicane_leg) + chicane_mid
    trail = bottom - chicane_lead - chicane_dx
    if chicane_lead <= 0 or trail <= 0:
        raise ValueError(
            f"Chicane does not fit on the bottom straight ({bottom:.2f} m). "
            f"Need lead ({chicane_lead:.2f} m) + chicane ({chicane_dx:.2f} m) "
            f"+ trail ({trail:.2f} m) all > 0. Reduce chicane size or increase width."
        )

    forty_five = np.pi / 4
    chicane = [
        Straight(chicane_lead),
        Arc(r, forty_five),
        Straight(chicane_leg),
        Arc(r, -forty_five),
        Straight(chicane_mid),
        Arc(r, -forty_five),
        Straight(chicane_leg),
        Arc(r, forty_five),
        Straight(trail),
    ]
    # rect_segments[0] is the original bottom straight; keep the other 7 edges.
    segments = chicane + list(rect_segments[1:])
    return segments, start_pose


def centerline_from_xy(x, y, ds: float = 1.0, closed: bool = True, smooth_window: int = 51):
    """
    Build an arc-length centerline (s, x, y, heading, curvature) from a
    polyline. Used for imported real circuits that are not Straight/Arc
    turtle-graphics.

    Heading is atan2 of consecutive segments; curvature is d(heading)/ds
    after optional Savitzky-Golay smoothing so GPS/OSM kinks do not create
    impossible corner-speed spikes.
    """
    from scipy.signal import savgol_filter

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or x.size != y.size:
        raise ValueError("Need at least 3 (x, y) points of equal length.")

    if closed and np.hypot(x[0] - x[-1], y[0] - y[-1]) > 1e-9:
        x = np.concatenate([x, x[:1]])
        y = np.concatenate([y, y[:1]])

    seg = np.hypot(np.diff(x), np.diff(y))
    s_raw = np.concatenate([[0.0], np.cumsum(seg)])
    s_uniform = np.arange(0.0, s_raw[-1], ds)
    if s_uniform.size < 3:
        raise ValueError("Polyline is shorter than 3 samples at the requested ds.")

    x_u = np.interp(s_uniform, s_raw, x)
    y_u = np.interp(s_uniform, s_raw, y)

    dx = np.empty_like(x_u)
    dy = np.empty_like(y_u)
    dx[:-1] = np.diff(x_u)
    dy[:-1] = np.diff(y_u)
    if closed:
        dx[-1] = x_u[0] - x_u[-1]
        dy[-1] = y_u[0] - y_u[-1]
    else:
        dx[-1] = dx[-2]
        dy[-1] = dy[-2]

    heading_raw = np.arctan2(dy, dx)
    heading = np.unwrap(heading_raw)
    n = heading.size
    ds_last = np.hypot(dx[-1], dy[-1])
    length = float(s_uniform[-1] + ds_last)

    # Unwrapped heading gains ±2π per lap, so it is not periodic. Subtract that
    # linear trend before wrap-mode smoothing, then add it back — otherwise the
    # start/finish seam becomes a fake curvature spike.
    net_turn = 0.0
    if closed:
        wrap_delta = np.unwrap(np.array([heading[-1], heading_raw[0]]))[1] - heading[-1]
        net_turn = (heading[-1] + wrap_delta) - heading[0]
        trend = net_turn * (s_uniform / length)
        residual = heading - trend
    else:
        residual = heading
        trend = 0.0

    w = int(smooth_window)
    if w % 2 == 0:
        w += 1
    if w >= 5 and n >= w:
        mode = "wrap" if closed else "interp"
        residual = savgol_filter(residual, w, polyorder=3, mode=mode)
    heading = residual + trend

    if closed:
        heading_next = np.empty(n)
        heading_prev = np.empty(n)
        heading_next[:-1] = heading[1:]
        heading_next[-1] = heading[0] + net_turn
        heading_prev[1:] = heading[:-1]
        heading_prev[0] = heading[-1] - net_turn
        ds_fwd = np.empty(n)
        ds_fwd[:-1] = np.diff(s_uniform)
        ds_fwd[-1] = ds_last
        ds_bwd = np.roll(ds_fwd, 1)
        curvature = (heading_next - heading_prev) / np.maximum(ds_fwd + ds_bwd, 1e-9)
    else:
        curvature = np.gradient(heading, s_uniform)

    return {
        "s": s_uniform,
        "x": x_u,
        "y": y_u,
        "heading": heading,
        "curvature": curvature,
    }


def resample_uniform(data: dict, ds: float) -> dict:
    """Resample a centerline dict onto a uniform arc-length grid via linear interpolation."""
    s = data["s"]
    s_uniform = np.arange(0.0, s[-1], ds)
    out = {"s": s_uniform}
    for key in ("x", "y", "curvature"):
        out[key] = np.interp(s_uniform, s, data[key])
    # heading needs unwrap-safe interpolation
    heading_unwrapped = np.unwrap(data["heading"])
    out["heading"] = np.interp(s_uniform, s, heading_unwrapped)
    return out
