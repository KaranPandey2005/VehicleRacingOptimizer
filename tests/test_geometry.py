import numpy as np
import pytest
from src.track.geometry import (
    Straight, Arc, build_centerline, rounded_rectangle_segments,
    chicane_circuit_segments, centerline_from_xy,
)
from src.track.track import Track


def test_straight_segment_curvature_zero():
    data = build_centerline([Straight(100.0)], start_pose=(0, 0, 0), ds=1.0)
    assert np.allclose(data["curvature"], 0.0)
    assert np.isclose(data["x"][-1], 100.0, atol=1e-6)
    assert np.isclose(data["y"][-1], 0.0, atol=1e-6)


def test_straight_segment_heading_constant():
    data = build_centerline([Straight(50.0)], start_pose=(0, 0, np.pi / 4), ds=1.0)
    assert np.allclose(data["heading"], np.pi / 4)


def test_arc_curvature_matches_one_over_radius():
    radius = 20.0
    data = build_centerline([Arc(radius, np.pi / 2)], start_pose=(0, 0, 0), ds=0.5)
    assert np.allclose(data["curvature"], 1.0 / radius, atol=1e-9)


def test_arc_left_turn_90_ends_at_expected_pose():
    radius = 30.0
    data = build_centerline([Arc(radius, np.pi / 2)], start_pose=(0, 0, 0), ds=0.2)
    # Starting at origin heading 0, turning left 90 deg on radius R ends at (R, R), heading pi/2
    assert np.isclose(data["x"][-1], radius, atol=1e-3)
    assert np.isclose(data["y"][-1], radius, atol=1e-3)
    assert np.isclose(data["heading"][-1], np.pi / 2, atol=1e-6)


def test_arc_right_turn_is_negative_curvature():
    data = build_centerline([Arc(15.0, -np.pi / 2)], start_pose=(0, 0, 0), ds=0.5)
    assert np.allclose(data["curvature"], -1.0 / 15.0, atol=1e-9)
    assert np.isclose(data["heading"][-1], -np.pi / 2, atol=1e-6)


def test_arc_length_matches_radius_times_angle():
    radius, angle = 40.0, 1.3
    data = build_centerline([Arc(radius, angle)], start_pose=(0, 0, 0), ds=0.5)
    assert np.isclose(data["s"][-1], radius * angle, atol=1e-2)


def test_rounded_rectangle_closes_exactly():
    segments, start_pose = rounded_rectangle_segments(500.0, 250.0, (40.0, 25.0, 60.0, 35.0))
    data = build_centerline(segments, start_pose=start_pose, ds=1.0)
    dx = data["x"][-1] - data["x"][0]
    dy = data["y"][-1] - data["y"][0]
    dist = np.hypot(dx, dy)
    assert dist < 0.5, f"Track did not close: gap = {dist:.4f} m"

    # heading should also return to (2*pi mod) the start heading
    dh = (data["heading"][-1] - data["heading"][0]) % (2 * np.pi)
    assert np.isclose(dh, 0.0, atol=1e-3) or np.isclose(dh, 2 * np.pi, atol=1e-3)


def test_rounded_rectangle_invalid_geometry_raises():
    with pytest.raises(ValueError):
        # radii too large for the given width/height
        rounded_rectangle_segments(50.0, 50.0, (40.0, 40.0, 40.0, 40.0))


def test_chicane_circuit_closes_exactly():
    segments, start_pose = chicane_circuit_segments(500.0, 250.0, (40.0, 25.0, 60.0, 35.0))
    data = build_centerline(segments, start_pose=start_pose, ds=1.0)
    dist = np.hypot(data["x"][-1] - data["x"][0], data["y"][-1] - data["y"][0])
    assert dist < 0.5, f"Chicane circuit did not close: gap = {dist:.4f} m"
    n_arcs = sum(1 for seg in segments if isinstance(seg, Arc))
    assert n_arcs == 8


def test_chicane_circuit_too_large_raises():
    with pytest.raises(ValueError):
        chicane_circuit_segments(
            500.0, 250.0, (40.0, 25.0, 60.0, 35.0),
            chicane_radius=80.0, chicane_leg=120.0, chicane_mid=120.0, chicane_lead=80.0,
        )


def test_chicane_track_class_is_closed():
    track = Track.chicane_circuit()
    assert track.is_closed()
    assert track.length > Track.rounded_rectangle().length


def test_track_class_builds_and_is_closed():
    track = Track.rounded_rectangle()
    assert track.is_closed()
    assert track.length > 0
    # boundaries should be offset by half width from centerline at a sample point
    i = 10
    dist_left = np.hypot(track.left_x[i] - track.x[i], track.left_y[i] - track.y[i])
    assert np.isclose(dist_left, track.width / 2, atol=1e-6)


def test_track_curvature_at_s_interpolates_reasonably():
    track = Track.rounded_rectangle()
    k0 = track.curvature_at_s(0.0)
    k_wrapped = track.curvature_at_s(track.length + 5.0)  # should wrap around
    k_direct = track.curvature_at_s(5.0)
    assert np.isclose(k_wrapped, k_direct, atol=1e-2)


def test_centerline_from_xy_circle_closes_with_expected_curvature():
    theta = np.linspace(0.0, 2 * np.pi, 250, endpoint=False)
    radius = 100.0
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    data = centerline_from_xy(x, y, ds=1.0, closed=True, smooth_window=21)
    dist = np.hypot(data["x"][-1] - data["x"][0], data["y"][-1] - data["y"][0])
    assert dist < 2.0
    assert np.isclose(data["s"][-1], 2 * np.pi * radius, atol=5.0)
    assert np.allclose(np.abs(data["curvature"]), 1.0 / radius, atol=0.005)


def test_spa_francorchamps_loads_and_is_closed():
    track = Track.spa_francorchamps()
    assert track.is_closed(tol=2.0)
    # Official F1 length is 7004 m; OSM-derived 2D centerline should be nearby.
    assert 6500.0 < track.length < 7500.0
    assert track.width > 8.0

