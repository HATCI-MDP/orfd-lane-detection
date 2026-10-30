"""Route scoring must measure offsets and departures against the authored road."""

import numpy as np
import pytest

from offroad_autonomy.runtime.route_metrics import (
    build_centerline,
    count_departures,
    lateral_offsets,
    rejection_stats,
    roads_near,
    travel_headings,
)


def _road(points, half_width=3.0, reverse=False):
    points = np.asarray(points, dtype=float)
    if reverse:
        points = points[::-1]
    direction = np.gradient(points, axis=0)
    direction /= np.hypot(*direction.T)[:, None]
    left = np.column_stack((-direction[:, 1], direction[:, 0]))
    edges = []
    for p, n in zip(points, left):
        edges.append(
            {
                "left": [*(p + half_width * n), 0.0],
                "middle": [*p, 0.0],
                "right": [*(p - half_width * n), 0.0],
            }
        )
    return {"edges": edges}


STRAIGHT = {"r": _road([[0, 0], [10, 0], [20, 0], [30, 0]])}


@pytest.mark.parametrize("reverse", [False, True])
def test_offset_is_positive_left_of_travel_whatever_the_node_order(reverse):
    roads = {"r": _road([[0, 0], [10, 0], [20, 0], [30, 0]], reverse=reverse)}
    xy = np.array([[5.0, 0.4], [10.0, 0.4], [15.0, -0.6], [20.0, -0.6]])
    offsets, half = lateral_offsets(xy, build_centerline(roads), travel_headings(xy))
    np.testing.assert_allclose(offsets, [0.4, 0.4, -0.6, -0.6], atol=1e-9)
    np.testing.assert_allclose(half, 3.0)


def test_offset_uses_the_perpendicular_to_the_nearest_segment():
    roads = {"r": _road([[0, 0], [10, 0], [10, 10]])}
    xy = np.array([[9.0, 4.0], [9.0, 6.0]])
    offsets, _ = lateral_offsets(xy, build_centerline(roads), np.array([[0, 1.0], [0, 1.0]]))
    # Driving north along the second segment, x = 9 is 1 m to its left.
    np.testing.assert_allclose(offsets, [1.0, 1.0])


def test_departure_counts_each_exit_from_the_road_once():
    offsets = np.array([0.0, 2.9, 3.2, 3.5, 1.0, -3.1, -0.5])
    assert count_departures(offsets, np.full(len(offsets), 3.0)) == 2


def test_starting_off_the_road_is_a_departure():
    assert count_departures(np.array([4.0, 4.0, 0.0]), np.full(3, 3.0)) == 1


def test_rejection_stats_skip_unevaluated_frames():
    stats = rejection_stats([True, False, False, None, False, True, False])
    assert (stats.evaluated, stats.rejected, stats.longest_streak) == (6, 4, 3)
    assert stats.rate_pct == pytest.approx(400 / 6)


def test_rejection_stats_without_evaluated_frames_report_no_rate():
    assert rejection_stats([None, None]).rate_pct is None


def test_roads_near_drops_roads_out_of_reach():
    roads = {"near": STRAIGHT["r"], "far": _road([[500, 500], [510, 500]])}
    assert set(roads_near(roads, (0.0, 0.0), 100.0)) == {"near"}


def test_no_scorable_road_is_refused():
    with pytest.raises(ValueError, match="No authored road"):
        build_centerline(
            {"short": {"edges": [{"left": [0, 0], "middle": [0, 0], "right": [0, 0]}]}}
        )
