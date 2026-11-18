from gfsem.data.obstacle_events import Intrinsics, ground_point

K = Intrinsics(533.7, 533.7, 638.4, 362.1)


def test_ground_point_straight_ahead() -> None:
    # A point 10 m ahead on flat ground for a level camera at 1.25 m: v = cy + fy*h/x
    x, y = ground_point(K.cx, K.cy + K.fy * 1.25 / 10.0, K, 1.25)
    assert abs(x - 10.0) < 1e-6 and abs(y) < 1e-6


def test_ground_point_left_is_positive_y() -> None:
    x, y = ground_point(K.cx - 100, K.cy + 50, K, 1.25)
    assert y > 0


def test_above_horizon_is_none() -> None:
    assert ground_point(640, K.cy - 5, K, 1.25) is None
