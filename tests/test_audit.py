import numpy as np

from gfsem.data.audit import side_by_side_score, summarise_timeline


def _texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8), dtype=np.uint8)
    img = np.kron(base, np.ones((8, 8), np.uint8))
    return np.repeat(img[..., None], 3, axis=2)


def test_sbs_detects_stereo_pair() -> None:
    left = _texture(360, 640, 0)
    right = np.roll(left, -8, axis=1)  # small disparity
    assert side_by_side_score(np.hstack([left, right])) > 0.8


def test_sbs_rejects_mono() -> None:
    assert side_by_side_score(_texture(360, 1280, 1)) < 0.3


def test_summary_static_and_events() -> None:
    rows = []
    for i in range(50):  # 10 s at 5 Hz
        t = i * 0.2
        rows.append({"t": t, "flow_px": 0.1 if t < 4 else 3.0,
                     "n_blue": 1 if 6 <= t < 9 else 0, "n_orange": 0, "excluded": 0})
    s = summarise_timeline(rows)
    assert 3.0 <= s["static_s"] <= 5.0
    assert s["obstacle_events"] == 1
