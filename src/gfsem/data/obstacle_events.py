"""Count obstacle encounters (people, bollards, cones) in track recordings for the Phase 0 inventory.

The track tests use people standing on the runway as obstacles, so a COCO person detector
(YOLO, offline only) is combined with the HSV bollard/cone hints from ``audit.py``. Each
detection's bottom-centre is projected to the ground with a flat-ground pinhole model
(camera height ``h``, zero pitch), so an obstacle is "in path" when it lies 2–20 m ahead
and within ±3 m laterally. Consecutive in-path samples form an *event*; with ECU speed
available an event is labelled ``stopped`` (car below 1.8 km/h while the obstacle is ahead)
or ``passed_left``/``passed_right`` (the obstacle was last seen close and off-centre while
moving), otherwise ``unresolved``. These are counts for planning, not labels.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gfsem.data.audit import iter_frames, obstacle_blobs

X_MIN, X_MAX, Y_MAX = 2.0, 20.0, 3.0
GAP_S = 1.0
MOVING_KMH = 1.8


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled(self, w: int, ref_w: int) -> "Intrinsics":
        s = w / ref_w
        return Intrinsics(self.fx * s, self.fy * s, self.cx * s, self.cy * s)


def ground_point(u: float, v: float, K: Intrinsics, h: float) -> tuple[float, float] | None:
    """Flat-ground projection of pixel (u, v) for a level camera at height h.

    Returns (x forward, y left) in metres, or None at/above the horizon.
    """
    dv = v - K.cy
    if dv <= 1e-3:
        return None
    x = K.fy * h / dv
    y = -(u - K.cx) * x / K.fx
    return x, y


@dataclass
class Event:
    recording: str
    t_start: float
    t_end: float
    kind: str  # person | bollard | cone
    n_samples: int
    x_min_m: float
    y_at_closest_m: float
    y_last_m: float
    x_last_m: float
    min_speed_kmh: float | None
    outcome: str
    thumb: str


def _speed_lookup(path: Path) -> Any:
    """Return f(t_rel_s) -> km/h for bags with /ECU/speed, else None."""
    if path.suffix != ".db3":
        return None
    from gfsem.data.rosbag_io import SPEED_TOPIC, BagReader

    bag = BagReader(path)
    try:
        if not bag.has(SPEED_TOPIC):
            return None
        ts, v = zip(*[(t, m.data) for t, m in bag.messages(SPEED_TOPIC)])
        t_rel = (np.array(ts) - bag.t0_ns) / 1e9
        va = np.array(v, dtype=float)
    finally:
        bag.close()
    return lambda t: float(np.interp(t, t_rel, va))


def detect_events(path: Path, rel: str, model: Any, K_ref: Intrinsics, ref_w: int, h: float,
                  out_dir: Path, dt: float = 0.2, exclude: list[list[float]] | None = None) -> list[Event]:
    speed = _speed_lookup(path)
    exclude = exclude or []
    samples: list[tuple[float, str, float, float, np.ndarray]] = []
    for t, frame in iter_frames(path, dt):
        if any(a <= t < b for a, b in exclude):
            continue
        H, W = frame.shape[:2]
        K = K_ref.scaled(W, ref_w)
        dets: list[tuple[str, float, float]] = []
        res = model.predict(frame, classes=[0], conf=0.4, verbose=False, imgsz=640)[0]
        persons = res.boxes.xyxy.cpu().numpy()
        for (x1, y1, x2, y2) in persons:
            dets.append(("person", (x1 + x2) / 2, y2))

        def inside_person(x: float, y: float, w: float, hh: float) -> bool:
            # Blue jeans / orange clothing were the main false positives in Phase 0.
            cx, cy = x + w / 2, y + hh / 2
            return any(px1 - 5 <= cx <= px2 + 5 and py1 - 5 <= cy <= py2 + 5 for px1, py1, px2, py2 in persons)

        blue, orange = obstacle_blobs(frame)
        dets += [("bollard", x + w / 2, y + hh) for x, y, w, hh in blue if not inside_person(x, y, w, hh)]
        dets += [("cone", x + w / 2, y + hh) for x, y, w, hh in orange if not inside_person(x, y, w, hh)]
        for kind, u, v in dets:
            g = ground_point(u, v, K, h)
            if g and X_MIN <= g[0] <= X_MAX and abs(g[1]) <= Y_MAX:
                samples.append((t, kind, g[0], g[1], frame))
    # group by kind into events separated by > GAP_S
    events: list[Event] = []
    for kind in ("person", "bollard", "cone"):
        ks = [s for s in samples if s[1] == kind]
        groups: list[list[tuple[float, str, float, float, np.ndarray]]] = []
        for s in ks:
            if groups and s[0] - groups[-1][-1][0] <= GAP_S:
                groups[-1].append(s)
            else:
                groups.append([s])
        for g in groups:
            if len(g) < 3:  # < ~0.6 s: likely a false positive
                continue
            closest = min(g, key=lambda s: s[2])
            last = g[-1]
            vmin = None
            if speed is not None:
                vmin = min(speed(s[0]) for s in g)
            if vmin is not None and vmin < MOVING_KMH:
                outcome = "stopped"
            elif last[2] < 8.0 and abs(last[3]) > 0.8:
                outcome = "passed_left" if last[3] < 0 else "passed_right"  # obstacle on the right → car went left
            else:
                outcome = "unresolved"
            thumb = out_dir / f"{Path(rel).stem}_{kind}_{g[0][0]:06.1f}.jpg"
            cv2.imwrite(str(thumb), cv2.resize(closest[4], (480, int(480 * closest[4].shape[0] / closest[4].shape[1]))))
            events.append(Event(rel, round(g[0][0], 1), round(last[0], 1), kind, len(g), round(closest[2], 1),
                                round(closest[3], 2), round(last[3], 2), round(last[2], 1),
                                None if vmin is None else round(vmin, 1), outcome, thumb.name))
    events.sort(key=lambda e: e.t_start)
    return events


def write_events(events: list[Event], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(Event.__dataclass_fields__))
        w.writeheader()
        for e in events:
            w.writerow(asdict(e))
