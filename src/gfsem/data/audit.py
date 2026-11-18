"""Phase 0 video audit.

For every video under ``data/raw/`` this module records container, resolution,
fps, duration, side-by-side stereo detection, embedded IMU/timestamps and
intrinsics (SVO only), and writes:

* ``data/inventory.csv``                 one row per file
* ``data/audit/<stem>_contact.jpg``       12-frame contact sheet
* ``data/audit/<stem>_timeline.csv``      per-second motion + obstacle-colour hints
* ``data/audit/<stem>_obstacles.jpg``     frames with the strongest bollard/cone hints

Classification (PROMPT.md §2): A = ZED SVO/SVO2, B = side-by-side stereo MP4,
C = monocular onboard, D = not an ego view. Rosbags (``.db3``, added after Phase 0
review) get R1 = ZED rectified RGB + ZED odometry, R2 = ZED RGB + ECU speed but no
odometry, X = no usable camera stream (< 2 Hz). A, B, R1, R2, X are detected automatically;
C vs D cannot be decided from pixels statistics alone, so non-stereo files get
``C?`` unless ``configs/audit_overrides.yaml`` sets the class after a visual
review. The overrides file can also exclude time ranges (e.g. pit-lane walking).

Nothing under ``data/raw/`` is ever written to.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import yaml

VIDEO_EXT = {".svo", ".svo2", ".mp4", ".mkv", ".webm", ".avi", ".mov", ".db3"}
MOVING_KMH = 1.8  # 0.5 m/s, the low-speed threshold of PROMPT.md §2
MIN_IMAGE_HZ = 2.0

# Motion: median LK flow magnitude (px at 640-wide) between samples ANALYSIS_DT apart.
ANALYSIS_DT = 0.2  # s, i.e. the 5 Hz sample rate used for training samples
STATIC_FLOW_PX = 0.6  # below this the car is treated as stationary

# HSV ranges (OpenCV: H in [0,180)). Tuned loosely; these are hints, not labels.
BLUE_LO, BLUE_HI = (98, 110, 50), (126, 255, 255)
ORANGE_LO, ORANGE_HI = (4, 140, 110), (20, 255, 255)


@dataclass
class VideoInfo:
    path: str
    container: str
    width: int
    height: int
    fps: float
    duration_s: float
    n_frames: int
    side_by_side: bool
    sbs_score: float
    has_imu: bool
    has_timestamps: bool
    intrinsics: str  # JSON (fx, fy, cx, cy, dist) or "" if unknown
    auto_class: str
    final_class: str
    domain: str = "?"  # track | urban | handheld | lab, from the manual review
    moving_s: float = 0.0
    static_s: float = 0.0
    excluded_s: float = 0.0
    usable_s: float = 0.0
    obstacle_seconds: int = 0
    obstacle_events: int = 0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- probing


def ffprobe(path: Path) -> dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out)


def _ratio(s: str | None) -> float:
    if not s or s in ("0/0",):
        return 0.0
    num, _, den = s.partition("/")
    return float(num) / float(den or 1)


def probe_container(path: Path) -> dict[str, Any]:
    """Container-level facts via ffprobe, falling back to decoding when metadata is missing."""
    meta = ffprobe(path)
    vs = next(s for s in meta["streams"] if s["codec_type"] == "video")
    fps = _ratio(vs.get("avg_frame_rate")) or _ratio(vs.get("r_frame_rate"))
    duration = float(meta["format"].get("duration") or vs.get("duration") or 0.0)
    n_frames = int(vs["nb_frames"]) if vs.get("nb_frames", "N/A") not in ("N/A", None) else 0
    if n_frames == 0:
        n_frames = count_frames(path)
    tags = meta["format"].get("tags", {})
    return {
        "container": meta["format"]["format_name"].split(",")[0] + f"/{vs['codec_name']}",
        "width": int(vs["width"]),
        "height": int(vs["height"]),
        "fps": round(fps, 3),
        "duration_s": round(duration, 2),
        "n_frames": n_frames,
        "creation_time": tags.get("creation_time", ""),
        "encoder": tags.get("encoder", ""),
        "has_audio": any(s["codec_type"] == "audio" for s in meta["streams"]),
    }


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return int(out.split(",")[0]) if out else 0


def probe_svo(path: Path) -> dict[str, Any]:
    """Open an SVO with the ZED SDK and read resolution, fps, calibration and sensor availability."""
    import pyzed.sl as sl  # imported lazily: only needed for class A

    init = sl.InitParameters()
    init.set_from_svo_file(str(path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"pyzed could not open {path}: {err}")
    try:
        info = cam.get_camera_information()
        cfg = info.camera_configuration
        left = cfg.calibration_parameters.left_cam
        n = cam.get_svo_number_of_frames()
        fps = float(cfg.fps)
        intr = {"fx": left.fx, "fy": left.fy, "cx": left.cx, "cy": left.cy, "dist": list(left.disto),
                "baseline_m": cfg.calibration_parameters.get_camera_baseline() / 1000.0}
        sensors = info.sensors_configuration
        has_imu = bool(sensors.is_sensor_available(sl.SENSOR_TYPE.ACCELEROMETER))
        return {
            "container": path.suffix.lstrip(".").upper(),
            "width": int(cfg.resolution.width),
            "height": int(cfg.resolution.height),
            "fps": fps,
            "duration_s": round(n / fps, 2) if fps else 0.0,
            "n_frames": int(n),
            "has_imu": has_imu,
            "has_timestamps": True,
            "intrinsics": json.dumps(intr),
            "camera_model": str(info.camera_model),
            "serial": int(info.serial_number),
        }
    finally:
        cam.close()


# --------------------------------------------------------------------------- frame access


def iter_frames(path: Path, dt: float) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_s, BGR frame) roughly every ``dt`` seconds by sequential decoding.

    Dispatches on file type: SVO via pyzed (rectified left image), ``.db3`` via the bag
    reader (ZED rectified RGB topic), anything else via OpenCV. Sequential reads are used
    instead of seeking because VP8/MPEG-4 seeking is unreliable.
    """
    suffix = path.suffix.lower()
    if suffix in (".svo", ".svo2"):
        yield from _iter_svo(path, dt)
        return
    if suffix == ".db3":
        from gfsem.data.rosbag_io import BagReader

        bag = BagReader(path)
        try:
            yield from bag.iter_images(dt)
        finally:
            bag.close()
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open {path}")
    next_t = 0.0
    idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        t = t_ms / 1000.0 if t_ms > 0 else idx / fps
        idx += 1
        if t + 1e-6 >= next_t:
            yield t, frame
            next_t += dt
    cap.release()


def _iter_svo(path: Path, dt: float) -> Iterator[tuple[float, np.ndarray]]:
    import pyzed.sl as sl

    init = sl.InitParameters()
    init.set_from_svo_file(str(path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NONE
    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"pyzed could not open {path}")
    mat = sl.Mat()
    t0: int | None = None
    next_t = 0.0
    try:
        while cam.grab() == sl.ERROR_CODE.SUCCESS:
            ns = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            t0 = ns if t0 is None else t0
            t = (ns - t0) / 1e9
            if t + 1e-6 < next_t:
                continue
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            yield t, mat.get_data()[:, :, :3].copy()  # ZED returns BGRA
            next_t = t + dt
    finally:
        cam.close()


# --------------------------------------------------------------------------- bags


def probe_bag(path: Path) -> dict[str, Any]:
    """Topic-level facts for a ROS 2 bag plus speed/odometry statistics.

    Moving time is taken from ``/ECU/speed`` (km/h) when present, else from odometry speed,
    and is only counted where the camera stream has no gap longer than 1 s.
    """
    from gfsem.data import rosbag_io as rio

    bag = rio.BagReader(path)
    try:
        img_topic = bag.first_of(rio.IMAGE_TOPICS)
        odom_topic = bag.first_of(rio.ODOM_TOPICS)
        steer_topic = bag.first_of(rio.STEER_TOPICS)
        out: dict[str, Any] = {
            "container": "ros2bag/sqlite3", "duration_s": round(bag.duration_s, 2),
            "image_topic": img_topic or "", "image_hz": round(bag.hz(img_topic), 2) if img_topic else 0.0,
            "odom_topic": odom_topic or "", "odom_hz": round(bag.hz(odom_topic), 2) if odom_topic else 0.0,
            "has_ecu_speed": bag.has(rio.SPEED_TOPIC), "steer_cmd_topic": steer_topic or "",
            "has_imu": bag.first_of(rio.IMU_TOPICS) is not None,
            "has_gps": bag.has("/ublox_gps_node/fix") or bag.has("/gps/filtered"),
            "has_timestamps": True, "bag_t0_ns": bag.t0_ns, "bag_t1_ns": bag.t1_ns,
            "width": 0, "height": 0, "n_frames": bag.topics[img_topic].count if img_topic else 0,
            "fps": round(bag.hz(img_topic), 2) if img_topic else 0.0, "intrinsics": "",
        }
        if img_topic:
            for _, img in bag.iter_images(1e9, img_topic):
                out["height"], out["width"] = img.shape[:2]
                break
        ci = bag.first_of(rio.CAMINFO_TOPICS)
        if ci:
            _, msg = next(bag.messages(ci))
            k = list(msg.k)
            out["intrinsics"] = json.dumps({"fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5],
                                            "dist": list(msg.d), "w": msg.width, "h": msg.height})
        # speed profile on a 0.1 s grid
        grid = np.arange(bag.t0_ns, bag.t1_ns, int(1e8))
        speed = None
        ecu_dist = odom_dist = float("nan")
        if out["has_ecu_speed"]:
            ts, v = zip(*[(t, m.data) for t, m in bag.messages(rio.SPEED_TOPIC)])
            ts_a, v_a = np.array(ts), np.array(v, dtype=float)
            speed = np.interp(grid, ts_a, v_a)
            ecu_dist = float(np.sum(np.clip(speed, 0, None) / 3.6 * 0.1))
            out["ecu_max_kmh"] = round(float(v_a.max()), 2)
        if odom_topic:
            pts = [(t, m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z)
                   for t, m in bag.messages(odom_topic)]
            if len(pts) > 2:
                a = np.array(pts, dtype=float)
                odom_dist = float(np.sum(np.linalg.norm(np.diff(a[:, 1:3], axis=0), axis=1)))
                if speed is None:
                    vt = np.linalg.norm(np.diff(a[:, 1:3], axis=0), axis=1) / np.maximum(np.diff(a[:, 0]) / 1e9, 1e-3)
                    speed = np.interp(grid, a[1:, 0], vt * 3.6)
        out["ecu_dist_m"] = round(ecu_dist, 1) if np.isfinite(ecu_dist) else ""
        out["odom_dist_m"] = round(odom_dist, 1) if np.isfinite(odom_dist) else ""
        if img_topic:
            its = bag.timestamps(img_topic)
            idx = np.clip(np.searchsorted(its, grid), 1, len(its) - 1)
            gap = np.minimum(np.abs(its[idx] - grid), np.abs(its[idx - 1] - grid)) / 1e9
            covered = gap < 0.5
        else:
            covered = np.zeros_like(grid, dtype=bool)
        if speed is not None:
            moving = speed > MOVING_KMH
            out["speed_moving_s"] = round(float(moving.sum()) * 0.1, 1)
            out["speed_usable_s"] = round(float((moving & covered).sum()) * 0.1, 1)
            out["mean_moving_kmh"] = round(float(speed[moving].mean()), 2) if moving.any() else 0.0
        return out
    finally:
        bag.close()


# --------------------------------------------------------------------------- analyses


def side_by_side_score(frame: np.ndarray) -> float:
    """Normalised cross-correlation between left and right halves (best over small shifts).

    A ZED side-by-side frame has two nearly identical views with a horizontal disparity,
    so the score is high (> 0.8). A monocular frame scores much lower.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (g.shape[1] // 4 * 2, g.shape[0] // 2)).astype(np.float32)
    w = g.shape[1] // 2
    left, right = g[:, :w], g[:, w:]
    best = -1.0
    for shift in range(0, 24, 2):
        a = left[:, shift:]
        b = right[:, : w - shift]
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        best = max(best, float((a * b).mean()))
    return best


def median_flow(prev: np.ndarray, cur: np.ndarray) -> float:
    """Median LK optical-flow magnitude on the lower 60 % of the image (road region), px @640w."""
    h = prev.shape[0]
    y0 = int(0.4 * h)
    p0 = cv2.goodFeaturesToTrack(prev[y0:], maxCorners=300, qualityLevel=0.01, minDistance=8)
    if p0 is None or len(p0) < 10:
        return float("nan")
    p0 = p0 + np.array([[[0, y0]]], dtype=np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
    good = st.ravel() == 1
    if good.sum() < 10:
        return float("nan")
    return float(np.median(np.linalg.norm((p1 - p0)[good].reshape(-1, 2), axis=1)))


def obstacle_blobs(frame: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Blue-bollard-like and orange-cone-like blobs below the horizon. Returns (x, y, w, h) boxes.

    Shape gates: bollards are upright cylinders (h/w 1.2–4.5), cones are roughly as tall as
    wide or taller; both must sit in the lower 60 % of the image and not span huge areas
    (which rejects painted blue pit floors and sky).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, W = frame.shape[:2]

    def boxes(lo: tuple[int, int, int], hi: tuple[int, int, int], ar_lo: float, ar_hi: float,
              min_area: int) -> list[tuple[int, int, int, int]]:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area or area > 0.03 * H * W:
                continue
            if y + h < 0.45 * H:  # entirely above the horizon band
                continue
            ar = h / max(w, 1)
            fill = area / float(w * h)
            if ar_lo <= ar <= ar_hi and fill > 0.45:
                out.append((int(x), int(y), int(w), int(h)))
        return out

    scale = W / 640.0
    blue = boxes(BLUE_LO, BLUE_HI, 1.2, 4.5, int(60 * scale * scale))
    orange = boxes(ORANGE_LO, ORANGE_HI, 0.7, 3.0, int(25 * scale * scale))
    return blue, orange


def _in_ranges(t: float, ranges: list[list[float]]) -> bool:
    return any(a <= t < b for a, b in ranges)


def analyse_timeline(path: Path, exclude: list[list[float]]) -> tuple[list[dict[str, Any]], list[tuple[float, np.ndarray, int]]]:
    """Per-sample motion and obstacle hints at ANALYSIS_DT. Returns rows and top obstacle frames."""
    rows: list[dict[str, Any]] = []
    prev_g: np.ndarray | None = None
    scored: list[tuple[float, np.ndarray, int]] = []
    for t, frame in iter_frames(path, ANALYSIS_DT):
        small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        flow = median_flow(prev_g, g) if prev_g is not None else float("nan")
        prev_g = g
        blue, orange = obstacle_blobs(small)
        rows.append({"t": round(t, 3), "flow_px": round(flow, 3) if not math.isnan(flow) else "",
                     "n_blue": len(blue), "n_orange": len(orange), "excluded": int(_in_ranges(t, exclude))})
        k = len(blue) + len(orange)
        if k:
            vis = small.copy()
            for (x, y, w, h) in blue:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 2)
            for (x, y, w, h) in orange:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 128, 255), 2)
            scored.append((t, vis, k))
    # keep the 12 highest-scoring frames, at least 2 s apart
    scored.sort(key=lambda s: -s[2])
    picked: list[tuple[float, np.ndarray, int]] = []
    for s in scored:
        if all(abs(s[0] - p[0]) >= 2.0 for p in picked):
            picked.append(s)
        if len(picked) == 12:
            break
    picked.sort(key=lambda s: s[0])
    return rows, picked


def summarise_timeline(rows: list[dict[str, Any]], fps_eff: float = 1 / ANALYSIS_DT) -> dict[str, Any]:
    """Moving/static/excluded seconds and obstacle 'events' (runs of ≥2 s with hints, ≤1 s gaps)."""
    dt = 1.0 / fps_eff
    moving = static = excluded = 0.0
    # Smooth the static decision over a 1 s window to ignore single noisy samples.
    flows = np.array([r["flow_px"] if r["flow_px"] != "" else np.nan for r in rows], dtype=float)
    k = int(round(fps_eff))
    smooth = np.array([np.nanmedian(flows[max(0, i - k // 2): i + k // 2 + 1]) if np.isfinite(
        flows[max(0, i - k // 2): i + k // 2 + 1]).any() else np.nan for i in range(len(flows))])
    for r, f in zip(rows, smooth):
        if r["excluded"]:
            excluded += dt
        elif np.isfinite(f) and f < STATIC_FLOW_PX:
            static += dt
        else:
            moving += dt
        r["static"] = int(bool(np.isfinite(f) and f < STATIC_FLOW_PX))
    secs: dict[int, int] = {}
    for r in rows:
        if not r["excluded"]:
            secs[int(r["t"])] = secs.get(int(r["t"]), 0) + r["n_blue"] + r["n_orange"]
    hit = sorted(s for s, n in secs.items() if n >= 2)  # ≥2 detections within that second
    events = 0
    run_start = prev = None
    for s in hit + [None]:  # type: ignore[list-item]
        if s is not None and prev is not None and s - prev <= 2:
            prev = s
            continue
        if run_start is not None and prev is not None and prev - run_start + 1 >= 2:
            events += 1
        run_start = prev = s
    return {"moving_s": round(moving, 1), "static_s": round(static, 1), "excluded_s": round(excluded, 1),
            "usable_s": round(moving, 1), "obstacle_seconds": len(hit), "obstacle_events": events}


# --------------------------------------------------------------------------- visuals


def _label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def tile(frames: list[tuple[str, np.ndarray]], cols: int = 4, width: int = 400) -> np.ndarray:
    tiles = []
    for text, f in frames:
        h = int(f.shape[0] * width / f.shape[1])
        tiles.append(_label(cv2.resize(f, (width, h)), text))
    if not tiles:
        return np.zeros((10, 10, 3), np.uint8)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i: i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def contact_sheet(path: Path, duration: float, n: int = 12) -> np.ndarray:
    targets = [duration * (i + 0.5) / n for i in range(n)]
    got: list[tuple[str, np.ndarray]] = []
    j = 0
    for t, frame in iter_frames(path, 0.1):
        if j < n and t >= targets[j]:
            got.append((f"{path.name}  t={t:6.1f}s", frame))
            j += 1
        if j == n:
            break
    return tile(got)


# --------------------------------------------------------------------------- driver


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("files", {}) or {}


def audit_file(path: Path, raw_root: Path, out_dir: Path, overrides: dict[str, dict[str, Any]]) -> VideoInfo:
    rel = str(path.relative_to(raw_root))
    ov = overrides.get(rel, {})
    suffix = path.suffix.lower()
    sbs, sbs_score = False, 0.0
    if suffix in (".svo", ".svo2"):
        p = probe_svo(path)
        auto = "A"
    elif suffix == ".db3":
        p = probe_bag(path)
        if p["image_hz"] < MIN_IMAGE_HZ:
            auto = "X"
        elif p["odom_topic"]:
            auto = "R1"
        elif p["has_ecu_speed"]:
            auto = "R2"
        else:
            auto = "C?"
    else:
        p = probe_container(path)
        p.update(has_imu=False, has_timestamps=False, intrinsics="")
        scores = []
        for i, (_, frame) in enumerate(iter_frames(path, max(p["duration_s"] / 6, 0.5))):
            scores.append(side_by_side_score(frame))
            if i >= 5:
                break
        sbs_score = float(np.median(scores)) if scores else 0.0
        aspect = p["width"] / p["height"]
        sbs = bool(aspect >= 3.0 and sbs_score > 0.8)
        auto = "B" if sbs else "C?"
    final = str(ov.get("class", auto))
    exclude = ov.get("exclude", []) or []

    stem = path.stem
    if auto == "X":
        rows, top = [], []
        summary = {"moving_s": 0.0, "static_s": 0.0, "excluded_s": 0.0, "usable_s": 0.0,
                   "obstacle_seconds": 0, "obstacle_events": 0}
    else:
        rows, top = analyse_timeline(path, exclude)
        summary = summarise_timeline(rows) if rows else {
            "moving_s": 0.0, "static_s": 0.0, "excluded_s": 0.0, "usable_s": 0.0,
            "obstacle_seconds": 0, "obstacle_events": 0}
        if "speed_usable_s" in p:
            # Bags: trust ECU/odometry speed over image flow for moving time, minus excluded ranges.
            summary["moving_s"] = p["speed_moving_s"]
            summary["usable_s"] = round(max(0.0, p["speed_usable_s"] - summary["excluded_s"]), 1)
    if final in ("D", "X", "dup"):
        summary["usable_s"] = 0.0
    if rows:
        with open(out_dir / f"{stem}_timeline.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        cv2.imwrite(str(out_dir / f"{stem}_contact.jpg"), contact_sheet(path, p["duration_s"]))
    if top:
        cv2.imwrite(str(out_dir / f"{stem}_obstacles.jpg"),
                    tile([(f"t={t:6.1f}s  hints={k}", f) for t, f, k in top], width=320))

    extra = {k: v for k, v in p.items() if k not in VideoInfo.__dataclass_fields__}
    return VideoInfo(
        path=rel, container=p["container"], width=p["width"], height=p["height"], fps=p["fps"],
        duration_s=p["duration_s"], n_frames=p["n_frames"], side_by_side=sbs, sbs_score=round(sbs_score, 3),
        has_imu=p["has_imu"], has_timestamps=p["has_timestamps"], intrinsics=p["intrinsics"],
        auto_class=auto, final_class=final, domain=str(ov.get("domain", "?")), notes=str(ov.get("notes", "")),
        extra=extra, **summary,
    )


def _audit_job(args: tuple[Path, Path, Path, dict[str, dict[str, Any]]]) -> VideoInfo:
    f, raw_root, out_dir, overrides = args
    print(f"[audit] {f.relative_to(raw_root)}", flush=True)
    try:
        return audit_file(f, raw_root, out_dir, overrides)
    except Exception as e:  # corrupt/empty recordings are logged, not fatal
        return VideoInfo(path=str(f.relative_to(raw_root)), container="?", width=0, height=0, fps=0.0,
                         duration_s=0.0, n_frames=0, side_by_side=False, sbs_score=0.0, has_imu=False,
                         has_timestamps=False, intrinsics="", auto_class="ERR", final_class="ERR",
                         notes=f"{type(e).__name__}: {e}"[:300])


def mark_duplicate_bags(infos: list[VideoInfo]) -> None:
    """Bags whose recording window overlaps an earlier-listed bag by > 90 % are copies.

    The copy with the most usable time is kept; the others get final_class ``dup``.
    """
    bags = [i for i in infos if "bag_t0_ns" in i.extra]
    bags.sort(key=lambda i: -i.usable_s)
    kept: list[VideoInfo] = []
    for b in bags:
        b0, b1 = b.extra["bag_t0_ns"], b.extra["bag_t1_ns"]
        for k in kept:
            k0, k1 = k.extra["bag_t0_ns"], k.extra["bag_t1_ns"]
            inter = max(0, min(b1, k1) - max(b0, k0))
            if inter > 0.9 * max(1, min(b1 - b0, k1 - k0)):
                b.final_class, b.usable_s = "dup", 0.0
                b.notes = (b.notes + f" duplicate of {k.path}").strip()
                break
        else:
            kept.append(b)


def write_inventory(infos: list[VideoInfo], inventory_csv: Path) -> None:
    with open(inventory_csv, "w", newline="") as fh:
        fields = [k for k in VideoInfo.__dataclass_fields__ if k != "extra"] + ["extra"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for info in infos:
            d = asdict(info)
            d["extra"] = json.dumps(d["extra"])
            w.writerow(d)


def read_inventory(inventory_csv: Path) -> list[VideoInfo]:
    types = {k: f.type for k, f in VideoInfo.__dataclass_fields__.items()}
    out = []
    for r in csv.DictReader(open(inventory_csv)):
        kw: dict[str, Any] = {}
        for k, v in r.items():
            t = str(types.get(k, "str"))
            if k == "extra":
                kw[k] = json.loads(v) if v else {}
            elif t in ("int",):
                kw[k] = int(float(v or 0))
            elif t in ("float",):
                kw[k] = float(v or 0)
            elif t in ("bool",):
                kw[k] = v == "True"
            elif k in types:
                kw[k] = v
        out.append(VideoInfo(**kw))
    return out


def reclassify(inventory_csv: Path, out_dir: Path, overrides_path: Path) -> list[VideoInfo]:
    """Re-apply ``audit_overrides.yaml`` (class, domain, notes, exclude) without decoding video again."""
    overrides = load_overrides(overrides_path)
    infos = read_inventory(inventory_csv)
    for i in infos:
        if i.final_class == "dup":
            continue
        ov = overrides.get(i.path, {})
        i.final_class = str(ov.get("class", i.auto_class))
        i.domain = str(ov.get("domain", i.domain))
        i.notes = str(ov.get("notes", ""))
        stem = Path(i.path).stem
        tl = out_dir / f"{stem}_timeline.csv"
        if tl.exists():
            exclude = ov.get("exclude", []) or []
            rows = list(csv.DictReader(open(tl)))
            for r in rows:
                r["t"] = float(r["t"])
                r["n_blue"], r["n_orange"] = int(r["n_blue"]), int(r["n_orange"])
                r["excluded"] = int(_in_ranges(r["t"], exclude))
            summ = summarise_timeline(rows)
            i.excluded_s = summ["excluded_s"]
            if "speed_usable_s" in i.extra:
                i.usable_s = round(max(0.0, i.extra["speed_usable_s"] - i.excluded_s), 1)
            else:
                i.usable_s = summ["usable_s"]
        if i.final_class in ("D", "X", "ERR"):
            i.usable_s = 0.0
    write_inventory(infos, inventory_csv)
    return infos


def run_audit(raw_root: Path, inventory_csv: Path, out_dir: Path, overrides_path: Path,
              workers: int = 6) -> list[VideoInfo]:
    from concurrent.futures import ProcessPoolExecutor

    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(overrides_path)
    files = sorted(p for p in raw_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)
    jobs = [(f, raw_root, out_dir, overrides) for f in files]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            infos = list(ex.map(_audit_job, jobs))
    else:
        infos = [_audit_job(j) for j in jobs]
    mark_duplicate_bags(infos)
    write_inventory(infos, inventory_csv)
    return infos
