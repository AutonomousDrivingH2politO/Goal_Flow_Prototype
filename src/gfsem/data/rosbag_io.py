"""Read-only access to ROS 2 sqlite3 bags (``.db3``) without a running ROS graph.

Messages are deserialised with ``rclpy.serialization`` and the Humble message
packages, so ``/opt/ros/humble/setup.bash`` (and, for ``zed_msgs``, the AD-SEM
install) must be sourced. The bag file is opened with ``mode=ro``; nothing is
ever written to it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

# Preferred camera topics, best first. Compressed is cheapest to read.
IMAGE_TOPICS = (
    "/zed/zed_node/rgb/color/rect/image/compressed",
    "/zed/zed_node/rgb/color/rect/image",
    "/zed/zed_node/rgb/image_rect_color/compressed",
    "/zed/zed_node/rgb/image_rect_color",
)
ODOM_TOPICS = ("/zed/zed_node/odom", "/odometry/filtered/local")
SPEED_TOPIC = "/ECU/speed"  # std_msgs/Float32, km/h
STEER_TOPICS = ("/commands/KalmanAngle", "/cmd_vel")  # commands, not measured steering
IMU_TOPICS = ("/zed/zed_node/imu/data",)
CAMINFO_TOPICS = ("/zed/zed_node/rgb/color/rect/image/camera_info", "/zed/zed_node/rgb/camera_info")


@lru_cache(maxsize=None)
def _msg_type(name: str) -> Any:
    from rosidl_runtime_py.utilities import get_message

    return get_message(name)


def deserialize(data: bytes, type_name: str) -> Any:
    from rclpy.serialization import deserialize_message

    return deserialize_message(data, _msg_type(type_name))


def image_to_bgr(msg: Any, type_name: str) -> np.ndarray:
    """Convert sensor_msgs/Image or CompressedImage to a BGR uint8 array."""
    if type_name.endswith("CompressedImage"):
        return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    enc = msg.encoding.lower()
    ch = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3, "mono8": 1}[enc]
    img = buf.reshape(msg.height, msg.step)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
    if enc == "bgra8":
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "mono8":
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


@dataclass
class TopicInfo:
    id: int
    name: str
    type: str
    count: int


class BagReader:
    """Minimal reader for one ROS 2 ``.db3`` file (or a bag directory containing one)."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        if path.is_dir():
            dbs = sorted(path.glob("*.db3"))
            if not dbs:
                raise FileNotFoundError(f"no .db3 in {path}")
            path = dbs[0]
        self.path = path
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        counts = dict(self.db.execute("select topic_id, count(*) from messages group by topic_id").fetchall())
        self.topics = {n: TopicInfo(i, n, t, counts.get(i, 0))
                       for i, n, t in self.db.execute("select id, name, type from topics")}
        t0, t1 = self.db.execute("select min(timestamp), max(timestamp) from messages").fetchone()
        self.t0_ns: int = t0 or 0
        self.t1_ns: int = t1 or 0

    @property
    def duration_s(self) -> float:
        return (self.t1_ns - self.t0_ns) / 1e9

    def has(self, topic: str) -> bool:
        return topic in self.topics and self.topics[topic].count > 0

    def first_of(self, candidates: tuple[str, ...]) -> str | None:
        return next((t for t in candidates if self.has(t)), None)

    def hz(self, topic: str) -> float:
        if not self.has(topic) or self.duration_s <= 0:
            return 0.0
        return self.topics[topic].count / self.duration_s

    def timestamps(self, topic: str) -> np.ndarray:
        """Bag receive timestamps (ns) for a topic."""
        tid = self.topics[topic].id
        return np.array([r[0] for r in self.db.execute(
            "select timestamp from messages where topic_id=? order by timestamp", (tid,))], dtype=np.int64)

    def messages(self, topic: str) -> Iterator[tuple[int, Any]]:
        info = self.topics[topic]
        for ts, data in self.db.execute(
                "select timestamp, data from messages where topic_id=? order by timestamp", (info.id,)):
            yield ts, deserialize(data, info.type)

    def iter_images(self, dt: float, topic: str | None = None) -> Iterator[tuple[float, np.ndarray]]:
        """Yield (t_s relative to bag start, BGR) roughly every ``dt`` seconds, decoding only those rows."""
        topic = topic or self.first_of(IMAGE_TOPICS)
        if topic is None:
            return
        info = self.topics[topic]
        rows = self.db.execute("select id, timestamp from messages where topic_id=? order by timestamp",
                               (info.id,)).fetchall()
        next_t = 0.0
        for rid, ts in rows:
            t = (ts - self.t0_ns) / 1e9
            if t + 1e-6 < next_t:
                continue
            (data,) = self.db.execute("select data from messages where id=?", (rid,)).fetchone()
            try:
                img = image_to_bgr(deserialize(data, info.type), info.type)
            except Exception:  # corrupt frame: skip it rather than abort the audit
                continue
            if img is None:
                continue
            yield t, img
            next_t = t + dt

    def close(self) -> None:
        self.db.close()
