#!/usr/bin/env python3
"""Phase 0: count obstacle encounters (people / bollards / cones) in the track recordings.

Reads data/inventory.csv (domain == track, usable classes) and writes
data/audit/obstacle_events.csv plus one thumbnail per event in data/audit/events/.
Source ROS first for bags (see audit_videos.py).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfsem.data.audit import load_overrides  # noqa: E402
from gfsem.data.obstacle_events import Intrinsics, detect_events, write_events  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory", type=Path, default=Path("data/inventory.csv"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--vehicle", type=Path, default=Path("configs/vehicle.yaml"))
    ap.add_argument("--overrides", type=Path, default=Path("configs/audit_overrides.yaml"))
    ap.add_argument("--weights", default="/home/bylogix/yolov8m.pt")
    ap.add_argument("--out", type=Path, default=Path("data/audit"))
    a = ap.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(a.vehicle.read_text())
    cam = cfg["camera"]
    k = cam["hd720"]
    K = Intrinsics(k["fx"], k["fy"], k["cx"], k["cy"])
    model = YOLO(a.weights)
    ov = load_overrides(a.overrides)
    thumbs = a.out / "events"
    thumbs.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.inventory))
            if r["domain"] == "track" and r["final_class"] in ("A", "B", "C", "R1", "R2")]
    events = []
    for r in rows:
        print(f"[events] {r['path']}", flush=True)
        ev = detect_events(a.raw / r["path"], r["path"], model, K, k["width"], cam["height_m"], thumbs,
                           exclude=ov.get(r["path"], {}).get("exclude"))
        events += ev
        write_events(events, a.out / "obstacle_events.csv")
    c = Counter((e.kind, e.outcome) for e in events)
    print("\nevents by kind/outcome:")
    for (kind, outcome), n in sorted(c.items()):
        print(f"  {kind:8s} {outcome:13s} {n}")


if __name__ == "__main__":
    main()
