#!/usr/bin/env python3
"""Phase 0: audit every recording under data/raw/ (videos, SVO2, ROS 2 bags).

Writes data/inventory.csv and contact sheets in data/audit/. For bags, source ROS first:
    source /opt/ros/humble/setup.bash && source ~/AD-SEM/install/setup.bash
--reclassify re-applies configs/audit_overrides.yaml to an existing inventory without decoding.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfsem.data.audit import reclassify, run_audit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--inventory", type=Path, default=Path("data/inventory.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/audit"))
    ap.add_argument("--overrides", type=Path, default=Path("configs/audit_overrides.yaml"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reclassify", action="store_true")
    a = ap.parse_args()

    if a.reclassify:
        infos = reclassify(a.inventory, a.out, a.overrides)
    else:
        infos = run_audit(a.raw, a.inventory, a.out, a.overrides, a.workers)
    per: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    print(f"\n{'file':70s} {'cls':4s} {'domain':8s} {'fps':>6s} {'dur s':>7s} {'usable':>7s}")
    for i in sorted(infos, key=lambda i: (i.domain, i.path)):
        print(f"{i.path[:70]:70s} {i.final_class:4s} {i.domain:8s} {i.fps:6.2f} {i.duration_s:7.1f} {i.usable_s:7.1f}")
        per[(i.domain, i.final_class)][0] += i.duration_s
        per[(i.domain, i.final_class)][1] += i.usable_s
        per[(i.domain, i.final_class)][2] += 1
    print("\nper domain/class:  files / total min / usable min")
    for (d, c), (tot, use, n) in sorted(per.items()):
        print(f"  {d:8s} {c:4s} {int(n):3d} {tot/60:7.2f} {use/60:7.2f}")


if __name__ == "__main__":
    main()
