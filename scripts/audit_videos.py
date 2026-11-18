#!/usr/bin/env python3
"""Phase 0: audit every video under data/raw/ and write data/inventory.csv + contact sheets."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfsem.data.audit import run_audit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--inventory", type=Path, default=Path("data/inventory.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/audit"))
    ap.add_argument("--overrides", type=Path, default=Path("configs/audit_overrides.yaml"))
    a = ap.parse_args()

    infos = run_audit(a.raw, a.inventory, a.out, a.overrides)
    per = defaultdict(lambda: [0.0, 0.0, 0])
    print(f"\n{'file':32s} {'cls':4s} {'res':10s} {'fps':>6s} {'dur s':>7s} {'moving':>7s} {'static':>7s} {'excl':>6s} {'obst ev':>7s}")
    for i in infos:
        print(f"{i.path:32s} {i.final_class:4s} {i.width}x{i.height:<5d} {i.fps:6.2f} {i.duration_s:7.1f} "
              f"{i.moving_s:7.1f} {i.static_s:7.1f} {i.excluded_s:6.1f} {i.obstacle_events:7d}")
        per[i.final_class][0] += i.duration_s
        per[i.final_class][1] += i.usable_s
        per[i.final_class][2] += i.obstacle_events
    print("\nper class:  total min / usable (moving, not excluded) min / obstacle-hint events")
    for c in sorted(per):
        tot, use, ev = per[c]
        print(f"  {c:3s} {tot/60:6.2f} / {use/60:6.2f} / {ev}")


if __name__ == "__main__":
    main()
