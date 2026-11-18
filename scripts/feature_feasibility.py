#!/usr/bin/env python3
"""Feature-extraction feasibility on the team footage (pre-Phase-2 check requested by Can).

Questions answered:
  1. Speed: does a frozen backbone run fast enough on the AGX Orin (fp32 / fp16, batch 1)?
  2. Storage: how large is the token cache for all usable track footage?
  3. Information: do frozen tokens *linearly* encode the drivable area on a held-out session?
     Target = TwinLiteNetPlus drivable mask averaged per patch; baseline = per-patch colour
     statistics + patch position with the same classifier.
  4. Picture: PCA of the tokens (first 3 components → RGB) next to the image and masks.

Frames: track recordings (classes A, R1, R2, C) at 1 Hz, excluded ranges and over-exposed frames
dropped. --sessions limits the recordings, --holdout picks the test recording(s). Source ROS first for bags.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gfsem.data.audit import iter_frames, load_overrides  # noqa: E402
from gfsem.data.features import (  # noqa: E402
    BackboneSpec, DinoV2Tokens, TwinLiteDrivable, mask_to_grid, preprocess)

HOLDOUT_KEY = "Morano_21_05_2026"


def overexposed(bgr: np.ndarray) -> bool:
    """Auto-exposure washout at bag start: most pixels near white."""
    small = cv2.resize(bgr, (160, 90))
    return bool((small.min(axis=2) > 235).mean() > 0.35)


def colour_features(bgr: np.ndarray, spec: BackboneSpec) -> np.ndarray:
    """Per-patch mean/std of RGB and HSV + normalised (row, col) → (rows, cols, 14)."""
    img = cv2.resize(bgr, (spec.width, spec.height), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x = np.concatenate([img, hsv], axis=2).astype(np.float32) / 255.0
    r, c, p = spec.grid[0], spec.grid[1], spec.patch
    x = x[: r * p, : c * p].reshape(r, p, c, p, 6)
    mean, std = x.mean(axis=(1, 3)), x.std(axis=(1, 3))
    rr, cc = np.meshgrid(np.linspace(0, 1, r), np.linspace(0, 1, c), indexing="ij")
    return np.concatenate([mean, std, rr[..., None], cc[..., None]], axis=2)


def bench(fn, x: torch.Tensor, n: int = 30) -> float:
    for _ in range(5):
        fn(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def train_probe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, epochs: int = 30,
                hidden: int = 0, seed: int = 0) -> dict[str, float]:
    """Per-patch classifier (linear if hidden == 0) on standardised features; BCE on soft targets."""
    torch.manual_seed(seed)
    dev = "cuda"
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t = torch.from_numpy((Xtr - mu) / sd).float().to(dev)
    Xte_t = torch.from_numpy((Xte - mu) / sd).float().to(dev)
    ytr_t = torch.from_numpy(ytr).float().to(dev)
    d = Xtr.shape[1]
    model: nn.Module = nn.Linear(d, 1) if hidden == 0 else nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Xtr_t)
    for _ in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 4096):
            idx = perm[i: i + 4096]
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xtr_t[idx]).squeeze(1), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte_t).squeeze(1)).cpu().numpy()
    pred, gt = p > 0.5, yte > 0.5
    inter, union = (pred & gt).sum(), (pred | gt).sum()
    return {"iou": float(inter / max(union, 1)), "acc": float((pred == gt).mean()), "probs": p}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, default=Path("data/inventory.csv"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--overrides", type=Path, default=Path("configs/audit_overrides.yaml"))
    ap.add_argument("--vehicle", type=Path, default=Path("configs/vehicle.yaml"))
    ap.add_argument("--cache", type=Path, default=Path("data/features/feasibility"))
    ap.add_argument("--out", type=Path, default=Path("reports/feature_feasibility"))
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="substrings of inventory paths to use (default: all usable track recordings)")
    ap.add_argument("--holdout", default=HOLDOUT_KEY, help="substring marking held-out recordings")
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    cfg = yaml.safe_load(a.vehicle.read_text())
    seg_repo = Path(cfg["segmentation"]["repo"])
    dspec = BackboneSpec()  # DINOv2 518×294 → 21×37
    dino = DinoV2Tokens(dspec).cuda()
    tl = TwinLiteDrivable(seg_repo, "large")

    # ---------------------------------------------------------------- 1. latency
    dummy = torch.randn(1, 3, dspec.height, dspec.width, device="cuda")
    lat = {"dinov2_vits14_fp32_ms": bench(dino, dummy)}
    with torch.autocast("cuda", dtype=torch.float16):
        lat["dinov2_vits14_fp16_ms"] = bench(dino, dummy)
    lat["dinov2_vits14_fp16_batch8_per_frame_ms"] = bench(
        lambda x: _autocast(dino, x), torch.randn(8, 3, dspec.height, dspec.width, device="cuda")) / 8
    lat["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    print("latency:", json.dumps({k: round(v, 1) for k, v in lat.items()}), flush=True)

    # ---------------------------------------------------------------- 2. extract
    ov = load_overrides(a.overrides)
    rows = [r for r in csv.DictReader(open(a.inventory))
            if r["domain"] == "track" and r["final_class"] in ("A", "B", "C", "R1", "R2") and float(r["usable_s"]) > 5]
    if a.sessions:
        rows = [r for r in rows if any(k in r["path"] for k in a.sessions)]
    usable_total_s = sum(float(r["usable_s"]) for r in rows)
    recs: list[dict] = []
    n_over = 0
    t_extract = 0.0
    for r in rows:
        path = a.raw / r["path"]
        excl = ov.get(r["path"], {}).get("exclude") or []
        D, C, M, imgs, ts = [], [], [], [], []
        for t, f in iter_frames(path, 1.0 / a.hz):
            if any(x0 <= t < x1 for x0, x1 in excl):
                continue
            if overexposed(f):
                n_over += 1
                continue
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                D.append(dino(preprocess([f], dspec.width, dspec.height, "cuda"))[0].half().cpu().numpy())
            t_extract += time.perf_counter() - t0
            p = tl(f)
            M.append(mask_to_grid(p, dspec))
            C.append(colour_features(f, dspec))
            imgs.append(cv2.resize(f, (dspec.width, dspec.height)))
            ts.append(t)
        if not D:
            continue
        stem = Path(r["path"]).stem
        np.savez(a.cache / f"{stem}.npz", dino=np.stack(D), t=np.array(ts))
        recs.append({"path": r["path"], "holdout": a.holdout in r["path"], "dino": np.stack(D),
                     "col": np.stack(C), "m_d": np.stack(M), "imgs": imgs})
        print(f"[feat] {r['path']}: {len(D)} frames", flush=True)
    n_frames = sum(len(x["dino"]) for x in recs)
    bytes_per = dspec.grid[0] * dspec.grid[1] * 384 * 2
    storage = {
        "frames_extracted": n_frames, "overexposed_dropped": n_over,
        "dinov2_bytes_per_frame_fp16": bytes_per,
        "usable_track_s": usable_total_s,
        "full_cache_5hz_gb": usable_total_s * 5 * bytes_per / 1e9,
        "full_cache_5hz_with_flip_gb": usable_total_s * 5 * bytes_per * 2 / 1e9,
        "extract_ms_per_frame_incl_preproc": 1000 * t_extract / max(n_frames, 1),
    }
    print("storage:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in storage.items()}), flush=True)

    # ---------------------------------------------------------------- 3. probes
    def stack(key: str, mkey: str, hold: bool) -> tuple[np.ndarray, np.ndarray]:
        X = np.concatenate([x[key].reshape(-1, x[key].shape[-1]) for x in recs if x["holdout"] == hold]).astype(np.float32)
        y = np.concatenate([x[mkey].reshape(-1) for x in recs if x["holdout"] == hold]).astype(np.float32)
        return X, y

    results: dict[str, dict[str, float]] = {}
    keep_probs = None
    for name, key, mkey, hidden in [("colour+position (linear)", "col", "m_d", 0),
                                    ("DINOv2 ViT-S/14 (linear)", "dino", "m_d", 0)]:
        Xtr, ytr = stack(key, mkey, False)
        Xte, yte = stack(key, mkey, True)
        ious, accs = [], []
        for seed in range(a.seeds):
            out = train_probe(Xtr, ytr, Xte, yte, hidden=hidden, seed=seed)
            ious.append(out["iou"])
            accs.append(out["acc"])
            if key == "dino" and seed == 0:
                keep_probs = out["probs"]
        results[name] = {"iou_mean": float(np.mean(ious)), "iou_std": float(np.std(ious)),
                         "acc_mean": float(np.mean(accs)), "n_train_patches": int(len(Xtr)),
                         "n_test_patches": int(len(Xte))}
        print(f"probe {name:28s} IoU {np.mean(ious):.3f} ± {np.std(ious):.3f}  acc {np.mean(accs):.3f}", flush=True)
    n_train_frames = sum(len(x["dino"]) for x in recs if not x["holdout"])
    n_test_frames = sum(len(x["dino"]) for x in recs if x["holdout"])

    # ---------------------------------------------------------------- 4. figure
    from sklearn.decomposition import PCA  # type: ignore[import-not-found]

    tr_tokens = np.concatenate([x["dino"].reshape(-1, 384) for x in recs if not x["holdout"]]).astype(np.float32)
    rng = np.random.default_rng(0)
    pca = PCA(3).fit(tr_tokens[rng.choice(len(tr_tokens), min(50000, len(tr_tokens)), replace=False)])
    test = [x for x in recs if x["holdout"]]
    flat_idx = []
    for ri, x in enumerate(test):
        for fi in range(len(x["dino"])):
            flat_idx.append((ri, fi))
    picks = [flat_idx[i] for i in np.linspace(0, len(flat_idx) - 1, 6).astype(int)]
    offsets = np.cumsum([0] + [len(x["dino"]) for x in test])
    G = dspec.grid[0] * dspec.grid[1]
    lo = hi = None
    tiles = []
    for ri, fi in picks:
        x = test[ri]
        img = x["imgs"][fi]
        tok = x["dino"][fi].reshape(-1, 384).astype(np.float32)
        z = pca.transform(tok)
        if lo is None:
            allz = pca.transform(tr_tokens[: 20000])
            lo, hi = np.percentile(allz, 2, axis=0), np.percentile(allz, 98, axis=0)
        z = np.clip((z - lo) / (hi - lo), 0, 1).reshape(dspec.grid[0], dspec.grid[1], 3)
        pca_img = cv2.resize((z[:, :, ::-1] * 255).astype(np.uint8), (dspec.width, dspec.height), interpolation=cv2.INTER_NEAREST)
        gt = cv2.resize(x["m_d"][fi], (dspec.width, dspec.height), interpolation=cv2.INTER_NEAREST)
        k = offsets[ri] + fi
        pr = cv2.resize(keep_probs[k * G:(k + 1) * G].reshape(dspec.grid), (dspec.width, dspec.height),
                        interpolation=cv2.INTER_NEAREST)

        def over(m: np.ndarray, col: tuple[int, int, int]) -> np.ndarray:
            o = img.copy()
            sel = m > 0.5
            o[sel] = (0.5 * o[sel] + 0.5 * np.array(col)).astype(np.uint8)
            return o

        row = [img, over(gt, (0, 200, 0)), over(pr, (200, 120, 0)), pca_img]
        for t_, txt in zip(row, ["image", "TwinLiteNet+ drivable (target)", "linear probe on DINOv2", "DINOv2 PCA"]):
            cv2.putText(t_, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.hstack(row))
    cv2.imwrite(str(a.out / "probe_examples.jpg"), np.vstack(tiles))

    summary = {"latency_ms": lat, "storage": storage, "probes": results,
               "frames": {"train": n_train_frames, "test_holdout": n_test_frames, "holdout_key": a.holdout},
               "recordings": [x["path"] for x in recs]}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", a.out / "summary.json")


def _autocast(m: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.float16):
        return m(x)


if __name__ == "__main__":
    main()
