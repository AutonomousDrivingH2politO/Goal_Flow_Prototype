"""Train BEV GoalFlow (and the regression baseline) on the simulator shards.

    python -m gfsem.train --train "data/sim/val_000[01].npz" --val "data/sim/val_0002.npz" \
        --stages A,B,reg --steps 300 --out runs/smoke

Stages: A = encoder + goal scorer (L_goal [+ aux]); B = generator with A frozen (L1 flow);
joint = everything; reg = regression baseline with its own encoder (L1).
AdamW, lr 3e-4 cosine, wd 0.01. Device: cuda > mps > cpu (fp32 on MPS, AMP on CUDA).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .data.sim_torch import SimFrames, load_shards  # noqa: E402
from .models.common import Normalizer, pick_device  # noqa: E402
from .models.goal_scorer import goal_loss, kmeans_vocabulary  # noqa: E402
from .models.goalflow import GoalFlow, ModelConfig  # noqa: E402


def cosine(step: int, total: int, warmup: int = 50) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


class Logger:
    def __init__(self, out: Path):
        self.f = open(out / "log.csv", "a", newline="")
        self.w = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(out / "tb"))
        except Exception:
            self.tb = None

    def log(self, stage: str, step: int, **kv):
        row = {"stage": stage, "step": step, "time": round(time.time(), 1),
               **{k: float(v) for k, v in kv.items()}}
        if self.w is None or set(row) - set(self.w.fieldnames):
            self.w = csv.DictWriter(self.f, sorted(set(row) | set(self.w.fieldnames if self.w else [])))
            self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()
        if self.tb:
            for k, v in kv.items():
                self.tb.add_scalar(f"{stage}/{k}", float(v), step)


def to(batch: dict, dev) -> dict:
    return {k: v.to(dev, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def infinite(loader):
    while True:
        for b in loader:
            yield b


def stage_params(model: GoalFlow, stage: str):
    perc = list(model.bev.parameters()) + list(model.ego.parameters())
    if stage == "A":
        return perc + list(model.scorer.parameters()) + (list(model.aux.parameters()) if model.aux else [])
    if stage == "B":
        return list(model.gen.parameters())
    if stage == "reg":
        return perc + list(model.reg.parameters())
    return [p for n, p in model.named_parameters() if not n.startswith("reg.")]


def losses(model: GoalFlow, b: dict, stage: str, args) -> tuple[torch.Tensor, dict]:
    frozen = stage == "B"
    with torch.set_grad_enabled(not frozen):
        F_bev, F_ego, fmap = model.perceive(b["bev"], b["ego"])
    if frozen:
        F_bev, F_ego = F_bev.detach(), F_ego.detach()
    out = {}
    total = torch.zeros((), device=b["bev"].device)
    if stage in ("A", "joint"):
        dis, dac = model.scorer(F_bev, F_ego)
        lg, parts = goal_loss(dis, dac, model.scorer.vocab, b["goal"], b["dac"], T=args.T,
                              w4=1.0, w5=args.w5)
        total = total + lg
        out.update(l_goal=lg.detach(), **parts)
        if model.aux is not None and "aux" in b:
            la = F.binary_cross_entropy_with_logits(model.aux(fmap), b["aux"])
            total = total + args.w_aux * la
            out["l_aux"] = la.detach()
    if stage in ("B", "joint"):
        lf = model.flow_loss(F_bev, F_ego, b["traj"], b["goal"])
        total = total + lf
        out["l_flow"] = lf.detach()
    if stage == "reg":
        lr_ = model.regression_loss(F_bev, F_ego, b["traj"])
        total = total + lr_
        out["l_reg"] = lr_.detach()
    return total, out


@torch.no_grad()
def evaluate(model: GoalFlow, loader, stage: str, args, dev, max_batches: int = 10) -> dict:
    model.eval()
    acc: dict[str, list] = {}
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = to(b, dev)
        _, parts = losses(model, b, stage, args)
        for k, v in parts.items():
            acc.setdefault(k, []).append(float(v))
        modes, n = b["modes"], b["n_modes"]
        mask = torch.arange(8, device=dev)[None] < n[:, None]
        if stage in ("B", "joint"):
            r = model.plan(b["bev"], b["ego"], M=32, n_steps=1)
            ade = (r["traj"][:, None] - modes).norm(dim=-1).mean(-1)            # (B, 8)
            acc.setdefault("minADE_modes", []).append(float(ade.masked_fill(~mask, 1e9).min(1).values.mean()))
            acc.setdefault("ADE_driven", []).append(float(ade[:, 0].mean()))
            s_ade = (r["samples"][:, :, None] - modes[:, None]).norm(dim=-1).mean(-1)   # (B, M, 8)
            acc.setdefault("minADE_samples", []).append(float(s_ade.min(1).values[:, 0].mean()))
        if stage in ("A", "joint"):
            dis, dac = model.scorer(*model.perceive(b["bev"], b["ego"])[:2])
            g = model.scorer.vocab[dis.argmax(-1)]
            acc.setdefault("goal_err", []).append(float((g[:, :2] - b["goal"][:, :2]).norm(dim=-1).mean()))
            acc.setdefault("dac_acc", []).append(float(((dac > 0).float() == b["dac"]).float().mean()))
        if stage == "reg":
            p = model.norm.inverse(model.reg(*model.perceive(b["bev"], b["ego"])[:2]))
            ade = (p[:, None] - modes).norm(dim=-1).mean(-1)
            acc.setdefault("ADE_driven", []).append(float(ade[:, 0].mean()))
            acc.setdefault("minADE_modes", []).append(float(ade.masked_fill(~mask, 1e9).min(1).values.mean()))
    model.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def train_stage(model: GoalFlow, stage: str, train_dl, val_dl, args, dev, log: Logger) -> list[dict]:
    params = stage_params(model, stage)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, args.steps))
    use_amp = dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    if stage == "B":            # frozen perception: keep BatchNorm statistics fixed
        model.bev.eval()
    hist = []
    it = infinite(train_dl)
    t0 = time.time()
    for step in range(args.steps):
        b = to(next(it), dev)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss, parts = losses(model, b, stage, args)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            row = dict(loss=float(loss), lr=sched.get_last_lr()[0], **{k: float(v) for k, v in parts.items()})
            log.log(stage, step, **row)
            hist.append(dict(step=step, **row))
            print(f"[{stage}] step {step:5d} loss {float(loss):.4f} " +
                  " ".join(f"{k} {float(v):.4f}" for k, v in parts.items()) +
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        if val_dl is not None and (step + 1) % args.eval_every == 0:
            ev = evaluate(model, val_dl, stage, args, dev)
            log.log(f"{stage}_val", step, **ev)
            print(f"[{stage}] val @ {step + 1}: " + " ".join(f"{k} {v:.3f}" for k, v in ev.items()), flush=True)
    return hist


def save(model: GoalFlow, path: Path, args, vocab: np.ndarray):
    torch.save(dict(state=model.state_dict(), cfg=asdict(model.cfg), vocab=vocab,
                    norm_lo=model.norm.lo.cpu().numpy(), norm_hi=model.norm.hi.cpu().numpy(),
                    args=vars(args)), path)


def load(path: str | Path, device="cpu") -> GoalFlow:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GoalFlow(ModelConfig(**ck["cfg"]), ck["vocab"], Normalizer(tuple(ck["norm_lo"]), tuple(ck["norm_hi"])))
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="glob of training shards")
    ap.add_argument("--val", default=None, help="glob of validation shards")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stages", default="A,B,reg")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--vocab-n", type=int, default=1024)
    ap.add_argument("--T", type=float, default=0.5, help="distance-score temperature [m]")
    ap.add_argument("--w5", type=float, default=0.005)
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--w-aux", type=float, default=1.0)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--t-dist", default="uniform")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--plots", type=int, default=6)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    dev = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device {dev}", flush=True)

    tr_shards = load_shards(args.train)
    va_shards = load_shards(args.val) if args.val else []
    base = SimFrames(tr_shards)
    targets = base.all_targets()
    vocab = kmeans_vocabulary(targets[:, -1, :], args.vocab_n, seed=args.seed)
    norm = Normalizer.fit(targets[:, :, :2])
    print(f"train frames {len(base)}, modes {len(targets)}, vocab {len(vocab)}, "
          f"norm lo {norm.lo.tolist()} hi {norm.hi.tolist()}", flush=True)
    np.save(args.out / "vocab.npy", vocab)

    train_ds = SimFrames(tr_shards, vocab=vocab, aux=args.aux)
    val_ds = SimFrames(va_shards, vocab=vocab, aux=args.aux) if va_shards else None
    mk = dict(batch_size=args.bs, num_workers=args.workers, persistent_workers=args.workers > 0,
              multiprocessing_context="spawn" if args.workers > 0 else None)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **mk)
    val_dl = DataLoader(val_ds, shuffle=False, **mk) if val_ds else None

    cfg = ModelConfig(D=args.D, sigma=args.sigma, t_dist=args.t_dist, aux_head=args.aux)
    model = GoalFlow(cfg, vocab, norm).to(dev)
    n_par = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("reg."))
    print(f"GoalFlow parameters: {n_par / 1e6:.2f} M", flush=True)
    log = Logger(args.out)
    summary = dict(params_M=n_par / 1e6, device=str(dev), frames=len(base), vocab=len(vocab), stages={})
    stages = args.stages.split(",")
    for st in stages:
        if st == "reg":
            reg_model = GoalFlow(ModelConfig(D=args.D, regression=True), vocab, norm).to(dev)
            hist = train_stage(reg_model, "reg", train_dl, val_dl, args, dev, log)
            save(reg_model, args.out / "regression.pt", args, vocab)
        else:
            hist = train_stage(model, st, train_dl, val_dl, args, dev, log)
            save(model, args.out / f"goalflow_{st}.pt", args, vocab)
        ev = evaluate(model if st != "reg" else reg_model, val_dl or train_dl, st, args, dev)
        summary["stages"][st] = dict(first=hist[0], last=hist[-1], val=ev)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if args.plots and any(s in stages for s in ("B", "joint")):
        from .viz import plot_samples
        plot_samples(model, val_ds or train_ds, args.out / "samples", n=args.plots, device=dev,
                     reg_model=reg_model if "reg" in stages else None)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
