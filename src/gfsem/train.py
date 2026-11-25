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
from .models.goal_scorer import goal_loss, kmeans_vocabulary, select_goal  # noqa: E402
from .models.goalflow import GoalFlow, ModelConfig  # noqa: E402
from .models.rollout import RolloutConfig, kappa_feasible, rollout  # noqa: E402


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


def goal_collides(g: torch.Tensor, obstacles: torch.Tensor, n_obs: torch.Tensor) -> torch.Tensor:
    """(B,) bool: does a Juno box at g* (x, y, theta; ego frame) overlap a true obstacle disc?"""
    from .sim.config import VehicleConfig
    veh = VehicleConfig.from_yaml()
    x0, x1 = veh.body_x
    h = veh.width / 2
    c, s_ = torch.cos(g[:, 2])[:, None], torch.sin(g[:, 2])[:, None]
    dx = obstacles[..., 0] - g[:, 0:1]
    dy = obstacles[..., 1] - g[:, 1:2]
    lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy
    qx, qy = lx.clamp(x0, x1), ly.clamp(-h, h)
    d = torch.hypot(lx - qx, ly - qy) - obstacles[..., 2]
    valid = torch.arange(obstacles.shape[1], device=g.device)[None] < n_obs[:, None]
    return ((d < 0) & valid).any(1)


def obstacles_near_path(path: torch.Tensor, obstacles: torch.Tensor, n_obs: torch.Tensor, dist: float = 2.0):
    """(B,) bool: any true obstacle within ``dist`` metres (surface) of the driven teacher path."""
    d = torch.cdist(obstacles[..., :2], path) - obstacles[..., 2:3]                    # (B, O, T)
    valid = torch.arange(obstacles.shape[1], device=path.device)[None] < n_obs[:, None]
    return ((d.min(-1).values < dist) & valid).any(1)


def _traj_metrics(acc: dict, executed, raw, modes, mask):
    """ADE / FDE / minADE-over-teacher-modes for the executed path (after the pure-pursuit + bicycle
    rollout) and the raw path, against the driven teacher mode (mode 0)."""
    for tag, tr in (("", executed), ("_raw", raw)):
        d = (tr[:, None] - modes).norm(dim=-1)                       # (B, 8, T)
        ade = d.mean(-1)
        acc.setdefault(f"ADE_driven{tag}", []).append(float(ade[:, 0].mean()))
        acc.setdefault(f"FDE_driven{tag}", []).append(float(d[:, 0, -1].mean()))
        acc.setdefault(f"minADE_modes{tag}", []).append(float(ade.masked_fill(~mask, 1e9).min(1).values.mean()))


def goal_mode_coverage(dis, dac, vocab, modes, n_modes, ks=(5, 10), radius=1.0, mass_radius=1.5,
                       mass_min=0.10) -> dict[str, list]:
    """Per frame with >= 2 teacher modes: do the top-k goals (final score) include a goal within
    ``radius`` of every mode's endpoint? Does the distance score put >= ``mass_min`` of its softmax
    mass within ``mass_radius`` of every mode? Returns lists of 0/1 per multi-mode frame."""
    score = F.log_softmax(dis, -1) + F.logsigmoid(dac)
    prob = torch.softmax(dis, -1)
    out: dict[str, list] = {f"cover@{k}": [] for k in ks}
    out["mass_on_every_mode"] = []
    ends = modes[:, :, -1, :2]                                           # (B, 8, 2)
    dist = torch.cdist(ends, vocab[None, :, :2].expand(ends.shape[0], -1, -1))   # (B, 8, N)
    for i in range(dis.shape[0]):
        m = int(n_modes[i])
        if m < 2:
            continue
        d = dist[i, :m]                                                  # (m, N)
        for k in ks:
            top = score[i].topk(min(k, score.shape[1])).indices
            out[f"cover@{k}"].append(float((d[:, top] <= radius).any(1).all()))
        mass = torch.stack([prob[i][d[j] <= mass_radius].sum() for j in range(m)])
        out["mass_on_every_mode"].append(float((mass >= mass_min).all()))
    return out


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
            r = model.plan(b["bev"], b["ego"], M=SAMPLES_M, n_steps=1)
            _traj_metrics(acc, r["traj"], r["traj_raw"], modes, mask)
            s_ade = (r["samples"][:, :, None] - modes[:, None]).norm(dim=-1).mean(-1)   # (B, M, 8) executed
            acc.setdefault("minADE_samples", []).append(float(s_ade.min(1).values[:, 0].mean()))
            s_raw = (r["samples_raw"][:, :, None] - modes[:, None]).norm(dim=-1).mean(-1)
            acc.setdefault("minADE_samples_raw", []).append(float(s_raw.min(1).values[:, 0].mean()))
            acc.setdefault("kappa_ok_raw", []).append(float(r["kappa_ok_raw"].mean()))
            acc.setdefault("kappa_ok_raw_1.1", []).append(float(kappa_feasible(r["samples_raw"], tol=1.1).float().mean()))
            g2 = r["goal"][:, None, :2]
            acc.setdefault("endpoint_to_gstar", []).append(float((r["samples"][:, :, -1] - g2).norm(dim=-1).mean()))
            acc.setdefault("endpoint_to_gstar_raw", []).append(float((r["samples_raw"][:, :, -1] - g2).norm(dim=-1).mean()))
        if stage in ("A", "joint"):
            dis, dac = model.scorer(*model.perceive(b["bev"], b["ego"])[:2])
            cov = goal_mode_coverage(dis, dac, model.scorer.vocab, modes, n)
            for k_, v_ in cov.items():
                acc.setdefault(k_, []).extend(v_)
            gs, _ = select_goal(dis, dac, model.scorer.vocab)
            if "obstacles" in b:
                col = goal_collides(gs, b["obstacles"], b["n_obstacles"])
                acc.setdefault("gstar_collision", []).extend(col.float().tolist())
                near = obstacles_near_path(modes[:, 0], b["obstacles"], b["n_obstacles"])
                pred, tgt = (dac > 0).float(), b["dac"]
                fr_acc = (pred == tgt).float().mean(-1)
                acc.setdefault("dac_acc_near_obst", []).extend(fr_acc[near].tolist())
                tp = ((pred == 1) & (tgt == 1)).float().sum(-1) / (tgt == 1).float().sum(-1).clamp_min(1)
                tn = ((pred == 0) & (tgt == 0)).float().sum(-1) / (tgt == 0).float().sum(-1).clamp_min(1)
                has_neg = (tgt == 0).any(-1)
                acc.setdefault("dac_balanced_acc", []).extend((0.5 * (tp + tn))[has_neg].tolist())
            g = model.scorer.vocab[dis.argmax(-1)]
            acc.setdefault("goal_err", []).append(float((g[:, :2] - b["goal"][:, :2]).norm(dim=-1).mean()))
            acc.setdefault("dac_acc", []).append(float(((dac > 0).float() == b["dac"]).float().mean()))
        if stage == "reg":
            p_raw = model.norm.inverse(model.reg(*model.perceive(b["bev"], b["ego"])[:2]))
            p_exec = rollout(p_raw, b["ego"][:, 0], b["ego"][:, 1], RolloutConfig())
            _traj_metrics(acc, p_exec, p_raw, modes, mask)
            acc.setdefault("kappa_ok_raw", []).append(float(kappa_feasible(p_raw).float().mean()))
            acc.setdefault("kappa_ok_raw_1.1", []).append(float(kappa_feasible(p_raw, tol=1.1).float().mean()))
    model.train()
    res = {k: float(np.mean(v)) if len(v) else float("nan") for k, v in acc.items()}
    res["n_multimode_frames"] = float(len(acc.get("mass_on_every_mode", [])))
    return res


# Stage A is selected on its validation objective (L_goal). goal_err (argmax goal vs the *driven*
# endpoint) was tried first and dropped: it jumps whenever the scorer prefers the other, equally
# valid teacher mode, so it stopped improving while L_goal, DAC accuracy and the g* collision rate
# all kept improving (2026-11-25 run, steps 2126-8422).
SAMPLES_M = 128      # one sample count everywhere: validation, evaluation scripts, renders
BEST_METRIC = {"A": "l_goal", "B": "ADE_driven", "joint": "ADE_driven", "reg": "ADE_driven"}


def save_state(path: Path, model, opt, sched, step, total, best, bad, args, vocab, metric=None):
    """Full training state (atomic write), for mid-stage resume."""
    tmp = path.with_suffix(".tmp")
    torch.save(dict(state=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step,
                    total=total, best=best, bad=bad, metric=metric, cfg=asdict(model.cfg), vocab=vocab,
                    norm_lo=model.norm.lo.cpu().numpy(), norm_hi=model.norm.hi.cpu().numpy(),
                    args=vars(args), saved=time.time()), tmp)
    os.replace(tmp, path)


def train_stage(model: GoalFlow, stage: str, train_dl, val_dl, args, dev, log: Logger,
                steps: int | None = None, best_path: Path | None = None, vocab=None,
                state_path: Path | None = None) -> list[dict]:
    steps = steps or args.steps
    budget_h = getattr(args, f"hours_{stage}", None)
    total = [steps]                      # mutable: a time budget can shrink it after warmup
    params = stage_params(model, stage)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, total[0]))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    if stage == "B":            # frozen perception: keep BatchNorm statistics fixed
        model.bev.eval()
    hist = []
    it = infinite(train_dl)
    t0 = time.time()
    best = float("inf")
    bad = 0                                  # evaluations without improvement (early stopping)
    t_eval = time.time()
    t_ckpt = time.time()
    step = 0
    t_meas = None
    if state_path is not None and state_path.exists():
        st = torch.load(state_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["state"])
        opt.load_state_dict(st["opt"])
        total[0] = st["total"]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: cosine(s_, total[0]))
        sched.load_state_dict(st["sched"])
        step, best, bad = st["step"], st["best"], st["bad"]
        if st.get("metric", "goal_err" if stage == "A" else BEST_METRIC[stage]) != BEST_METRIC[stage]:
            print(f"[{stage}] selection metric changed to {BEST_METRIC[stage]}: best and patience reset", flush=True)
            best, bad = float("inf"), 0
        budget_h = None                       # the step total was already fitted to the budget
        print(f"[{stage}] resumed at step {step}/{total[0]} (best {BEST_METRIC[stage]} {best:.3f})", flush=True)
        if stage == "B":
            model.bev.eval()
    while step < total[0]:
        b = to(next(it), dev)
        accum = getattr(args, f"accum_{stage}", None) or 1
        opt.zero_grad(set_to_none=True)
        if accum > 1:   # gradient accumulation: same effective batch, 1/accum of the activation memory
            n = b["bev"].shape[0]
            loss_sum, parts_sum = 0.0, {}
            for c in range(accum):
                sl = slice(c * n // accum, (c + 1) * n // accum)
                mb = {k: (v[sl] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n else v) for k, v in b.items()}
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    l_c, p_c = losses(model, mb, stage, args)
                scaler.scale(l_c / accum).backward()
                loss_sum += float(l_c.detach()) / accum
                for k, v in p_c.items():
                    parts_sum[k] = parts_sum.get(k, 0.0) + float(v) / accum
            loss = torch.tensor(loss_sum)
            parts = parts_sum
        else:
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss, parts = losses(model, b, stage, args)
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if budget_h and step == 10:
            t_meas = time.time()
        if budget_h and step == 60 and t_meas is not None:
            sps = (time.time() - t_meas) / 50
            fit = int(budget_h * 3600 / max(sps, 1e-3))
            total[0] = max(100, min(steps, fit))
            print(f"[{stage}] {sps:.2f} s/step -> {total[0]} steps fit the {budget_h:.1f} h budget", flush=True)
        if step % args.log_every == 0 or step == total[0] - 1:
            row = dict(loss=float(loss.detach()), lr=sched.get_last_lr()[0], **{k: float(v) for k, v in parts.items()})
            log.log(stage, step, **row)
            hist.append(dict(step=step, **row))
            el = time.time() - t0
            print(f"[{stage}] step {step:6d}/{total[0]} loss {float(loss.detach()):.4f} " +
                  " ".join(f"{k} {float(v):.4f}" for k, v in parts.items()) +
                  f"  ({el:.0f}s, {el / (step + 1):.2f}s/step)", flush=True)
        due = val_dl is not None and (step + 1) % args.eval_every == 0 or (args.eval_minutes and time.time() - t_eval > 60 * args.eval_minutes)
        if val_dl is not None and (due or step == total[0] - 1):
            due = True
            t_eval = time.time()
            ev = evaluate(model, val_dl, stage, args, dev, max_batches=args.val_batches)
            log.log(f"{stage}_val", step, **ev)
            print(f"[{stage}] val @ {step + 1}: " + " ".join(f"{k} {v:.3f}" for k, v in ev.items()), flush=True)
            m = ev.get(BEST_METRIC[stage], float("inf"))
            if m < best:
                best, bad = m, 0
                if best_path is not None:
                    save(model, best_path, args, vocab)
                    print(f"[{stage}] new best {BEST_METRIC[stage]} {m:.3f} -> {best_path.name}", flush=True)
            else:
                bad += 1
            if stage == "B":
                model.bev.eval()
            if args.patience and bad >= args.patience:
                print(f"[{stage}] early stop: no {BEST_METRIC[stage]} improvement in {bad} evaluations", flush=True)
                step += 1
                break
        step += 1
        if state_path is not None and (time.time() - t_ckpt > 60 * args.ckpt_minutes or due):
            save_state(state_path, model, opt, sched, step, total[0], best, bad, args, vocab, BEST_METRIC[stage])
            t_ckpt = time.time()
    if state_path is not None:
        save_state(state_path, model, opt, sched, step, total[0], best, bad, args, vocab, BEST_METRIC[stage])
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
    ap.add_argument("--train", required=True, help="glob of training shards, or mm:<dir> (memmap split)")
    ap.add_argument("--val", default=None, help="glob of validation shards, or mm:<dir>")
    ap.add_argument("--scene-index", default=None, help="comma-separated scenes_<split>.npz files")
    ap.add_argument("--steps-A", type=int, default=None)
    ap.add_argument("--steps-B", type=int, default=None)
    ap.add_argument("--steps-reg", type=int, default=None)
    ap.add_argument("--steps-joint", type=int, default=None)
    ap.add_argument("--val-frames", type=int, default=4000, help="fixed random validation subset")
    ap.add_argument("--val-batches", type=int, default=1000)
    ap.add_argument("--resume", action="store_true", help="skip stages whose checkpoint exists")
    ap.add_argument("--hours-A", type=float, default=None, help="wall-clock budget for stage A")
    ap.add_argument("--hours-B", type=float, default=None)
    ap.add_argument("--hours-reg", type=float, default=None)
    ap.add_argument("--hours-joint", type=float, default=None)
    ap.add_argument("--eval-minutes", type=float, default=None, help="also evaluate every N minutes")
    ap.add_argument("--ckpt-minutes", type=float, default=30.0, help="full training-state checkpoint interval")
    ap.add_argument("--patience", type=int, default=0, help="early stop after N evaluations without improvement")
    ap.add_argument("--accum-A", type=int, default=1, help="gradient accumulation micro-batches in stage A")
    ap.add_argument("--accum-B", type=int, default=1)
    ap.add_argument("--accum-reg", type=int, default=1)
    ap.add_argument("--accum-joint", type=int, default=1)
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

    def source(spec):
        if spec is None:
            return []
        if spec.startswith("mm:"):
            from .data.sim_torch import MemmapSplit
            return [MemmapSplit(spec[3:])]
        return load_shards(spec)

    tr_shards, va_shards = source(args.train), source(args.val)
    si = None
    if args.scene_index:
        from .data.scene_index import SceneIndex
        si = SceneIndex(args.scene_index.split(","))
    base = SimFrames(tr_shards)
    if (args.out / "vocab.npy").exists() and args.resume:
        vocab = np.load(args.out / "vocab.npy")
        nl = np.load(args.out / "norm.npy")
        norm = Normalizer(tuple(nl[0]), tuple(nl[1]))
    else:
        targets = base.all_targets()
        vocab = kmeans_vocabulary(targets[:, -1, :], args.vocab_n, seed=args.seed)
        norm = Normalizer.fit(targets[:, :, :2])
        np.save(args.out / "vocab.npy", vocab)
        np.save(args.out / "norm.npy", np.stack([norm.lo.numpy(), norm.hi.numpy()]))
        print(f"modes {len(targets)}", flush=True)
    print(f"train frames {len(base)}, vocab {len(vocab)}, norm lo {norm.lo.tolist()} hi {norm.hi.tolist()}", flush=True)

    train_ds = SimFrames(tr_shards, vocab=vocab, aux=args.aux, scene_index=si)
    val_ds = SimFrames(va_shards, vocab=vocab, aux=args.aux, scene_index=si) if va_shards else None
    mk = dict(batch_size=args.bs, num_workers=args.workers, persistent_workers=args.workers > 0,
              multiprocessing_context="spawn" if args.workers > 0 else None)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **mk)
    val_dl = None
    if val_ds is not None:
        from torch.utils.data import Subset
        sub = np.random.default_rng(1234).choice(len(val_ds), min(args.val_frames, len(val_ds)), replace=False)
        val_dl = DataLoader(Subset(val_ds, sorted(sub.tolist())), shuffle=False, **mk)

    cfg = ModelConfig(D=args.D, sigma=args.sigma, t_dist=args.t_dist, aux_head=args.aux)
    model = GoalFlow(cfg, vocab, norm).to(dev)
    reg_model = None
    n_par = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("reg."))
    print(f"GoalFlow parameters: {n_par / 1e6:.2f} M", flush=True)
    log = Logger(args.out)
    summary = dict(params_M=n_par / 1e6, device=str(dev), frames=len(base), vocab=len(vocab), stages={})
    stages = args.stages.split(",")
    t_all = time.time()
    for st in stages:
        t_stage = time.time()
        steps = getattr(args, f"steps_{st}", None) or args.steps
        ck = args.out / ("regression.pt" if st == "reg" else f"goalflow_{st}.pt")
        best = args.out / ("regression_best.pt" if st == "reg" else f"goalflow_{st}_best.pt")
        if st == "reg":
            if args.resume and ck.exists():
                reg_model = load(best if best.exists() else ck, dev)
                print("resume: regression loaded", flush=True)
                continue
            reg_model = GoalFlow(ModelConfig(D=args.D, regression=True), vocab, norm).to(dev)
            hist = train_stage(reg_model, "reg", train_dl, val_dl, args, dev, log, steps, best, vocab,
                               args.out / "state_reg.pt")
            save(reg_model, ck, args, vocab)
            reg_model = load(best if best.exists() else ck, dev)
        else:
            if args.resume and ck.exists():
                model.load_state_dict(torch.load(best if best.exists() else ck, map_location=dev,
                                                 weights_only=False)["state"])
                print(f"resume: stage {st} loaded", flush=True)
                continue
            hist = train_stage(model, st, train_dl, val_dl, args, dev, log, steps, best, vocab,
                               args.out / f"state_{st}.pt")
            save(model, ck, args, vocab)
            if best.exists():          # continue from the best checkpoint of this stage
                model.load_state_dict(torch.load(best, map_location=dev, weights_only=False)["state"])
        summary["stages"][st] = dict(first=hist[0] if hist else None, last=hist[-1] if hist else None,
                                     seconds=round(time.time() - t_stage))
    # final report, in the agreed order: goal multimodality, curvature feasibility, ADE
    if val_dl is not None:
        report = {}
        if any(s in stages for s in ("A", "joint")):
            report["goal"] = evaluate(model, val_dl, "A", args, dev, max_batches=args.val_batches)
        if any(s in stages for s in ("B", "joint")):
            report["trajectory"] = evaluate(model, val_dl, "B", args, dev, max_batches=args.val_batches)
        if reg_model is not None:
            report["regression"] = evaluate(reg_model, val_dl, "reg", args, dev, max_batches=args.val_batches)
        summary["final_val"] = report
    summary["train_seconds_this_run"] = round(time.time() - t_all)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    if args.plots and any(s in stages for s in ("B", "joint")):
        from .viz import plot_samples
        plot_samples(model, val_ds or train_ds, args.out / "samples", n=args.plots, device=dev, reg_model=reg_model)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
