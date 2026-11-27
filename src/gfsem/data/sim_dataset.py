"""Phase L1: imitation dataset from the privileged lattice-corridor teacher.

Per scene (deterministic in the seed):
* one noise level drawn from the training mix; the teacher drives the scene closed loop
  (privileged, plan once) while the LiDAR model produces observations at that level;
* frames at 5 Hz, each with the compact Observation (no raster), the true pose and the targets:
  mode 0 is the driven 4 s future (16 poses at 0.25 s, ego frame at t); further modes come from
  the lattice inside the top-K distinct A* routes (re-solved every ``alt_every`` frames and
  re-tracked from the current state in between), deduplicated by homotopy signature;
* about ``p_recovery`` of frames are recovery states: the pose is perturbed (lateral +-1 m,
  heading +-15 deg), observed from there, and the target is a fresh teacher plan.

Shards are .npz with ragged arrays flattened plus offsets.
"""
from __future__ import annotations

import math
import os
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..planners.corridor import input_from_privileged
from ..planners.corridor_lattice import LatticeCorridorConfig, LatticeCorridorPlanner, side_signature_local
from ..sim.closed_loop import CorridorPrivileged, run_episode
from ..sim.config import NOISE_LEVELS, SimConfig
from ..sim.geometry import footprint_corners, poses_from_frame, poses_to_frame, to_frame
from ..sim.lattice import LatticeConfig
from ..sim.lidar import Observation, privileged_view, scan
from ..sim.scene import Scene, generate_scene
from ..sim.vehicle import VehicleState, pure_pursuit, step

SPLITS = {"train": (100_000, 15_000), "val": (200_000, 1_500), "test": (300_000, 1_500)}


def data_root() -> Path:
    return Path(os.environ.get("GFSEM_DATA", Path(__file__).resolve().parents[3] / "data")) / "sim"


@dataclass(frozen=True)
class DatasetConfig:
    hz: float = 5.0
    horizon_steps: int = 16
    horizon_dt: float = 0.25
    K: int = 8
    alt_every: int = 10                # frames between top-K route re-solves (2 s)
    alt_beam: int = 300
    alt_dev_max: float = 0.5           # an alternative must still pass this close to the current pose
    p_recovery: float = 0.3
    rec_lateral: float = 1.0
    rec_heading_deg: float = 15.0
    noise_mix: tuple = (("L0", 0.15), ("L1", 0.25), ("L2", 0.25), ("L3", 0.2), ("L4", 0.15))
    scenes_per_shard: int = 50


def track_future(path_w: np.ndarray, st: VehicleState, cfg: SimConfig, dcfg: DatasetConfig,
                 v_target: float) -> np.ndarray:
    """Simulate pure pursuit on ``path_w`` (world) from ``st``; return 16 future poses (world)."""
    s = st.copy()
    dt = cfg.control.dt
    per = int(round(dcfg.horizon_dt / dt))
    out = []
    idx = 0
    for k in range(dcfg.horizon_steps * per):
        delta, idx = pure_pursuit(s.x, s.y, s.yaw, path_w, cfg.control.lookahead, cfg.vehicle.wheelbase, idx)
        s = step(s, v_target, delta, dt, cfg.vehicle)
        if (k + 1) % per == 0:
            out.append((s.x, s.y, s.yaw))
    return np.array(out)


def _sig_hash(sig: tuple) -> int:
    return zlib.crc32(repr(sig).encode()) & 0x7FFFFFFF


def _obs_arrays(o: Observation) -> dict:
    return dict(points=o.points.astype(np.float32), labels=o.labels.astype(np.uint8),
                scan=o.scan_ranges.astype(np.float16), edge=o.edge_points.astype(np.float32),
                edge_side=o.edge_side.astype(np.int8), goal=o.goal_line.astype(np.float32),
                ego=o.ego_state.astype(np.float32), odom=o.pose_odom.astype(np.float32))


def generate_scene_frames(seed: int, cfg: SimConfig | None = None, dcfg: DatasetConfig | None = None) -> dict:
    cfg = cfg or SimConfig()
    dcfg = dcfg or DatasetConfig()
    rng = np.random.default_rng([seed, 11])
    names, w = zip(*dcfg.noise_mix)
    level = str(rng.choice(names, p=np.array(w) / sum(w)))
    scene = generate_scene(seed, cfg.scene, cfg.vehicle)
    teacher = LatticeCorridorPlanner(cfg.vehicle, cfg.scene, mode="privileged")
    ep = run_episode(scene, CorridorPrivileged(teacher), level, cfg, record=True)
    traj = ep.trajectory  # (T,5) states after each dt step
    if ep.outcome == "stopped":   # the car stays where it stopped: pad so braking frames get a future
        last = traj[-1].copy()
        last[3] = 0.0
        traj = np.concatenate([traj, np.repeat(last[None], 100, 0)])
    dt = cfg.control.dt
    per_h = int(round(dcfg.horizon_dt / dt))
    per_f = int(round(1.0 / (dcfg.hz * dt)))
    alt_cfg = LatticeCorridorConfig(lattice=LatticeConfig(beam=dcfg.alt_beam))
    frames = []
    alts_w: list[np.ndarray] = []
    rec_rng = np.random.default_rng([seed, 12])
    obs_rng = np.random.default_rng([seed, 13])
    n_modes_scene = set()
    for fi, rec in enumerate(ep.records[:: per_f // cfg.control.replan_every]):
        t = rec["t"]
        k = int(round(t / dt))
        if k + dcfg.horizon_steps * per_h - 1 >= len(traj):
            break                                   # no full 4 s future
        pose = rec["pose"]
        prev = traj[k - 1] if k > 0 else np.array([*pose, cfg.scene.spawn_speed, 0.0])
        st = VehicleState(pose[0], pose[1], pose[2], float(prev[3]), float(prev[4]))
        view = privileged_view(scene, pose)
        recovery = bool(rec_rng.random() < dcfg.p_recovery)
        obs = rec["obs"]
        modes_w = []
        if recovery:
            pp = _perturb(scene, pose, rec_rng, dcfg, cfg)
            if pp is None:
                recovery = False
            else:
                pl = LatticeCorridorPlanner(cfg.vehicle, cfg.scene, lcfg=alt_cfg, mode="privileged")
                st_r = VehicleState(pp[0], pp[1], pp[2], st.v, 0.0)
                inp = input_from_privileged(privileged_view(scene, pp))
                inp.pose_odom = pp
                res = pl.plan(inp, st.v, 0.0)
                if res.brake or len(res.path) < 2:
                    recovery = False
                else:
                    pose, st = pp, st_r
                    view = privileged_view(scene, pose)
                    obs = scan(scene, pose, NOISE_LEVELS[level], obs_rng, cfg.lidar, speed=st.v)
                    modes_w.append(track_future(poses_from_frame(res.path, pose)[:, :2], st, cfg, dcfg,
                                                cfg.scene.target_speed))
        if not recovery:
            modes_w.append(traj[k + per_h - 1:k + dcfg.horizon_steps * per_h:per_h, :3].copy())
            # alternatives from the top-K distinct routes
            if fi % dcfg.alt_every == 0:
                pl = LatticeCorridorPlanner(cfg.vehicle, cfg.scene, lcfg=alt_cfg, mode="privileged")
                inp = input_from_privileged(view)
                alts_w = [poses_from_frame(p, pose) for p in pl.teacher_paths(inp, dcfg.K, st, dcfg.K)]
            for a in alts_w:
                d = np.hypot(*(a[:, :2] - pose[:2]).T)
                i0 = int(np.argmin(d))
                if d[i0] > dcfg.alt_dev_max or len(a) - i0 < 4:
                    continue
                modes_w.append(track_future(a[i0:, :2], st, cfg, dcfg, cfg.scene.target_speed))
        # ego-frame targets, dedupe by homotopy signature (true obstacles near the 4 s future)
        cen = view["centres"]
        targets, sigs = [], []
        for m in modes_w:
            m_e = poses_to_frame(m, pose)
            full = np.concatenate([[[0.0, 0.0, 0.0]], m_e])
            sg = _sig_hash(side_signature_local(full, cen))
            if sg in sigs:
                continue
            targets.append(m_e)
            sigs.append(sg)
        n_modes_scene.update(sigs)
        frames.append(dict(t=t, pose=pose.astype(np.float32), recovery=recovery,
                           targets=np.array(targets, np.float32), sigs=np.array(sigs, np.int64),
                           obs=_obs_arrays(obs)))
    return dict(seed=seed, level=level, feasible=scene.feasible, outcome=ep.outcome,
                frames=frames, n_homotopy=len(n_modes_scene),
                kappa_levels=list(teacher.kappa_log))


def _perturb(scene: Scene, pose, rng, dcfg: DatasetConfig, cfg: SimConfig):
    for _ in range(10):
        dl = rng.uniform(-dcfg.rec_lateral, dcfg.rec_lateral)
        dh = math.radians(rng.uniform(-dcfg.rec_heading_deg, dcfg.rec_heading_deg))
        p = np.array([pose[0] - math.sin(pose[2]) * dl, pose[1] + math.cos(pose[2]) * dl, pose[2] + dh])
        from ..sim.geometry import box_circle_clearance
        if scene.n and box_circle_clearance(p[None], scene.centres, scene.radii + 0.25,
                                            cfg.vehicle.body_x, cfg.vehicle.width).min() < 0:
            continue
        _, dc = scene.track.frenet(footprint_corners(p[None], cfg.vehicle.body_x, cfg.vehicle.width)[0])
        if np.abs(dc).max() > scene.track.width / 2 - 0.1:
            continue
        return p
    return None


def pack_shard(scenes: list[dict]) -> dict:
    """Flatten a list of scene dicts into arrays with offsets."""
    cols = {k: [] for k in ("points", "labels", "scan", "edge", "edge_side", "goal", "ego", "odom")}
    off = {"points": [0], "edge": [0], "targets": [0]}
    meta = {k: [] for k in ("seed", "t", "pose", "recovery", "level")}
    T, S = [], []
    for sc in scenes:
        for f in sc["frames"]:
            o = f["obs"]
            for k in cols:
                cols[k].append(o[k])
            off["points"].append(off["points"][-1] + len(o["points"]))
            off["edge"].append(off["edge"][-1] + len(o["edge"]))
            T.append(f["targets"])
            S.append(f["sigs"])
            off["targets"].append(off["targets"][-1] + len(f["targets"]))
            meta["seed"].append(sc["seed"])
            meta["t"].append(f["t"])
            meta["pose"].append(f["pose"])
            meta["recovery"].append(f["recovery"])
            meta["level"].append(int(sc["level"][1]))
    if not T:
        return {}
    out = dict(
        points=np.concatenate(cols["points"]), labels=np.concatenate(cols["labels"]),
        points_off=np.array(off["points"], np.int64),
        edge=np.concatenate(cols["edge"]), edge_side=np.concatenate(cols["edge_side"]),
        edge_off=np.array(off["edge"], np.int64),
        scan=np.stack(cols["scan"]), goal=np.stack(cols["goal"]), ego=np.stack(cols["ego"]),
        odom=np.stack(cols["odom"]),
        targets=np.concatenate(T), sigs=np.concatenate(S), targets_off=np.array(off["targets"], np.int64),
        seed=np.array(meta["seed"], np.int64), t=np.array(meta["t"], np.float32),
        pose=np.stack(meta["pose"]), recovery=np.array(meta["recovery"], bool),
        level=np.array(meta["level"], np.int8),
        scene_seed=np.array([s["seed"] for s in scenes], np.int64),
        scene_n_homotopy=np.array([s["n_homotopy"] for s in scenes], np.int16),
        scene_feasible=np.array([s["feasible"] for s in scenes], bool),
        scene_outcome=np.array([s["outcome"] for s in scenes]),
    )
    return out


def shard_seeds(split: str, shard: int, dcfg: DatasetConfig) -> list[int]:
    base, n = SPLITS[split]
    a = shard * dcfg.scenes_per_shard
    return [base + i for i in range(a, min(a + dcfg.scenes_per_shard, n))]


def n_shards(split: str, dcfg: DatasetConfig) -> int:
    return math.ceil(SPLITS[split][1] / dcfg.scenes_per_shard)


def build_shard(split: str, shard: int, out_dir: Path, dcfg: DatasetConfig | None = None) -> dict:
    dcfg = dcfg or DatasetConfig()
    t0 = time.time()
    scenes = [generate_scene_frames(s, dcfg=dcfg) for s in shard_seeds(split, shard, dcfg)]
    arr = pack_shard(scenes)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{split}_{shard:04d}.npz"
    tmp = out_dir / f".{split}_{shard:04d}.tmp.npz"
    np.savez_compressed(tmp, **arr)
    tmp.rename(f)
    return dict(split=split, shard=shard, file=str(f), frames=int(len(arr.get("seed", []))),
                bytes=f.stat().st_size, seconds=time.time() - t0,
                homotopy=[int(x) for x in arr.get("scene_n_homotopy", [])],
                seeds=[s["seed"] for s in scenes])


def observation_from_shard(arr: dict, i: int, lidar=None) -> Observation:
    """Rebuild frame ``i``'s Observation from a loaded shard (raw data; rasterise at load time).
    Sensor geometry (mount, ray angles, blind zone) comes from ``LidarConfig``."""
    from ..sim.config import LidarConfig
    lc = lidar or LidarConfig()
    p0, p1 = arr["points_off"][i], arr["points_off"][i + 1]
    e0, e1 = arr["edge_off"][i], arr["edge_off"][i + 1]
    import math
    n = len(arr["scan"][i])                  # ray count from the stored scan (0.2 deg = 601, EMX 0.08 deg = 1501)
    h = math.radians(lc.hfov_deg) / 2
    ang = np.array([-h, -h + 2 * h / (n - 1)])
    return Observation(
        points=arr["points"][p0:p1], labels=arr["labels"][p0:p1],
        scan_ranges=arr["scan"][i].astype(np.float32),
        edge_points=arr["edge"][e0:e1], edge_side=arr["edge_side"][e0:e1],
        goal_line=arr["goal"][i], ego_state=arr["ego"][i],
        sensor_xy=(lc.mount_x, lc.mount_y), ray_angle0=float(ang[0]), ray_step=float(ang[1] - ang[0]),
        blind_zone=lc.blind_zone, pose_odom=arr["odom"][i])


def targets_from_shard(arr: dict, i: int) -> tuple[np.ndarray, np.ndarray]:
    t0, t1 = arr["targets_off"][i], arr["targets_off"][i + 1]
    return arr["targets"][t0:t1], arr["sigs"][t0:t1]
