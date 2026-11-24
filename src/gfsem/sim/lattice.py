"""Shared collision world and forward-arc lattice search (ego or any planar frame).

Used by the lattice corridor planner (teacher and observed baseline). Juno's box is tested
exactly against inflated obstacle discs; track edges are polylines with inward normals and a
box corner beyond an edge counts as a collision (track exit).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .config import VehicleConfig
from .geometry import footprint_corners
from .vehicle import resample_polyline


@dataclass(frozen=True)
class LatticeConfig:
    step: float = 1.0                 # arc length per primitive [m]
    n_kappa: int = 7
    kappa_frac: float = 0.8           # |kappa| <= kappa_frac * kappa_max
    inflation: float = 0.25           # obstacle inflation = door clearance (one margin everywhere)
    n_samples: int = 4                # collision samples per primitive (every 0.25 m)
    xy_res: float = 0.25
    psi_res_deg: float = 3.0
    beam: int = 800
    max_heading_dev_deg: float = 120.0   # vs the goal-line normal (curved tracks need slack)
    max_dk_idx: int = 2               # curvature continuity: |kappa index change| per primitive
    w_clear: float = 1.0              # cost per metre driven with clearance below clear_pref
    clear_pref: float = 1.0
    w_kappa: float = 0.5              # cost per curvature change (in units of kappa_max) per step
    layer_factor: float = 1.6         # max layers = factor * distance-to-goal / step


def arc(poses: np.ndarray, k: np.ndarray, l: float) -> np.ndarray:
    """Advance (N,3) poses along arcs of curvature k (N,) for length l."""
    th = poses[:, 2]
    small = np.abs(k) < 1e-9
    ks = np.where(small, 1.0, k)
    dx = np.where(small, l * np.cos(th), (np.sin(th + k * l) - np.sin(th)) / ks)
    dy = np.where(small, l * np.sin(th), (np.cos(th) - np.cos(th + k * l)) / ks)
    return np.stack([poses[:, 0] + dx, poses[:, 1] + dy, th + k * l], -1)


class CollisionWorld:
    def __init__(self, centres: np.ndarray, radii: np.ndarray, left: np.ndarray, right: np.ndarray,
                 veh: VehicleConfig, edge_ds: float = 0.25):
        self.veh = veh
        self.x_rng = veh.body_x
        self.xc = (self.x_rng[0] + self.x_rng[1]) / 2
        self.half_diag = math.hypot((self.x_rng[1] - self.x_rng[0]) / 2, veh.width / 2)
        self.C = np.asarray(centres, float).reshape(-1, 2)
        self.R = np.asarray(radii, float).reshape(-1)
        self.ctree = cKDTree(self.C) if len(self.C) else None
        self.edges = []
        for e, side in ((left, 1), (right, -1)):
            e = resample_polyline(np.asarray(e, float), edge_ds)
            if len(e) < 2:
                continue
            t = np.gradient(e, axis=0)
            t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
            # inward normal: right of travel for the left edge, left of travel for the right edge
            n = np.stack([t[:, 1], -t[:, 0]], -1) if side == 1 else np.stack([-t[:, 1], t[:, 0]], -1)
            self.edges.append((cKDTree(e), e, n))

    def clearance(self, poses: np.ndarray, inflation: float) -> np.ndarray:
        """(N,) clearance to the nearest inflated obstacle (inf if none near); -1 on track exit."""
        poses = np.atleast_2d(poses)
        N = len(poses)
        out = np.full(N, np.inf)
        if self.ctree is not None:
            c, s = np.cos(poses[:, 2]), np.sin(poses[:, 2])
            cen = poses[:, :2] + self.xc * np.stack([c, s], -1)
            k = min(6, len(self.C))
            ub = self.half_diag + self.R.max() + inflation + 2.0
            d, j = self.ctree.query(cen, k=k, distance_upper_bound=ub)
            d, j = d.reshape(N, k), j.reshape(N, k)
            valid = np.isfinite(d)
            jj = np.where(valid, j, 0)
            cx, cy = self.C[jj, 0], self.C[jj, 1]
            dx, dy = cx - poses[:, 0:1], cy - poses[:, 1:2]
            lx = c[:, None] * dx + s[:, None] * dy
            ly = -s[:, None] * dx + c[:, None] * dy
            h = self.veh.width / 2
            qx = np.clip(lx, self.x_rng[0], self.x_rng[1])
            qy = np.clip(ly, -h, h)
            dist = np.hypot(lx - qx, ly - qy)
            inside = (lx > self.x_rng[0]) & (lx < self.x_rng[1]) & (np.abs(ly) < h)
            dist = np.where(inside, -1.0, dist) - self.R[jj] - inflation
            dist = np.where(valid, dist, np.inf)
            out = dist.min(1)
        if self.edges:
            cs = footprint_corners(poses, self.x_rng, self.veh.width).reshape(-1, 2)
            exit_ = np.zeros(len(cs), bool)
            for tree, e, n in self.edges:
                _, i = tree.query(cs)
                exit_ |= ((cs - e[i]) * n[i]).sum(1) < 0
            out = np.where(exit_.reshape(N, 4).any(1), -1.0, out)
        return out

    def collides(self, poses: np.ndarray, inflation: float) -> np.ndarray:
        return self.clearance(poses, inflation) < 0


def lattice_search(world: CollisionWorld, start, goal: tuple[np.ndarray, np.ndarray], cfg: LatticeConfig,
                   allowed_fn=None, start_kappa: float = 0.0) -> np.ndarray | None:
    """Forward arcs from ``start`` until crossing the goal segment (gL, gR).

    Returns dense poses (K,3) every step / n_samples metres, or None. ``allowed_fn`` maps body
    centres (N,2) to a bool mask (corridor restriction).
    """
    veh = world.veh
    kmax = veh.kappa_max * cfg.kappa_frac
    ks = np.linspace(-kmax, kmax, cfg.n_kappa)
    gL, gR = np.asarray(goal[0], float), np.asarray(goal[1], float)
    gvec = gL - gR
    glen = float(np.linalg.norm(gvec))
    gdir = gvec / max(glen, 1e-9)
    nrm = np.array([gdir[1], -gdir[0]])
    start = np.asarray(start, float)
    if np.dot(nrm, (gL + gR) / 2 - start[:2]) < 0:
        nrm = -nrm
    n_heading = math.atan2(nrm[1], nrm[0])
    dist0 = float(np.dot(gR - start[:2], nrm))
    n_layers = int(max(dist0, 1.0) * cfg.layer_factor / cfg.step) + 4
    psi_res = math.radians(cfg.psi_res_deg)
    ns = cfg.n_samples
    fr = np.arange(1, ns + 1) / ns

    front = start[None, :]
    front_k = np.array([start_kappa])
    front_g = np.zeros(1)
    layers = [(front, None, None)]          # (poses, parent idx, kappa)
    visited: set[int] = set()
    for _ in range(n_layers):
        F = len(front)
        if F == 0:
            return None
        par = np.repeat(np.arange(F), len(ks))
        ki = np.tile(np.arange(len(ks)), F)
        kk = ks[ki]
        if cfg.max_dk_idx < len(ks):
            fk = np.argmin(np.abs(front_k[:, None] - ks[None, :]), 1)
            m = np.abs(ki - fk[par]) <= cfg.max_dk_idx
            par, kk = par[m], kk[m]
        base = front[par]
        samples = np.stack([arc(base, kk, cfg.step * f) for f in fr], 1)   # (F*nk, ns, 3)
        end = samples[:, -1]
        ok = np.ones(len(end), bool)
        dpsi = np.abs((end[:, 2] - n_heading + np.pi) % (2 * np.pi) - np.pi)
        ok &= dpsi < math.radians(cfg.max_heading_dev_deg)
        if allowed_fn is not None:
            cen = end[:, :2] + world.xc * np.stack([np.cos(end[:, 2]), np.sin(end[:, 2])], -1)
            ok &= allowed_fn(cen)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0:
            return None
        clr = world.clearance(samples[idx].reshape(-1, 3), cfg.inflation).reshape(len(idx), ns)
        good = (clr >= 0).all(1)
        idx, clr = idx[good], clr[good]
        if len(idx) == 0:
            return None
        end, par, kk = end[idx], par[idx], kk[idx]
        g = (front_g[par]
             + cfg.w_clear * cfg.step * np.clip(cfg.clear_pref - clr.min(1), 0, None)
             + cfg.w_kappa * np.abs(kk - front_k[par]) / max(kmax, 1e-9))
        prog = (end[:, :2] - gR) @ nrm
        lat = (end[:, :2] - gR) @ gdir
        done = (prog >= 0) & (lat >= -0.5) & (lat <= glen + 0.5)
        if done.any():
            cand = np.nonzero(done)[0]
            i = int(cand[np.argmin(g[cand])])
            layers.append((end, par, kk))
            return _reconstruct(layers, i, cfg)
        key = (np.round(end[:, 0] / cfg.xy_res).astype(np.int64) * 2_000_003
               + np.round(end[:, 1] / cfg.xy_res).astype(np.int64) * 1009
               + np.round(end[:, 2] / psi_res).astype(np.int64))
        order = np.argsort(g, kind="stable")          # cheapest representative per cell
        _, first = np.unique(key[order], return_index=True)
        first = order[first]
        first = np.array([i for i in first if int(key[i]) not in visited], dtype=np.int64)
        if len(first) == 0:
            return None
        visited.update(int(x) for x in key[first])
        if len(first) > cfg.beam:
            rank = g[first] - prog[first]             # cost-to-come + distance-to-goal
            first = first[np.argsort(rank)[:cfg.beam]]
        layers.append((end[first], par[first], kk[first]))
        front, front_k, front_g = end[first], kk[first], g[first]
    return None


def _reconstruct(layers, i: int, cfg: LatticeConfig) -> np.ndarray:
    chain = []
    for lvl in range(len(layers) - 1, 0, -1):
        poses, par, kk = layers[lvl]
        chain.append((int(par[i]), float(kk[i])))
        i = int(par[i])
    chain = chain[::-1]
    pose = layers[0][0][0]
    out = [pose]
    fr = np.arange(1, cfg.n_samples + 1) / cfg.n_samples
    for _, k in chain:
        seg = np.stack([arc(pose[None], np.array([k]), cfg.step * f)[0] for f in fr])
        out.extend(seg)
        pose = seg[-1]
    return np.array(out)
