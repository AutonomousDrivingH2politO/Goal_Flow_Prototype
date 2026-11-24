"""Observed road edges as both the planners and the BEV raster use them.

One definition so the learned planner's drivable channel and the corridor planners see the same
road: edge points binned to 1 m (median), then unobserved stretches of one edge filled from the
opposite edge offset by the nominal track width (observed points always win).
"""
from __future__ import annotations

import numpy as np


def bin_edge(e: np.ndarray, bin_w: float = 1.0) -> np.ndarray:
    if len(e) == 0:
        return np.zeros((0, 2))
    e = e[np.argsort(e[:, 0])]
    b = np.floor(e[:, 0] / bin_w).astype(int)
    out = []
    for v in np.unique(b):
        m = b == v
        out.append([np.median(e[m, 0]), np.median(e[m, 1])])
    return np.array(out)


def offset_edge(e: np.ndarray, d: float) -> np.ndarray:
    """Offset a polyline by d to its left (negative = right).

    The track is straight or constant-curvature, so the edge is first fitted with a cubic in its
    own chord frame and the *fitted* curve is offset: offsetting noisy points by a track width
    multiplies any tangent error twelvefold.
    """
    n = len(e)
    if n < 2:
        return np.zeros((0, 2))
    c = e[-1] - e[0]
    L = np.linalg.norm(c)
    if L < 1e-6:
        return np.zeros((0, 2))
    u = c / L
    R = np.array([[u[0], u[1]], [-u[1], u[0]]])          # world -> chord frame
    q = (e - e[0]) @ R.T
    # degree from the edge's length: cubic ends swing on short noisy edges
    deg = int(min(1 if L < 12 else 2 if L < 25 else 3, n - 1))
    coef = np.polyfit(q[:, 0], q[:, 1], deg)
    res = np.abs(q[:, 1] - np.polyval(coef, q[:, 0]))       # one round of outlier rejection
    keep = res <= max(3.0 * 1.4826 * np.median(res), 0.15)
    if keep.sum() > deg + 1 and not keep.all():
        coef = np.polyfit(q[keep, 0], q[keep, 1], deg)
    y = np.polyval(coef, q[:, 0])
    dy = np.polyval(np.polyder(coef), q[:, 0])
    t = np.stack([np.ones_like(dy), dy], -1)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], -1)
    off = np.stack([q[:, 0], y], -1) + d * nrm
    return off @ R + e[0]


def complete_edges(left: np.ndarray, right: np.ndarray, width: float, gap: float = 2.0,
                   min_sep: float = 1.0):
    """Fill unobserved stretches of one edge from the other, offset by the nominal track width.
    Observed points always win; synthetic ones only where no observed point is within ``gap``.
    A synthetic point must also lie at least ``width - min_sep`` from every point of the edge it
    was offset from (a correct offset is about one width away; anything closer is a bad tangent)."""
    def fill(own, src, synth):
        if len(synth) == 0:
            return own
        ds = np.min(np.hypot(synth[:, None, 0] - src[None, :, 0], synth[:, None, 1] - src[None, :, 1]), 1)
        synth = synth[ds >= width - min_sep]
        if len(own) and len(synth):
            d = np.min(np.hypot(synth[:, None, 0] - own[None, :, 0], synth[:, None, 1] - own[None, :, 1]), 1)
            synth = synth[d > gap]
        out = np.concatenate([own, synth]) if len(own) else synth
        return out[np.argsort(out[:, 0])] if len(out) else out
    return (fill(left, right, offset_edge(right, width)), fill(right, left, offset_edge(left, -width)))


def observed_edges(edge_points: np.ndarray, edge_side: np.ndarray, track_width: float,
                   bin_w: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) polylines, sorted by x, binned and completed."""
    left = bin_edge(np.asarray(edge_points, float)[edge_side > 0], bin_w)
    right = bin_edge(np.asarray(edge_points, float)[edge_side < 0], bin_w)
    return complete_edges(left, right, track_width)
