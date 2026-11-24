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
    """Offset a polyline by d to its left (negative = right)."""
    if len(e) < 2:
        return np.zeros((0, 2))
    t = np.gradient(e, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return e + d * np.stack([-t[:, 1], t[:, 0]], -1)


def complete_edges(left: np.ndarray, right: np.ndarray, width: float, gap: float = 2.0):
    """Fill unobserved stretches of one edge from the other, offset by the nominal track width.
    Observed points always win; synthetic ones only where no observed point is within ``gap``."""
    def fill(own, synth):
        if len(synth) == 0:
            return own
        if len(own):
            d = np.min(np.hypot(synth[:, None, 0] - own[None, :, 0], synth[:, None, 1] - own[None, :, 1]), 1)
            synth = synth[d > gap]
        out = np.concatenate([own, synth]) if len(own) else synth
        return out[np.argsort(out[:, 0])]
    return fill(left, offset_edge(right, width)), fill(right, offset_edge(left, -width))


def observed_edges(edge_points: np.ndarray, edge_side: np.ndarray, track_width: float,
                   bin_w: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) polylines, sorted by x, binned and completed."""
    left = bin_edge(np.asarray(edge_points, float)[edge_side > 0], bin_w)
    right = bin_edge(np.asarray(edge_points, float)[edge_side < 0], bin_w)
    return complete_edges(left, right, track_width)
