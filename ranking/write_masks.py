#!/usr/bin/env python3
"""Mask output: ranks the raw detections by the reranker probability (or by the raw score) and writes each kept box
as nine points, its 28 x 28 energy map sampled on a 3 x 3 grid inside the box.
usage: write_masks.py <tag> [cap] [rank=mcp|raw]
"""
import os, sys, json, collections
import numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
TAG = sys.argv[1] if len(sys.argv) > 1 else "m2c_ep8"
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 1      # boxes per frame and class
RANK = sys.argv[3] if len(sys.argv) > 3 else "mcp"
PER_FRAME = 6                                          # boxes per frame
GRID = 3                                               # points per side
IMG_W, IMG_H = 360, 180                                # panorama of the detector and of the output


def grid_mask(emap, box, n=GRID):
    """Energy map of one box sampled at the centers of an n x n grid. Returns n * n points (x, y, intensity)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    t = torch.from_numpy(np.asarray(emap, np.float32))[None, None]
    g = F.interpolate(t, size=(n, n), mode="bilinear", align_corners=False)[0, 0].numpy()
    lo, hi = g.min(), g.max()
    g = np.ones_like(g) if hi - lo < 1e-8 else (g - lo) / (hi - lo)     # min-max per box
    pts = []
    for iy in range(n):
        for ix in range(n):
            x = min(max(x1 + (ix + 0.5) / n * (x2 - x1), 0.0), IMG_W - 1.0)
            y = min(max(y1 + (iy + 0.5) / n * (y2 - y1), 0.0), IMG_H - 1.0)
            pts.append([round(x, 2), round(y, 2), round(float(g[iy, ix]), 4)])
    return pts


if __name__ == "__main__":
    DET = os.path.join(HERE, "raw_dets", TAG)
    OUT = os.path.join(HERE, "predictions", f"{TAG}_{RANK}_cap{CAP}")
    os.makedirs(OUT, exist_ok=True)
    sc = np.load(os.path.join(HERE, f"mcp_scores_{TAG}.npz"))
    prob, sidx, seqs = sc["pmcp"], sc["sidx"], list(sc["seqs"])
    for si, seq in enumerate(seqs):
        z = np.load(os.path.join(DET, seq + ".npz"))
        fi, cl, bx, em = z["fi"], z["cls"], z["box"], z["emap"]
        key = prob[sidx == si] if RANK == "mcp" else z["score"]
        by_fc = collections.defaultdict(list)
        for k in range(len(fi)):
            by_fc[(int(fi[k]), int(cl[k]))].append(k)
        kept = collections.defaultdict(list)
        for (f, c), ks in by_fc.items():
            for k in sorted(ks, key=lambda k: -key[k])[:CAP]:
                kept[f].append((float(key[k]), k, c))
        anns = []
        for f, lst in kept.items():
            for s, k, c in sorted(lst, key=lambda t: -t[0])[:PER_FRAME]:
                anns.append({"metadata_frame_index": f, "category_id": c, "score": s,
                             "segmentation": [grid_mask(em[k], bx[k])]})
        json.dump({"annotations": anns}, open(os.path.join(OUT, seq + ".json"), "w"))
    print(f"{len(seqs)} clips written to {OUT}")
