#!/usr/bin/env python3
"""Predictions: raw detections ranked by the reranker probability, and each box output as points
across its predicted energy instead of a single peak.
usage: predict.py <tag> [cap]
"""
import os, sys, json, collections
import numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(__file__)
TAG = sys.argv[1] if len(sys.argv) > 1 else "m2c_ep8"
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 3      # boxes per frame and class
PER_FRAME = 6                                          # boxes per frame
N_POINTS = 20                                          # points per box
IMG_W, IMG_H = 360, 180                                # detector panorama
EVAL_W, EVAL_H = 100, 50                               # grid of the reference energy field


def mask_points(emap, box, n_points=N_POINTS):
    """Energy map of one box resized to the box on the reference grid; the n strongest cells as (x, y, energy)."""
    sx, sy = EVAL_W / IMG_W, EVAL_H / IMG_H
    ex1 = int(np.clip(round(float(box[0]) * sx), 0, EVAL_W - 1))
    ey1 = int(np.clip(round(float(box[1]) * sy), 0, EVAL_H - 1))
    ex2 = int(np.clip(round(float(box[2]) * sx), 0, EVAL_W))
    ey2 = int(np.clip(round(float(box[3]) * sy), 0, EVAL_H))
    bw, bh = max(ex2 - ex1, 1), max(ey2 - ey1, 1)
    t = torch.from_numpy(emap.astype(np.float32))[None, None]
    be = F.interpolate(t, size=(bh, bw), mode="bilinear", align_corners=False)[0, 0].numpy()
    lo, hi = be.min(), be.max()
    norm = np.ones_like(be) if hi - lo < 1e-8 else (be - lo) / (hi - lo)
    flat = norm.ravel(); n = min(n_points, flat.size)
    top = np.argpartition(flat, -n)[-n:]; top = top[np.argsort(-flat[top])]
    ox, oy = IMG_W / EVAL_W, IMG_H / EVAL_H
    pts = []
    for k in top:
        py, px = divmod(int(k), bw)
        pts.append([round(min(float((ex1 + px + 0.5) * ox), IMG_W - 1.0), 2),
                    round(min(float((ey1 + py + 0.5) * oy), IMG_H - 1.0), 2),
                    round(float(norm[py, px]), 4)])
    return pts


if __name__ == "__main__":
    DET = os.path.join(HERE, "raw_dets", TAG)
    OUT = os.path.join(HERE, "predictions", TAG)
    os.makedirs(OUT, exist_ok=True)
    sc = np.load(os.path.join(HERE, f"mcp_scores_{TAG}.npz"))
    prob, sidx, seqs = sc["pmcp"], sc["sidx"], list(sc["seqs"])
    for si, seq in enumerate(seqs):
        z = np.load(os.path.join(DET, seq + ".npz"))
        fi, cl, bx, em = z["fi"], z["cls"], z["box"], z["emap"]
        p = prob[sidx == si]
        by_fc = collections.defaultdict(list)
        for k in range(len(fi)):
            by_fc[(int(fi[k]), int(cl[k]))].append(k)
        kept = collections.defaultdict(list)
        for (f, c), ks in by_fc.items():
            for k in sorted(ks, key=lambda k: -p[k])[:CAP]:
                kept[f].append((float(p[k]), k, c))
        anns = []
        for f, lst in kept.items():
            for s, k, c in sorted(lst, key=lambda t: -t[0])[:PER_FRAME]:
                anns.append({"metadata_frame_index": f, "category_id": c, "score": s,
                             "segmentation": [mask_points(em[k], bx[k])]})
        json.dump({"annotations": anns}, open(os.path.join(OUT, seq + ".json"), "w"))
    print(f"{len(seqs)} clips written to {OUT}")
