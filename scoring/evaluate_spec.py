#!/usr/bin/env python3
"""
Spec scorer: the DCASE 2026 Task 3 metric as written on the task page.
(The code scorer is evaluate.py of the baseline repository; this file is the strict spec counterpart.)

Spec: a spherical Gaussian blob (sigma 6 deg) for each [x, y, intensity] on a 360 x 180 canvas,
  binary mask M = 1[E >= tau max E], tau 0.10, then COCO mAP (binary mask IoU, 0.50:0.05:0.95)
  and AP50; Pearson r on the union of matched pairs; great-circle Hungarian matching (T_DOA 20 deg).

Notes:
- The Gaussian is approximated by gaussian_filter(sigma 6) with longitude wrap.
  The 1/cos widening near the poles is not modeled; most sources are near the equator.
- mAP matching is COCO style (per frame and class, greedy by score, binary IoU), 101-point AP, class macro.
"""
import os, json, argparse, glob
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.optimize import linear_sum_assignment

H, W = 180, 360
SIGMA, TAU = 6.0, 0.10
IOU_THRS = np.round(np.arange(0.50, 0.951, 0.05), 2)   # COCO 10 thresholds
N_CLASSES = 13
T_DOA = 20.0

def load_anns(path):
    with open(path) as f:
        return json.load(f).get("annotations", [])

_MARGIN = int(np.ceil(3 * SIGMA))   # crop margin for the Gaussian tail

def render_energy(seg):
    """seg=list of polygons (each = list of [x,y,intensity]) → (H,W) energy float32 or None.
    Crops only the elevation range of the instance (+-3 sigma); azimuth keeps the full width for the wrap."""
    pts = [p for poly in seg if poly for p in poly]
    if not pts:
        return None
    a = np.asarray(pts, dtype=np.float32)
    x = np.clip(np.round(a[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(a[:, 1]).astype(int), 0, H - 1)
    inten = a[:, 2].astype(np.float32)
    y0 = max(0, int(y.min()) - _MARGIN)
    y1 = min(H, int(y.max()) + _MARGIN + 1)
    Pc = np.zeros((y1 - y0, W), dtype=np.float32)
    np.add.at(Pc, (y - y0, x), inten)
    if Pc.max() <= 0:
        return None
    Ec = gaussian_filter(Pc, sigma=SIGMA, mode=["constant", "wrap"])
    E = np.zeros((H, W), dtype=np.float32)
    E[y0:y1] = Ec
    return E

def to_mask(E):
    if E is None:
        return None
    m = E.max()
    return (E >= TAU * m) if m > 1e-9 else None

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    return float(inter) / float(np.logical_or(a, b).sum())

def centroid_latlon(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (90.0 - ys.mean(), xs.mean())

def great_circle_deg(p, q):
    lat1, lon1, lat2, lon2 = map(np.radians, (p[0], p[1], q[0], q[1]))
    c = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

def compute_ap(scores, tps, n_gt):
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, float))
    tp = np.asarray(tps, float)[order]
    tpc = np.cumsum(tp); fpc = np.cumsum(1.0 - tp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        p = prec[rec >= r]
        ap += p.max() if p.size else 0.0
    return ap / 101.0

def group_fc(anns):
    d = {}
    for a in anns:
        d.setdefault((int(a["metadata_frame_index"]), int(a["category_id"])), []).append(a)
    return d

def evaluate(pred_dir, gt_dir, verbose=False):
    gt_files = {os.path.basename(p)[:-len("_std.json")]: p
                for p in glob.glob(os.path.join(gt_dir, "**", "*_std.json"), recursive=True)}
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.json")))

    det = {c: [] for c in range(N_CLASSES)}     # (score, tpvec[len IOU_THRS])
    n_gt = {c: 0 for c in range(N_CLASSES)}
    pearson_vals = []
    matched_seqs = 0

    for pf in pred_files:
        seq = os.path.basename(pf)[:-5]
        if seq not in gt_files:
            continue
        matched_seqs += 1
        pred_fc, gt_fc = group_fc(load_anns(pf)), group_fc(load_anns(gt_files[seq]))
        for (fi, cls), gl in gt_fc.items():
            if 0 <= cls < N_CLASSES:
                n_gt[cls] += len(gl)
        for (fi, cls) in (set(pred_fc) | set(gt_fc)):
            if not (0 <= cls < N_CLASSES):
                continue
            gE = [render_energy(a["segmentation"]) for a in gt_fc.get((fi, cls), [])]
            gts = [(E, to_mask(E)) for E in gE if E is not None]
            gts = [(E, m) for (E, m) in gts if m is not None]
            preds = []
            for a in pred_fc.get((fi, cls), []):
                E = render_energy(a["segmentation"]); m = to_mask(E)
                if m is not None:
                    preds.append((float(a.get("score", 0.0)), E, m))
            if not preds:
                continue
            preds.sort(key=lambda t: -t[0])
            ng = len(gts)
            iou_mat = np.zeros((len(preds), ng), np.float32)
            for i, (_, _, pm) in enumerate(preds):
                for j, (_, gm) in enumerate(gts):
                    iou_mat[i, j] = mask_iou(pm, gm)
            # COCO greedy per threshold (independent)
            tpvecs = [np.zeros(len(IOU_THRS), bool) for _ in preds]
            for ti, thr in enumerate(IOU_THRS):
                taken = np.zeros(ng, bool)
                for i in range(len(preds)):
                    best_j, best = -1, thr
                    for j in range(ng):
                        if not taken[j] and iou_mat[i, j] >= best:
                            best, best_j = iou_mat[i, j], j
                    if best_j >= 0:
                        taken[best_j] = True; tpvecs[i][ti] = True
            for (sc, _, _), tpv in zip(preds, tpvecs):
                det[cls].append((sc, tpv))
            # Pearson: great-circle Hungarian on matched pairs, energy over 2px dilated union
            if ng:
                ri, cj = linear_sum_assignment(-iou_mat)
                for i, j in zip(ri, cj):
                    if iou_mat[i, j] <= 0:
                        continue
                    pE, pm = preds[i][1], preds[i][2]
                    gE, gm = gts[j]
                    pc, gc = centroid_latlon(pm), centroid_latlon(gm)
                    if pc and gc and great_circle_deg(pc, gc) > T_DOA:
                        continue
                    union = binary_dilation(np.logical_or(pm, gm), iterations=2)
                    pv, gv = pE[union], gE[union]
                    if pv.size >= 2 and pv.std() > 1e-9 and gv.std() > 1e-9:
                        r = np.corrcoef(pv, gv)[0, 1]
                        if not np.isnan(r):
                            pearson_vals.append(float(r))

    per_class, macro = {}, np.zeros(len(IOU_THRS)); valid = 0
    for c in range(N_CLASSES):
        if n_gt[c] == 0:
            per_class[c] = None; continue
        scores = [s for s, _ in det[c]]
        tpmat = np.array([t for _, t in det[c]]) if det[c] else np.zeros((0, len(IOU_THRS)), bool)
        aps = np.array([compute_ap(scores, tpmat[:, ti] if tpmat.size else [], n_gt[c])
                        for ti in range(len(IOU_THRS))])
        per_class[c] = {"mAP": float(np.nanmean(aps)), "AP50": float(aps[0]), "n_gt": int(n_gt[c])}
        macro += aps; valid += 1
    macro = macro / max(valid, 1)
    return {
        "scorer": "spec(gaussian σ=6°, binary τ=10%, 360x180, COCO IoU 0.50:0.05:0.95)",
        "matched_seqs": matched_seqs, "valid_classes": valid,
        "macro_mAP": float(macro.mean()) if valid else float("nan"),
        "macro_AP50": float(macro[0]) if valid else float("nan"),
        "macro_pearson_r": float(np.mean(pearson_vals)) if pearson_vals else float("nan"),
        "n_pearson_pairs": len(pearson_vals),
        "per_class": {str(k): v for k, v in per_class.items()},
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = evaluate(args.pred_dir, args.gt_dir)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
