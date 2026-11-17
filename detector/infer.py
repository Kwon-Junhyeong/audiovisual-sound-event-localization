#!/usr/bin/env python3
"""Inference and scoring: load a checkpoint, run dev-test (Track A: RGB = 0), write submission JSON,
score with the spec and code scorers, and report micro and macro mAP and Pearson r.
"""
import os, sys, json, glob, argparse, subprocess, time
sys.path.insert(0, os.path.dirname(__file__))
import common as C
C.setup_paths()
import numpy as np, torch
import torch.nn.functional as F
from acoustic_features import _latent_to_equirect_single

IMG_W, IMG_H = 360, 180
EVAL_W, EVAL_H = 100, 50
OUT_W, OUT_H = 360, 180
SUB_ROOT = os.path.join(C.IMPL, "submissions")
HARNESS = os.path.join(C.TASK3, "scoring")

def extract_peaks(emap_raw, box_xyxy, n_peaks=20):
    sx, sy = EVAL_W / IMG_W, EVAL_H / IMG_H
    ex1 = int(np.clip(round(float(box_xyxy[0]) * sx), 0, EVAL_W - 1))
    ey1 = int(np.clip(round(float(box_xyxy[1]) * sy), 0, EVAL_H - 1))
    ex2 = int(np.clip(round(float(box_xyxy[2]) * sx), 0, EVAL_W))
    ey2 = int(np.clip(round(float(box_xyxy[3]) * sy), 0, EVAL_H))
    bw, bh = max(ex2 - ex1, 1), max(ey2 - ey1, 1)
    t = torch.from_numpy(emap_raw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(bh, bw), mode="bilinear", align_corners=False)
    be = t[0, 0].numpy()
    emin, emax = be.min(), be.max()
    norm = np.ones_like(be) if emax - emin < 1e-8 else (be - emin) / (emax - emin)
    flat = norm.ravel(); n_peaks = min(n_peaks, flat.size)
    top = np.argpartition(flat, -n_peaks)[-n_peaks:]; top = top[np.argsort(-flat[top])]
    oxs, oys = OUT_W / EVAL_W, OUT_H / EVAL_H
    tri = []
    for fi in top:
        py, px = divmod(int(fi), bw)
        ox = min(float((ex1 + px + 0.5) * oxs), OUT_W - 1.0)
        oy = min(float((ey1 + py + 0.5) * oys), OUT_H - 1.0)
        tri.append([round(ox, 2), round(oy, 2), round(float(norm[py, px]), 4)])
    return tri

_FRAME_MISS = {"n": 0}
_IMAGENET_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
def _fill_rgb(mode):
    if mode == "blindfold":
        return _IMAGENET_MEAN_T.expand(3, IMG_H, IMG_W).clone()  # Track A: same as the blindfold in training
    return torch.zeros(3, IMG_H, IMG_W)                          # zero: literal 0

def frame_image(ext, wav_path, fi, frames_base=None, rgb_fill="blindfold"):
    ac = ext.get_frame_bands(wav_path, fi)          # (9,180,360)
    if frames_base:
        split = os.path.basename(os.path.dirname(wav_path)); seq = os.path.basename(wav_path)[:-4]
        fp = os.path.join(frames_base, split, seq, f"{seq}_{int(fi):04d}.jpg")
        if os.path.isfile(fp):
            from PIL import Image
            import numpy as _np
            img = Image.open(fp).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
            rgb = torch.from_numpy(_np.asarray(img, dtype="float32") / 255.0).permute(2, 0, 1)
        else:
            _FRAME_MISS["n"] += 1; rgb = _fill_rgb(rgb_fill)
    else:
        rgb = _fill_rgb(rgb_fill)
    return torch.cat([rgb, ac], dim=0)

def infer_clip(model, ext, wav_path, nframes, dev, score_thr, per_class_cap, batch=48, frames_base=None, rgb_fill="blindfold"):
    anns = []
    model.eval()
    with torch.no_grad():
        for s in range(0, nframes, batch):
            idxs = list(range(s, min(s + batch, nframes)))
            imgs = [frame_image(ext, wav_path, fi, frames_base, rgb_fill).to(dev) for fi in idxs]
            with torch.amp.autocast("cuda", enabled=True):
                dets = model(imgs)
            for fi, det in zip(idxs, dets):
                boxes = det.get("boxes"); labels = det.get("labels"); scores = det.get("scores")
                emaps = det.get("energy_maps"); dists = det.get("distances", None)
                if boxes is None or len(boxes) == 0:
                    continue
                boxes = boxes.cpu().numpy(); labels = labels.cpu().numpy()
                scores = scores.cpu().numpy()
                emaps = emaps.cpu().numpy() if emaps is not None else np.zeros((len(boxes), 28, 28))
                dists = dists.cpu().numpy() if dists is not None else None
                keep = np.where(scores >= score_thr)[0]
                # per-class cap (top by score)
                by_cls = {}
                for di in keep:
                    by_cls.setdefault(int(labels[di]), []).append(di)
                for cls, dis in by_cls.items():
                    if cls <= 0:   # 0=background
                        continue
                    dis = sorted(dis, key=lambda d: -scores[d])[:per_class_cap]
                    for di in dis:
                        tri = extract_peaks(emaps[di], boxes[di], 20)
                        a = {"metadata_frame_index": int(fi), "category_id": int(cls) - 1,
                             "score": float(scores[di]), "segmentation": [tri]}
                        if dists is not None:
                            a["distance"] = float(dists[di] * 500.0)   # dist_norm
                        anns.append(a)
    return anns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="m2a_eval")
    ap.add_argument("--n_clips", type=int, default=0, help="0=all test clips")
    ap.add_argument("--score_thr", type=float, default=0.05)
    ap.add_argument("--per_class_cap", type=int, default=5)
    ap.add_argument("--skip_spec", action="store_true", help="skip the spec scorer (code scorer only, faster)")
    ap.add_argument("--frames_base", default="", help="frame directory for inference with video (empty: no video)")
    ap.add_argument("--rgb_fill", default="blindfold", choices=["blindfold", "zero"],
                    help="RGB fill without video: blindfold (ImageNet mean, as in training) or zero")
    args = ap.parse_args()
    os.makedirs(C.WORK, exist_ok=True); os.chdir(C.WORK)
    dev = "cuda"; torch.cuda.set_per_process_memory_fraction(0.5, 0)
    C.build_mic_layout()
    ext = C.make_extractor(dev)
    model = C.build_model(dev, freeze_encoder=True)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"]); print(f"[infer] loaded {args.ckpt} (epoch {ck.get('epoch')})")

    sub_dir = os.path.join(SUB_ROOT, args.tag); os.makedirs(sub_dir, exist_ok=True)
    test_wavs = sorted(glob.glob(os.path.join(C.MIC, "dev-test-*", "*.wav")))
    if args.n_clips:
        test_wavs = test_wavs[:args.n_clips]
    t0 = time.time()
    for wi, wav in enumerate(test_wavs):
        seq = os.path.basename(wav)[:-4]; split = os.path.basename(os.path.dirname(wav))
        cache = os.path.join(C.CACHE, split, seq + ".npy")
        nframes = int(np.load(cache, mmap_mode="r").shape[0]) if os.path.exists(cache) else 0
        if nframes == 0:
            continue
        anns = infer_clip(model, ext, wav, nframes, dev, args.score_thr, args.per_class_cap,
                          frames_base=(args.frames_base or None), rgb_fill=args.rgb_fill)
        json.dump({"annotations": anns}, open(os.path.join(sub_dir, seq + ".json"), "w"))
        if wi < 2 or wi % 10 == 0:
            print(f"  [{wi+1}/{len(test_wavs)}] {seq}: frames={nframes} preds={len(anns)} ({time.time()-t0:.0f}s)")
    print(f"[infer] submissions → {sub_dir} ({time.time()-t0:.0f}s)")

    # spec scorer (slow, skip with --skip_spec)
    spec = {}
    if not args.skip_spec:
        sys.path.insert(0, HARNESS)
        import evaluate_spec
        spec = evaluate_spec.evaluate(sub_dir, C.LABELS_BASE)
    # code scorer (repository evaluate.py, unmodified)
    code_out = os.path.join(C.WORK, f"codeeval_{args.tag}")
    os.makedirs(code_out, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(HARNESS, "evaluate_code.py"),
                    "--pred_dir", sub_dir, "--gt_dir", C.LABELS_BASE,
                    "--frames_base", os.path.join(C.WORK, "layout", "frames"),
                    "--output_dir", code_out], capture_output=True)
    code = {}
    cj = os.path.join(code_out, "metrics.json")
    if os.path.exists(cj):
        cd = json.load(open(cj))
        code = {"macro_mAP": cd.get("macro_mAP"), "macro_AP50": cd.get("macro_AP50"),
                "micro_mAP": cd.get("micro_mAP"), "macro_pearson_r": cd.get("macro_pearson_r"),
                "micro_pearson_r": cd.get("micro_pearson_r")}
    spec_rep = {k: spec.get(k) for k in ("macro_mAP", "macro_AP50", "macro_pearson_r", "n_pearson_pairs", "matched_seqs")} if spec else {}
    report = {"ckpt": args.ckpt, "epoch": ck.get("epoch"), "n_test_clips": len(test_wavs),
              "spec": spec_rep, "code": code}
    print("[infer] DUAL-SCORE:", json.dumps(report, ensure_ascii=False))
    with open(os.path.join(C.WORK, f"eval_{args.tag}.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
