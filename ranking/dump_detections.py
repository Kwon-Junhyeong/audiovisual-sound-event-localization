#!/usr/bin/env python3
"""Raw detections before filtering: low-threshold inference on the dev-test clips, saved as box, score and class
in ranking/raw_dets/<seq>.npz, so the reranking runs on the CPU from this cache.
usage: dump_detections.py <ckpt> <n_clips> [score_thresh] [det_per_img]
"""
import os, sys, glob, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector"))
import common as C
C.setup_paths()
import numpy as np, torch

CKPT=sys.argv[1]; NCLIP=int(sys.argv[2]) if len(sys.argv)>2 else 12
STH=float(sys.argv[3]) if len(sys.argv)>3 else 0.01
DPI=int(sys.argv[4]) if len(sys.argv)>4 else 100
SPLIT=os.environ.get("SPLIT","dev-test")     # dev-test, or dev-train for calibration
SUF="" if SPLIT=="dev-test" else "_calib"
TAG=os.path.basename(CKPT)[:-4]+SUF
OUT=os.path.join(os.path.dirname(__file__),"raw_dets",TAG); os.makedirs(OUT,exist_ok=True)
FRAMES=os.path.join(C.IMPL,"frames_dev")
MEMFRAC=float(os.environ.get("MEMFRAC","0.4"))
dev="cuda"; torch.cuda.set_per_process_memory_fraction(MEMFRAC,0)
C.build_mic_layout(); C.use_jpg_frames(); C.use_blindfold_missing()
ext=C.make_extractor(dev)
model=C.build_model(dev, freeze_encoder=True); model.eval()
ck=torch.load(CKPT,map_location=dev); model.load_state_dict(ck["model"])
rh=model.det.roi_heads; rh.score_thresh=STH; rh.detections_per_img=DPI
print(f"[dump] {TAG} ep{ck.get('epoch')} thr={STH} det/img={DPI} → {OUT}",flush=True)

wavs=sorted(glob.glob(os.path.join(C.MIC,f"{SPLIT}-*","*.wav")))[:NCLIP]
for wi,wav in enumerate(wavs):
    seq=os.path.basename(wav)[:-4]; split=os.path.basename(os.path.dirname(wav))
    cache=os.path.join(C.CACHE,split,seq+".npy")
    if not os.path.exists(cache): continue
    if os.path.exists(os.path.join(OUT,seq+".npz")): print(f"  skip {seq}"); continue
    nfr=int(np.load(cache,mmap_mode="r").shape[0])
    FI=[]; CL=[]; SC=[]; BX=[]; EM=[]
    with torch.no_grad():
        B=8
        for s in range(0,nfr,B):
            idxs=list(range(s,min(s+B,nfr)))
            imgs=[C.frame_image(ext,wav,fi,FRAMES).to(dev) for fi in idxs]
            with torch.amp.autocast("cuda",enabled=True): dets=model(imgs)
            for fi,d in zip(idxs,dets):
                bx=d.get("boxes"); lb=d.get("labels"); sc=d.get("scores"); em=d.get("energy_maps")
                if bx is None or len(bx)==0: continue
                bx=bx.cpu().numpy(); lb=lb.cpu().numpy(); sc=sc.cpu().numpy()
                em=em.cpu().numpy().astype(np.float16) if em is not None else None
                for k in range(len(bx)):
                    c=int(lb[k])-1
                    if not(0<=c<13): continue
                    FI.append(fi); CL.append(c); SC.append(float(sc[k])); BX.append(bx[k].tolist())
                    EM.append(em[k] if em is not None else np.zeros((28,28),np.float16))
    np.savez_compressed(os.path.join(OUT,seq+".npz"),
        fi=np.array(FI,np.int32), cls=np.array(CL,np.int32), score=np.array(SC,np.float32),
        box=np.array(BX,np.float32), emap=np.array(EM,np.float16), nframes=nfr)
    print(f"  [{wi+1}/{len(wavs)}] {seq}: frames={nfr} raw_dets={len(FI)}",flush=True)
print("[dump] done",flush=True)
