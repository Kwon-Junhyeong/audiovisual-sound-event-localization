#!/usr/bin/env python3
"""Submission: raw detections ranked by the reranker probability per frame and class, capped, peaks extracted,
written as submission JSON and scored with the code and spec scorers.
usage: submit.py <tag> [cap] [rank=mcp|score] [n_clips]
"""
import os, sys, json, glob, collections, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "detector"))
import common as C
C.setup_paths()
import numpy as np
from infer import extract_peaks
TAG=sys.argv[1] if len(sys.argv)>1 else "m2c_ep8"
CAP=int(sys.argv[2]) if len(sys.argv)>2 else 3
RANK=sys.argv[3] if len(sys.argv)>3 else "mcp"
NCL=int(sys.argv[4]) if len(sys.argv)>4 else 15
DET=os.path.join(os.path.dirname(__file__),"raw_dets",TAG)
HARNESS=os.path.join(C.TASK3,"scoring")
seqs=sorted([os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DET,'*.npz'))])[:NCL]
# reranker probabilities and best IoU, in the feature order
sc=np.load(os.path.join(os.path.dirname(__file__),f"mcp_scores_{TAG}.npz"),allow_pickle=True)
pmcp=sc["pmcp"]; sidx=sc["sidx"]; mseqs=list(sc["seqs"])
fc=np.load(os.path.join(os.path.dirname(__file__),f"mcp_feats_{TAG}.npz"))
bestiou_all=fc["bestiou"]   # same order as sidx
subdir=os.path.join(os.path.dirname(__file__),"submissions",f"{TAG}_{RANK}_cap{CAP}")
os.makedirs(subdir,exist_ok=True)
for si,seq in enumerate(seqs):
    z=np.load(os.path.join(DET,seq+".npz")); fi=z["fi"]; cl=z["cls"]; rawsc=z["score"]; bx=z["box"]; em=z["emap"]
    # this clip
    if seq in mseqs:
        gi=mseqs.index(seq); pm=pmcp[sidx==gi]; bo=bestiou_all[sidx==gi]
        if len(pm)!=len(fi): pm=rawsc; bo=rawsc
    else: pm=rawsc; bo=rawsc
    key = {"mcp":pm,"score":rawsc,"oracle":bo}.get(RANK, pm)
    byfc=collections.defaultdict(list)
    for k in range(len(fi)): byfc[(int(fi[k]),int(cl[k]))].append(k)
    anns=[]
    PERFRAME=int(os.environ.get("PERFRAME_CAP","6"))   # detections per frame (keeps the files under 20 MB)
    kept=collections.defaultdict(list)                  # f -> [(key,k,c)]
    for (f,c),ks in byfc.items():
        for k in sorted(ks,key=lambda k:-key[k])[:CAP]:
            kept[int(f)].append((float(key[k]),k,int(c)))
    NPEAKS=int(os.environ.get("NPEAKS","20"))
    for f,lst in kept.items():
        for sval,k,c in sorted(lst,key=lambda t:-t[0])[:PERFRAME]:
            tri=extract_peaks(em[k].astype(np.float32), bx[k], NPEAKS)
            anns.append({"metadata_frame_index":f,"category_id":c,
                         "score":sval,"segmentation":[tri]})
    json.dump({"annotations":anns}, open(os.path.join(subdir,seq+".json"),"w"))
print(f"[submit] {subdir} clips={len(seqs)} (rank={RANK} cap={CAP})")
# scoring: code (repository evaluate.py, unmodified) and spec
code_out=os.path.join(subdir,"_codeeval"); os.makedirs(code_out,exist_ok=True)
subprocess.run([sys.executable, os.path.join(HARNESS,"evaluate_code.py"),
                "--pred_dir",subdir,"--gt_dir",C.LABELS_BASE,
                "--frames_base",os.path.join(C.WORK,"layout","frames"),"--output_dir",code_out],
               capture_output=True)
cj=os.path.join(code_out,"metrics.json"); code={}
if os.path.exists(cj):
    cd=json.load(open(cj)); code={k:cd.get(k) for k in ("macro_mAP","macro_AP50","micro_mAP","macro_pearson_r","micro_pearson_r")}
sys.path.insert(0,HARNESS); import evaluate_spec
spec=evaluate_spec.evaluate(subdir, C.LABELS_BASE)
spec_rep={k:spec.get(k) for k in ("macro_mAP","macro_AP50","macro_pearson_r")} if spec else {}
print(json.dumps({"tag":TAG,"rank":RANK,"cap":CAP,"code":code,"spec":spec_rep},ensure_ascii=False,indent=2))
json.dump({"tag":TAG,"rank":RANK,"cap":CAP,"code":code,"spec":spec_rep},
          open(os.path.join(os.path.dirname(__file__),f"submit_{TAG}_{RANK}_cap{CAP}.json"),"w"),ensure_ascii=False,indent=2)
