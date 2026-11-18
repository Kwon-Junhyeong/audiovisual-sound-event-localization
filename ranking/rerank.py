#!/usr/bin/env python3
"""Reranking: per-detection features from the raw detections; a gradient-boosting model cross-fitted over clip folds
predicts whether each candidate matches a reference event, and that probability replaces the detector score.
- features: raw score, energy map (prominence, concentration, entropy, max, mean, std), box geometry,
  co-detection density, SRP-PHAT direction dominance. label: matches a reference event at the IoU threshold.
usage: rerank.py <tag> [iou]"""
import os, sys, json, glob, collections
import numpy as np
TASK3=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
GT=os.environ.get("DCASE_LABELS", os.path.join(TASK3,"data","labels_dev"))
TAG=sys.argv[1] if len(sys.argv)>1 else "m2c_ep8"
IOU=float(sys.argv[2]) if len(sys.argv)>2 else 0.25
DET=os.path.join(os.path.dirname(__file__),"raw_dets",TAG)
GCACHE=os.path.join(os.path.dirname(__file__),"gate_cache")
FEATCACHE=os.path.join(os.path.dirname(__file__),f"mcp_feats_{TAG}.npz")
NC=13
def bbox(seg):
    pts=[p for poly in seg if poly for p in poly]
    if not pts: return None
    a=np.asarray(pts,float); return [a[:,0].min(),a[:,1].min(),a[:,0].max(),a[:,1].max()]
def gfc(anns):
    d=collections.defaultdict(list)
    for a in anns: d[(int(a["metadata_frame_index"]),int(a["category_id"]))].append(a)
    return d
def iou_mat(D,G):   # D:(n,4) G:(m,4) -> (n,m)
    if len(D)==0 or len(G)==0: return np.zeros((len(D),len(G)))
    x0=np.maximum(D[:,0:1],G[:,0]); y0=np.maximum(D[:,1:2],G[:,1])
    x1=np.minimum(D[:,2:3],G[:,2]); y1=np.minimum(D[:,3:4],G[:,3])
    iw=np.clip(x1-x0,0,None); ih=np.clip(y1-y0,0,None); inter=iw*ih
    ad=((D[:,2]-D[:,0])*(D[:,3]-D[:,1]))[:,None]; ag=((G[:,2]-G[:,0])*(G[:,3]-G[:,1]))[None,:]
    return inter/np.maximum(ad+ag-inter,1e-9)

gtf={os.path.basename(p)[:-len('_std.json')]:p for p in glob.glob(os.path.join(GT,'**','*_std.json'),recursive=True)}
seqs=sorted([os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DET,"*.npz")) if os.path.basename(p)[:-4] in gtf])

if os.path.exists(FEATCACHE):
    z=np.load(FEATCACHE,allow_pickle=True)
    F=z["F"]; sidx=z["sidx"]; bestgid=z["bestgid"]; bestiou=z["bestiou"]; score=z["score"]; ngt=z["ngt"]; biou=z["biou"]
    print(f"[mcp] feat cache hit {FEATCACHE} dets={len(F)}")
else:
    Fl=[]; sidxl=[]; bgidl=[]; biou_l=[]; scl=[]; ngt=np.zeros(NC,int); gid_base=0
    for si,seq in enumerate(seqs):
        z=np.load(os.path.join(DET,seq+".npz")); fi=z["fi"]; cl=z["cls"]; sc=z["score"]; bx=z["box"]; em=z["emap"].astype(np.float32)
        G=gfc(json.load(open(gtf[seq])).get("annotations",[]))
        gca=os.path.join(GCACHE,seq+".npz"); clo=dif=None
        if os.path.exists(gca): g=np.load(gca); clo=g["clo"]; dif=g["dif"]
        # reference boxes and global ids
        gtbox={}; gid_map={}
        for (f,c),gl in G.items():
            if not(0<=c<NC): continue
            bs=[bbox(a["segmentation"]) for a in gl]; bs=[b for b in bs if b]
            if bs:
                gtbox[(f,c)]=np.array(bs,float)
                gid_map[(f,c)]=np.arange(gid_base,gid_base+len(bs)); gid_base+=len(bs)
                ngt[c]+=len(bs)
        # energy-map features
        e=em.reshape(len(em),-1); s=e.sum(1)+1e-9; mx=e.max(1); mn=e.mean(1); std=e.std(1)
        p=e/s[:,None]; ent=-(p*np.log(p+1e-12)).sum(1)/np.log(e.shape[1])
        k=max(1,int(0.1*e.shape[1])); conc=np.partition(e,-k,axis=1)[:,-k:].sum(1)/s
        prom=mx/(mn+1e-6)
        EW=int(round(e.shape[1]**0.5)); ctr=(EW-1)/2.0; mxd=(2*ctr**2)**0.5
        pk=e.argmax(1); poff=np.sqrt((pk//EW-ctr)**2+(pk%EW-ctr)**2)/max(mxd,1e-6)  # offset of the energy peak from the box centre
        area=(bx[:,2]-bx[:,0])*(bx[:,3]-bx[:,1])/(360*180); elev=(bx[:,1]+bx[:,3])/2/180
        asp=(bx[:,2]-bx[:,0])/np.maximum(bx[:,3]-bx[:,1],1)
        fcount=collections.Counter(fi.tolist()); fccount=collections.Counter(zip(fi.tolist(),cl.tolist()))
        fcnt=np.array([fcount[int(x)] for x in fi]); fccnt=np.array([fccount[(int(fi[i]),int(cl[i]))] for i in range(len(fi))])
        cv=np.array([clo[int(f)] if (clo is not None and int(f)<len(clo) and np.isfinite(clo[int(f)])) else 0.0 for f in fi])
        dv=np.array([dif[int(f)] if (dif is not None and int(f)<len(dif) and np.isfinite(dif[int(f)])) else 0.0 for f in fi])
        # best reference match and local NMS dominance
        bg=np.full(len(fi),-1,int); bi=np.zeros(len(fi)); nmsr=np.zeros(len(fi))
        byfc=collections.defaultdict(list)
        for i in range(len(fi)): byfc[(int(fi[i]),int(cl[i]))].append(i)
        for (f,c),idl in byfc.items():
            idl=np.array(idl)
            if (f,c) in gtbox:
                M=iou_mat(bx[idl],gtbox[(f,c)]); j=M.argmax(1); v=M[np.arange(len(idl)),j]
                bg[idl]=gid_map[(f,c)][j]; bi[idl]=v
            if 1<len(idl)<=300:
                Ms=iou_mat(bx[idl],bx[idl])>0.3; sci=sc[idl]
                higher=(sci[None,:]>sci[:,None])&Ms; n_ov=Ms.sum(1)-1
                nmsr[idl]=higher.sum(1)/np.maximum(n_ov,1)
        feats=np.stack([sc,area,elev,asp,fcnt,fccnt,cv,dv,mx,mn,prom,conc,ent,std,poff,nmsr],1)
        Fl.append(feats); sidxl.append(np.full(len(fi),si)); bgidl.append(bg); biou_l.append(bi); scl.append(sc)
        print(f"  feat {si+1}/{len(seqs)} {seq} dets={len(fi)}",flush=True)
    F=np.concatenate(Fl).astype(np.float32); sidx=np.concatenate(sidxl); bestgid=np.concatenate(bgidl)
    bestiou=np.concatenate(biou_l); score=np.concatenate(scl); biou=bestiou
    np.savez_compressed(FEATCACHE,F=F,sidx=sidx,bestgid=bestgid,bestiou=bestiou,score=score,ngt=ngt,biou=biou)
    print(f"[mcp] feat cache saved {FEATCACHE} dets={len(F)}")

# SRP-PHAT direction feature (same order as the features)
SRPDIR=os.path.join(os.path.dirname(__file__),"srp_cache",TAG)
if all(os.path.exists(os.path.join(SRPDIR,s+".npy")) for s in seqs):
    srp_all=np.concatenate([np.load(os.path.join(SRPDIR,s+".npy")) for s in seqs]).astype(np.float32)
    if len(srp_all)==len(F): F=np.column_stack([F, srp_all])
y=(bestiou>=IOU).astype(int)
print(f"[mcp] {TAG} dets={len(F)} feats={F.shape[1]} matchable@{IOU}={int(y.sum())} ({y.mean()*100:.1f}%)")

# match probability, cross-fitted over three clip folds
from sklearn.ensemble import HistGradientBoostingClassifier
fold=np.array([s%3 for s in range(len(seqs))])
pmcp=np.zeros(len(F))
for fo in range(3):
    tr=np.isin(sidx,np.where(fold!=fo)[0]); te=np.isin(sidx,np.where(fold==fo)[0])
    if te.sum()==0 or y[tr].sum()==0: continue
    clf=HistGradientBoostingClassifier(max_iter=120,max_depth=6,learning_rate=0.1,
         l2_regularization=1.0,min_samples_leaf=100,early_stopping=True,validation_fraction=0.1)
    clf.fit(F[tr],y[tr]); pmcp[te]=clf.predict_proba(F[te])[:,1]
np.savez_compressed(os.path.join(os.path.dirname(__file__),f"mcp_scores_{TAG}.npz"),
                    pmcp=pmcp.astype(np.float32), sidx=sidx, seqs=np.array(seqs))
