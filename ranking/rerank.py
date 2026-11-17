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

# SRP-PHAT direction features (same order as the features)
USE_SRP = os.environ.get("USE_SRP","1")=="1"
SRPDIR=os.path.join(os.path.dirname(__file__),"srp_cache",TAG)
if USE_SRP and all(os.path.exists(os.path.join(SRPDIR,s+".npy")) for s in seqs):
    srp_all=np.concatenate([np.load(os.path.join(SRPDIR,s+".npy")) for s in seqs]).astype(np.float32)
    if len(srp_all)==len(F):
        F=np.column_stack([F, srp_all]); print(f"[mcp] +SRP feature (dets={len(F)})")
    else: print(f"[mcp] SRP len mismatch {len(srp_all)} vs {len(F)} — skip")
y=(bestiou>=IOU).astype(int)
NGT=int(ngt.sum())
print(f"[mcp] {TAG} dets={len(F)} feats={F.shape[1]} matchable@{IOU}={int(y.sum())} ({y.mean()*100:.1f}%) NGT={NGT}")

import time as _t
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
MODE=os.environ.get("MCP_MODE","clf")  # clf, regr (best IoU) or perclass
nseq=len(seqs); fold=np.array([s%3 for s in range(nseq)])
pmcp=np.zeros(len(F))
# class array for the perclass mode
_cl=[]
for s in seqs: _cl.append(np.load(os.path.join(DET,s+".npz"))["cls"])
CLS_=np.concatenate(_cl)
for fo in range(3):
    t0=_t.time()
    tr=np.isin(sidx,np.where(fold!=fo)[0]); te=np.isin(sidx,np.where(fold==fo)[0])
    if te.sum()==0 or y[tr].sum()==0: continue
    if MODE=="regr":
        rg=HistGradientBoostingRegressor(max_iter=150,max_depth=6,learning_rate=0.1,
             l2_regularization=1.0,min_samples_leaf=100,early_stopping=True,validation_fraction=0.1)
        rg.fit(F[tr],bestiou[tr]); pmcp[te]=rg.predict(F[te])
    elif MODE=="perclass":
        for c in range(NC):
            mtr=tr&(CLS_==c); mte=te&(CLS_==c)
            if mte.sum()==0: continue
            if mtr.sum()<200 or y[mtr].sum()<5: pmcp[mte]=F[mte][:,0]; continue  # fallback=score
            clf=HistGradientBoostingClassifier(max_iter=100,max_depth=5,min_samples_leaf=60,early_stopping=True,validation_fraction=0.12)
            clf.fit(F[mtr],y[mtr]); pmcp[mte]=clf.predict_proba(F[mte])[:,1]
    else:
        clf=HistGradientBoostingClassifier(max_iter=120,max_depth=6,learning_rate=0.1,
             l2_regularization=1.0,min_samples_leaf=100,early_stopping=True,validation_fraction=0.1)
        clf.fit(F[tr],y[tr]); pmcp[te]=clf.predict_proba(F[te])[:,1]
    print(f"  fold{fo} fit {_t.time()-t0:.0f}s tr={tr.sum()} pos={int(y[tr].sum())} mode={MODE}",flush=True)
np.savez_compressed(os.path.join(os.path.dirname(__file__),f"mcp_scores_{TAG}.npz"),
                    pmcp=pmcp.astype(np.float32), sidx=sidx, seqs=np.array(seqs))

# fast AP: best reference per detection computed once, greedy claim per ranking
clsarr=None  # classes come from the raw detections
# rebuild the class array if it is not cached
if "clsarr" not in dir() or clsarr is None:
    cl_list=[]
    for si,seq in enumerate(seqs):
        z=np.load(os.path.join(DET,seq+".npz")); cl_list.append(z["cls"])
    clsarr=np.concatenate(cl_list)
RR=np.linspace(0,1,101)
def ap101(tp, n_gt):   # tp: 0/1 in rank order, 101-point interpolation
    tpc=np.cumsum(tp); fpc=np.cumsum(1-tp); rec=tpc/n_gt; prec=tpc/np.maximum(tpc+fpc,1e-9)
    mpre=np.maximum.accumulate(prec[::-1])[::-1]
    idx=np.searchsorted(rec,RR,side='left'); vals=np.zeros(101)
    ok=idx<len(rec); vals[ok]=mpre[idx[ok]]
    return float(vals.mean())
def greedy_tp(order, thr):
    bg=bestgid[order]; bi=bestiou[order]; tp=np.zeros(len(order))
    cand=np.where(bi>=thr)[0]   # matchable detections only, in rank order
    bgc=bg[cand].astype(int); claimed=set()
    for k,r in enumerate(cand):
        g=bgc[k]
        if g>=0 and g not in claimed: claimed.add(g); tp[r]=1
    return tp
def ap_rank(key, thr):
    aps=[]; allk=[]; allt=[]
    for c in range(NC):
        if ngt[c]==0: continue
        idx=np.where(clsarr==c)[0]
        if len(idx)==0: aps.append(0.0); continue
        order=idx[np.argsort(-key[idx])]; tp=greedy_tp(order,thr)
        aps.append(ap101(tp,ngt[c])); allk.append(key[order]); allt.append(tp)
    macro=float(np.mean(aps)) if aps else float("nan")
    K=np.concatenate(allk); T=np.concatenate(allt); o=np.argsort(-K)
    micro=ap101(T[o],NGT)
    return round(macro,4),round(micro,4)

import time as _t2
t0=_t2.time(); cur=ap_rank(score,IOU); mcp=ap_rank(pmcp,IOU); orc=ap_rank(bestiou,IOU)
print(f"  ap_rank x3 {_t2.time()-t0:.0f}s",flush=True)
def rec(a,b,o): return round(100*(b-a)/max(o-a,1e-9),1)
out={"tag":TAG,"iou":IOU,"n_det":len(F),"matchable_pct":round(float(y.mean()),3),
 "macro_current":cur[0],"macro_mcp":mcp[0],"macro_oracle":orc[0],"macro_recovery_pct":rec(cur[0],mcp[0],orc[0]),
 "micro_current":cur[1],"micro_mcp":mcp[1],"micro_oracle":orc[1],"micro_recovery_pct":rec(cur[1],mcp[1],orc[1])}
print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
json.dump(out,open(os.path.join(os.path.dirname(__file__),f"mcp_rerank_{TAG}_{IOU}.json"),"w"),ensure_ascii=False,indent=2)
clf=HistGradientBoostingClassifier(max_iter=120,max_depth=6,early_stopping=True).fit(F,y)
from sklearn.inspection import permutation_importance
ss=np.random.RandomState(0).choice(len(F),min(25000,len(F)),replace=False)
imp=permutation_importance(clf,F[ss],y[ss],n_repeats=2,random_state=0)
names=["score","area","elev","asp","fcnt","fccnt","closure","diffuse","emax","emean","eprom","econc","eentropy","estd","poff","nmsr"]
if F.shape[1]==17: names=names+["srp"]
elif F.shape[1]==18: names=names+["srp","srp_rank"]
order=np.argsort(-imp.importances_mean)
print("feat_importance:",{names[i]:round(float(imp.importances_mean[i]),4) for i in order[:8]})
json.dump(out,open(os.path.join(os.path.dirname(__file__),f"mcp_rerank_{TAG}_{IOU}.json"),"w"),ensure_ascii=False,indent=2)
