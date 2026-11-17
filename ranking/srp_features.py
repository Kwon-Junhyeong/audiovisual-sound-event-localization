#!/usr/bin/env python3
"""SRP-PHAT at the direction of each raw detection, saved as ranking/srp_cache/<tag>/<seq>.npy
in the order of the raw detections. Used as reranker features.
usage: srp_features.py <tag> [n_clips]"""
import os, sys, glob
import numpy as np
from scipy.signal import stft
TASK3=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
D4=os.path.join(TASK3,"work","data_4ch")
TAG=sys.argv[1] if len(sys.argv)>1 else "m2c_ep8"
NCL=int(sys.argv[2]) if len(sys.argv)>2 else 99
DET=os.path.join(os.path.dirname(__file__),"raw_dets",TAG)
OUT=os.path.join(os.path.dirname(__file__),"srp_cache",TAG); os.makedirs(OUT,exist_ok=True)
SR,FPS=24000,10; NPER=1024; AVG=2; C=343.0; NUP=8192; UPS=NUP//NPER; BEST_CONV=1
EIG=[(55,45),(125,315),(125,135),(55,225)]
def unit(inc,az):
    t=np.radians(inc); p=np.radians(az); return np.array([np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)])
POS=np.array([0.042*unit(i,a) for i,a in EIG])
PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
DPOS=np.array([POS[a]-POS[b] for a,b in PAIRS])   # (6,3)
NL=129; LOFF=64   # LAGS=-64..64
def doa(az,el):  # conv1: (x,-y,z)
    az=np.radians(az); el=np.radians(el)
    v=np.stack([np.cos(el)*np.cos(az), -np.cos(el)*np.sin(az), np.sin(el)],-1)
    return v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-9)
def clip_gcc(wav):
    import soundfile as sf
    x,sr=sf.read(wav,dtype="float32"); x=x.T
    f,t,Z=stft(x,fs=SR,nperseg=NPER,noverlap=NPER-512,axis=-1)
    wf=np.floor(t*FPS).astype(int); nF=int(wf.max())+1
    G=np.zeros((nF,6,NL),np.float32)
    for vf in range(nF):
        sel=np.where((wf>=vf-AVG)&(wf<=vf+AVG))[0]
        if len(sel)<3: continue
        Zs=Z[:,:,sel]
        for pi,(a,b) in enumerate(PAIRS):
            S=np.mean(Zs[a]*np.conj(Zs[b]),axis=1); S=S/(np.abs(S)+1e-8)
            g=np.fft.irfft(S,n=NUP); G[vf,pi]=np.concatenate([g[-64:],g[:65]])
    return G,nF
# direction grid for contrast and rank: 24 azimuths x 8 elevations
_gaz=np.linspace(-180,165,24); _gel=np.linspace(-78,78,8)
_GA,_GE=np.meshgrid(_gaz,_gel); GRID=doa(_GA.ravel(),_GE.ravel())   # (192,3)
GTAU=(-(GRID@DPOS.T)/C*SR*UPS); GPOS=np.clip(GTAU+LOFF,0,NL-1.001)
GF0=np.floor(GPOS).astype(int); GW=GPOS-GF0; NG=GRID.shape[0]; AR6=np.arange(6)
def interp_srp(Gframe, f0, w):   # Gframe:(6,NL), f0/w:(...,6) → srp(...)
    g0=Gframe[AR6,f0]; g1=Gframe[AR6,np.minimum(f0+1,NL-1)]
    return (g0*(1-w)+g1*w).sum(-1)

wavf={os.path.basename(p)[:-4]:p for p in glob.glob(os.path.join(D4,'**','*.wav'),recursive=True)}
seqs=sorted([os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DET,'*.npz'))])[:NCL]
for si,seq in enumerate(seqs):
    if os.path.exists(os.path.join(OUT,seq+".npy")): print(f"skip {seq}"); continue
    if seq not in wavf: print(f"no wav {seq}"); continue
    G,nF=clip_gcc(wavf[seq])
    z=np.load(os.path.join(DET,seq+".npz")); fi=z["fi"]; bx=z["box"]
    cx=(bx[:,0]+bx[:,2])/2; cy=(bx[:,1]+bx[:,3])/2
    d=doa(cx-180.0, 90.0-cy)                       # (N,3)
    tau=-(d@DPOS.T)/C*SR*UPS                        # (N,6) upsample-lag
    pos=np.clip(tau+LOFF,0,NL-1.001); f0=np.floor(pos).astype(int); w=pos-f0
    ff=np.clip(fi,0,nF-1)
    Gd=G[ff]                                        # (N,6,NL)
    ar=np.arange(len(fi))[:,None]; pr=np.arange(6)[None,:]
    g0=Gd[ar,pr,f0]; g1=Gd[ar,pr,np.minimum(f0+1,NL-1)]
    srp=((g0*(1-w)+g1*w).sum(1)).astype(np.float32) # (N,)
    # SRP on the grid per frame, then the dominance rank of the detection direction
    uf=np.unique(ff); gmap={int(f):i for i,f in enumerate(uf)}
    grid_srp=np.zeros((len(uf),NG),np.float32)
    for i,f in enumerate(uf):
        Gf=G[f]                                     # (6,NL)
        gg0=Gf[AR6[None,:],GF0]; gg1=Gf[AR6[None,:],np.minimum(GF0+1,NL-1)]
        grid_srp[i]=(gg0*(1-GW)+gg1*GW).sum(1)      # (NG,)
    srp_rank=np.empty(len(fi),np.float32)
    for k in range(len(fi)):
        gs=grid_srp[gmap[int(ff[k])]]; srp_rank[k]=float((gs<srp[k]).mean())
    out=np.stack([srp,srp_rank],1).astype(np.float32)   # (N,2)
    np.save(os.path.join(OUT,seq+".npy"), out)
    print(f"[{si+1}/{len(seqs)}] {seq} dets={len(fi)} srp={srp.mean():.4f} rank={srp_rank.mean():.3f}",flush=True)
print("[srp_dump] done",flush=True)
