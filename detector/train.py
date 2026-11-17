#!/usr/bin/env python3
"""Detector training on the organizers' baseline recipe.
- freeze_mode: full, baseline (layer2 frozen, layer4 from epoch 3, layer3 from epoch 7) or frozen.
- per-group AdamW learning rates, BN of frozen layers in eval mode, energy_mult raises the energy loss.
- gradient accumulation, fp16, a checkpoint and the validation loss every epoch.
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(__file__))
import common as C
C.setup_paths(opt_einsum=True)
import torch
from torch.utils.data import DataLoader

CKPT_DIR = os.path.join(C.IMPL, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

def move_t(targets, dev):
    return [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]

_IMAGENET_MEAN = None
def apply_rgb(images, mode, blindfold_p, dev):
    """rgb_mode: zero (Track A, RGB = 0), dropout (real RGB with 15% blindfold) or real."""
    global _IMAGENET_MEAN
    import random
    if mode == "zero":
        for im in images:
            im[:3, :, :] = 0.0
    elif mode == "dropout":
        if _IMAGENET_MEAN is None:
            _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(3, 1, 1)
        for im in images:
            if random.random() < blindfold_p:
                im[:3, :, :] = _IMAGENET_MEAN
    # real: as it is
    return images

def total_loss(ld, energy_mult):
    base = sum(v for v in ld.values() if torch.is_tensor(v) and v.ndim == 0)
    if energy_mult != 1.0:
        extra = 0.0
        for k in ("loss_energy_mask", "loss_fullmap"):
            if k in ld and torch.is_tensor(ld[k]):
                extra = extra + ld[k]
        base = base + (energy_mult - 1.0) * extra
    return base

@torch.no_grad()
def val_loss(model, loader, dev, max_batches, energy_mult, rgb_mode, blindfold):
    model.train()
    tot, n = 0.0, 0
    for images, targets in loader:
        if sum(t["boxes"].shape[0] for t in targets) == 0:
            continue
        # validation: real frames for audiovisual training, zero for Track A
        vm = "real" if rgb_mode == "dropout" else rgb_mode
        images = apply_rgb([im.to(dev) for im in images], vm, blindfold, dev); targets = move_t(targets, dev)
        with torch.amp.autocast("cuda", enabled=True):
            tot += float(total_loss(model(images, targets), energy_mult)); n += 1
        if n >= max_batches:
            break
    return tot / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--frames_per_epoch", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze_mode", default="full", choices=["full", "baseline", "frozen"])
    ap.add_argument("--energy_mult", type=float, default=3.0)
    ap.add_argument("--frames_base", default="", help="frame directory with video (empty: zero RGB layout)")
    ap.add_argument("--rgb_mode", default="zero", choices=["zero", "dropout", "real"])
    ap.add_argument("--blindfold", type=float, default=0.15)
    ap.add_argument("--val_batches", type=int, default=120)
    ap.add_argument("--ckpt_every", type=int, default=1)
    ap.add_argument("--tag", default="m2b")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()
    MET = os.path.join(C.WORK, f"train_{args.tag}.jsonl")

    os.makedirs(C.WORK, exist_ok=True); os.chdir(C.WORK)
    dev = "cuda"; torch.cuda.set_per_process_memory_fraction(0.5, 0)
    C.build_mic_layout()
    ext = C.make_extractor(dev)
    fb = args.frames_base or None
    if fb:
        C.use_jpg_frames()         # .jpg frames
        C.use_blindfold_missing()  # missing frames filled with the ImageNet mean
        print(f"[train] video frames: {fb} (rgb_mode={args.rgb_mode}, missing=blindfold)")
    fpe = None if args.frames_per_epoch <= 0 else args.frames_per_epoch   # 0=full-data(no subsample)
    train_ds, _ = C.build_dataset("train", ext, frames_per_epoch=fpe, frames_base=fb)
    val_ds, _ = C.build_dataset("test", ext, frames_per_epoch=args.val_batches * 3 * args.batch_size, frames_base=fb)
    from model import collate_fn
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = C.build_model(dev, freeze_mode=args.freeze_mode)
    opt = C.build_optimizer(model, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    start_ep = 1
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); start_ep = ck["epoch"] + 1
        print(f"[train] resumed @ {ck['epoch']}")

    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] tag={args.tag} freeze={args.freeze_mode} energy_mult={args.energy_mult} "
          f"batch={args.batch_size}x{args.grad_accum} | trainable={nt/1e6:.1f}M | "
          f"train_pos={len(train_ds)} val_pos={len(val_ds)}", flush=True)

    best = float("inf")
    for ep in range(start_ep, args.epochs + 1):
        train_ds.reset_epoch()
        C.apply_freeze(model, args.freeze_mode, epoch=ep)
        model.train(); C.freeze_bn(model)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time(); run = 0.0; nb = 0; opt.zero_grad(set_to_none=True)
        for images, targets in train_loader:
            if sum(t["boxes"].shape[0] for t in targets) == 0:
                continue
            images = apply_rgb([im.to(dev) for im in images], args.rgb_mode, args.blindfold, dev); targets = move_t(targets, dev)
            with torch.amp.autocast("cuda", enabled=True):
                loss = total_loss(model(images, targets), args.energy_mult) / args.grad_accum
            scaler.scale(loss).backward()
            run += float(loss) * args.grad_accum; nb += 1
            if nb % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if nb % 400 == 0:
                print(f"  {args.tag} ep{ep} b{nb} loss={run/nb:.3f} peakVRAM={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
        tr = run / max(nb, 1)
        vl = val_loss(model, val_loader, dev, args.val_batches, args.energy_mult, args.rgb_mode, args.blindfold)
        peak = torch.cuda.max_memory_allocated() / 1e9
        rec = {"epoch": ep, "train_loss": round(tr, 4), "val_loss": round(vl, 4),
               "peak_VRAM_GB": round(peak, 2), "epoch_s": round(time.time() - t0, 1),
               "n_batches": nb, "trainable_M": round(nt / 1e6, 1)}
        print(f"[train] {args.tag} EP{ep}: train={tr:.3f} val={vl:.3f} peakVRAM={peak:.2f}GB {rec['epoch_s']:.0f}s", flush=True)
        open(MET, "a").write(json.dumps(rec) + "\n")
        torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(CKPT_DIR, f"{args.tag}_last.pth"))
        if ep % args.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "epoch": ep}, os.path.join(CKPT_DIR, f"{args.tag}_ep{ep}.pth"))
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "epoch": ep, "val_loss": vl}, os.path.join(CKPT_DIR, f"{args.tag}_best.pth"))
    print("[train] DONE")

if __name__ == "__main__":
    main()
