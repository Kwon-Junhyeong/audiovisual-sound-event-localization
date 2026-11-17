#!/usr/bin/env python3
"""Shared setup for the detector: paths, layout, dataset and model builders. Used by train and infer."""
import os, sys, glob, types

TASK3 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMPL = os.path.join(TASK3, "work")
REPO = os.environ.get("BASELINE_DIR", os.path.join(TASK3, "baseline"))
LABELS_BASE = os.environ.get("DCASE_LABELS", os.path.join(TASK3, "data", "labels_dev"))
DATA4CH = os.path.join(IMPL, "data_4ch")
CACHE = os.path.join(IMPL, "features_cache", "latent_fp16")
UPLAM = os.path.join(REPO, "UpLAM.pth")
WORK = os.path.join(IMPL, "experiments", "m2")
LAYOUT = os.path.join(WORK, "layout")
MIC = os.path.join(LAYOUT, "mic")
FRAMES = os.path.join(LAYOUT, "frames")

os.environ.setdefault("TORCH_HOME", os.path.join(IMPL, "torch_cache"))

def setup_paths(opt_einsum=True):
    """opt_einsum on for training with cached latents, off only for building the cache."""
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    pl = os.path.join(IMPL, "pylibs")
    if pl not in sys.path:
        sys.path.append(pl)
    if "mpl_toolkits.basemap" not in sys.modules:
        _bm = types.ModuleType("mpl_toolkits.basemap")
        _bm.Basemap = type("B", (), {"__init__": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))})
        sys.modules["mpl_toolkits.basemap"] = _bm
    import torch
    try:
        torch.backends.opt_einsum.enabled = bool(opt_einsum)
    except Exception:
        pass

def build_mic_layout():
    """Symlinks mic_base/<split_dir>/<seq>.wav into work/data_4ch."""
    os.makedirs(MIC, exist_ok=True)
    os.makedirs(FRAMES, exist_ok=True)
    for w in glob.glob(os.path.join(DATA4CH, "**", "*.wav"), recursive=True):
        split = os.path.basename(os.path.dirname(w))
        seq = os.path.basename(w)
        dd = os.path.join(MIC, split); os.makedirs(dd, exist_ok=True)
        dst = os.path.join(dd, seq)
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(w, dst)
        os.makedirs(os.path.join(FRAMES, split), exist_ok=True)

def make_extractor(device="cuda"):
    from cached_extractor import CachedExtractor
    return CachedExtractor(cache_root=CACHE, mic_base=MIC, num_bands=9,
                           fallback_uplam=UPLAM, device=device)

FRAMES_REAL = os.path.join(IMPL, "frames_dev")   # JPEG frames extracted with ffmpeg

def use_jpg_frames():
    """Use .jpg frames instead of .png (patches model.frame_path; the repository is not modified)."""
    import model as _m
    def _fp(seq_dir, seq_name, idx):
        return os.path.join(seq_dir, f"{seq_name}_{int(idx):04d}.jpg")
    _m.frame_path = _fp

def use_blindfold_missing():
    """Fills missing frames (12 dev-train-sony clips without video) with the ImageNet mean instead of noise,
    as in training, so no noise reaches the detector."""
    import model as _m, numpy as _np, torch as _t
    MEAN = _t.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    orig = _m.EnergySegDataset.load_frame_tensor
    def patched(self, seq_dir, seq_name, frame_idx, warn_missing=True):
        if not os.path.isfile(_m.frame_path(seq_dir, seq_name, frame_idx)):
            wav_path = _m.wav_path_from_seq_dir(seq_dir, self.frames_base, self.mic_base)
            ac = self.acoustic_extractor.get_frame_bands(wav_path, frame_idx)
            rgb = MEAN.expand(3, self.img_h, self.img_w).clone()
            return _t.cat([rgb, ac], dim=0)
        return orig(self, seq_dir, seq_name, frame_idx, warn_missing)
    _m.EnergySegDataset.load_frame_tensor = patched

def build_dataset(split_keyword, extractor, frames_per_epoch=None, augmentor=None, frames_base=None):
    from model import get_sequence_infos, EnergySegDataset
    fb = frames_base or FRAMES
    infos = get_sequence_infos(split_keyword, LABELS_BASE, fb)
    ds = EnergySegDataset(infos, frames_base=fb, mic_base=MIC,
                          acoustic_extractor=extractor, frames_per_epoch=frames_per_epoch,
                          img_w=360, img_h=180, dist_norm=500.0, augmentor=augmentor)
    return ds, infos

def build_model(device="cuda", freeze_encoder=True, freeze_mode=None):
    """freeze_mode: frozen (whole backbone body), baseline (organizers' schedule: layer2 always frozen,
    layer1 always trained, layer4 from epoch 3, layer3 from epoch 7) or full. freeze_encoder=False means full."""
    import torch
    from model import EnergyInstanceModel
    m = EnergyInstanceModel(num_classes=14, n_channels=12, img_w=360, img_h=180,
                            class_weights=torch.ones(14)).to(device)
    mode = freeze_mode or ("frozen" if freeze_encoder else "full")
    apply_freeze(m, mode, epoch=99)   # initial state (a large epoch applies the full schedule)
    return m

def apply_freeze(m, mode, epoch=99):
    """The organizers' apply_backbone_freeze. full: everything trained, frozen: body frozen, baseline: schedule."""
    body = m.det.backbone.body
    # always trained: dual stem, bn1, acoustic_norm, scale
    for mod in (body.conv1.conv1_rgb, body.conv1.conv1_audio, body.bn1, m.acoustic_norm):
        for p in mod.parameters():
            p.requires_grad_(True)
    body.conv1.rgb_scale.requires_grad_(True)
    body.conv1.audio_raw.requires_grad_(True)
    for p in m.det.backbone.fpn.parameters():
        p.requires_grad_(True)
    if mode == "full":
        for L in (body.layer1, body.layer2, body.layer3, body.layer4):
            for p in L.parameters():
                p.requires_grad_(True)
    elif mode == "frozen":
        for L in (body.layer1, body.layer2, body.layer3, body.layer4):
            for p in L.parameters():
                p.requires_grad_(False)
    else:  # baseline schedule
        for p in body.layer1.parameters(): p.requires_grad_(True)
        for p in body.layer2.parameters(): p.requires_grad_(False)   # always frozen
        for p in body.layer4.parameters(): p.requires_grad_(epoch >= 3)
        for p in body.layer3.parameters(): p.requires_grad_(epoch >= 7)

def freeze_bn(m):
    """BN statistics of frozen layers kept in eval mode (the organizers' freeze_backbone_bn)."""
    import torch.nn as nn
    body = m.det.backbone.body
    for L in (body.layer1, body.layer2, body.layer3, body.layer4):
        if not any(p.requires_grad for p in L.parameters()):
            for mm in L.modules():
                if isinstance(mm, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    mm.eval()

def build_optimizer(m, lr=1e-4):
    """Per-group learning rates with AdamW, as the organizers set them; trainable parameters only."""
    import torch.optim as optim
    body = m.det.backbone.body
    groups = [
        ("stem_rgb",  list(body.conv1.conv1_rgb.parameters()) + [body.conv1.rgb_scale], lr*0.2,  1e-4),
        ("stem_audio",list(body.conv1.conv1_audio.parameters()) + [body.conv1.audio_raw], lr*2.0, 0.0),
        ("norms",     list(body.bn1.parameters()) + list(m.acoustic_norm.parameters()), lr*1.0, 1e-4),
        ("bb_l1",     list(body.layer1.parameters()), lr*0.1,  1e-4),
        ("bb_l2",     list(body.layer2.parameters()), lr*0.05, 1e-4),
        ("bb_l3",     list(body.layer3.parameters()), lr*0.05, 1e-4),
        ("bb_l4",     list(body.layer4.parameters()), lr*0.1,  1e-4),
        ("fpn",       list(m.det.backbone.fpn.parameters()), lr*0.2, 1e-4),
        ("rpn",       list(m.det.rpn.parameters()), lr*0.1, 5e-3),
        ("box_head",  list(m.det.roi_heads.box_head.parameters()), lr*0.02, 5e-2),
        ("box_pred",  list(m.det.roi_heads.box_predictor.parameters()), lr*0.02, 5e-2),
        ("custom",    list(m.energy_head.parameters()) + list(m.energy_decoder.parameters()) +
                      list(m.distance_head.parameters()), lr*0.5, 1e-2),
    ]
    pg = [{"params": p, "lr": l, "weight_decay": wd, "name": n} for (n, p, l, wd) in groups if p]
    return optim.AdamW(pg)
