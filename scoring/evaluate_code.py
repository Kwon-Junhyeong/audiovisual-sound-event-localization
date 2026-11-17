#!/usr/bin/env python3
"""Code scorer: runs evaluate.py of the baseline repository unmodified.

A compatibility shim is added only when the installed matplotlib lacks colormaps.get_cmap.
Same CLI as evaluate.py: --pred_dir --gt_dir --frames_base [--output_dir --iou_thresholds ...].
Default IoU thresholds 0.25, 0.50, 0.75, soft IoU, 100 x 50 rendering.
"""
import os, sys, runpy, matplotlib

# matplotlib compatibility shim (the repository code is not modified)
try:
    matplotlib.colormaps.get_cmap
except AttributeError:
    type(matplotlib.colormaps).get_cmap = lambda self, name=None: self[name]

REPO_EVAL = os.path.join(os.environ.get("BASELINE_DIR", os.path.join(os.path.dirname(__file__), "..", "baseline")), "evaluate.py")

sys.argv = ["evaluate.py"] + sys.argv[1:]
runpy.run_path(REPO_EVAL, run_name="__main__")
