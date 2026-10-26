"""Recompute the pooled EER of every run with the official ASVspoof 2021 eval package.

Usage:
    git clone https://github.com/asvspoof-challenge/2021
    python stats/official_check.py --package 2021/eval-package --keys <keys dir>

The package expects a score for every trial, so files that only hold the 'eval' phase are
padded with score 0 for the progress/hidden trials; the 'eval' subset filter drops them.
Only the EER is checked (no min t-DCF, which needs the C012 file from asvspoof.org).
"""
import argparse
import contextlib
import io
import sys
import tempfile
from pathlib import Path

from eer_utils import RUNS, DF_OVERRIDES, SCORES

parser = argparse.ArgumentParser()
parser.add_argument("--package", required=True)
parser.add_argument("--keys", required=True)
args = parser.parse_args()

sys.path.insert(0, str(Path(args.package).resolve()))
import matplotlib, matplotlib.cm  # noqa: E401
if not hasattr(matplotlib.cm, "get_cmap"):  # removed in newer matplotlib, still used by the package
    matplotlib.cm.get_cmap = matplotlib.colormaps.get_cmap
import config, main  # noqa: E401

_init = config.ConfigLA.__init__
def _init_eer_only(self):
    _init(self)
    self.flag_tDCF = False
config.ConfigLA.__init__ = _init_eer_only


def padded(path, track, tmp):
    all_ids = [l.split()[1] for l in open(Path(args.keys) / track / "CM" / "trial_metadata.txt")]
    scores = dict(l.split() for l in open(path))
    out = Path(tmp) / path.name
    with open(out, "w") as f:
        for i in all_ids:
            f.write(f"{i} {scores.get(i, 0)}\n")
    return out


files = [((m, r), t, SCORES / p.format(track=t)) for (m, r), p in RUNS.items() for t in ("LA", "DF")]
files += [((m, r + " (no TTA)"), "DF", SCORES / p) for (m, r), p in DF_OVERRIDES.items()]

with tempfile.TemporaryDirectory() as tmp:
    for (model, run), track, path in files:
        with contextlib.redirect_stdout(io.StringIO()):
            _, eer = main.evaluation_API(str(padded(path, track, tmp)), track, "eval", label_dir=args.keys)
        print(f"{model:6s} {run:18s} {track}  pooled EER = {eer[-1, -1] * 100:.4f}%", flush=True)
