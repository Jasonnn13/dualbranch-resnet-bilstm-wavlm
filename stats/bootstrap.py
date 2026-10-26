"""95% stratified, paired bootstrap of the EER for every run (reviewer point 1).

Usage:
    python stats/bootstrap.py --keys <dir with LA/CM/trial_metadata.txt and DF/CM/trial_metadata.txt>

Writes results/bootstrap_eer.csv and results/bootstrap_delta.csv.
Keys: ASVspoof 2021 LA/DF keys from https://www.asvspoof.org/index2021.html
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from eer_utils import RUNS, score_path, load_eval_keys, load_aligned_scores, compute_eer

N_BOOT = 2000
BOOT_SEED = 0
OUT = Path(__file__).resolve().parent.parent / "results"

parser = argparse.ArgumentParser()
parser.add_argument("--keys", required=True)
parser.add_argument("--tta", action="store_true", help="use the TTA file for Hybrid seed 123 DF")
args = parser.parse_args()

rng = np.random.default_rng(BOOT_SEED)
eer_rows, delta_rows = [], []

for track in ["LA", "DF"]:
    ids, labels = load_eval_keys(Path(args.keys) / track / "CM" / "trial_metadata.txt")
    bona_idx, spoof_idx = np.where(labels == 0)[0], np.where(labels == 1)[0]
    runs = {k: load_aligned_scores(score_path(*k, track, use_notta=not args.tta), ids) for k in RUNS}
    print(f"{track}: {len(ids)} eval trials ({len(bona_idx)} bona fide, {len(spoof_idx)} spoof)", flush=True)

    point = {k: compute_eer(s[bona_idx], s[spoof_idx]) for k, s in runs.items()}
    boot = {k: np.empty(N_BOOT) for k in runs}
    # Stratified + paired: every run is scored on the same resampled trials
    for b in range(N_BOOT):
        ib = rng.choice(bona_idx, size=len(bona_idx), replace=True)
        isp = rng.choice(spoof_idx, size=len(spoof_idx), replace=True)
        for k, s in runs.items():
            boot[k][b] = compute_eer(s[ib], s[isp])
        if (b + 1) % 200 == 0:
            print(f"  {track} {b + 1}/{N_BOOT}", flush=True)

    for (model, run), e in point.items():
        lo, hi = np.percentile(boot[(model, run)], [2.5, 97.5])
        eer_rows.append(dict(track=track, model=model, run=run, eer=100 * e, ci_low=100 * lo, ci_high=100 * hi))
    for run in sorted({r for _, r in runs}):
        d = 100 * (boot[("Hybrid", run)] - boot[("WavLM", run)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = max((d >= 0).mean(), 1 / N_BOOT)  # one-sided P(delta >= 0); floor 1/B
        delta_rows.append(dict(track=track, run=run,
                               delta=100 * (point[("Hybrid", run)] - point[("WavLM", run)]),
                               ci_low=lo, ci_high=hi, p=p))

eer = pd.DataFrame(eer_rows).round(4)
delta = pd.DataFrame(delta_rows).round(4)
eer.to_csv(OUT / "bootstrap_eer.csv", index=False)
delta.to_csv(OUT / "bootstrap_delta.csv", index=False)
print(eer.to_string(index=False))
print(delta.to_string(index=False))
print(eer.groupby(["track", "model"])["eer"].agg(["mean", "std"]).round(2))
