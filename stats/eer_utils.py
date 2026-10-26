import numpy as np
import pandas as pd
from pathlib import Path

SCORES = Path(__file__).resolve().parent.parent / "results" / "scores"

# (model, run) -> score file pattern; {track} is LA or DF.
# All files are *_inverted: higher score = more bona fide (official convention).
RUNS = {
    ("Hybrid", "run1"): "hybrid/run1/cm_scores_2021_{track}_eval_inverted.txt",
    ("Hybrid", "seed42"): "hybrid/seed42/cm_scores_2021_{track}_eval_inverted_hybrid_seed42.txt",
    ("Hybrid", "seed123"): "hybrid/seed123/cm_scores_2021_{track}_eval_inverted_hybrid_seed123.txt",
    ("WavLM", "run1"): "wavlm/run1/cm_scores_2021_{track}_eval_inverted.txt",
    ("WavLM", "seed42"): "wavlm/seed42/cm_scores_2021_{track}_eval_inverted_wavlm_seed42.txt",
    ("WavLM", "seed123"): "wavlm/seed123/cm_scores_2021_{track}_eval_inverted_wavlm_seed123.txt",
}
# Hybrid seed 123 DF re-scored without TTA (same checkpoint); used in the paper instead of the TTA file.
DF_OVERRIDES = {
    ("Hybrid", "seed123"): "hybrid/seed123_df_notta/cm_scores_2021_DF_eval_inverted_hybrid_seed123_notta.txt",
}


def score_path(model, run, track, use_notta=True):
    if track == "DF" and use_notta and (model, run) in DF_OVERRIDES:
        return SCORES / DF_OVERRIDES[(model, run)]
    return SCORES / RUNS[(model, run)].format(track=track)


def load_eval_keys(key_file):
    """Trials of the official 'eval' phase only (column 8). Label 1 = spoof, 0 = bona fide."""
    ids, labels = [], []
    with open(key_file) as f:
        for line in f:
            p = line.split()
            if p[7] == "eval":
                ids.append(p[1])
                labels.append(1 if p[5] == "spoof" else 0)
    return ids, np.array(labels)


def load_aligned_scores(path, ids):
    scores = pd.read_csv(path, sep=" ", header=None, names=["id", "score"], index_col="id")["score"]
    aligned = scores.reindex(ids).to_numpy()
    if np.isnan(aligned).any():
        raise ValueError(f"{path} is missing {np.isnan(aligned).sum()} eval trials")
    return aligned


def compute_eer(bona, spoof):
    """Sort-based EER, same as compute_eer in the official ASVspoof 2021 eval package."""
    n_b, n_s = len(bona), len(spoof)
    scores = np.concatenate([bona, spoof])
    is_spoof = np.concatenate([np.zeros(n_b), np.ones(n_s)])
    is_spoof = is_spoof[np.argsort(scores, kind="mergesort")]
    frr = np.concatenate([[0.0], np.cumsum(1 - is_spoof) / n_b])  # bona fide called spoof
    far = np.concatenate([[1.0], 1 - np.cumsum(is_spoof) / n_s])  # spoof accepted
    idx = np.argmin(np.abs(frr - far))
    return (frr[idx] + far[idx]) / 2
