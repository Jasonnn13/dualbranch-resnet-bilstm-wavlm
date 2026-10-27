# Multi-run results: Hybrid vs WavLM (ASVspoof 2021 LA and DF)

Official ASVspoof 2021 EER (%) on the `eval` phase trials only (column 8 of the key files): LA 148,176 trials (14,816 bona fide), DF 533,928 trials (14,869 bona fide). Models are trained on ASVspoof 2019 LA only.

Every number below comes from the score files in [`scores/`](scores) via [`../stats/bootstrap.py`](../stats/bootstrap.py) (output: [`bootstrap_eer.csv`](bootstrap_eer.csv), [`bootstrap_delta.csv`](bootstrap_delta.csv)). The point EERs were also recomputed with the official ASVspoof 2021 eval package ([`../stats/official_check.py`](../stats/official_check.py)) and match to 4 decimals.

## Summary (3 runs per model)

| Model | LA EER | DF EER |
|---|---|---|
| WavLM | **3.90 ± 0.55** | 8.59 ± 0.43 |
| Hybrid | 4.41 ± 1.06 | **7.69 ± 0.79** |
| Mean paired ΔEER (Hybrid − WavLM, pp) | +0.51 ± 1.48 | −0.90 ± 1.16 |

Mean ± sample std (n − 1) over the unseeded run, seed 42 and seed 123.

## EER per run [95% CI]

| Run | Hybrid LA | WavLM LA | Hybrid DF | WavLM DF |
|---|---|---|---|---|
| unseeded | 5.58 [5.21–5.91] | 3.62 [3.47–3.77] | 7.63 [7.42–7.87]* | 8.31 [8.08–8.49] |
| seed 42 | 4.12 [3.92–4.33] | 3.54 [3.38–3.69] | 8.52 [8.35–8.66]* | 8.38 [8.17–8.54] |
| seed 123 | 3.53 [3.37–3.70] | 4.54 [4.39–4.70] | 6.93 [6.66–7.14] | 9.09 [8.86–9.31] |

\* Evaluated with test-time augmentation (see below).

## Paired ΔEER = Hybrid − WavLM (pp) [95% CI], negative = Hybrid better

| Run | LA | p | DF | p |
|---|---|---|---|---|
| unseeded | +1.96 [+1.64, +2.27] | 1.000 | −0.69 [−0.85, −0.47] | 0.0005 |
| seed 42 | +0.59 [+0.41, +0.78] | 1.000 | +0.14 [−0.00, +0.28] | 0.973 |
| seed 123 | −1.01 [−1.17, −0.86] | 0.0005 | −2.16 [−2.34, −2.01] | 0.0005 |

On DF, the Hybrid is significantly better in 2 of 3 runs (unseeded, seed 123); in seed 42 the difference is not significant. On LA, the Hybrid is significantly better only in seed 123; WavLM is significantly better in the unseeded run and seed 42.

## Method

- **Runs:** the unseeded run is the original run without a fixed seed (Hybrid: Kaggle `benedictusjason/hybrid-thesis` Version 82, scriptVersionId 322316772, the run reported in the first submission; WavLM: `alpretdexter/wavlm-thesis`). Seeds 42 and 123 fix every random source (weight init, data order, augmentation). All runs of a model use the same architecture and hyperparameters; the best checkpoint is chosen on the 2019 LA dev EER. Exception: the Hybrid unseeded run was scored with the weights after the last epoch (30), whose 2019 dev EER (0.1962%) equals that of the best checkpoint (epoch 26). Its log ([`logs/hybrid_unseeded.log`](logs/hybrid_unseeded.log)) holds the cell outputs taken from the version's `__results__.html`, because Kaggle no longer serves the raw session log for that version.
- **Bootstrap:** 95% stratified bootstrap, B = 2000, `numpy.random.default_rng(0)`. Bona fide and spoof trials are resampled separately. All runs of a track are scored on the same resamples, so ΔEER is paired. The CI is the 2.5th–97.5th percentile.
- **p:** one-sided, the share of resamples with Δ ≥ 0. 0.0005 (= 1/B) is the smallest value B = 2000 can show.
- **Scores:** `*_inverted.txt` files follow the official convention (higher = more bona fide). The unseeded files hold every trial (eval + progress + hidden); the seed files hold the `eval` trials only.

## Test-time augmentation (TTA)

No run uses TTA except the Hybrid unseeded and seed 42 DF evals, which average two passes (the second shifted by 1,000 samples). To check its effect, the Hybrid seed 123 DF run was re-scored with the same checkpoint with and without TTA:

| Hybrid seed 123 DF | EER | Score file |
|---|---|---|
| without TTA (used above) | 6.93 [6.66–7.14] | `scores/hybrid/seed123_df_notta/` |
| with TTA | 6.85 | `scores/hybrid/seed123/` (DF file) |

TTA changes the EER by 0.09 pp, well below the spread between runs (6.93–8.52 for Hybrid DF).

## Training details

| Run | Best epoch | 2019 dev EER | Notebook |
|---|---|---|---|
| Hybrid unseeded | 26 / 30 (scored with epoch 30 weights) | 0.1962% | [Kaggle V82](https://www.kaggle.com/code/benedictusjason/hybrid-thesis?scriptVersionId=322316772), log: [`hybrid_unseeded.log`](logs/hybrid_unseeded.log) |
| Hybrid seed 42 | 24 / 30 | 0.0785% | [`hybrid_seed42.ipynb`](../notebooks/revision/hybrid_seed42.ipynb) |
| Hybrid seed 123 | 12 (early stop at 22) | 0.1962% | [`hybrid_seed123.ipynb`](../notebooks/revision/hybrid_seed123.ipynb), DF without TTA: [`hybrid_seed123_df_notta.ipynb`](../notebooks/revision/hybrid_seed123_df_notta.ipynb) |
| WavLM seed 42 | 14 / 15 | 0.2960% | [`wavlm_seed42.ipynb`](../notebooks/revision/wavlm_seed42.ipynb), DF resume: [`wavlm_seed42_df_resume.ipynb`](../notebooks/revision/wavlm_seed42_df_resume.ipynb) |
| WavLM seed 123 | 15 / 15 | 0.4710% | [`wavlm_seed123.ipynb`](../notebooks/revision/wavlm_seed123.ipynb), DF: [`wavlm_seed123_df.ipynb`](../notebooks/revision/wavlm_seed123_df.ipynb) |

All runs used Kaggle T4 ×2. Kaggle stops a session after 12 h, so long DF evals were resumed in a second session that skipped files already scored.
