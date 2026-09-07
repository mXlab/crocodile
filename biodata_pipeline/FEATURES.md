# Feature Glossary

Two extractor families exist in this pipeline, producing two *different*
feature schemas (not two implementations of the same one) — see
[PIPELINE.md](../PIPELINE.md) for the full architecture discussion:

1. **Continuous schema (73 features)** — `modules/continuous_feature_extractor.py`'s
   `EnhancedContinuousFeatureExtractor` (Module 3 in the docs below), used by
   `scripts/extract_continuous_features.py`. The original extractor, online/
   real-time-safe by construction. Documented in full below, including a
   real ANOVA discriminability ranking of all 73 features.
2. **NeuroKit2 schema (53 features)** — `modules/batch_feature_extractor.py`
   (offline, NeuroKit2-based; the current pipeline default) and
   `modules/online_feature_extractor.py` (online/causal, computes the exact
   same 53 feature definitions from real-time-safe signal derivation — see
   its module docstring). Documented in its own section below, including its
   own ANOVA ranking and a feature-by-feature mapping back to the continuous
   schema (which features are shared, dropped, or new between the two).

Written directly from the current source, not from memory or an earlier
spec — see the "History" note at the bottom for why that distinction
matters here.

## Continuous schema — 73 features (`modules/continuous_feature_extractor.py`)

17 EDA, 18 cardiac, 33 respiratory, 5 multimodal. Sampling rate 100 Hz; one
row emitted per `feature_interval_s` (default 1s), computed from a rolling
30s context window. Filters and rolling histories persist for the entire
session (never reset mid-session) — early-session values are statistically
thinner (fewer history samples feeding the rolling stats) than later ones.

## Shared conventions

A handful of patterns repeat across most features; documented once here
instead of on every row below:

| Pattern | Meaning |
|---|---|
| `*_trend_10s` / `*_trend_full` | Linear regression slope (`np.polyfit(x, y, 1)[0]`) of the quantity over recent history (`_10s`) or the full session so far (`_full`). Sign = direction, magnitude = rate. |
| `*_coefficient_of_variation` | `std / mean * 100`. Dimensionless relative variability, comparable across subjects with different absolute baselines. |
| `*_level_indicator` | Discretized personal z-score: `-1` if the current value is more than 1 std below the subject's own rolling mean, `+1` if more than 1 std above, `0` otherwise. Session-relative, not an absolute physiological threshold. |
| `*_normalized_*` | Continuous z-score against the subject's own rolling history. |
| `*_scaled_*` | Same z-score, clipped and mapped into `[0, 1]` (contrast-stretched around ±2 std). |
| `*_rate_of_change` | Change in the quantity across its rolling history, normalized to a per-minute rate. |
| `*_recent_*` | Computed from roughly the last 5-10s only, vs. the full current window or full session history. |

## EDA — 17 features (`modules/continuous_feature_extractor.py:232`)

Input: `gsr` channel. Split into a tonic (SCL) and phasic (SCR) component.

**SCL (tonic, slow skin-conductance level)** — a very slow low-pass filter
(`alpha=0.005`) applied sample-by-sample, accumulated in a 60s rolling
history:

| Feature | Computation | Intent |
|---|---|---|
| `eda.scl_mean` | Mean of the low-pass-filtered signal, last 10s | Tonic arousal baseline |
| `eda.scl_median` | Median of the same 10s window | Robust to phasic leak-through |
| `eda.scl_std` | Std of the 10s window | How much the *tonic* level itself fluctuates |
| `eda.scl_range` | Peak-to-peak of the 10s window | |
| `eda.scl_trend_10s` | Slope over the 10s window | Rising vs. settling arousal |
| `eda.scl_trend_full` | Slope over the full session history (>100 samples) | Longer-horizon arousal trend |

**SCR (phasic, fast skin-conductance response)** — high-pass filtered
(`alpha=0.1`), rectified (negative deflections clipped to 0), peaks detected
via `scipy.find_peaks` (≥0.5s apart, height > `mean + 0.5·std`):

| Feature | Computation | Intent |
|---|---|---|
| `eda.scr_mean` | Mean rectified SCR signal, last 30s | Phasic activity baseline |
| `eda.scr_std` | Std of the rectified signal | |
| `eda.scr_event_count_10s` | Count of detected peaks in the last 10s | |
| `eda.scr_frequency` | That count ÷ 10s | Events/sec — Boucsein (2012): >0.5/s indicates high arousal |
| `eda.scr_onset_count_5s` | Count of peaks in the last 5s | |
| `eda.scr_mean_peak_amplitude` | Mean height of all peaks in the 30s window | |
| `eda.scr_recent_max_amplitude` | Max peak height among peaks in the last 5s | |
| `eda.scr_recent_mean_amplitude` | Mean peak height among peaks in the last 5s | |
| `eda.scr_max_rise_rate` | Largest peak's amplitude gain over the ~0.1s before it, per second | How abruptly a response fired |
| `eda.scr_event_clustering` | `1 - min(CV of inter-peak intervals, 1)` | Near 1 = regular/clustered bursts; near 0 = sparse/irregular. Benedek & Kaernbach (2010): clustering distinguishes sustained anxiety from single-event surprise |
| `eda.eda_instability_10s` | Variance of the first derivative of the SCR signal, last 10s | Moment-to-moment jaggedness of the phasic signal |

## Cardiac — 18 features (`modules/continuous_feature_extractor.py:380`)

Input: `heart` (PPG) channel, z-normalized over a 30s window. R-peaks via
`scipy.find_peaks` (≥0.4s apart → max 150 BPM).

**Heart rate** (from inter-beat intervals):

| Feature | Computation | Intent |
|---|---|---|
| `cardiac.hr_mean` | 60 / mean inter-beat interval (s) | Current heart rate (BPM) |
| `cardiac.hr_median` | 60 / median inter-beat interval | Robust HR estimate |
| `cardiac.hr_std` | Std of per-beat instantaneous HR in the window | |
| `cardiac.hr_trend_10s` | Slope of HR over ~last 10s of history | Kreibig (2010): HR acceleration in fear, anger, positive surprise |
| `cardiac.hr_trend_full` | Slope of HR over full session history (>10 samples) | |
| `cardiac.hr_delta_10s` | Latest HR − HR from ~10s ago | |
| `cardiac.hr_recent_max` | Max HR in the last 10s | |
| `cardiac.hr_recent_spike` | That max − session-median HR | How far the latest spike sits above baseline |
| `cardiac.hr_max_acceleration` | Largest \|sample-to-sample HR change\| across full session (>100 samples) | Sharpest HR jump seen so far this session |
| `cardiac.bpm_rate_of_change` | BPM change from oldest to newest of the last 50 beats, per minute | |
| `cardiac.bpm_coefficient_of_variation` | CV (%) of BPM across the last 50 beats | |

**HRV** (heart-rate variability, from R-R intervals — needs ≥3 in history):

| Feature | Computation | Intent |
|---|---|---|
| `cardiac.hrv_rmssd` | RMS of successive R-R interval differences | Classic vagal-tone / parasympathetic marker. Levenson (2014): high HRV = regulation or recovery |
| `cardiac.hrv_sdnn` | Std of R-R intervals | Overall HRV |
| `cardiac.hrv_pnn50` | % of successive R-R diffs exceeding 50ms | Vagal-tone marker |
| `cardiac.hrv_cv` | CV (%) of R-R intervals | |

**PPG waveform amplitude** (peak-to-trough of the pulse waveform itself —
peripheral blood-volume pulse strength, related to vasoconstriction /
sympathetic tone, distinct from heart *rate*):

| Feature | Computation | Intent |
|---|---|---|
| `cardiac.ppg_amplitude_level_indicator` | -1/0/+1 vs. rolling amplitude history | |
| `cardiac.ppg_amplitude_rate_of_change` | Change in pulse amplitude across rolling history, per minute | |
| `cardiac.ppg_amplitude_coefficient_of_variation` | CV (%) of pulse amplitude across rolling history | |

## Respiratory — 33 features (`modules/continuous_feature_extractor.py:591`)

Input: `respiration` channel, adaptively normalized to a continuously-updated
`[min, max]` that slowly tracks the subject's own observed range across the
whole session (not fixed physical units). Breaths detected as peaks
(inhalation apexes, ≥2s apart) and troughs (exhalation endpoints) via
`scipy.find_peaks`.

**General / normalization:**

| Feature | Computation | Intent |
|---|---|---|
| `respiratory.resp_normalized` | Mean signal rescaled to the adaptive `[0,1]` range | |
| `respiratory.resp_scaled` | Signal re-centered ±2 std of its own recent mean, clipped to `[0,1]` | Contrast-stretched variant |
| `respiratory.resp_currently_exhaling` | 1.0 if latest trough came after latest peak | Current breath phase |

**Breathing rate:**

| Feature | Computation | Intent |
|---|---|---|
| `respiratory.resp_rate_mean` | 60 / mean inter-breath interval | Breaths per minute |
| `respiratory.resp_rate_median` | 60 / median inter-breath interval | |
| `respiratory.resp_rate_std` | Std of per-breath instantaneous rate | |
| `respiratory.resp_normalized_rpm` | z-score of current RPM vs. rolling RPM history | |
| `respiratory.resp_scaled_rpm` | Same z-score, clipped to `[0,1]` | |
| `respiratory.resp_rpm_level_indicator` | -1/0/+1 vs. rolling RPM history (needs ≥5) | |
| `respiratory.resp_rate_trend_10s` | Slope of RPM over the last ~5 breaths | |
| `respiratory.resp_rate_trend_full` | Slope of RPM over the full rolling history | |
| `respiratory.resp_rpm_rate_of_change` | RPM change oldest→newest in history, per minute | |
| `respiratory.resp_rpm_coefficient_of_variation` | CV (%) of RPM across history | |

**Breath amplitude** (peak-to-trough depth of each breath):

| Feature | Computation | Intent |
|---|---|---|
| `respiratory.resp_amplitude_mean` | Mean breath depth, current window | Homma (2008): breath depth = relaxation marker |
| `respiratory.resp_amplitude_median` | Median breath depth, current window | |
| `respiratory.resp_amplitude_range` | Peak-to-peak of breath depths, current window | |
| `respiratory.resp_amplitude_std` | Std of breath depths, current window | |
| `respiratory.resp_normalized_amplitude` | z-score of latest breath depth vs. rolling history | |
| `respiratory.resp_scaled_amplitude` | Same z-score, clipped to `[0,1]` | |
| `respiratory.resp_amplitude_level_indicator` | -1/0/+1 vs. rolling history (needs ≥5) | |
| `respiratory.resp_amplitude_rate_of_change` | Breath-depth change across rolling history, per minute | |
| `respiratory.resp_amplitude_coefficient_of_variation` | CV (%) of breath depth across rolling history | |
| `respiratory.resp_amplitude_variability_10s` | Std of breath depths *within the current window only* | Shorter-horizon version of `resp_amplitude_std` |
| `respiratory.resp_amplitude_trend_10s` | Slope of breath depth across breaths in the current window | |
| `respiratory.resp_amplitude_trend_full` | Slope of breath depth across the full rolling history | |
| `respiratory.resp_amplitude_spike_5s` | Max of the last 3 breath depths − session mean breath depth | Overshoot magnitude (deep breath / gasp / sigh candidate) |

**Special events:**

| Feature | Computation | Intent |
|---|---|---|
| `respiratory.resp_exhale_ratio` | Mean fraction of each breath cycle spent exhaling (trough→next-peak duration ÷ full breath duration) | Boiten (1994): low variability + normal exhale ratio = controlled emotion |
| `respiratory.resp_sigh_count_5s` | Count of recent breaths deeper than `mean + 2·std` of rolling amplitude | Vlemincx (2013): sighs → sadness, relief, resignation |
| `respiratory.resp_sigh_frequency` | That count ÷ 5s | |
| `respiratory.resp_pause_detected_5s` | 1.0 if any recent inter-breath interval exceeded `mean + 2·std` | Masaoka (1997): respiratory pause → surprise/fear marker |
| `respiratory.resp_gasp_detected_5s` | 1.0 if any recent interval was abnormally short (`< mean - 1.5·std`, and >1s to exclude noise) | Rapid, gasp-like breath |
| `respiratory.resp_variability_cv` | CV (%) of breath-to-breath intervals, current window | Boiten (1994): fear vs. anger discriminator |
| `respiratory.resp_variability_cv_10s` | **Identical formula and identical input (`intervals_s`) as `resp_variability_cv` above** — only the minimum-sample gate differs (≥2 vs ≥3) | See caveat below — this is not actually a distinct 10s-windowed computation |

## Multimodal — 5 features (`modules/continuous_feature_extractor.py:944`)

Hand-set linear combinations of the features above, with hardcoded
normalization constants. Unlike most of the single-modality features (which
have direct literature support per-feature, cited above), these composites
are heuristic and **not individually validated against emotion labels** —
treat them as engineering approximations, not established psychophysiological
measures.

| Feature | Computation | Intent |
|---|---|---|
| `multimodal.arousal_index` | `clip((scl_norm + hr_norm + resp_norm)/3, 0, 1)` where `scl_norm=scl_mean/10`, `hr_norm=(hr_mean-60)/40`, `resp_norm=(resp_rate_mean-12)/8` | Higher EDA + HR + breathing rate → higher arousal. **See caveat below — found constant (zero-variance) on real reference data.** |
| `multimodal.valence_proxy` | `(hrv_norm + resp_var_norm)/2` where `hrv_norm=1-clip(hrv_cv/50,0,1)`, `resp_var_norm=1-clip(resp_variability_cv/50,0,1)` | Lower HR variability + more regular breathing → more positive valence. **Debatable assumption** — HRV is far better established in the literature as an arousal/regulation marker than a valence marker; this proxy isn't literature-grounded the way the single-modality features above are. |
| `multimodal.regulation_index` | Mean of three stability terms: `1-clip(\|scl_trend_10s\|*100,0,1)`, `1-clip(\|hr_trend_10s\|/10,0,1)`, `1-clip(\|resp_rate_trend_10s\|,0,1)` | High when EDA/HR/breathing rate are all flat — "how well-regulated" the current state is |
| `multimodal.instability_index` | `1 - regulation_index` | Pure complement — carries no information beyond `regulation_index` |
| `multimodal.event_detected` | OR of: SCR frequency > 0.5/s, HR spike > 10 BPM, sigh detected, pause detected | Flags that *something* physiologically notable just happened |

## ANOVA Ranking

All 73 features ranked by one-way ANOVA F-statistic (`sklearn.feature_selection.f_classif`),
run on the **reference (actress/Laurence) data only**, restricted to the
three emotions shared with the calibration subject used throughout this
pipeline's cross-subject alignment work (`anx`/`neu`/`sad` — see
[README.md](README.md)'s cross-subject alignment section). This is the same
convention `train_transformer.py`'s Ridge `--n-features` option and
`scripts/validate_heldout_emotion.py --n-features` use: discriminability is
judged in the actress' own space (the alignment target), never the incoming
subject's calibration data.

To regenerate: extract Laurence's combined features
(`emotion_biodata_laurence_main_*.csv`), filter to the emotions you care
about, and run ANOVA on that — either via this exact snippet or via
`modules.feature_analyzer.FeatureAnalyzer(df).compute_feature_importance(method='anova')`
(a more general tool already in this repo, see `scripts/analyze_features.py`).
A feature with `nan` F-score (constant within the filtered rows) sorts to
the **back** here, not the front — an earlier version of the feature-selection
code in both files above had a real bug where `NaN` scores sorted to the
front; fixed, see `scripts/train_transformer.py`.

| Rank | F-score | p-value | Feature |
|---|---|---|---|
| 1 | 1021.82 | 2.22e-218 | `cardiac.hr_recent_max` |
| 2 | 678.70 | 2.65e-171 | `eda.scl_mean` |
| 3 | 669.21 | 8.55e-170 | `eda.scl_median` |
| 4 | 530.98 | 3.81e-146 | `cardiac.hr_median` |
| 5 | 475.32 | 1.33e-135 | `cardiac.hr_trend_full` |
| 6 | 329.78 | 1.88e-104 | `cardiac.hr_mean` |
| 7 | 322.79 | 8.42e-103 | `cardiac.hr_recent_spike` |
| 8 | 268.65 | 1.95e-89 | `cardiac.ppg_amplitude_coefficient_of_variation` |
| 9 | 223.00 | 2.78e-77 | `cardiac.hr_max_acceleration` |
| 10 | 151.33 | 3.06e-56 | `respiratory.resp_amplitude_coefficient_of_variation` |
| 11 | 107.13 | 7.98e-42 | `respiratory.resp_rate_median` |
| 12 | 101.38 | 7.45e-40 | `cardiac.hrv_cv` |
| 13 | 93.54 | 3.95e-37 | `respiratory.resp_scaled` |
| 14 | 92.34 | 1.04e-36 | `eda.scr_std` |
| 15 | 61.01 | 2.44e-25 | `cardiac.hrv_rmssd` |
| 16 | 59.81 | 6.86e-25 | `multimodal.valence_proxy` |
| 17 | 55.66 | 2.55e-23 | `respiratory.resp_gasp_detected_5s` |
| 18 | 53.96 | 1.14e-22 | `cardiac.hrv_sdnn` |
| 19 | 50.32 | 2.81e-21 | `respiratory.resp_sigh_count_5s` |
| 20 | 50.32 | 2.81e-21 | `respiratory.resp_sigh_frequency` |
| 21 | 46.59 | 7.75e-20 | `respiratory.resp_normalized` |
| 22 | 45.99 | 1.32e-19 | `eda.scl_range` |
| 23 | 44.91 | 3.50e-19 | `respiratory.resp_amplitude_median` |
| 24 | 43.73 | 1.01e-18 | `respiratory.resp_amplitude_spike_5s` |
| 25 | 39.81 | 3.48e-17 | `eda.scl_std` |
| 26 | 38.20 | 1.49e-16 | `respiratory.resp_rpm_coefficient_of_variation` |
| 27 | 34.34 | 5.13e-15 | `multimodal.regulation_index` |
| 28 | 34.34 | 5.13e-15 | `multimodal.instability_index` |
| 29 | 32.55 | 2.67e-14 | `eda.scr_event_clustering` |
| 30 | 25.81 | 1.40e-11 | `respiratory.resp_variability_cv_10s` |
| 31 | 25.31 | 2.23e-11 | `cardiac.hr_std` |
| 32 | 23.76 | 9.68e-11 | `cardiac.hrv_pnn50` |
| 33 | 22.07 | 4.77e-10 | `respiratory.resp_variability_cv` |
| 34 | 21.16 | 1.13e-09 | `eda.scr_mean_peak_amplitude` |
| 35 | 19.34 | 6.36e-09 | `respiratory.resp_rate_std` |
| 36 | 18.71 | 1.16e-08 | `respiratory.resp_amplitude_mean` |
| 37 | 18.29 | 1.74e-08 | `respiratory.resp_rate_mean` |
| 38 | 17.80 | 2.76e-08 | `eda.scr_max_rise_rate` |
| 39 | 15.20 | 3.35e-07 | `respiratory.resp_pause_detected_5s` |
| 40 | 10.22 | 4.14e-05 | `cardiac.bpm_coefficient_of_variation` |
| 41 | 8.61 | 2.00e-04 | `respiratory.resp_rpm_level_indicator` |
| 42 | 7.86 | 4.18e-04 | `respiratory.resp_currently_exhaling` |
| 43 | 7.37 | 6.77e-04 | `respiratory.resp_normalized_rpm` |
| 44 | 6.75 | 1.24e-03 | `respiratory.resp_scaled_rpm` |
| 45 | 6.49 | 1.60e-03 | `respiratory.resp_rate_trend_10s` |
| 46 | 6.36 | 1.81e-03 | `eda.scr_mean` |
| 47 | 5.89 | 2.89e-03 | `cardiac.hr_delta_10s` |
| 48 | 5.17 | 5.90e-03 | `respiratory.resp_amplitude_level_indicator` |
| 49 | 4.76 | 8.85e-03 | `respiratory.resp_exhale_ratio` |
| 50 | 4.06 | 1.76e-02 | `cardiac.ppg_amplitude_level_indicator` |
| 51 | 3.63 | 2.70e-02 | `multimodal.event_detected` |
| 52 | 3.30 | 3.74e-02 | `eda.scr_event_count_10s` |
| 53 | 3.30 | 3.74e-02 | `eda.scr_frequency` |
| 54 | 3.07 | 4.71e-02 | `eda.scl_trend_10s` |
| 55 | 2.87 | 5.72e-02 | `respiratory.resp_scaled_amplitude` |
| 56 | 2.31 | 9.95e-02 | `respiratory.resp_rpm_rate_of_change` |
| 57 | 2.06 | 1.28e-01 | `eda.scr_recent_max_amplitude` |
| 58 | 1.86 | 1.56e-01 | `eda.scr_onset_count_5s` |
| 59 | 1.83 | 1.60e-01 | `respiratory.resp_rate_trend_full` |
| 60 | 1.79 | 1.68e-01 | `respiratory.resp_amplitude_trend_full` |
| 61 | 1.69 | 1.85e-01 | `respiratory.resp_amplitude_rate_of_change` |
| 62 | 1.23 | 2.94e-01 | `eda.scl_trend_full` |
| 63 | 1.10 | 3.32e-01 | `eda.scr_recent_mean_amplitude` |
| 64 | 0.87 | 4.18e-01 | `eda.eda_instability_10s` |
| 65 | 0.79 | 4.56e-01 | `cardiac.bpm_rate_of_change` |
| 66 | 0.64 | 5.27e-01 | `respiratory.resp_amplitude_range` |
| 67 | 0.61 | 5.42e-01 | `cardiac.ppg_amplitude_rate_of_change` |
| 68 | 0.59 | 5.57e-01 | `respiratory.resp_normalized_amplitude` |
| 69 | 0.55 | 5.75e-01 | `cardiac.hr_trend_10s` |
| 70 | 0.38 | 6.81e-01 | `respiratory.resp_amplitude_trend_10s` |
| 71 | 0.11 | 8.95e-01 | `respiratory.resp_amplitude_variability_10s` |
| 72 | 0.11 | 8.95e-01 | `respiratory.resp_amplitude_std` |
| 73 | constant | n/a | `multimodal.arousal_index` |

**Reading this ranking:**

- **The top is dominated by cardiac and EDA features** (`hr_recent_max`,
  `scl_mean`/`scl_median`, `hr_median`) — respiratory features don't appear
  until rank 10, and most of the bottom half (ranks ~60+, p > 0.05) is
  respiratory or trend/rate-of-change features that don't statistically
  distinguish `anx`/`neu`/`sad` at all in this dataset. Consistent with the
  heavy redundancy noted in [README.md](README.md)'s dimensionality
  discussion — many of the 73 features are non-discriminating noise for this
  particular emotion set.
- **`multimodal.arousal_index` is last, literally constant** (zero variance)
  in this reference data — see the caveat below.
- **`multimodal.valence_proxy` (rank 16) is comfortably in the useful range**
  despite its weaker theoretical grounding — see the caveat below.

## Caveats found while writing this glossary

- **`resp_variability_cv` and `resp_variability_cv_10s` compute the exact same
  value** whenever both are active (identical formula on the identical
  `intervals_s` array — the `_10s` suffix implies a distinct short-window
  computation that doesn't actually exist in the code). Not a crash risk, but
  worth knowing before treating them as two independent signals in an
  ANOVA/importance ranking — they'll always be perfectly correlated (see
  ranks 30 and 33 above: same underlying quantity, slightly different F-score
  only because of a different NaN-filtering sample count).
- **`multimodal.arousal_index` was found to be constant (F-score `-inf`,
  zero variance) on Laurence's `anx`/`neu`/`sad` reference data** (rank 73
  above). Given the hardcoded normalization constants in its formula (`/10`,
  `/40`, `/8`), this is plausibly a scale mismatch for this subject/session
  rather than a genuine absence of arousal variation — worth checking the raw
  `scl_mean`/`hr_mean`/`resp_rate_mean` ranges for this dataset against the
  assumed 60-100 BPM / 12-20 breaths-per-min ranges before trusting this
  feature anywhere.
- **`multimodal.valence_proxy` ranked 16th of 73** by ANOVA F-score despite
  resting on a weaker theoretical foundation than the arousal features above
  it — worth independent scrutiny before leaning on it for anything
  valence-related (e.g. the valence-arousal reframing discussed in
  [README.md](README.md)).

## NeuroKit2 schema — 53 features (`modules/batch_feature_extractor.py` / `modules/online_feature_extractor.py`)

17 EDA, 17 cardiac, 19 respiratory, no multimodal composites. Sampling rate
100 Hz, one row per second. Two computation methods produce this *exact*
same 53-feature schema (same names, same window sizes, same formulas) —
`BatchFeatureExtractor` derives the underlying signals (tonic/phasic, HR/
quality, breath amplitude/RVT/symmetry/phase) offline via NeuroKit2 (sees
the whole session at once); `OnlineFeatureExtractor` subclasses it and
overrides only that derivation step with real-time-safe causal algorithms.
See `online_feature_extractor.py`'s module docstring for exactly which
approximations that involves, and
[PIPELINE.md](../PIPELINE.md#can-an-online-extractor-match-neurokit2s-51-feature-schema)
for a feature-by-feature agreement measurement between the two (mean
correlation 0.52 — strong on smooth aggregate features, weak on anything
built from exact event timing).

Most `_10s`/`_60s`/`_5s`/`_full` window suffixes and `*_trend_*`/`*_cv_*`
patterns follow the same conventions as the continuous schema above (see
"Shared conventions"). Values are in raw sensor units, not physical units
(µS, ms, etc.) — normalization is deliberately deferred to modeling
(Stage 5), not baked into extraction (see `batch_feature_extractor.py`'s
module docstring).

**EDA** — tonic (`EDA_Tonic`) / phasic (`EDA_Phasic`) decomposition, plus a
per-SCR-onset event history (amplitude, rise time, recovery time):

| Feature | Computation | Intent |
|---|---|---|
| `eda.tonic_level` | Mean tonic component, last 10s | Slow arousal baseline |
| `eda.tonic_std_10s` / `tonic_range_10s` | Std / peak-to-peak of tonic, last 10s | |
| `eda.tonic_trend_10s` / `tonic_trend_full` | Slope of tonic, last 10s / full session (≥30s) | |
| `eda.phasic_mean_10s` / `phasic_std_10s` | Mean / std of phasic component, last 10s | Overall phasic activity level |
| `eda.instability_10s` | Variance of phasic's first derivative, last 10s | Moment-to-moment jaggedness |
| `eda.scr_rate_60s` / `scr_event_count_10s` | Count of SCR onsets, last 60s / 10s | |
| `eda.scr_recent_max_amplitude_5s` | Max onset amplitude among onsets in last 5s | |
| `eda.scr_event_clustering_60s` | `1 - min(CV of inter-onset intervals, 1)`, needs ≥3 onsets in 60s | Near 1 = clustered bursts |
| `eda.seconds_since_onset` | Time since the most recent SCR onset (session start if none yet) | **See caveat below** |
| `eda.last_onset_amplitude` / `last_onset_risetime` / `last_onset_recoverytime` | Amplitude / rise time / peak-to-50%-decay time of the most recent onset | Recovery time is a genuinely new measure vs. the continuous schema — NeuroKit2 provides it directly. **See caveat below** |
| `eda.mean_onset_amplitude_full` | Running mean of all onset amplitudes seen so far this session | **See caveat below** |

**Cardiac** — PPG rate/quality/pulse-amplitude, HRV from inter-beat intervals:

| Feature | Computation | Intent |
|---|---|---|
| `cardiac.hr_mean_10s` / `hr_median_10s` / `hr_std_10s` | Instantaneous PPG-derived HR stats, last 10s | |
| `cardiac.hr_trend_10s` / `hr_trend_full` | Slope of HR, last 10s / full session | |
| `cardiac.hr_delta_10s` | Latest HR − HR ~10s ago | |
| `cardiac.hr_recent_max_10s` / `hr_recent_spike_10s` | Max HR in last 10s / that max − session median so far | |
| `cardiac.hr_max_acceleration_full` | Largest \|sample-to-sample HR change\| across the full session so far (≥30s) | |
| `cardiac.quality_mean_10s` | Mean PPG signal-quality index, last 10s | NeuroKit2 template-matching SQI (batch) or an IBI-regularity proxy (online — the weakest approximation of the two extractors, see PIPELINE.md) |
| `cardiac.hrv_sdnn_60s` / `hrv_rmssd_60s` / `hrv_pnn50_60s` / `hrv_cv_60s` | Standard HRV metrics on inter-beat intervals, last 60s (≥3 beats) | Vagal tone / parasympathetic markers |
| `cardiac.bpm_cv_60s` | CV(%) of instantaneous HR, last 60s | |
| `cardiac.ppg_amplitude_mean_10s` / `ppg_amplitude_cv_10s` | Mean / CV(%) of peak-to-preceding-trough pulse amplitude, last 10s | Peripheral pulse strength, distinct from rate |

**Respiratory** — breath rate/amplitude/RVT/symmetry/phase from trough-
peak-trough cycles (troughs de-duplicated at ≥1.5s apart — see
`_respiratory_aggregate`'s docstring for why):

| Feature | Computation | Intent |
|---|---|---|
| `respiratory.rate_mean_10s` / `rate_median_10s` | Mean / median instantaneous breath rate, last 10s | Median added 2026-09 — see schema-unification note below |
| `respiratory.rate_std_10s` | Std of breath rate, last 10s | |
| `respiratory.rate_trend_10s` / `rate_trend_full` | Slope of breath rate, last 10s / full session | |
| `respiratory.amplitude_mean_10s` / `amplitude_median_10s` / `amplitude_std_10s` / `amplitude_range_10s` | Breath-depth stats, last 10s | |
| `respiratory.amplitude_cv_10s` | CV(%) of breath depth, last 10s | Added 2026-09 — see schema-unification note below |
| `respiratory.amplitude_spike_5s` | Max of last 3 breath depths − session mean depth so far (≥3 breaths) | Overshoot magnitude (deep breath / gasp / sigh candidate) |
| `respiratory.rvt_mean_10s` | Mean respiratory volume-per-time, last 10s | NeuroKit2 native measure, no continuous-schema equivalent |
| `respiratory.symmetry_risedecay_mean_10s` | Mean inhale-duration ÷ cycle-duration ratio, last 10s | NeuroKit2 native measure, no continuous-schema equivalent |
| `respiratory.exhale_ratio_10s` | Fraction of last 10s spent in exhale phase | |
| `respiratory.sigh_count_5s` / `sigh_frequency_5s` | Count (/ ÷5s) of breaths deeper than mean+2·std of depths so far, in last 5s (≥5 breaths) | |
| `respiratory.pause_detected_5s` | 1.0 if any recent (5s) inter-breath interval exceeded mean+2·std of intervals so far | |
| `respiratory.gasp_detected_5s` | 1.0 if any recent interval was abnormally short (<mean−1.5·std, >1s) | |
| `respiratory.cv_60s` | CV(%) of inter-breath intervals, last 60s | Consolidates the continuous schema's `resp_variability_cv`/`resp_variability_cv_10s` duplicate-formula bug into one clean feature |

### ANOVA ranking (53 features, same Laurence `anx`/`neu`/`sad` methodology as above)

| Rank | F-score | p-value | Feature |
|---|---|---|---|
| 1 | 442767.87 | 0.00e+00 | `eda.last_onset_recoverytime` — **artifact, see caveat below, do not trust this rank** |
| 2 | 5953.66 | 0.00e+00 | `eda.last_onset_risetime` — **artifact, see caveat below** |
| 3 | 4528.86 | 0.00e+00 | `eda.mean_onset_amplitude_full` — **artifact, see caveat below** |
| 4 | 1371.26 | 9.14e-256 | `eda.tonic_trend_full` |
| 5 | 671.98 | 3.09e-170 | `eda.tonic_level` |
| 6 | 581.80 | 3.14e-155 | `cardiac.hr_max_acceleration_full` |
| 7 | 360.46 | 1.63e-111 | `cardiac.hrv_cv_60s` |
| 8 | 355.92 | 1.73e-110 | `cardiac.bpm_cv_60s` |
| 9 | 275.92 | 2.71e-91 | `eda.seconds_since_onset` — same caveat, milder |
| 10 | 246.43 | 1.25e-83 | `cardiac.hrv_pnn50_60s` |
| 11 | 227.56 | 1.55e-78 | `cardiac.hr_trend_full` |
| 12 | 223.59 | 1.91e-77 | `eda.last_onset_amplitude` — same caveat, milder |
| 13 | 222.59 | 3.60e-77 | `cardiac.hrv_sdnn_60s` |
| 14 | 193.39 | 6.66e-69 | `cardiac.hrv_rmssd_60s` |
| 15 | 180.31 | 4.57e-65 | `respiratory.symmetry_risedecay_mean_10s` |
| 16 | 177.00 | 4.40e-64 | `cardiac.ppg_amplitude_mean_10s` |
| 17 | 117.34 | 2.89e-45 | `respiratory.exhale_ratio_10s` |
| 18 | 77.70 | 1.72e-31 | `cardiac.hr_std_10s` |
| 19 | 74.43 | 2.64e-30 | `respiratory.rate_trend_full` |
| 20 | 70.55 | 6.94e-29 | `eda.tonic_range_10s` |
| 21 | 70.13 | 9.96e-29 | `respiratory.amplitude_range_10s` |
| 22 | 69.66 | 1.47e-28 | `cardiac.hr_mean_10s` |
| 23 | 64.39 | 1.32e-26 | `cardiac.hr_median_10s` |
| 24 | 64.22 | 1.54e-26 | `eda.tonic_std_10s` |
| 25 | 63.71 | 2.37e-26 | `respiratory.rvt_mean_10s` |
| 26 | 62.34 | 7.77e-26 | `respiratory.amplitude_std_10s` |
| 27 | 60.45 | 3.94e-25 | `cardiac.quality_mean_10s` |
| 28 | 50.52 | 2.36e-21 | `respiratory.rate_median_10s` |
| 29 | 47.70 | 2.88e-20 | `cardiac.hr_recent_max_10s` |
| 30 | 47.08 | 4.98e-20 | `respiratory.rate_mean_10s` |
| 31 | 46.39 | 9.27e-20 | `cardiac.hr_recent_spike_10s` |
| 32 | 35.98 | 1.14e-15 | `respiratory.amplitude_spike_5s` |
| 33 | 23.78 | 9.46e-11 | `respiratory.amplitude_cv_10s` |
| 34 | 21.27 | 1.02e-09 | `respiratory.amplitude_mean_10s` |
| 35 | 20.71 | 1.72e-09 | `respiratory.sigh_count_5s` |
| 36 | 20.71 | 1.72e-09 | `respiratory.sigh_frequency_5s` |
| 37 | 20.37 | 2.38e-09 | `eda.phasic_std_10s` |
| 38 | 20.01 | 3.35e-09 | `respiratory.amplitude_median_10s` |
| 39 | 19.59 | 5.02e-09 | `cardiac.ppg_amplitude_cv_10s` |
| 40 | 7.15 | 8.35e-04 | `respiratory.cv_60s` |
| 41 | 6.99 | 9.80e-04 | `eda.instability_10s` |
| 42 | 5.82 | 3.09e-03 | `respiratory.rate_std_10s` |
| 43 | 3.36 | 3.51e-02 | `eda.phasic_mean_10s` |
| 44 | 2.61 | 7.43e-02 | `eda.tonic_trend_10s` |
| 45 | 1.64 | 1.94e-01 | `eda.scr_event_count_10s` |
| 46 | 1.64 | 1.94e-01 | `eda.scr_recent_max_amplitude_5s` |
| 47 | 1.64 | 1.94e-01 | `eda.scr_rate_60s` |
| 48 | 0.66 | 5.20e-01 | `respiratory.pause_detected_5s` |
| 49 | 0.36 | 6.98e-01 | `cardiac.hr_trend_10s` |
| 50 | 0.24 | 7.86e-01 | `cardiac.hr_delta_10s` |
| 51 | 0.12 | 8.83e-01 | `respiratory.rate_trend_10s` |
| 52 | constant | n/a | `respiratory.gasp_detected_5s` |
| 53 | constant | n/a | `eda.scr_event_clustering_60s` |

**Caveat found while building this ranking — the top-3 scores are an
artifact, not a real result.** `eda.last_onset_recoverytime`'s F-score
(442767.87 — a ~74x jump over rank 2, itself suspiciously large) comes from
near-zero *within-class* variance: it's a per-onset attribute that holds
its value constant until the next SCR onset fires, so during a long quiet
stretch that happens to fall entirely within one emotion segment, every row
in that segment gets the exact identical value (confirmed directly: `anx`
rows are *all* 667.42, std=0.0; `neu` rows are *all* 0.0, std=0.0). ANOVA
reads this as perfect separation, but it's actually separating *which
temporal segment of the session* the rows came from, not the emotion itself
— the same non-independent-samples trap the blocked-vs-LOSO methodology
work elsewhere in this pipeline (`PIPELINE.md`, `EXPERIMENTS.md`) exists to
guard against. `last_onset_risetime` and `mean_onset_amplitude_full` (ranks
2-3) show the identical pattern (confirmed: near-zero within-class std for
`anx`/`neu`). `last_onset_amplitude` (rank 12) and `seconds_since_onset`
(rank 9) are milder versions of the same issue — treat all five "last-onset-
history" features' ranks here as unreliable, not as evidence they
discriminate emotion. This doesn't affect the schema-unification decision
below (neither of the two ported features is in this family), but it does
mean a future feature-selection pass on this schema should exclude or
specially handle the per-onset-history family rather than trust this
ranking for them.

### Cross-reference: continuous (73) vs. NeuroKit2 (51→53) schemas

Comparing column names directly, only 4 features match exactly
(`cardiac.hr_delta_10s`, `cardiac.hr_trend_10s`, `cardiac.hr_trend_full`,
`eda.scr_event_count_10s`) — but many more are the same underlying quantity
renamed when this schema was designed. Cross-referencing both feature lists
against the continuous schema's ANOVA ranking above:

- **41 of 73 continuous features have a NeuroKit2-schema counterpart**
  (renamed, sometimes with an explicit window size added — e.g.
  `eda.scl_mean`→`eda.tonic_level`, `cardiac.hrv_rmssd`→`cardiac.hrv_rmssd_60s`).
- **32 were dropped**: mostly respiratory `*_normalized_*`/`*_scaled_*`/
  `*_level_indicator` variants and all 5 `multimodal.*` composites (already
  flagged above as unvalidated / one literally constant).
- **10 are new to the NeuroKit2 schema**: mostly the EDA onset-history
  family (`last_onset_*`, `seconds_since_onset`, `mean_onset_amplitude_full`
  — see the ANOVA caveat above before trusting these) plus
  `respiratory.rvt_mean_10s` and `respiratory.symmetry_risedecay_mean_10s`.

**Schema-unification decision (2026-09):** of the 32 dropped continuous
features, `resp_rate_median` (old rank 11/73) and
`resp_amplitude_coefficient_of_variation` (old rank 10/73) scored highly
enough and were distinct enough from what's already kept to test. Ported as
`respiratory.rate_median_10s` and `respiratory.amplitude_cv_10s` (both
tables above). Re-ranked in the new 53-feature ANOVA above at 28/53 and
33/53 respectively — still solidly significant, confirming the old ranking
held up in the new representation. But retraining Stage 5 end-to-end showed
**no measurable R² improvement** (0.448 vs. 0.457, within fold noise) —
kept anyway since they're legitimate and cheap, not because they proved
useful for the regression task. Full discussion:
[PIPELINE.md](../PIPELINE.md#feature-schema-unification-porting-value-from-the-73-feature-set).
Other high scorers were considered and rejected: `eda.scl_median` likely
just remeasures the already-kept `tonic_level`; `resp_scaled`/
`resp_normalized` would reintroduce pre-normalized features this schema
deliberately avoids (Stage 5 already rescales everything downstream).

## History

An earlier report, `docs/implementation/Feature_Extraction_Validation_Report.md`
(2026-02-06), documents a 54-feature version of this pipeline (13 cardiac, 17
EDA, 19 respiratory, 5 multimodal) with real literature citations for a subset
of features — several of those citations are reused above. It predates the 18
cardiac / 33 respiratory feature set implemented today; treat its narrative
interpretation sections as historical context, not a current feature count.

This file originally lived at `docs/module_guides/module3_feature_extractor.md`
(the path `docs/implementation/system_architecture.md` referenced as already
existing and "✅ DONE" before it was actually written) and was moved here, to
`FEATURES.md`, for visibility alongside `README.md`; `system_architecture.md`
has been updated to point here.
