# Feature Glossary

Documents every feature produced by `modules/continuous_feature_extractor.py`'s
`EnhancedContinuousFeatureExtractor` (Module 3 in the docs below) — the
extractor used by `scripts/extract_continuous_features.py` and everything
downstream of it (see [README.md](README.md) workflow 1). Written directly
from the current source, not from memory or an earlier spec — see the
"History" note at the bottom for why that distinction matters here. Also
includes a real ANOVA discriminability ranking of all 73 features (below).

**Total: 73 features** — 17 EDA, 18 cardiac, 33 respiratory, 5 multimodal.
Sampling rate 100 Hz; one row emitted per `feature_interval_s` (default 1s),
computed from a rolling 30s context window. Filters and rolling histories
persist for the entire session (never reset mid-session) — early-session
values are statistically thinner (fewer history samples feeding the rolling
stats) than later ones.

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
