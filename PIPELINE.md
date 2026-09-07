# Crocodile Pipeline: How the Pieces Fit Together

This is the map. For implementation depth on any one piece, follow the links —
don't duplicate them here.

## The goal

Crocodile is an interactive installation: a user's physiological signals
(heart rate, EDA, respiration) drive real-time generation of a face — a method
actress' avatar — via a trained StyleGAN2 model. The user's
biodata becomes the avatar's emotional state.

This requires a **runtime pipeline** that turns a live user's biodata signals 
into a (generated) face of the actress in the installation:

```
  [raw user biodata] ── b_u ──▶ [feature extraction] ── x_u ──▶ [user-actress alignment] 
     ── x_a ──▶ [actress-features-to-W regressor] ── W ──▶ StyleGAN2 ────▶ face
```

In order to achieve this, we need an **offline preprocessing pipeline** to build training
data and models from the actress' own recordings. This pipeline consists in two parts:

1. biodata_pipeline:
  * Converts raw biodata to features ``[feature extraction]``
  * Performs user-actress alignment ``[user-actress alignment]``
2. latent_pipeline: 
  * Creates a regressor that converts the actress biodata features to latent space ``[actress-features-to-W regressor]``
  * Provides an encoder that converts an image to latent space ``[actress-image-to-W encoder]``

```
  biodata_pipeline:  raw biodata CSVs ──▶ [feature extraction]

  latent_pipeline:   video frames ──▶ [actress-image-to-W encoder] ──▶ biodata_w_dataset.csv
                                                                             │
                     [actress-features-to-W regressor]  ◀────────────────────┘
```

`[user-actress alignment]` doesn't appear in this diagram: it isn't built from
the actress' recordings alone, it also needs a calibration recording from a
second person. That training happens offline too, just not as part of this
actress-only artifact chain — see `biodata_pipeline/README.md` workflow 3.

## Where each piece fits

### `biodata_pipeline/` — the physiological side

Turns raw 100Hz sensor recordings into feature vectors, and separately, learns
to map a *new* subject's physiological feature space onto the actress'.

- **Feature extraction**: raw `heart`/`gsr`/`respiration` CSV → 73 features/second
  (`continuous_features.csv`). This is what feeds `latent_pipeline` Stage 4 below.
- **Windowing + classification eval**: sanity-checks that emotions are separable
  in the extracted features (not on the pipeline's critical path to the avatar).
- **Cross-subject alignment**: trains a transformer (Ridge / Optimal Transport)
  mapping a new subject's feature space onto the actress' reference space. **This
  is the runtime pipeline's `[user-actress alignment]` step** — a user's biodata
  has to pass through this before anything downstream can make sense of it,
  since the actress-features-to-W regressor (once built) will only understand
  biodata shaped like the actress'.

→ Details: `biodata_pipeline/README.md`

### `latent_pipeline/` — the visual side

Inverts the actress' own video frames into the W-space of an *already-trained*,
frozen StyleGAN2 model (`models/finalModel_Crocodile.pkl`, 2048×2048, `w_dim=512`),
then attaches her synchronized biodata to those W vectors. This is the direct
implementation of `crocodile_pipeline_handoff.md`'s Stages 1–4 (that doc predates
`latent_pipeline` and used placeholder pool names/parameters — see the
correspondence table below).

| Stage | Script | Purpose |
|---|---|---|
| 1. Frame extraction | `scripts/stage1_extract.py` | Video → labeled frame pools |
| 2A. Synthetic pre-training | `scripts/stage2a_train_synthetic.py` | Warm-start: supervised MSE(encoder(image), W) on 10k generator-sampled (image, W) pairs — no LPIPS, no frozen-generator backprop, ~5min/epoch. Optional but recommended: real footage alone is too little data to train the encoder from random init. |
| 2B. Real-frame fine-tuning | `scripts/stage2b_train_frames.py` | **The actual encoder training.** Fine-tunes a CNN (face image → 512-dim W vector) on real actress frames, backpropagating through the frozen StyleGAN2 with LPIPS + MSE + diversity + temporal + emotion-contrastive losses. Does **not** auto-load 2A's weights — pass `--pretrained outputs/train_synthetic/best.pt` explicitly, or it starts from random init. |
| 3. Validation | `scripts/stage3_validate.py` | Compare CNN inversion vs. slow optimization-based inversion |
| 4. Biodata attachment | `scripts/stage4_assemble.py` | Encode all biodata-pool frames → join with `biodata_pipeline`'s `continuous_features.csv` → `data/biodata_w_dataset.csv` |
| 5. Biodata→W regressor | `scripts/stage5_train_regressor.py` | Fits biodata → W (MLP, blocked-shuffle CV) on `biodata_w_dataset.csv`. |
| 6. Offline user-to-latent test | `scripts/stage6_user_to_latent_test.py` | End-to-end test of a *non-actress* subject's pre-recorded biodata → aligned features → predicted W → StyleGAN2 face. Offline/batch only — see terminology note below. |

→ Details: `latent_pipeline/PLAN.md` (architecture, directory layout, corrected
parameters) and `crocodile_pipeline_handoff.md` (original CNN architecture, loss
function code, training-loop reference — still accurate for *how* Stage 2 works,
just not the source of truth for pool names or current status)

### `training_gan/` — legacy, superseded

`train_with_biodata.py` trains a **small conditional GAN from scratch** (128px,
custom architecture), conditioned directly on biodata — no StyleGAN2, no W-space,
no encoder. Its history goes back to 2019–2020, predating the StyleGAN2 model and
`latent_pipeline` by years. **Confirmed superseded** by the `latent_pipeline`
approach — not part of the active critical path. `CLAUDE.md` and
`training_gan/README.md` have been updated to reflect this.

## Terminology: old doc vs. current code

`crocodile_pipeline_handoff.md` predates the actual pool/session structure. When
reading it, translate:

| Handoff doc | Actual code |
|---|---|
| Pool A (4 videos + biodata) | `session_1S`, `session_2S`, `session_3S`, `session_4S` |
| Pool B (1 video, no biodata) | `session_1X` |
| Pool GAN (2500 images) | `gan_images` (from `Diverse/`) |
| 30fps video, 2160×2160 | 24fps, 500×500 `.mov` (originals were 2160×2160) |
| StyleGAN2 1024px, 14–18 layers | StyleGAN2 2048px, 20 layers (`num_ws`) |
| 20 hand-specified biodata features | 73 features from `EnhancedContinuousFeatureExtractor` |
| "participant" | "user" |

## Current status (as of this session — 2026-09-04)

Reconstructed from checkpoints/logs, not memory — verify before trusting if more
time has passed.

- **`biodata_pipeline`**: extraction, windowing/eval, and alignment (Ridge / OT
  class-conditional, best NPA 54.2%) all have working code and real output data.
  Everything since the last commit (2026-02-10) is now committed as of this
  session.
- **`latent_pipeline`**: Stages 0–5 (setup through dataset/dataloader) done.
  Stage 2A (`stage2a_train_synthetic.py`) completed 20/20 epochs on 2026-02-20 (val_mse
  0.0046). Stage 2B (`stage2b_train_frames.py`) ran through epoch 14/20 by 2026-02-25 but
  its checkpoints (`outputs/best.pt`/`latest.pt`) were only actually saved up to
  epoch 10 — a log-overwrite bug (`training_log.json` was replaced instead of
  appended to on each `--resume`) masked this until this session, when it was
  fixed. Stage 2B resume is now running from epoch 10, both locally and on the
  Alliance cluster Rorqual (`latent_pipeline/cluster/submit_train_rorqual.sh`) —
  these are two independent, diverging checkpoint lineages by design; compare
  `training_log.json` from both before picking one to continue from. Stage 3
  (validate) and Stage 4 (assemble) are coded but blocked on Stage 2B finishing.
- **Stage 5 (biodata→W regressor)**: done. `stage5_train_regressor.py` trains a
  small MLP (256,128) on the NeuroKit2 batch feature set (53 features as of
  the schema-unification work below, was 51; mean val R²=0.448, was 0.457 —
  within fold-to-fold noise, not a regression) under blocked-shuffle k-fold
  CV, saving `regressor.joblib` + a visual original/generated comparison grid.
- **Offline user-to-latent pipeline**: done and tested end-to-end on a subject
  other than the actress (Erin) — `apply_transformer.py` (alignment) +
  `stage6_user_to_latent_test.py` (regressor + StyleGAN2 render). This proves
  the four offline pieces (feature extraction, cross-subject alignment,
  regressor, StyleGAN2) compose correctly on non-actress biodata.
- **Runtime pipeline (live)**: still not built. "Runtime" is reserved
  specifically for continuously incoming sensor data — online feature
  extraction (each second computed from only that second's and earlier
  samples, no lookahead), per-sample alignment + regression + render running
  in a loop. The *offline* user-to-latent pipeline above proves the same
  four pieces work together, but reads a pre-recorded CSV as a batch;
  nothing yet wires them to live data.

### Feature extractor comparison: NeuroKit2 batch vs. continuous

`biodata_pipeline` has two feature extractors: `continuous_feature_extractor.py`
(the original — "continuous" because it streams sample-by-sample using only
past/current data, no lookahead, so it can eventually run in a real-time
installation) and `batch_feature_extractor.py` (added this session — NeuroKit2-
based, offline-only, sees each session's whole signal at once). Re-ran the
entire offline user-to-latent pipeline with the continuous extractor (73
features) in place of the NeuroKit2 batch extractor, keeping everything else
identical (same MLP architecture, same blocked-CV protocol, same OT
class-conditional alignment method, same subject — Erin) to isolate the
extractor as the only variable. Config: `latent_pipeline/configs/continuous_compare.yaml`;
outputs in `latent_pipeline/outputs/stage5_regressor_continuous/`.

| Extractor | Features | Stage 5 val R² (mean ± std) |
|---|---|---|
| NeuroKit2 batch | 51 | **0.457** ± 0.06 |
| Continuous (`continuous_feature_extractor.py`) | 73 | 0.313 ± 0.04 |

NeuroKit2 batch features win clearly, consistent with the earlier Ridge-only
comparison (0.272 vs 0.153) that originally motivated the switch — this
confirms the gap holds under the stronger MLP model too, not just Ridge.
Qualitatively, the continuous extractor's generated faces for Erin showed
*more* dramatic expression swings than the NeuroKit2 version, but that reads
as noise rather than signal: it's the extractor with the lower R², so the
wider swings are consistent with a less-constrained, less-accurate
regressor rather than better emotional expressiveness. The continuous
extractor remains the only real-time-compatible option and isn't going
away — this result just confirms NeuroKit2 batch processing is the right
choice whenever offline processing is available (training, and any
pre-recorded calibration step).

### Can an online extractor match NeuroKit2's 51-feature schema?

The comparison above answers "which existing extractor is better" but not
"how good could a real-time-compatible one be if it computed the *same* 51
features NeuroKit2 does, just online?" — the continuous extractor's 73
features aren't even the same features, so that comparison can't isolate
computation method from feature design. To isolate that, `batch_feature_extractor.py`
was refactored (behavior-preserving, verified bit-identical) to separate
"derive raw physiological signals" (tonic/phasic, HR/quality, breath
amplitude/RVT/symmetry/phase — currently via NeuroKit2, sees the whole
session at once) from "compute the 51 windowed features from those signals"
(already online-safe, since each second only reads its inputs up to that
point). `modules/online_feature_extractor.py`'s `OnlineFeatureExtractor`
subclasses it, overriding only the signal-derivation step with real-time-safe
algorithms (causal Butterworth filters via `sosfilt` with steady-state
initial conditions to avoid cold-start transients; a bespoke slope-threshold
SCR onset/peak detector calibrated from each session's own first 30s;
rolling z-scored peak/trough detection for cardiac and respiratory events,
adapted from `EnhancedContinuousFeatureExtractor`'s existing approach) — so
the 51 feature *definitions* are guaranteed identical between the two, and
`scripts/compare_extractors.py` measures purely how well the online signal
derivation approximates NeuroKit2's offline one, feature by feature, on the
same 4 actress sessions (Pearson correlation + normalized MAE, first 30s of
each session excluded as warm-up).

**Result: mean correlation 0.52 across 51 features, but split sharply by
feature type**, not evenly degraded:

- **Strong agreement (corr > 0.85, 9 features)**: EDA tonic-level and its
  trend/std/range, phasic std, `eda.instability_10s`, PPG amplitude
  mean/CV, `cardiac.hrv_pnn50_60s` — all smooth, continuous aggregate
  statistics that don't depend on precisely locating individual discrete
  events in time.
- **Weak agreement (corr < 0.4, 20 features)**: nearly everything built on
  exact event timing/counting -- SCR onset detection (`scr_event_count_10s`,
  `scr_rate_60s`, `last_onset_*`, `seconds_since_onset`), HR itself and its
  short-window derivatives (`hr_mean/median/trend/delta/recent_max_10s`),
  and most respiratory rate/variability/binary-event features
  (`rate_mean/std/trend_10s`, `cv_60s`, `exhale_ratio_10s`,
  `symmetry_risedecay_mean_10s`, sigh/pause/gasp detection).

**Interpretation**: a real-time-safe extractor recovers the *slow, smooth*
physiological trends well but a few-sample timing jitter in causal peak/
onset detection is enough to scramble short-window (5-10s) event-count and
rate features, even when the underlying signal trend is well recovered
(visible directly in `extractor_comparison_timeseries.png` -- e.g.
`cardiac.hr_mean_10s` tracks the same contour but noisier; respiratory rate
diverges more sharply during noisier stretches). This isn't a fixable
one-line bug so much as an inherent trade-off of causal vs. offline
detection -- NeuroKit2 itself needed the batch extractor's own trough
de-duplication fix for an analogous over-detection failure mode, so some of
this gap is fundamental to real-time biosignal event detection, not unique
to this prototype. Given the offline user-to-latent pipeline's own
NeuroKit2-batch-vs-continuous result above, the practical conclusion is the
same either way: batch/NeuroKit2 processing remains the right choice
whenever the pipeline can afford to be offline (training, and any
pre-recorded calibration step); this result specifically tells us which
*categories* of feature would need the most caution if the same 51-feature
schema were ever computed live -- a live regressor would be safer leaning
on the smooth/trend features shown here to survive causal computation well,
not on the event-count/rate ones that don't.

(The numbers above predate the schema-unification work just below, which
grew the schema to 53 features; re-running the comparison afterward gave
mean correlation 0.50 — same overall pattern, unchanged conclusion. The
newly-added `respiratory.amplitude_cv_10s` lands solidly in the weak group,
corr=0.02, expected since it's a ratio of two quantities the online
detector already struggles with individually; `rate_median_10s` is
comparable to the already-weak `rate_mean_10s`, corr=0.29.)

### Feature schema unification: porting value from the 73-feature set

The continuous extractor's 73 features and the NeuroKit2 schema's 51 (now
53, see below) aren't the same feature set under different names — comparing
column names directly, only 4 match exactly. But many more are the *same
underlying quantity*, renamed when the NeuroKit2 schema was designed (e.g.
`eda.scl_mean`→`eda.tonic_level`, `cardiac.hrv_rmssd`→`cardiac.hrv_rmssd_60s`).
Cross-referencing both feature lists against `FEATURES.md`'s existing ANOVA
ranking (computed on the continuous extractor's 73 features, Laurence's
`anx`/`neu`/`sad` data):

- **41 of the 73 continuous features have a NeuroKit2-schema counterpart**
  (mostly renamed, sometimes with an explicit window size added).
- **32 were dropped** in the NeuroKit2 redesign — mostly respiratory
  `*_normalized_*`/`*_scaled_*`/`*_level_indicator` variants and the 5
  heuristic `multimodal.*` composites (already flagged in `FEATURES.md` as
  unvalidated, one of them literally constant on real data).
- **10 are genuinely new** to the NeuroKit2 schema (mostly the EDA SCR
  onset-history family — `last_onset_amplitude/risetime/recoverytime`,
  `seconds_since_onset` — which NeuroKit2 provides directly and the
  continuous extractor never computed at all).

Of the 32 dropped, two ranked high enough in the old ANOVA (`resp_rate_median`
rank 11/73, F=107; `resp_amplitude_coefficient_of_variation` rank 10/73,
F=151) and were distinct enough from what's already kept (not just a
mean/std pair the model could already derive) to be worth testing. Ported
as `respiratory.rate_median_10s` and `respiratory.amplitude_cv_10s` into
`batch_feature_extractor.py`'s respiratory aggregate step — which
`OnlineFeatureExtractor` inherits automatically, so both extractors gained
the two features for free. (Other high scorers were rejected: `eda.scl_median`
is likely just measuring the same slow tonic signal as the already-kept
`tonic_level`; `resp_scaled`/`resp_normalized` would reintroduce
pre-normalized features the schema deliberately avoids, since Stage 5
already rescales everything downstream.)

**Validation, in two steps** (a feature's old rank doesn't guarantee it
still discriminates once recomputed against the NeuroKit2 tonic/phasic/HR/
breath-cycle signals, which are numerically different from the continuous
extractor's own filters):
1. Re-ran ANOVA on the extended 53-feature set (same Laurence
   `anx`/`neu`/`sad` methodology): both features remained solidly
   significant (`rate_median_10s` rank 28/53, F=50.5, p=2.4e-21;
   `amplitude_cv_10s` rank 33/53, F=23.8, p=9.5e-11) — the old ranking's
   hypothesis held up in the new representation.
2. Retrained Stage 5 (MLP, same config) on the 53-feature dataset: mean val
   R²=0.448 vs. 0.457 at 51 features — **no measurable improvement**, well
   within the ~0.05 fold-to-fold std. Despite being real, significant
   univariate discriminators, they added no marginal value to a model that
   already had 51 correlated features to work with -- the ANOVA-vs-actual-
   task gap this project has run into before (see the alignment-method
   comparisons in `EXPERIMENTS.md`).

**Decision: kept both anyway** — legitimate, cheap, harmless (difference is
noise, not regression), and documents the honest null result rather than
hiding it. Not evidence to keep porting further down the ANOVA list without
similarly validating on the actual task each time.

## Picking this back up

With Stages 1–6 and the offline user-to-latent pipeline all working, the
critical path forward is:
1. Widen the offline pipeline test beyond Erin's 3 emotion labels
   (anx/neu/sad) to a subject/recording covering more of the emotion range,
   to see how the regressor + alignment behave outside that overlap
2. Decide whether the cross-subject alignment step needs improvement — the
   generated faces for a new subject show less expression variation than
   Stage 5's own actress-held-out visual check, which stacks two lossy steps
   (OT alignment + regression) instead of one
3. Only once the offline chain is trusted does building the live "runtime
   pipeline" become a real question: wire the online/continuous feature
   extractor + `apply_transformer.py`'s per-sample equivalent + the
   regressor + StyleGAN2 into something that runs continuously on live
   user data

`training_gan/` is legacy and sits outside this critical path entirely.
