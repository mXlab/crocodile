# Crocodile Pipeline: How the Pieces Fit Together

This is the map. For implementation depth on any one piece, follow the links —
don't duplicate them here. For step-by-step install/test instructions for the
live pipeline specifically (including what's private and needs to come from a
teammate rather than this repo), see [INSTALL.md](INSTALL.md).

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
- **Offline user-to-latent pipeline**: done and tested end-to-end on a
  subject other than the actress — Erin's recording, aligned against the
  actress' (Laurence's) reference space — via `apply_transformer.py`
  (alignment) + `stage6_user_to_latent_test.py` (regressor + StyleGAN2
  render). This proves the four offline pieces (feature extraction,
  cross-subject alignment, regressor, StyleGAN2) compose correctly on
  non-actress biodata.
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

### Online extractor vs. continuous: a three-way comparison

Two comparisons existed before this one: NeuroKit2 batch vs. online
(feature-by-feature correlation, above) and NeuroKit2 batch vs. continuous
(Stage 5 R², earlier in this doc). Missing was the comparison that actually
matters for deciding the online extractor's future: **does its real-time-
safe approximation of the NeuroKit2 schema actually beat the continuous
extractor** — the only other real-time-safe option — **on the real task**,
not just on paper? Ran the same offline user-to-latent pipeline a third way:
`OnlineFeatureExtractor` output → `latent_pipeline/configs/online_compare.yaml`
→ Stage 4/5, identical MLP + blocked-CV setup as the other two.

| Extractor | Features | Real-time-safe? | Stage 5 val R² (mean ± std) |
|---|---|---|---|
| NeuroKit2 batch | 53 | No (offline) | **0.448** ± 0.02 |
| NeuroKit2 online | 53 | Yes | 0.417 ± 0.04 |
| Continuous | 73 | Yes | 0.313 ± n/a |

(NeuroKit2 online's number was 0.431 when first measured here; revised
down slightly to 0.417 after the live-readiness work below fixed a small
look-ahead leak in how `process_session()` computed its output — see that
section for why the revision is expected and small, not a sign something
broke.)

**The online extractor stays reasonably close to full offline NeuroKit2
quality (0.417 vs. 0.448) while being real-time-safe, and clearly beats the
continuous extractor by a wide margin (0.417 vs. 0.313).** This is a
meaningfully different conclusion than looking only at the feature-by-
feature correlation comparison above would suggest (mean correlation ~0.5,
split sharply by feature type) -- on the metric that actually matters, the
aggregate effect of the weak features is small enough that the online
extractor is a legitimate candidate to become the extractor the eventual
live installation actually uses, without the offline/online train-serve
mismatch that would come from training Stage 5 on NeuroKit2 batch features
and hoping the causal approximation is close enough at inference time.

**A real bug was caught and fixed getting here, not just tuning**: the first
online run only kept 2264/4296 rows (47% dropped to NaN) and crashed
rendering the visual grid, driven almost entirely by `respiratory.amplitude_spike_5s`
(1967 NaN rows). Root cause: `_respiratory_aggregate`'s sigh/spike threshold
computed `d_mean, d_std = depths_so_far.mean(), depths_so_far.std()` (plain,
not NaN-safe) over `breath_depths`, which for the online extractor legitimately
contains a leading NaN (amplitude is only known from a cycle's *closing*
trough onward — see `_respiratory_cycle_signals` — so the very first trough
in any session precedes the first known amplitude value). That one NaN
poisoned `d_mean`/`d_std` for the *entire rest of the session*, not just the
first few seconds. Fixed by switching to `np.nanmean`/`np.nanstd` in
`batch_feature_extractor.py` (verified as a no-op for the batch path itself,
since NeuroKit2's amplitude signal rarely has this gap) — NaN dropout fell
from 47% to ~1%, val R² rose from 0.340 to 0.431, and the visual grid
rendered cleanly. Worth remembering next time an online-extractor result
looks suspiciously weak: check for NaN propagation before concluding the
approximation itself is the problem.

**Caveat carried over from the feature-by-feature comparison above still
applies**: the online extractor's weakest features are still the
event-count/rate ones (SCR onsets, HR short-window derivatives, breath
rate/variability) — this three-way result says the *aggregate* effect on
the downstream task is small, not that every individual feature survived
causal computation equally well. If a future model change leans harder on
those specific weak features (e.g. explicit feature selection favoring
them), the online/batch gap could reopen.

### Live-readiness: making OnlineFeatureExtractor actually callable on live data

The three-way comparison above was measured by calling
`process_session(whole_recorded_session_df)` once per session — the whole
recording handed over in a single Python call. Asked directly ("if we send
live raw data to this, will it work?") and checked empirically rather than
assumed, the honest answer at the time was **no**, for two separate reasons,
both now fixed:

1. **No incremental API / unbounded per-call cost.** `process_session()`
   required the whole array up front, with no way to feed new samples as
   they arrive. The two components that did real forward-scanning work --
   the EDA SCR state machine and the peak/trough detector -- held no state
   between calls, so any attempt to call them repeatedly on a growing
   buffer would reprocess everything from scratch each time.
2. **A real "backdating" bug**, found by feeding the same extractor a 500s
   prefix and the full ~1030s session and diffing the overlapping rows --
   a genuinely causal system must produce identical output there, and it
   didn't (up to ~2.5 bpm on `cardiac.hr_delta_10s`). A peak/onset can only
   be confirmed ~0.2s after it physically occurs (correct, unavoidable
   latency), but once confirmed, its effect was attributed back to the
   peak's own timestamp -- letting a row's features benefit from data that
   arrived *after* that row's own moment, which a live query at that exact
   moment would not have had.

**Fix**: `online_feature_extractor.py` was rewritten around two new
persistent-state primitives -- `_CausalFilterState` (carries `sosfilt`'s
`zi` across calls) and `_PeakDetectorState`/`_ScrDetectorState` (carry
`last_idx`/state-machine variables across calls, so each new second's worth
of samples is processed exactly once, not re-scanned from session start
every time) -- plus a one-row extraction of `BatchFeatureExtractor`'s three
aggregate functions (`_eda_aggregate_one_row` etc., verified bit-identical
to the multi-row versions on real data before anything else changed) so a
new row's features can be computed without redoing every prior row. On top
of these, `OnlineFeatureExtractor` now exposes:

- **`calibrate(calibration_df)`**: primes filter state and the SCR
  amplitude threshold from a separate calibration recording, per the
  exhibition's per-visitor calibration period (see memory) -- instead of
  process_session()'s fallback of using the live session's own first 30s.
- **`push(chunk_df) -> list[dict]`**: feed new raw samples (any chunk
  size, no alignment requirement), get back zero or more newly-finalized
  feature rows. Each row is finalized using state that has seen precisely
  up to that row's own sample boundary and is never revisited afterward --
  this discipline is what fixes the backdating bug, not a special-cased
  timestamp correction.
- **`process_session()` is now implemented on top of `push()`**, not a
  separate code path -- offline replay and live use can never drift apart,
  and the fix applies to both. This was a deliberate design decision
  (confirmed before implementation): the alternative, keeping
  `process_session()` untouched and adding `push()`/`calibrate()`
  alongside it, would have been lower-risk but left two parallel
  implementations to keep in sync and left the backdating bug in the
  numbers used for training data.

**Verified, not assumed** (`scripts/test_online_causality.py`, all passing):
prefix-vs-full now gives exactly zero difference on overlapping rows (was
~2.5); feeding `push()` 37-sample chunks (deliberately not aligned to the
100-sample feature step) gives byte-identical output to `process_session()`
on the same data; the fixed detector components' per-row cost stays roughly
flat across a session (0.17ms → 0.37ms first-10-rows vs. last-10-rows on a
~1030s session, 2.1x growth) rather than growing with session length.
End-to-end `push()` cost does still grow somewhat with session length
(measured up to ~46ms/row by the end of a ~1788s session) because several
features are inherently full-history quantities (`*_trend_full`,
`hr_max_acceleration_full`, the zero-order-hold reconstruction of hr/
quality/amplitude arrays) that this work deliberately did not optimize
further -- acceptable given this project's realistic session lengths are
minutes, not hours, and 46ms is still far under the 1Hz feature-interval
budget (see "out of scope" below).

**Explicitly out of scope for this work** (so it doesn't sprawl further):
wiring `push()` to actual live sensor/OSC/serial input (this only makes the
extractor *class* itself callable incrementally -- a real acquisition loop
is separate future work); bounded-memory ring buffers (growing arrays are
fine at exhibition session lengths); adaptive re-calibration mid-session.

## Live pipeline: biodata (OSC) → W (OSC) → Autolume

The runtime pipeline mentioned throughout this doc is now built, in its own
top-level **`live_pipeline/`** folder — separate from `biodata_pipeline/`
and `latent_pipeline/`, which hold reusable modules (`modules/`, `models/`)
plus their own offline scripts/experiments (`scripts/`). `live_pipeline/`'s
scripts import those modules as a library (`sys.path`-inserting the
relevant pipeline directory, same pattern those pipelines' own scripts
already use for their `modules/`/`models/`) rather than living inside
either one, so the "live runtime" layer isn't tangled up with training/
analysis code. It has no venv of its own — each script still runs under
whichever of the two existing venvs it needs (see below), via wrapper
scripts (`run_live.sh`, `run_replay.sh`, `run_debug_viewer.sh`) that pin
the right interpreter so you don't have to remember which is which.

One extraction happened as part of this split: `live_pipeline.py` needs
`load_transformer()`, which used to live in `biodata_pipeline/scripts/
train_transformer.py` (a script, not a module) — moved into a new
`biodata_pipeline/modules/alignment_transformer.py` (the transformer
classes + `load_transformer`), leaving `train_transformer.py` as just its
training CLI. `apply_transformer.py`, `validate_transformer.py`, and
`validate_heldout_emotion.py` were updated to import from the new module
too.

### OSC ports

All default to `127.0.0.1`; override per-script with `--*-host`/`--*-port`
flags if you need to run on a different machine or dodge a collision.

| Port | Direction | Used by | Carries |
|---|---|---|---|
| **9000** | `live_pipeline.py` listens; `session_control.py`/control panel and `replay_biodata_as_osc.py` both send | Session control (`/crocodile/session/*`, `/crocodile/calibration/*`, `/crocodile/live/*`) **and** raw biodata (`/crocodile/biodata`) — two different message streams sharing one port, disambiguated only by OSC address, not by port |
| **9001** | `live_pipeline.py` sends; `crocodile-control-module.js` (the latent controller) listens | Three disambiguated-by-address uses: session-status broadcasts (`/crocodile/session/status` → `[phase, session_id]`), fired after every state transition; the visitor's live W vector (`/crocodile/latent/user` → 512 floats, from `live_pipeline.py`); and the control panel's own widget-control addresses, namespaced under `/crocodile/control/` (`/crocodile/control/grid/select`, `/crocodile/control/transition/*`, `/crocodile/control/mix/amount`, `/crocodile/control/noise/amount`) |
| **1338** | `crocodile-control-module.js` (the latent controller) sends; `latent_osc_debug_viewer.py`/`latent_osc_debug_receiver.py`/Autolume listen | W output (`/crocodile/latent/final` → 512 floats). Not really ours to renumber — 1338 is Autolume's own default OSC-input port |
| **8090** | Open Stage Control's HTTP server (not OSC) | Browser UI for the control panel | Bumped from Open Stage Control's own default `8080` to dodge a local port collision (see `run_control_panel.sh`) |

The one non-obvious part of this scheme is **9001's now-heavy overload** —
session-status broadcasts, the visitor's live W vector, and the control
panel's own widget messages all arrive on the same port, separated only by
OSC address (the same disambiguation-by-address trick 9000 already uses for
control vs. raw biodata). 1338 and 8090 are both externally constrained
(Autolume's default, and an unrelated app already on 8080) rather than
chosen for memorability; 9000/9001 are the two actually "ours," kept
sequential on purpose.

- **`live_pipeline.py`** — the core program. Runs under
  `biodata_pipeline/venv` (needs `OnlineFeatureExtractor` + the alignment
  transformer; no torch). A **persistent OSC server**, not a one-shot
  script — loads the alignment transformer + Stage 5 regressor once, then
  handles any number of visitors in sequence through an explicit session
  state machine (`SessionState`, one instance for the process's lifetime):
  ```
  IDLE --session/start--> READY --calibration/start--> CALIBRATING
                            |                                |
                            |                       calibration/stop
                            |                                v
                            |                           CALIBRATED
                            |                                |
                            +-----------live/start------------+
                                         |
                                         v
                                       LIVE --calibration/recalibrate--> (stays LIVE)
                                         |
                              (any state) session/end
                                         v
                                       IDLE
  ```
  `session/start` creates a fresh `OnlineFeatureExtractor` per visitor (no
  state leaks between sessions) and, if `--calibration-csv` was given at
  server startup, immediately primes it from that file (`extractor.
  calibrate()`) before returning to READY — for when a suitable
  calibration recording already exists (e.g. a generic baseline, or
  reusing a visitor's own earlier recording) and the live calibration
  phase can be skipped entirely. The live `calibration/start`/`stop` phase
  is still available afterward too, and composes cleanly on top if used —
  `calibrate()` is just a `push()` call with the rows discarded, same as
  what the live phase does, so CSV-priming and live-priming stack rather
  than conflict. From `calibration/start` onward (or from `live/start`
  directly, if the operator skips the live calibration phase entirely),
  every incoming biodata sample is pushed through `extractor.push()`
  continuously for the rest of the session (through CALIBRATING,
  CALIBRATED, and LIVE) — only what happens to the *returned* finalized
  rows differs (discarded until LIVE, then aligned + regressed + sent as
  W). Biodata arriving in IDLE/READY is ignored. An invalid transition
  (e.g. `live/start` while IDLE) is logged and ignored, never crashes the
  server — see the module docstring for the full OSC control-address
  table and the reasoning for keeping everything (biodata + control) on
  one synchronous `BlockingOSCUDPServer` (serializes control and data
  handling for free, no locks needed). Sends each finalized row's
  512-float W vector to the **latent controller**
  (`crocodile-control-module.js`, via `/crocodile/latent/user` on port
  9001) rather than to Autolume directly — the controller composites the
  visitor's vector with the operator's emotion selection (mix, noise,
  transitions) before forwarding a single blended vector on to
  **Autolume** — a separate live StyleGAN performance app
  (`/home/tats/Documents/workspace/autolume`, not part of this repo;
  distinct from `stylegan_Autolume`, the bare synthesis library
  `latent_pipeline` imports directly). Autolume owns rendering and display
  entirely — neither this script nor the controller ever touches
  StyleGAN2. Autolume's latent-vector OSC handler expects exactly 512
  floats and treats them as W directly only if its "project" checkbox is
  left unchecked (it defaults to Z-space with an optional Z→W mapping
  step, which our output must bypass) — that instruction now applies to
  the controller's output (`/crocodile/latent/final` on port 1338), not to
  `live_pipeline.py` directly. `live_pipeline.py` also broadcasts
  `[state, session_id]` to a separate `/crocodile/session/status` address
  after every transition, for an operator control surface to confirm
  actual server state.
- **`session_control.py`** (run via `run_session_control.sh`, same
  interpreter-pinning pattern as the other wrapper scripts) — sends one
  session-control OSC message and exits (`--start-session [ID]`,
  `--start-calibration`, `--set-calibration-emotion [LABEL]`,
  `--stop-calibration`, `--start-live`, `--recalibrate`,
  `--refit-transformer`, `--end-session`). A CLI stand-in for a real
  control surface, and useful for scripted test sequences.
- **`crocodile-control-panel.json`** + **`run_control_panel.sh`** — an
  actual GUI control surface, built with
  [Open Stage Control](https://openstagecontrol.ammd.net/) (already used
  for this project's other installations, e.g. Xenolalia's own OSC panel).
  Three tabs: **Session** — one button per session-control message (`Start
  Calibration`, `Stop Calibration`, `Start Live`, `Recalibrate`, `End
  Session`), a text field for the optional session ID (sends
  `/crocodile/session/start` on Enter), and a status display bound to
  `/crocodile/session/status`; **Emotion Grid** — a thumbnail matrix (one
  button per emotion/image cell, headed by a row of emotion-name labels)
  wired to `/crocodile/control/grid/select`, plus transition mode/duration/
  start-stop controls and a target display; **Mixing** — actress/visitor
  mix and noise-amount faders (`/crocodile/control/mix/amount`,
  `/crocodile/control/noise/amount`). The Emotion Grid and Mixing
  tabs talk to `crocodile-control-module.js` (the latent controller, on
  port 9001), not to `live_pipeline.py`. `run_control_panel.sh` launches
  it pre-wired to `live_pipeline.py`'s default ports (sends to 9000,
  listens on 9001) and loads the latent controller's custom module — run
  both and the panel's buttons drive the session state machine directly,
  while the grid/mixing tabs drive the latent controller. Verified in
  this session: the session file loads without error and every widget
  renders with the correct label/position/wiring (confirmed via a
  headless Chromium screenshot); real click-through wasn't independently
  confirmed in that same headless pass (a GPU/compositor limitation of
  the sandboxed test environment, not a property of the file) — the
  button interaction pattern (`mode: momentary`) is otherwise standard,
  documented Open Stage Control behavior also used elsewhere in this
  project's other panels.
- **`replay_biodata_as_osc.py`** — runs under `biodata_pipeline/venv`.
  Sends a recorded raw biodata CSV out as OSC at real-time (or faster)
  pace, standing in for real sensor hardware. No hardware/OSC protocol for
  how biodata will actually arrive live is confirmed anywhere in this repo
  (legacy Arduino OSC code was removed in the newest firmware iteration in
  favor of serial-only, and the exact serial format isn't documented
  either) — `live_pipeline.py`'s input protocol is this project's own
  design, documented in both scripts' docstrings, for a future hardware
  bridge to match. Also logs the recording's `emotion`/`feeling_it`
  ground-truth columns to stdout whenever they change, if present, for
  eyeballing the pipeline's output against what the subject was actually
  feeling during testing.
- **`latent_osc_debug_viewer.py`** — optional, separate process. Runs under
  `latent_pipeline/.venv` (needs torch/StyleGAN2 — the only script here
  that does, and the only one needing the private StyleGAN2 checkpoint).
  Listens to the same W-over-OSC stream and renders it locally via this
  project's own StyleGAN2 code, for visual sanity-checking without
  Autolume running.
- **`latent_osc_debug_receiver.py`** — optional, separate process. Runs under
  `biodata_pipeline/venv` — no torch/StyleGAN2/checkpoint needed at all,
  unlike the viewer above. Listens to the same W-over-OSC stream and just
  reports receipt stats (count, rate, vector norm), for confirming the OSC
  plumbing works before the private StyleGAN2 checkpoint is even available
  (see INSTALL.md).
- **`generate_synthetic_biodata.py`** — optional. Runs under
  `biodata_pipeline/venv`. Uses NeuroKit2's own signal simulators
  (`ppg_simulate`/`eda_simulate`/`rsp_simulate`) to produce a CSV with the
  same columns `replay_biodata_as_osc.py` expects, as a stand-in when no
  real biodata recording is available (real recordings, like the StyleGAN2
  checkpoint, aren't shared on GitHub — see INSTALL.md). The resulting W
  vectors aren't physiologically meaningful (no real subject/emotion signal
  underlies them) — this only tests that data flows correctly end-to-end.
- **`live_pipeline/data/`** — reusable test fixtures, all sliced from
  Erin's raw recording (`biodata_pipeline/data/raw/
  emotion_biodata_erin_2026-02-09_labeled.csv`) so calibration and "live"
  replay never reuse the same raw samples (see bug 1 below for why that
  matters): `erin_calibration_segment.csv`/`erin_live_segment.csv` (first
  60s / the non-overlapping remainder — general-purpose), and the
  scenario-specific slices used in "Three usage scenarios" above
  (`erin_calibration_long_neu.csv`/`erin_live_after_long_neu.csv`,
  `erin_calibration_anx.csv`, `erin_calibration_sad.csv`,
  `erin_live_scenario3.csv`). Not tracked in git (real biodata) — generate
  synthetic equivalents with `generate_synthetic_biodata.py`, or re-slice
  from the raw recording, if you don't have them.

**Recording sessions (`--record-dir`)**: optional. If given,
`session/start` opens `{record-dir}/{session_id}.csv` and appends every
biodata sample from `calibration/start` onward (`heart, gsr, respiration,
session_phase, timestamp` — `session_phase` is `calib` or `live`,
`timestamp` is wall-clock, both ignored by every existing offline tool,
which only reads `heart`/`gsr`/`respiration`). Row-by-row flush, so a
crash mid-session doesn't lose the recording. Directly reusable by the
existing offline toolchain (`extract_continuous_features_batch.py`,
`train_transformer.py`, etc.) unchanged, for later retraining/analysis.

**Recalibration during LIVE**: of the three things calibration primes,
two (the causal filters, the cardiac/respiratory peak detectors) are
already continuously self-adapting for the rest of the session and need
no explicit action. The exception is the EDA/SCR amplitude threshold
(`_ScrDetectorState.amplitude_min`), computed once from the first ~30s
pushed through it and then frozen — so a visitor's EDA baseline drifting
over a long session leaves the SCR detector calibrated to stale
conditions. `OnlineFeatureExtractor.recalibrate()` resets just that
threshold (the rise/decline state machine and onset history are
untouched), triggered on demand via `/crocodile/calibration/recalibrate`
(LIVE only) — deliberately **on-demand, not continuous/rolling**: a naive
rolling threshold would fold real SCR events into its own baseline-noise
estimate, inflating the threshold right after a genuine response burst (a
feedback loop suppressing detection exactly when it matters). It also
doesn't pause W output or change `phase` — interrupting the visuals for
~30s every recalibration would be worse than a briefly stale threshold.

**Live per-visitor alignment fit (`--reference-features`)**: optional.
Until this, every session reused one static, pre-trained
`[user-actress alignment]` transformer (`--transformer`) regardless of
visitor — a poor personalization story given the whole point of that step
is correcting for each visitor's own physiological baseline. If
`--reference-features` is given (the actress' — Laurence's — online-schema
feature CSV, `biodata_pipeline/data/processed/continuous_features_online.csv`
— her own biodata, recorded synchronized with the video frames the
regressor was trained on; NOT `erin_features_online.csv`, Erin is a
separate test subject used only to validate cross-subject alignment, not
the actress), each session's own calibration recording is used to fit a fresh transformer
against it at `calibration/stop` (and on demand via
`/crocodile/calibration/refit`, which re-fits from the same stored
calibration buffer without re-recording it) — replacing `--transformer`
for that session only; a fresh session always starts back on the static
fallback. Building this surfaced two constraints on what "live fit" can
actually mean:

- **A passive calibration recording has no emotion ground truth**, but
  the previously-only method, `ClassConditionalOTTransformer`, needs
  per-emotion labels on both sides (`_common_setup()` in
  `alignment_transformer.py` requires ≥2 common emotion labels, or raises
  `ValueError`). The live protocol can go either way: an un-cued
  calibration tags every row `neu` by default; an operator can call
  `/crocodile/calibration/set_emotion <label>` between induced-emotion
  segments during a longer, scripted multi-emotion calibration (the
  intended richer path — a few minutes, ~3 induced emotions) to produce
  real per-emotion labels instead.
- **Even the "class-blind" methods (`ot_global`/`coral`) still hard-required
  ≥2 emotion labels**, purely as an artifact of `_common_setup()` — even
  though their actual covariance fit pools all rows together regardless of
  label (only the diagnostic per-emotion prototypes used the split). Fixed
  by adding a class-blind pooled fallback (`_common_setup_or_pool()`) to
  both: below 2 common labels, pool all of `subject_df` (optionally
  restricted to one reference `emotion`) instead of raising. They still
  need enough *total* samples for a well-conditioned 53×53 covariance
  estimate (~1400 off-diagonal terms) though — badly underdetermined from
  a very short window (tens of samples). For that shorter-window case, a
  fifth transformer, `ZScoreTransformer`, covers it: diagonal-only —
  matches each feature's mean *and* variance independently (a per-feature
  translation *and* scale correction, not just the former), reusing the
  same `StandardScaler`-based `sub_scaler`/`ref_scaler` pattern the other
  four already use, minus the covariance/OT step in between. Variance is
  still a univariate per-feature statistic (53 independent scalars, not a
  53×53 matrix), so it stays well-conditioned from a much smaller window
  than full-covariance methods need, and needs no emotion labels on the
  subject side at all.

`live_pipeline.py` auto-selects between all three (`--live-transformer-method
auto`, the default), in order: `ClassConditionalOTTransformer` if the
calibration buffer has ≥2 emotion labels with ≥`--min-samples-per-emotion`
rows each (a guided, multi-emotion calibration); else `CORALTransformer` if
the buffer has ≥`--min-samples-for-covariance` rows *total*, regardless of
labels (a single-baseline calibration long enough for a real covariance
fit); else `ZScoreTransformer` (short and/or unlabeled). The same
underlying mechanism (the optional per-row emotion tag, plus how much data
was collected) naturally produces whichever tier matches how that
particular visitor was actually calibrated — see "Three usage scenarios"
below. Any method can also be forced explicitly regardless of what the
data would otherwise justify. A failed fit (still-insufficient data, a
linalg error) is logged and never crashes the server — the previous
transformer just stays active.

### Three usage scenarios

**Just want to confirm the pipeline runs, without picking a calibration
strategy?** See README.md's "Simplest possible test" — a pre-fit synthetic
transformer + a synthetic replay CSV, both committed, skip calibration
entirely (straight from `session/start` to `live/start`). The three
scenarios below are about *calibration quality* once that's working and you
have a real visitor to align to; they're not the on-ramp.

The same calibration mechanism (a `CALIBRATING` phase, optionally tagged
with `calibration/set_emotion`) naturally supports three different ways of
running a session, differing only in how the calibration phase is used —
`live_pipeline.py` doesn't need to be told which one you're doing, `auto`
figures it out from the resulting data. All three below are simulated with
real data — Erin's raw recording
(`biodata_pipeline/data/raw/emotion_biodata_erin_2026-02-09_labeled.csv`,
134,744 samples: a clean 345s `neu` block, then 345s `anx`, 326s `sad`,
then another 331s `neu`) sliced into the fixtures used, all already in
`live_pipeline/data/` — and all three were verified end-to-end while
writing this (75/75 W vectors received across the three sessions below,
`auto` picking the method named in each case).

Start the server once, shared by all three (no `--transformer` needed —
each scenario's live fit is what actually produces output; add
`--record-dir` too if you want each session's raw biodata/fit saved):
```bash
live_pipeline/run_live.sh \
    --regressor latent_pipeline/outputs/stage5_regressor_online/regressor.joblib \
    --reference-features biodata_pipeline/data/processed/continuous_features_online.csv
```

1. **Just send data, no guided calibration → `zscore`.** A short, un-cued
   calibration — `erin_calibration_segment.csv` is 60s of real `neu`
   baseline (the first 6,000 raw samples) — lands every row on the default
   `neu` tag, one implicit "class." Well under
   `--min-samples-for-covariance` (default 300), so `auto` picks
   `ZScoreTransformer`.
   ```bash
   live_pipeline/run_session_control.sh --start-session visitor-1
   live_pipeline/run_session_control.sh --start-calibration
   live_pipeline/run_replay.sh --input live_pipeline/data/erin_calibration_segment.csv --speed 1.0
   live_pipeline/run_session_control.sh --stop-calibration
   live_pipeline/run_session_control.sh --start-live
   live_pipeline/run_replay.sh --input live_pipeline/data/erin_live_segment.csv --speed 1.0 --limit 2000
   live_pipeline/run_session_control.sh --end-session
   ```
   Confirmed: `Fitting z-score alignment on 39 subject -> 356 reference
   samples ... Live transformer fit: zscore on 60 calibration rows
   ({'neu': 60})`. Cheapest calibration, but see "Known limitation" below
   — `zscore`'s lack of covariance correction can produce visually glitchy
   output.

2. **Calibrate a real (covariance-aware) model, still without emotion
   classes → `coral`.** Same un-cued calibration as above, just a *longer*
   one — `erin_calibration_long_neu.csv` is 320s of real `neu` (raw
   samples 0-32,000), just past the 300-row threshold.
   `erin_live_after_long_neu.csv` is the 25s immediately following
   (samples 32,000-34,500, still all `neu`, non-overlapping with the
   calibration slice — reusing overlapping raw samples between calibration
   and live creates a filter-state discontinuity, see the two-bugs section
   below).
   ```bash
   live_pipeline/run_session_control.sh --start-session visitor-2
   live_pipeline/run_session_control.sh --start-calibration
   live_pipeline/run_replay.sh --input live_pipeline/data/erin_calibration_long_neu.csv --speed 1.0
   live_pipeline/run_session_control.sh --stop-calibration
   live_pipeline/run_session_control.sh --start-live
   live_pipeline/run_replay.sh --input live_pipeline/data/erin_live_after_long_neu.csv --speed 1.0
   live_pipeline/run_session_control.sh --end-session
   ```
   Confirmed: `Pooling class-blind (1 common emotion(s) with reference --
   reference restricted to 'neu') ... Fitting CORAL on 299 subject -> 356
   reference samples ... Live transformer fit: coral on 320 calibration
   rows ({'neu': 320})`. Or force it regardless of duration:
   `--live-transformer-method coral` (useful to compare against `zscore`
   on the exact same short recording from scenario 1).

3. **Guide the visitor through N emotions, then calibrate per-emotion →
   `ot_classconditional`.** The richest option, using Erin's real
   `anx`/`neu`/`sad` blocks directly:
   `erin_calibration_segment.csv` (60s `neu`, reused from scenario 1),
   `erin_calibration_anx.csv` (60s `anx`, raw samples 34,522-40,522) and
   `erin_calibration_sad.csv` (60s `sad`, raw samples 69,044-75,044) — each
   a clean single-emotion block from Erin's actual recording, cued to
   match with `calibration/set_emotion` before replaying it.
   `erin_live_scenario3.csv` (60s `neu`, from the *second* `neu` block,
   samples 101,644-104,644) is disjoint from every calibration chunk used.
   ```bash
   live_pipeline/run_session_control.sh --start-session visitor-3
   live_pipeline/run_session_control.sh --start-calibration
   live_pipeline/run_session_control.sh --set-calibration-emotion neu
   live_pipeline/run_replay.sh --input live_pipeline/data/erin_calibration_segment.csv --speed 1.0
   live_pipeline/run_session_control.sh --set-calibration-emotion anx
   live_pipeline/run_replay.sh --input live_pipeline/data/erin_calibration_anx.csv --speed 1.0
   live_pipeline/run_session_control.sh --set-calibration-emotion sad
   live_pipeline/run_replay.sh --input live_pipeline/data/erin_calibration_sad.csv --speed 1.0
   live_pipeline/run_session_control.sh --stop-calibration
   live_pipeline/run_session_control.sh --start-live
   live_pipeline/run_replay.sh --input live_pipeline/data/erin_live_scenario3.csv --speed 1.0
   live_pipeline/run_session_control.sh --end-session
   ```
   Confirmed: `Common emotions: ['anx', 'neu', 'sad'] ... Live transformer
   fit: ot_classconditional on 180 calibration rows ({'neu': 60, 'anx': 60,
   'sad': 60})`. Needs the longest calibration (here, 3 minutes total
   across 3 emotions) and the richest fit (a separate map per emotion),
   but doesn't have `zscore`'s out-of-distribution risk either — see
   "Known limitation" below for why covariance-aware methods (this one and
   scenario 2's) don't have that problem.

**Known limitation of `zscore`: can produce visually glitchy output** —
observed in real testing (a `zscore`-fit session's generated faces showed
more visual breakup/distortion than a `ClassConditionalOTTransformer`-fit
one on comparable data). This is the direct cost of being diagonal-only:
`zscore` matches each feature's mean and variance *independently*, but not
cross-feature correlation. If two features move together in the actress'
data but not in the visitor's, z-scoring can combine feature values in a
way that never actually co-occurred in the actress' training data —
genuinely out-of-distribution input to the Stage 5 regressor (an MLP,
which doesn't degrade gracefully outside its training distribution). The
covariance-aware methods (`ClassConditionalOTTransformer`, and `ot_global`/
`coral` if ever used) reshape joint structure too, not just marginals, so
they don't have this failure mode — it's specifically the tradeoff made to
get a method that's fittable from a short, unlabeled calibration window
(see above). Not necessarily a bug to engineer away, depending on intent —
for an installation, "occasionally glitchy" from `zscore`-calibrated
visitors could be an interesting expressive mode in its own right rather
than something to eliminate; that's an artistic call, not a technical one.
To confirm this is really what's happening in a given case: compare
W-vector norms (printed by both debug scripts) between a `zscore` session
and a `ClassConditionalOTTransformer` one on the same replayed data —
noticeably larger/more erratic norms under `zscore` support this
explanation over some other cause (a bad calibration fit, replay-speed
packet loss, etc.).

**Two real bugs were caught building this, both against the same
mechanism** (an artificially fast `--speed` in `replay_biodata_as_osc.py`
made a slow, systemic bug look like fast-UDP packet loss at first, so both
needed isolating before either was clearly diagnosed):

1. **`calibrate()` never stepped the cardiac/respiratory peak detectors
   through the calibration recording** — it only extended the causal
   filters, so a visitor's calibration-period heartbeats/breaths were
   invisible to the HRV/HR features for a while after the live portion
   began (their confirmed-peaks history picked up only a `lookback_s`-sized
   tail of calibration on the very first live row, not the full
   incrementally-built history `push()` would have accumulated). Fixed by
   making `calibrate()` a thin wrapper around `push()` (discarding the
   returned rows) instead of a separate, bespoke priming path — the two
   can no longer drift apart, structurally. Caught by a new fourth check in
   `scripts/test_online_causality.py` (`calibrate-then-push consistency`,
   now 4/4 passing), which exists specifically because building
   `live_pipeline.py` surfaced it — none of the first three checks
   exercised `calibrate()` at all.
2. **Testing methodology, not a code bug**: replaying a session at 50-80x
   speed over `replay_biodata_as_osc.py` can overwhelm
   `live_pipeline.py`'s synchronous `BlockingOSCUDPServer`, silently
   dropping/reordering UDP packets — confirmed by reproducing clean output
   for the identical data and code path with the OSC layer bypassed
   entirely (direct in-process `push()` calls), while the OSC-mediated run
   at 80x showed ~25% of rows with wildly implausible W norms (hundreds to
   thousands, vs. the normal ~10). At real-time (1x) speed, matching actual
   sensor rate, this doesn't occur — confirmed clean end-to-end. Documented
   prominently in `replay_biodata_as_osc.py`'s docstring so a future
   high-speed test result isn't mistaken for a pipeline defect.

**Explicitly out of scope, still**: wiring to actual sensor hardware (the
real protocol isn't confirmed); any change to Autolume itself; multi-
visitor/concurrent-session handling (one global session at a time); an
actual operator GUI/control-surface app (only the OSC protocol +
`session_control.py`'s CLI stand-in exist); automatic/periodic
recalibration (on-demand only, see above).

## Picking this back up

With Stages 1–6, the offline user-to-latent pipeline, and now the live OSC
pipeline all working, the remaining path is:
1. Widen the offline pipeline test beyond Erin's 3 emotion labels
   (anx/neu/sad) to a subject/recording covering more of the emotion range,
   to see how the regressor + alignment behave outside that overlap
2. Decide whether the cross-subject alignment step needs improvement — the
   generated faces for a new subject show less expression variation than
   Stage 5's own actress-held-out visual check, which stacks two lossy steps
   (OT alignment + regression) instead of one
3. Confirm the real sensor hardware's actual live communication protocol
   (serial? OSC? something else — not established anywhere in this repo)
   and either adapt it to match `live_pipeline.py`'s input protocol or
   build a small bridge between the two
4. Confirm the Open Stage Control panel (`crocodile-control-panel.json`)
   click-through in a real (non-headless) session against a running
   `live_pipeline.py`, and refine its layout/status-display formatting
   once actually seen running
5. Get Autolume actually running against `live_pipeline.py`'s output
   end-to-end (tested so far only against `latent_osc_debug_viewer.py` and
   `--log-only`) and confirm the "project unchecked" configuration note
   above in practice

`training_gan/` is legacy and sits outside this critical path entirely.
