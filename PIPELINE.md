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
  small MLP (256,128) on the NeuroKit2 batch feature set under blocked-shuffle
  k-fold CV (mean val R²=0.457), saving `regressor.joblib` + a visual
  original/generated comparison grid.
- **Offline user-to-latent pipeline**: done and tested end-to-end on a subject
  other than the actress (Erin) — `apply_transformer.py` (alignment) +
  `stage6_user_to_latent_test.py` (regressor + StyleGAN2 render). This proves
  the four offline pieces (feature extraction, cross-subject alignment,
  regressor, StyleGAN2) compose correctly on non-actress biodata.
- **Runtime pipeline (live)**: still not built. "Runtime" is reserved
  specifically for continuously incoming sensor data — causal/online feature
  extraction, per-sample alignment + regression + render running in a loop.
  The *offline* user-to-latent pipeline above proves the same four pieces
  work together, but reads a pre-recorded CSV as a batch; nothing yet wires
  them to live data.

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
   pipeline" become a real question: wire causal feature extraction +
   `apply_transformer.py`'s per-sample equivalent + the regressor + StyleGAN2
   into something that runs continuously on live user data

`training_gan/` is legacy and sits outside this critical path entirely.
