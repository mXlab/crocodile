# Crocodile Pipeline: How the Pieces Fit Together

This is the map. For implementation depth on any one piece, follow the links —
don't duplicate them here.

## The goal

Crocodile is an interactive installation: a participant's physiological signals
(heart rate, EDA, respiration) drive real-time generation of a face — a method
actress' avatar — via a trained StyleGAN2 model. The participant's
biodata becomes the avatar's emotional state.

Getting there requires an **offline preprocessing pipeline** (build training
data + models from the actress' own recordings) and a **runtime pipeline** (turn
a live participant's signals into a face at the installation). Only the first
one has working code today; the second is still mostly a design.

```
PREPROCESSING (offline, built against the actress' data)
  biodata_pipeline:  raw sensor CSVs ──▶ physiological features
                                              │
  latent_pipeline:   video frames ──▶ W-space encoder ──▶ biodata_w_dataset.csv
                                                                  │
                     [Stage 5 — biodata→W regressor]  ◀──────────┘   NOT YET BUILT

RUNTIME (live, not yet built)
  participant biodata ──▶ [alignment]† ──▶ [W regressor] ──▶ frozen StyleGAN2 ──▶ face
```
† the alignment model itself (biodata_pipeline's cross-subject transformer) already
exists and works — it just isn't wired into a runtime script yet.

## Where each piece fits

### `biodata_pipeline/` — the physiological side

Turns raw 100Hz sensor recordings into feature vectors, and separately, learns
to map a *new* subject's physiological feature space onto the actress'.

- **Feature extraction**: raw `heart`/`gsr`/`respiration` CSV → 67 features/second
  (`continuous_features.csv`). This is what feeds `latent_pipeline` Stage 4 below.
- **Windowing + classification eval**: sanity-checks that emotions are separable
  in the extracted features (not on the pipeline's critical path to the avatar).
- **Cross-subject alignment**: trains a transformer (Ridge / Optimal Transport)
  mapping a new subject's feature space onto the actress' reference space. **This
  is the runtime pipeline's "alignment" box** — a participant's biodata has to
  pass through this before anything downstream can make sense of it, since the
  W-regressor (once built) will only understand biodata shaped like the actress'.

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
| 2. Encoder training | `scripts/train_synthetic.py`, `scripts/train_frames.py` | Train a CNN that maps a face image → 512-dim W vector, using the frozen StyleGAN2 as a differentiable decoder |
| 3. Validation | `scripts/stage3_validate.py` | Compare CNN inversion vs. slow optimization-based inversion |
| 4. Biodata attachment | `scripts/stage4_assemble.py` | Encode all biodata-pool frames → join with `biodata_pipeline`'s `continuous_features.csv` → `data/biodata_w_dataset.csv` |
| 5. Biodata→W regressor | — | **Not built.** Next real piece of work once Stage 2 finishes: fit biodata → W on `biodata_w_dataset.csv`. |

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
| 20 hand-specified biodata features | 67 features from `EnhancedContinuousFeatureExtractor` |

## Current status (as of this session — 2026-09-04)

Reconstructed from checkpoints/logs, not memory — verify before trusting if more
time has passed.

- **`biodata_pipeline`**: extraction, windowing/eval, and alignment (Ridge / OT
  class-conditional, best NPA 54.2%) all have working code and real output data.
  Everything since the last commit (2026-02-10) is now committed as of this
  session.
- **`latent_pipeline`**: Stages 0–5 (setup through dataset/dataloader) done.
  Stage 2 encoder training (`train_frames.py`) **stalled mid-epoch-14-of-20 on
  2026-02-25**, ~6 hours after a suspicious 11-hour gap, no crash log, no process
  running now. Checkpoints (`outputs/best.pt`/`latest.pt`) are from epoch ~10,
  so resuming loses ~4 epochs (~4 hours of compute). Stage 3 (validate) and
  Stage 4 (assemble) are coded but blocked on Stage 2 finishing.
- **Stage 5 (biodata→W regressor)**: not started — no code anywhere references it.
- **Runtime pipeline**: not built. The alignment model exists (`biodata_pipeline`)
  but isn't wired to anything live.

## Picking this back up

The critical path to a first end-to-end result is:
1. Resume/finish `latent_pipeline/scripts/train_frames.py` (from `latest.pt`,
   note the `training_log.json` overwrite issue flagged earlier — fix before a
   long run so you don't lose history again on a restart)
2. Run Stage 3 validation, then Stage 4 assembly → `biodata_w_dataset.csv`
3. Build Stage 5 (biodata→W regressor) — the one genuinely unbuilt piece
4. Only then does "runtime pipeline" become a real question: wire
   `biodata_pipeline`'s alignment transformer + the new regressor + the frozen
   StyleGAN2 into something that runs on live participant data

`training_gan/` is legacy and sits outside this critical path entirely.
