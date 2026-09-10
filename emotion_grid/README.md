# Emotion Grid

Selects a diverse, per-emotion subset of real actress frames from
`latent_pipeline/data/biodata_w_dataset.csv` and exports thumbnails + their
W-vectors for a future grid-based interface (Pd / OpenStageControl /
TouchDesigner): click an image, get its latent vector, drive Autolume.

This tool only does selection and export. The grid interface itself is
built separately in whichever tool is chosen.

## Usage

```bash
python emotion_grid/build_grid.py
python emotion_grid/build_grid.py --images-per-emotion 16
python emotion_grid/build_grid.py --skip-grimace  # omit the pri/lau categories
```

The base 20-emotion selection only needs pandas/numpy/sklearn (e.g.
`biodata_pipeline/venv`). Including the pri/lau categories (the default)
also needs torch/cv2 to run the encoder on demand — use
`latent_pipeline/.venv` for that.

## Selection method

For each emotion (excluding `none`):

1. Candidate pool = frames with that `emotion_label` and `feeling_it == 1`
   (the actress' pedal-press confirmation), except `gri`, `pai`, and `sup`,
   which have zero-to-one `feeling_it == 1` frames in the dataset and so
   fall back to all frames for that emotion regardless of the flag.
2. If the pool has fewer frames than `--images-per-emotion`, all are kept.
3. Otherwise, the pool's W-vectors are clustered with KMeans
   (`k = --images-per-emotion`), and the frame closest to each cluster
   centroid (medoid) is selected — this spans the emotion's visual range
   without picking pure outliers.

## Grimace session (session_1X)

The actress also did a session of facial variations/extreme expressions
without any biodata recorded (`latent_pipeline/data/frames/session_1X/`).
It has per-frame emotion annotations (`latent_pipeline/data/annotations.py`)
but no precomputed W-vectors, so its selected frames are run through the
trained encoder checkpoint (`latent_pipeline/outputs/best.pt`) on demand.

Most of this session's labels already exist among the biodata sessions
(including `gri`, which is this session's own name for "grimace" — not to
be confused with grief) and are left to those existing categories rather
than duplicated. Only two labels are unique to session_1X and get their
own category, diversity-selected the same way as above:

- `pri`, `lau` — pride and laughing, which don't appear in the other four
  sessions.

`feeling_it` is left blank for these two categories — the flag doesn't
apply since no biodata/pedal was recorded for this session.

## Label reference (`latent_pipeline/data/emotion_labels.csv`, tracked in git
— not private)

`latent_pipeline/data/emotion_labels.csv` maps each 3-letter code to its
full name (`code, full_name`), taken from the original recording scripts
(`luana-Crocodile-with-data/script_1S.py` for the biodata sessions,
`script_1X.py` for `pri`/`lau`) — not guessed, since e.g. `con` is
"concentration" (not "contentment") and `dst` is "distrust" (not
"distress"). It lives there rather than in `emotion_grid/` since it
documents the vocabulary used by `annotations.py` and
`biodata_w_dataset.csv` directly, not just this tool's output.

It's a static vocabulary, not a per-run output, so it's tracked in git
despite the repo's blanket `*.csv` ignore rule (see the exception at the
bottom of `.gitignore`).

`build_grid.py` copies it into `--output-dir` alongside `manifest.csv` on
every run (and warns if any category in that run's manifest has no entry
in it), so the two files can be handed to TouchDesigner/Pd/OpenStageControl
together — the manifest itself only carries the 3-letter code.

## Output (`emotion_grid/data/`, gitignored — private, derived from real
actress footage)

- `manifest.csv` — one row per selected image: `id, emotion, pool_name,
  frame_number, feeling_it, thumbnail_path, w_000...w_511` (same `w_XXX`
  convention as `biodata_w_dataset.csv`).
- `emotion_labels.csv` — copy of the label reference above, for convenience.
- `thumbnails/<emotion>/<id>.png` — copies of the selected 256x256 frames.

CSV + a plain image folder was chosen over JSON so any of the candidate
interface tools (TouchDesigner Table DAT, Pd `[csvparse]`, OpenStageControl)
can load it without extra parsing code.
