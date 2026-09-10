#!/usr/bin/env python3
"""Select a diverse set of images per emotion and export them for a grid-based
emotion interface (Pd / OpenStageControl / TouchDesigner).

Reads latent_pipeline's biodata_w_dataset.csv (frame -> W-vector -> emotion
label), and for each emotion picks a small, visually diverse subset of frames
by clustering their W-vectors and taking the frame closest to each cluster's
centroid. Copies the corresponding thumbnails and writes a manifest CSV.

Also pulls two extra categories -- 'pri' and 'lau', which only exist in the
"grimace" session (session_1X: the actress running through facial
variations/extreme expressions, recorded without biodata). That session has
no precomputed W-vectors, so its frames are run through the trained encoder
checkpoint on demand -- this requires torch/cv2 (run this script with
latent_pipeline/.venv for that to work; use --skip-grimace to omit it
entirely). Labels session_1X shares with the biodata sessions (including
'gri', which already covers this session's grimace expressions) are left
to their existing categories and not duplicated here.

Usage:
    python emotion_grid/build_grid.py
    python emotion_grid/build_grid.py --images-per-emotion 16
    python emotion_grid/build_grid.py --skip-grimace
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
LATENT_PIPELINE_DIR = os.path.join(REPO_ROOT, "latent_pipeline")

# Emotions with no (or essentially no) feeling_it==1 frames in the dataset;
# for these we fall back to selecting from all frames regardless of the flag.
FEELING_IT_FALLBACK_EMOTIONS = {"gri", "pai", "sup"}

EXCLUDED_EMOTIONS = {"none"}

# The only session_1X ("grimace" session) labels with no equivalent in the
# biodata sessions. Every other label in that session (including 'gri',
# which already means grimace) is left to its existing category.
GRIMACE_OWN_LABELS = {"pri", "lau"}

# Reference file mapping each 3-letter code to its full name -- lives next
# to annotations.py (the source of this vocabulary) and biodata_w_dataset.csv
# (the dataset whose emotion_label column these codes describe).
EMOTION_LABELS_CSV = os.path.join(LATENT_PIPELINE_DIR, "data", "emotion_labels.csv")


def select_diverse_medoids(w_vectors, k, seed):
    """Cluster w_vectors into k groups and return the index of the frame
    closest to each cluster centroid (medoid). Falls back to returning all
    indices if there are fewer rows than k."""
    n = len(w_vectors)
    if n <= k:
        return list(range(n))

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(w_vectors)

    medoid_indices = []
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        centroid = kmeans.cluster_centers_[cluster_id]
        cluster_points = w_vectors[cluster_indices]
        dists = np.linalg.norm(cluster_points - centroid, axis=1)
        medoid_indices.append(cluster_indices[np.argmin(dists)])

    return medoid_indices


def export_selection(category, frame_paths, w_vectors, feeling_it_values, selected_indices,
                      thumbnails_dir, w_cols, extra_cols):
    """Copy the selected frames' thumbnails and return their manifest rows."""
    category_thumb_dir = os.path.join(thumbnails_dir, category)
    os.makedirs(category_thumb_dir, exist_ok=True)

    rows = []
    for i, idx in enumerate(selected_indices):
        image_id = f"{category}_{i:02d}"
        dst_filename = f"{image_id}.png"
        dst_path = os.path.join(category_thumb_dir, dst_filename)
        shutil.copyfile(frame_paths[idx], dst_path)

        entry = {
            "id": image_id,
            "emotion": category,
            "feeling_it": feeling_it_values[idx],
            "thumbnail_path": os.path.join("thumbnails", category, dst_filename),
        }
        entry.update(extra_cols[idx])
        for d, col in enumerate(w_cols):
            entry[col] = w_vectors[idx, d]
        rows.append(entry)
    return rows


def build_grid_layout(manifest_df, labels_df, output_dir):
    """Build the row-major, rectangular grid_layout.json structure consumed by
    the Open Stage Control emotion grid UI. Unlike manifest.csv, this carries
    no W-vectors (kept server-side only) and uses absolute thumbnail paths."""
    name_by_code = dict(zip(labels_df["code"], labels_df["full_name"]))

    emotion_order = list(dict.fromkeys(manifest_df["emotion"]))  # first-seen order, de-duped
    rows_by_emotion = {
        emotion: manifest_df[manifest_df["emotion"] == emotion].to_dict("records")
        for emotion in emotion_order
    }
    images_per_emotion = max(len(rows) for rows in rows_by_emotion.values())

    emotions = [
        {"code": code, "name": name_by_code.get(code, code)}
        for code in emotion_order
    ]

    cells = []
    for row in range(images_per_emotion):
        for emotion in emotion_order:
            rows = rows_by_emotion[emotion]
            if row >= len(rows):
                cells.append(None)
                continue
            r = rows[row]
            cells.append({
                "id": r["id"],
                "emotion": emotion,
                "thumbnailPath": os.path.join(output_dir, r["thumbnail_path"]),
            })

    layout = {
        "emotions": emotions,
        "images_per_emotion": images_per_emotion,
        "cells": cells,
    }
    layout_path = os.path.join(output_dir, "grid_layout.json")
    with open(layout_path, "w") as f:
        json.dump(layout, f, indent=2)
    return layout_path


def gather_grimace_candidates(frames_dir):
    """Bucket session_1X frames labeled 'pri' or 'lau' -- the only two labels
    in that session with no equivalent among the biodata sessions. Every
    other frame (including ones labeled 'gri', already covered by the
    biodata sessions' own grimace category) is left out."""
    if LATENT_PIPELINE_DIR not in sys.path:
        sys.path.insert(0, LATENT_PIPELINE_DIR)
    from data import annotations

    buckets = {label: [] for label in GRIMACE_OWN_LABELS}
    for fname in sorted(os.listdir(frames_dir)):
        if not fname.endswith(".png"):
            continue
        frame_number = int(fname[len("frame_"):-len(".png")])
        label = annotations.get_emotion("1X", frame_number)
        if label not in GRIMACE_OWN_LABELS:
            continue

        frame_path = os.path.join(frames_dir, fname)
        buckets[label].append((frame_number, frame_path))

    return buckets


def encode_frames(frame_paths, checkpoint_path, batch_size=16):
    """Run the trained image-to-W encoder on a list of 256x256 frame paths."""
    import cv2
    import torch

    if LATENT_PIPELINE_DIR not in sys.path:
        sys.path.insert(0, LATENT_PIPELINE_DIR)
    from models.encoder import EmotionEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EmotionEncoder(channels=(32, 64, 128, 256, 512), w_dim=512, dropout=0.3).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    def load_image(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0

    all_w = []
    for i in range(0, len(frame_paths), batch_size):
        batch = torch.stack([load_image(p) for p in frame_paths[i:i + batch_size]]).to(device)
        with torch.no_grad():
            all_w.append(encoder(batch).cpu().numpy())
    return np.concatenate(all_w, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Select diverse per-emotion images and export a grid manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(REPO_ROOT, "latent_pipeline", "data", "biodata_w_dataset.csv"),
        help="Path to biodata_w_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(SCRIPT_DIR, "data"),
        help="Directory to write manifest.csv and thumbnails/ into",
    )
    parser.add_argument(
        "--images-per-emotion",
        type=int,
        default=12,
        help="Target number of diverse images to select per emotion",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for KMeans clustering",
    )
    parser.add_argument(
        "--grimace-frames-dir",
        default=os.path.join(LATENT_PIPELINE_DIR, "data", "frames", "session_1X"),
        help="Directory of extracted session_1X (grimace) frames",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        default=os.path.join(LATENT_PIPELINE_DIR, "outputs", "best.pt"),
        help="Trained image-to-W encoder checkpoint, used to encode grimace frames",
    )
    parser.add_argument(
        "--skip-grimace",
        action="store_true",
        help="Skip the grimace/pri/lau categories (session_1X has no precomputed W-vectors, "
             "so including them requires torch/cv2 -- run with latent_pipeline/.venv)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    w_cols = sorted(
        [c for c in df.columns if c.startswith("w_")],
        key=lambda c: int(c.split("_")[1]),
    )
    print(f"Loaded {len(df)} frames, {len(w_cols)} W dimensions")

    df = df[~df["emotion_label"].isin(EXCLUDED_EMOTIONS)]

    thumbnails_dir = os.path.join(args.output_dir, "thumbnails")
    os.makedirs(thumbnails_dir, exist_ok=True)

    manifest_rows = []
    for emotion in sorted(df["emotion_label"].unique()):
        emotion_df = df[df["emotion_label"] == emotion]
        if emotion not in FEELING_IT_FALLBACK_EMOTIONS:
            candidates = emotion_df[emotion_df["feeling_it"] == 1]
        else:
            candidates = emotion_df
        candidates = candidates.reset_index(drop=True)

        if len(candidates) == 0:
            print(f"  {emotion}: no candidate frames, skipping")
            continue

        w_vectors = candidates[w_cols].to_numpy()
        selected_indices = select_diverse_medoids(w_vectors, args.images_per_emotion, args.seed)
        print(f"  {emotion}: {len(candidates)} candidates -> {len(selected_indices)} selected")

        frame_paths = [os.path.join(REPO_ROOT, p) for p in candidates["frame_path"]]
        feeling_it_values = candidates["feeling_it"].tolist()
        extra_cols = [
            {"pool_name": row["pool_name"], "frame_number": row["frame_number"]}
            for _, row in candidates.iterrows()
        ]
        manifest_rows.extend(export_selection(
            emotion, frame_paths, w_vectors, feeling_it_values, selected_indices,
            thumbnails_dir, w_cols, extra_cols,
        ))

    if not args.skip_grimace and os.path.isdir(args.grimace_frames_dir):
        print("\nGrimace session (session_1X):")
        grimace_buckets = gather_grimace_candidates(args.grimace_frames_dir)
        for category, entries in grimace_buckets.items():
            if not entries:
                print(f"  {category}: no candidate frames, skipping")
                continue
            frame_numbers, frame_paths = zip(*entries)
            w_vectors = encode_frames(list(frame_paths), args.encoder_checkpoint)
            selected_indices = select_diverse_medoids(w_vectors, args.images_per_emotion, args.seed)
            print(f"  {category}: {len(entries)} candidates -> {len(selected_indices)} selected")

            feeling_it_values = [None] * len(entries)
            extra_cols = [
                {"pool_name": "session_1X", "frame_number": fn} for fn in frame_numbers
            ]
            manifest_rows.extend(export_selection(
                category, list(frame_paths), w_vectors, feeling_it_values, selected_indices,
                thumbnails_dir, w_cols, extra_cols,
            ))

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    labels_df = pd.read_csv(EMOTION_LABELS_CSV)
    known_codes = set(labels_df["code"])
    undocumented = sorted(set(manifest_df["emotion"].unique()) - known_codes)
    if undocumented:
        print(f"\nWARNING: no full-name entry in {EMOTION_LABELS_CSV} for: {undocumented}")
    shutil.copyfile(EMOTION_LABELS_CSV, os.path.join(args.output_dir, "emotion_labels.csv"))

    layout_path = build_grid_layout(manifest_df, labels_df, args.output_dir)
    print(f"Grid layout: {layout_path}")

    print(f"\nSaved {len(manifest_df)} images across {manifest_df['emotion'].nunique()} emotions")
    print(f"Manifest: {manifest_path}")
    print(f"Labels reference: {os.path.join(args.output_dir, 'emotion_labels.csv')}")
    print(f"Thumbnails: {thumbnails_dir}")


if __name__ == "__main__":
    main()
