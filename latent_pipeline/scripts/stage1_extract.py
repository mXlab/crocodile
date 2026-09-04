#!/usr/bin/env python3
"""Stage 1: Extract frames from video pools and diverse images, build manifests.

Iterates over all pool.yaml descriptors in data/frames/, extracts frames based
on each pool's type, and generates training/biodata manifests.

Usage:
    python latent_pipeline/scripts/stage1_extract.py --config latent_pipeline/configs/default.yaml
"""

import argparse
import math
import os
import sys

import cv2
import pandas as pd
import yaml
from glob import glob
from tqdm import tqdm

# Add repo root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

from data.annotations import get_emotion_label, get_feeling_it, is_invalid


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_pools(frames_dir):
    """Find all pool.yaml descriptors and return list of pool configs."""
    pools = []
    for pool_yaml in sorted(glob(os.path.join(frames_dir, '*/pool.yaml'))):
        with open(pool_yaml) as f:
            pool = yaml.safe_load(f)
        pool['_dir'] = os.path.dirname(pool_yaml)
        pools.append(pool)
    return pools


def extract_video_frames(pool, repo_root, frame_size, extract_fps):
    """Extract frames from a video pool at the specified rate.

    Returns list of dicts with frame metadata.
    """
    video_path = os.path.join(repo_root, pool['source'])
    if not os.path.exists(video_path):
        print(f"  WARNING: Video not found: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps if video_fps > 0 else 0

    print(f"  Video: {video_path}")
    print(f"  FPS: {video_fps}, Total frames: {total_frames}, Duration: {duration_s:.1f}s")

    # Extract 1 frame per second: use frame numbers that are multiples of fps
    fps = pool.get('fps', 24)
    frame_interval = fps  # 1 frame per second = every fps-th frame
    session_key = pool.get('session_key')

    records = []
    skipped_invalid = 0
    frame_numbers = list(range(0, total_frames, frame_interval))

    for frame_number in tqdm(frame_numbers, desc=f"  Extracting {pool['name']}", unit='frame'):
        # Skip invalid (nul) frames — makeup, clapperboard, tech issues
        if session_key and is_invalid(session_key, frame_number):
            skipped_invalid += 1
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to target size
        frame_resized = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)

        # Save frame
        frame_filename = f"frame_{frame_number:06d}.png"
        frame_path = os.path.join(pool['_dir'], frame_filename)
        cv2.imwrite(frame_path, frame_resized)

        timestamp_s = frame_number / fps

        # Get annotations if available
        emotion_label = 'none'
        feeling_it_val = 0.0
        if session_key:
            emotion_label = get_emotion_label(session_key, frame_number)
            fi = get_feeling_it(session_key, frame_number)
            feeling_it_val = 1.0 if fi else 0.0

        records.append({
            'frame_path': os.path.relpath(frame_path, REPO_ROOT),
            'pool_name': pool['name'],
            'frame_number': frame_number,
            'timestamp_s': timestamp_s,
            'emotion_label': emotion_label,
            'feeling_it': feeling_it_val,
        })

    cap.release()
    print(f"  Extracted {len(records)} frames (skipped {skipped_invalid} invalid)")
    return records


def extract_diverse_images(pool, repo_root, frame_size):
    """Copy and downsample diverse images pool.

    Returns list of dicts with frame metadata.
    """
    source_dir = os.path.join(repo_root, pool['source'])
    if not os.path.exists(source_dir):
        print(f"  WARNING: Source directory not found: {source_dir}")
        return []

    image_files = sorted([
        f for f in os.listdir(source_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"  Source: {source_dir}, Found {len(image_files)} images")

    records = []
    for i, img_file in enumerate(tqdm(image_files, desc=f"  Processing {pool['name']}", unit='img')):
        img_path = os.path.join(source_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_resized = cv2.resize(img, (frame_size, frame_size), interpolation=cv2.INTER_AREA)

        out_filename = f"frame_{i:06d}.png"
        out_path = os.path.join(pool['_dir'], out_filename)
        cv2.imwrite(out_path, img_resized)

        records.append({
            'frame_path': os.path.relpath(out_path, REPO_ROOT),
            'pool_name': pool['name'],
            'frame_number': i,
            'timestamp_s': 0.0,
            'emotion_label': 'none',
            'feeling_it': 0.0,
        })

    print(f"  Processed {len(records)} images")
    return records


def build_biodata_manifest(training_records, pools, config):
    """Build the biodata mapping manifest from biodata pool frames.

    Only includes frames from pools with has_biodata=True where
    biodata exists (frame timestamp >= recording_start_time).
    """
    # Load continuous features to validate indices
    features_path = os.path.join(config['paths']['repo_root'],
                                 config['paths']['continuous_features'])
    features_df = pd.read_csv(features_path)

    # Build lookup: (session_id, timestamp) -> row index
    features_lookup = {}
    for idx, row in features_df.iterrows():
        key = (row['session_id'], float(row['timestamp']))
        features_lookup[key] = idx

    # Pool properties lookup
    pool_lookup = {p['name']: p for p in pools}

    biodata_records = []
    for rec in training_records:
        pool = pool_lookup.get(rec['pool_name'])
        if pool is None or not pool.get('has_biodata', False):
            continue

        fps = pool.get('fps', 24)
        recording_start = pool.get('recording_start_time', 0)
        session_id = pool.get('continuous_features_session_id')

        video_second = rec['frame_number'] / fps
        biodata_second = video_second - recording_start

        # Skip frames before biodata recording started
        if biodata_second < 0:
            continue

        # Find matching row in continuous features
        biodata_ts = math.floor(biodata_second)
        feature_key = (session_id, float(biodata_ts))
        feature_idx = features_lookup.get(feature_key)

        if feature_idx is None:
            continue

        biodata_records.append({
            'frame_path': rec['frame_path'],
            'pool_name': rec['pool_name'],
            'frame_number': rec['frame_number'],
            'timestamp_s': rec['timestamp_s'],
            'biodata_timestamp_s': biodata_second,
            'continuous_feature_idx': feature_idx,
            'emotion_label': rec['emotion_label'],
            'feeling_it': rec['feeling_it'],
        })

    return biodata_records


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Frame extraction')
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml',
                        help='Path to config YAML')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only discover pools, do not extract')
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = config['paths']['repo_root']
    frames_dir = os.path.join(repo_root, config['paths']['frames_dir'])
    metadata_dir = os.path.join(repo_root, config['paths']['metadata_dir'])
    frame_size = config['video']['frame_size']
    extract_fps = config['video']['extract_fps']

    os.makedirs(metadata_dir, exist_ok=True)

    # Discover pools
    pools = discover_pools(frames_dir)
    print(f"Discovered {len(pools)} pools:")
    for p in pools:
        print(f"  - {p['name']} (type={p['type']}, biodata={p.get('has_biodata', False)})")

    if args.dry_run:
        return

    # Extract frames from each pool
    all_records = []
    for pool in pools:
        print(f"\nProcessing pool: {pool['name']}")
        if pool['type'] == 'video':
            records = extract_video_frames(pool, repo_root, frame_size, extract_fps)
        elif pool['type'] == 'images':
            records = extract_diverse_images(pool, repo_root, frame_size)
        else:
            print(f"  Unknown pool type: {pool['type']}, skipping")
            continue
        all_records.extend(records)

    # Save CNN training manifest
    training_df = pd.DataFrame(all_records)
    training_path = os.path.join(metadata_dir, 'cnn_training_manifest.csv')
    training_df.to_csv(training_path, index=False)
    print(f"\nCNN training manifest: {training_path} ({len(training_df)} rows)")

    # Build and save biodata mapping manifest
    biodata_records = build_biodata_manifest(all_records, pools, config)
    biodata_df = pd.DataFrame(biodata_records)
    biodata_path = os.path.join(metadata_dir, 'biodata_mapping_manifest.csv')
    biodata_df.to_csv(biodata_path, index=False)
    print(f"Biodata mapping manifest: {biodata_path} ({len(biodata_df)} rows)")

    # Summary statistics
    print("\n=== Summary ===")
    for pool_name in training_df['pool_name'].unique():
        pool_df = training_df[training_df['pool_name'] == pool_name]
        print(f"  {pool_name}: {len(pool_df)} frames")

    if len(biodata_df) > 0:
        print(f"\nBiodata frames by pool:")
        for pool_name in biodata_df['pool_name'].unique():
            pool_df = biodata_df[biodata_df['pool_name'] == pool_name]
            print(f"  {pool_name}: {len(pool_df)} frames")

    # Spot-check: print first 5 entries
    print("\nSpot-check (first 5 training entries):")
    print(training_df.head().to_string())

    if len(biodata_df) > 0:
        print("\nSpot-check (first 5 biodata entries):")
        print(biodata_df.head().to_string())


if __name__ == '__main__':
    main()
