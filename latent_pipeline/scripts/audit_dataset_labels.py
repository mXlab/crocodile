#!/usr/bin/env python3
"""Audit the training manifest: break down 'none' labels into fine-grained categories.

Cross-references cnn_training_manifest.csv with Luana's annotations to show
exactly what the 'none' catch-all contains (nul, war, tra, coo, neu, unlabeled).

Usage:
    python latent_pipeline/scripts/audit_dataset_labels.py
"""

import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)

from data.annotations import get_emotion, SESSION_EMOTIONS

MANIFEST = os.path.join(REPO_ROOT, 'latent_pipeline/data/metadata/cnn_training_manifest.csv')

df = pd.read_csv(MANIFEST)

print(f"Total manifest rows: {len(df)}")
print(f"\n=== Current emotion_label distribution ===")
print(df['emotion_label'].value_counts().to_string())

# For session pools, resolve the fine-grained label
session_pools = df[df['pool_name'].str.startswith('session_')]

print(f"\n=== Session pools only: {len(session_pools)} frames ===")
print(session_pools['emotion_label'].value_counts().to_string())

# Break down "none" into fine-grained labels
none_frames = session_pools[session_pools['emotion_label'] == 'none'].copy()
print(f"\n=== Breaking down {len(none_frames)} 'none' frames ===")

fine_labels = []
for _, row in none_frames.iterrows():
    session = row['pool_name'].replace('session_', '')
    frame_num = int(row['frame_number'])
    fine = get_emotion(session, frame_num)
    fine_labels.append(fine if fine else 'unlabeled')

none_frames['fine_label'] = fine_labels

print("\nFine-grained breakdown of 'none' frames:")
print(none_frames['fine_label'].value_counts().to_string())

print("\nPer-session breakdown:")
for pool in sorted(none_frames['pool_name'].unique()):
    pool_none = none_frames[none_frames['pool_name'] == pool]
    print(f"\n  {pool} ({len(pool_none)} 'none' frames):")
    print(f"    {pool_none['fine_label'].value_counts().to_string()}")

# Show what we'd keep vs remove
print("\n\n=== Proposed filtering ===")
print("REMOVE (invalid/nul): frames with makeup, clapperboard, tech issues")
nul_count = (none_frames['fine_label'] == 'nul').sum()
print(f"  nul frames to remove: {nul_count}")

print("\nKEEP as 'none' (non-emotion but valid actress visible):")
for label in ['war', 'tra', 'coo', 'neu']:
    count = (none_frames['fine_label'] == label).sum()
    if count > 0:
        print(f"  {label}: {count}")

unlabeled = (none_frames['fine_label'] == 'unlabeled').sum()
if unlabeled:
    print(f"\n  unlabeled (no annotation coverage): {unlabeled}")

# Show impact on training set
non_session = df[~df['pool_name'].str.startswith('session_')]
emotion_frames = session_pools[session_pools['emotion_label'] != 'none']
valid_none = none_frames[none_frames['fine_label'] != 'nul']

total_after = len(non_session) + len(emotion_frames) + len(valid_none)
print(f"\n=== Impact summary ===")
print(f"  Current total:  {len(df)} frames")
print(f"  After removing nul: {total_after} frames (removed {len(df) - total_after})")
print(f"  Non-session pools (gan_images, diverse): {len(non_session)}")
print(f"  Session emotion frames: {len(emotion_frames)}")
print(f"  Session valid 'none' (war/tra/neu): {len(valid_none)}")
