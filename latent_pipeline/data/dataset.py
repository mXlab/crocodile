"""CrocodileEncoderDataset and custom BatchSampler for encoder training.

Loads from cnn_training_manifest.csv, reads pool.yaml descriptors to determine
pool properties, and provides images + metadata for loss routing.
"""

import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from glob import glob
from torch.utils.data import Dataset, Sampler


# All unique emotion labels across all sessions (excluding 'none')
EMOTION_LABELS = [
    'rlx', 'con', 'cfd', 'gra', 'joy', 'sup', 'sun', 'dst', 'anx',
    'fea', 'dsg', 'ang', 'sha', 'pai', 'dsp', 'sad', 'tir', 'rlf',
    'aro', 'gri', 'pri', 'lau',
]
EMOTION_TO_INT = {label: i for i, label in enumerate(EMOTION_LABELS)}
NONE_EMOTION_IDX = -1


def load_pool_properties(frames_dir):
    """Load all pool.yaml files and return {pool_name: properties_dict}."""
    pool_props = {}
    for pool_yaml in glob(os.path.join(frames_dir, '*/pool.yaml')):
        with open(pool_yaml) as f:
            pool = yaml.safe_load(f)
        pool_props[pool['name']] = {
            'has_biodata': pool.get('has_biodata', False),
            'has_emotions': pool.get('has_emotions', False),
            'has_temporal_continuity': pool.get('has_temporal_continuity', False),
            'has_feeling_it': pool.get('has_feeling_it', False),
            'skip_seconds': pool.get('skip_seconds', 0),
        }
    return pool_props


class CrocodileEncoderDataset(Dataset):
    """Dataset for encoder training from extracted frames.

    Returns:
        image: (3, H, W) tensor in [-1, 1] where H=W=target_size (or original size)
        metadata: dict with pool_name, frame_number, emotion_label (int),
                  feeling_it (float), has_biodata, has_emotions,
                  has_temporal_continuity, is_consecutive
    """

    def __init__(self, manifest_path, frames_dir, repo_root, pool_names=None,
                 target_size=None, val_fraction=None, split='all', seed=42):
        """
        Args:
            manifest_path: Path to cnn_training_manifest.csv
            frames_dir: Path to data/frames/ directory (contains pool.yaml files)
            repo_root: Repository root (for resolving relative frame paths)
            pool_names: Optional list of pool names to include (None = all)
            target_size: Optional int to resize images on the fly (e.g. 128)
            val_fraction: If set, randomly hold out this fraction for val (e.g. 0.1 = 10%).
                          Train and val sets are deterministic given the same seed.
            split: 'all' (default), 'train', or 'val'. Only used when val_fraction is set.
            seed: Random seed for the train/val split (default 42).
        """
        self.repo_root = repo_root
        self.target_size = target_size
        self.pool_props = load_pool_properties(frames_dir)

        # Load manifest
        df = pd.read_csv(manifest_path)
        if pool_names is not None:
            df = df[df['pool_name'].isin(pool_names)].reset_index(drop=True)

        # Filter out clapperboard/pre-recording frames using skip_seconds
        keep = []
        for idx, row in df.iterrows():
            props = self.pool_props.get(row['pool_name'], {})
            skip = props.get('skip_seconds', 0)
            if row['timestamp_s'] >= skip:
                keep.append(idx)
        df = df.loc[keep].reset_index(drop=True)

        # Random train/val split
        if val_fraction is not None and split != 'all':
            val_df = df.sample(frac=val_fraction, random_state=seed)
            if split == 'val':
                df = val_df
            else:
                df = df.drop(val_df.index)
            df = df.reset_index(drop=True)

        self.manifest = df

        # Build consecutive-frame index for temporal loss
        # Two frames are consecutive if same pool, frame_number differs by exactly fps (24)
        self._build_consecutive_pairs()

    def _build_consecutive_pairs(self):
        """Pre-compute which manifest rows form consecutive pairs."""
        self.is_consecutive = [False] * len(self.manifest)
        self.prev_idx = [-1] * len(self.manifest)

        # Group by pool, sorted by frame_number
        for pool_name, group in self.manifest.groupby('pool_name'):
            props = self.pool_props.get(pool_name, {})
            if not props.get('has_temporal_continuity', False):
                continue

            sorted_indices = group.sort_values('frame_number').index.tolist()
            for i in range(1, len(sorted_indices)):
                curr_idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                curr_fn = self.manifest.loc[curr_idx, 'frame_number']
                prev_fn = self.manifest.loc[prev_idx, 'frame_number']
                # At 1fps extraction from 24fps video, consecutive = 24 frames apart
                if curr_fn - prev_fn == 24:
                    self.is_consecutive[curr_idx] = True
                    self.prev_idx[curr_idx] = prev_idx

        # Also build a list of indices that are consecutive (for sampler)
        self.consecutive_indices = [
            i for i in range(len(self.manifest)) if self.is_consecutive[i]
        ]

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        pool_name = row['pool_name']
        props = self.pool_props.get(pool_name, {})

        # Load image
        frame_path = os.path.join(self.repo_root, row['frame_path'])
        img = cv2.imread(frame_path)
        if img is None:
            raise FileNotFoundError(f"Could not load: {frame_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # On-the-fly resize if target_size differs from loaded size
        if self.target_size is not None and img.shape[0] != self.target_size:
            img = cv2.resize(img, (self.target_size, self.target_size),
                             interpolation=cv2.INTER_AREA)

        # Convert to tensor in [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0

        # Emotion label as int
        emotion_str = row['emotion_label']
        emotion_idx = EMOTION_TO_INT.get(emotion_str, NONE_EMOTION_IDX)

        metadata = {
            'pool_name': pool_name,
            'frame_number': int(row['frame_number']),
            'emotion_label': emotion_idx,
            'feeling_it': float(row['feeling_it']),
            'has_biodata': props.get('has_biodata', False),
            'has_emotions': props.get('has_emotions', False),
            'has_temporal_continuity': props.get('has_temporal_continuity', False),
            'is_consecutive': self.is_consecutive[idx],
            'prev_idx': self.prev_idx[idx],
            'manifest_idx': idx,
        }

        return img, metadata


class TemporalAwareSampler(Sampler):
    """Custom sampler that yields ~50% consecutive-pair batches and ~50% random batches.

    For consecutive batches, yields pairs [prev_idx, curr_idx] to ensure
    temporal loss gets meaningful gradient.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.consecutive_indices = dataset.consecutive_indices
        self.n = len(dataset)

    def __iter__(self):
        # Build batch list
        batches = []

        # Consecutive-pair batches (~50% of total)
        consec = list(self.consecutive_indices)
        random.shuffle(consec)
        pair_batch_size = max(1, self.batch_size // 2)  # pairs take 2 slots each
        for i in range(0, len(consec), pair_batch_size):
            batch_consec = consec[i:i + pair_batch_size]
            batch = []
            for idx in batch_consec:
                prev = self.dataset.prev_idx[idx]
                if prev >= 0:
                    batch.extend([prev, idx])
            if batch:
                batches.append(batch[:self.batch_size])

        # Random batches (roughly same count)
        all_indices = list(range(self.n))
        random.shuffle(all_indices)
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if batch:
                batches.append(batch)

        # Shuffle all batches
        random.shuffle(batches)

        for batch in batches:
            yield from batch

    def __len__(self):
        # Approximate: consecutive pairs + random
        n_consec_batches = len(self.consecutive_indices) // max(1, self.batch_size // 2)
        n_random_batches = self.n // self.batch_size
        return (n_consec_batches + n_random_batches) * self.batch_size


ALL_REAL_POOLS = ['session_1S', 'session_2S', 'session_3S', 'session_4S', 'session_1X', 'diverse']
"""All real recording sessions plus the curated diverse image set used to train StyleGAN."""


def get_train_val_pools():
    """Return pool lists for scripts that still use pool-based filtering (quick_recon etc.).

    For training, use CrocodileEncoderDataset with val_stride instead — this gives
    a cross-session split that covers all emotion types in both train and val.
    """
    return ALL_REAL_POOLS, ALL_REAL_POOLS
