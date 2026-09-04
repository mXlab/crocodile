# Crocodile: Latent Vector Pipeline — Claude Code Implementation Handoff

> **Start with [`PIPELINE.md`](PIPELINE.md)** for current architecture, status,
> and how this fits with `biodata_pipeline`/`training_gan`. This document is
> kept for its Stage 1–2 implementation detail (CNN architecture, loss
> functions, training loop) — its pool names, parameters (video FPS/resolution,
> StyleGAN2 layer count), and "Stage 5 is future work" framing are outdated;
> see `PIPELINE.md`'s terminology table and `latent_pipeline/PLAN.md` for the
> corrected values actually in use.

## Project Context

**Crocodile** is an interactive art installation in which a participant's physiological
biodata (heart rate, EDA, respiration) drives real-time generation of facial expressions
via a StyleGAN2 model trained on actress Laurence Dauphinais. The installation creates
a psychosomatic mirror: the participant activates the avatar's emotions through
empathetic connection.

The core technical challenge is building a pipeline that:
1. Inverts Dauphinais' video frames into the W-space of a trained StyleGAN2 model
2. Associates those W vectors with synchronized physiological biodata
3. Trains a biodata→W regressor that can be used at runtime with participant data

This document covers **Stages 1 and 2** of that pipeline: frame extraction and CNN
encoder training. The biodata→W regressor (Stage 5) will be addressed separately.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING PIPELINE                       │
│                                                                  │
│  Stage 1: Frame Extraction                                       │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                    │
│  │ Pool A   │   │ Pool B   │   │ Pool GAN │                    │
│  │ 4 videos │   │ 1 video  │   │ 2500 img │                    │
│  │ +biodata │   │ no bio   │   │ no meta  │                    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘                    │
│       │              │              │                            │
│  Stage 2: CNN Encoder Training (all pools → image→W mapping)    │
│  ┌────▼──────────────▼──────────────▼──────────────────────┐   │
│  │  CNN Encoder  →  W vector  →  Frozen StyleGAN2  →  img  │   │
│  │  (train from scratch, loss computed vs original image)   │   │
│  └────────────────────────────┬─────────────────────────────┘   │
│                               │                                  │
│  Stage 3: Validation          │                                  │
│  Stage 4: Biodata Attachment  │ (Pool A only)                   │
│  Stage 5: Biodata→W Regressor │                                  │
└───────────────────────────────┘

RUNTIME PIPELINE (not built here):
Participant biodata → alignment → W regressor → StyleGAN2 → face
```

---

## Data Structure

### Source Files

```
Pool A (synchronized — PRIMARY DATA SOURCE):
    video_session_01.mp4  +  biodata_session_01.csv
    video_session_02.mp4  +  biodata_session_02.csv
    video_session_03.mp4  +  biodata_session_03.csv
    video_session_04.mp4  +  biodata_session_04.csv
    Total duration: ~90 minutes
    Content: Dauphinais performing emotional states
    Emotion labels: available per session segment
    feeling_it flag: available (marks confirmed genuine emotional experience)

Pool B (grimaces/variations — ENCODER TRAINING ONLY):
    video_session_05.mp4  (no biodata file)
    Content: facial variations, grimaces, extreme expressions
    Emotion labels: available (approximate)
    Duration: unknown, estimate 15-30 minutes

Pool GAN (diversity-selected — ENCODER TRAINING ONLY):
    2500 pre-extracted images at 2160×2160 resolution
    Content: visually diverse frames from Dauphinais recordings
    Frame position in original videos: UNKNOWN (metadata lost)
    These are the images used to train the StyleGAN2 model
```

### Biodata File Format

The biodata CSV files have the following structure (from `sample_session_features.csv`):

```
Columns (20 features total):
  Heart:
    heart_normalized        - normalized heart signal [0,1]
    heart_bmp               - heart beats per minute (raw)
    heart_amplitude_change  - change in PPG amplitude
    heart_bpm_change        - change in BPM

  EDA:
    eda_scr                 - electrodermal skin conductance response (phasic)
    eda_scl                 - electrodermal skin conductance level (tonic)

  Respiration:
    resp_normalized         - normalized respiration signal
    resp_scaled             - scaled respiration signal
    resp_exhaling           - binary exhale indicator [0,1]
    resp_normalized_amplitude
    resp_scaled_amplitude
    resp_amplitude_level
    resp_amplitude_change
    resp_amplitude_variability
    resp_rpm                - respiration rate (breaths per minute)
    resp_normalized_rpm
    resp_scaled_rpm
    resp_rpm_level
    resp_rpm_change
    resp_rpm_variability

Sampling: one row per biodata sample (approximately 30fps, matching video)
```

The recording file (`sample_session_recording.csv`) contains raw sensor values with
columns: `[sensor_value_1, sensor_value_2, timestamp_ms, emotion_label_id]`

### Biodata Window Parameters

```
Window size:  20-30 seconds of signal per feature computation
Window stride: 2-5 seconds between consecutive windows
→ Features are already computed and stored in the biodata CSV
→ Each row in the features CSV corresponds to one window position
→ The frame position in the video tells you which biodata row to use
```

**Critical**: Window overlap means that consecutive biodata samples share signal.
When training the biodata→W regressor (Stage 5), use temporal super-segment
grouping in cross-validation to prevent data leakage — same approach used in
the emotion classifier.

---

## Stage 1: Frame Extraction

### Goals

Extract frames from all video sources at appropriate rates, producing:
1. A **CNN training set** (high volume, all pools, no biodata needed)
2. A **biodata→W mapping set** (lower rate, Pool A only, with precise biodata attachment)

### Frame Counts

```
CNN Training Set (target: 8,000-11,000 frames):
  Pool A @ 1fps:  ~5,400 frames  (90 min × 60s × 1fps)
  Pool B @ 1fps:  ~1,000-2,000 frames (estimated)
  Pool GAN:       2,500 pre-extracted images
  Total:          ~8,900-9,900 frames

Biodata→W Mapping Set (target: 1,000-2,700 frames):
  Pool A @ 1 frame/2-5s: ~1,080-2,700 frames
  Pool A only — frame position ↔ biodata attachment guaranteed
```

### Output Directory Structure

```
data/
├── frames/
│   ├── cnn_training/
│   │   ├── pool_a/
│   │   │   ├── session_01/
│   │   │   │   ├── frame_000060.png   (filename encodes frame number)
│   │   │   │   ├── frame_000120.png
│   │   │   │   └── ...
│   │   │   ├── session_02/
│   │   │   ├── session_03/
│   │   │   └── session_04/
│   │   ├── pool_b/
│   │   │   └── session_05/
│   │   └── pool_gan/
│   │       ├── img_0001.png
│   │       └── ...
│   └── biodata_mapping/
│       ├── session_01/
│       │   ├── frame_000300.png
│       │   └── ...
│       ├── session_02/
│       ├── session_03/
│       └── session_04/
├── metadata/
│   ├── cnn_training_manifest.csv
│   └── biodata_mapping_manifest.csv
```

### Metadata CSV Schema

**`cnn_training_manifest.csv`**:
```
frame_path, session_id, pool, frame_number, timestamp_ms, emotion_label, has_biodata
frames/cnn_training/pool_a/session_01/frame_000060.png, session_01, A, 60, 2000, joy, True
frames/cnn_training/pool_b/session_05/frame_000060.png, session_05, B, 60, 2000, grimace, False
frames/cnn_training/pool_gan/img_0001.png, unknown, GAN, -1, -1, unknown, False
```

**`biodata_mapping_manifest.csv`**:
```
frame_path, session_id, frame_number, timestamp_ms, biodata_row_index, emotion_label, feeling_it
frames/biodata_mapping/session_01/frame_000300.png, session_01, 300, 10000, 42, joy, True
```

The `biodata_row_index` field is critical — it is the row index in the corresponding
biodata CSV file that aligns with this frame's timestamp. This is the link between
image and physiology.

### Frame Extraction Parameters

```python
# CNN training extraction
CNN_FPS = 1.0                    # 1 frame per second
TARGET_SIZE = (256, 256)         # downsample from 2160×2160
INTERPOLATION = cv2.INTER_AREA   # best quality for downsampling

# Biodata mapping extraction
MAPPING_STRIDE_SECONDS = 2.0     # 1 frame every 2 seconds (adjustable to 5s)
TARGET_SIZE = (256, 256)         # same as CNN training

# Session boundary handling
# NEVER extract frames that span session boundaries
# Each session is a separate video file — this is handled by processing files independently
```

### Implementation Notes

- Use `cv2.VideoCapture` for frame extraction
- Compute `biodata_row_index` by matching `frame_timestamp_ms` to the closest
  timestamp in the biodata CSV — use nearest-neighbor matching with a tolerance
  of ±100ms
- Preserve original frame numbers (not extracted frame index) in filenames
  so frame positions in the original video are always recoverable
- Pool GAN images are already extracted — just copy and register in manifest,
  set `frame_number = -1` and `biodata_row_index = -1` to indicate unknown position
- Verify that 2160×2160 → 256×256 downsampling preserves face centering
  (spot-check ~10 frames visually before running full extraction)

---

## Stage 2: CNN Encoder Training

### Goal

Train a CNN that maps a 256×256 face image to a 512-dimensional W vector in the
trained StyleGAN2 latent space. The frozen StyleGAN2 generator acts as a
differentiable decoder, providing gradient signal through reconstruction loss.

The encoder is a **preprocessing tool only** — it is not used at runtime.
Its sole purpose is to efficiently invert thousands of frames into W vectors
so that biodata can be attached to them.

### StyleGAN2 Integration

```python
# Load your trained StyleGAN2 generator
import pickle
import torch

with open('path/to/your/stylegan2.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

# Freeze all generator parameters
for param in G.parameters():
    param.requires_grad = False
G.eval()

# Generate image from W vector
# W shape: (batch_size, 512)
# G.synthesis expects W in shape (batch_size, num_layers, 512) for W+ space
# For W space: broadcast the same vector to all layers
def generate_from_w(G, w_batch):
    # w_batch: (B, 512)
    num_layers = G.num_ws  # typically 14-18 depending on resolution
    w_broadcast = w_batch.unsqueeze(1).repeat(1, num_layers, 1)  # (B, L, 512)
    imgs = G.synthesis(w_broadcast, noise_mode='const')
    return imgs  # (B, 3, H, W), range [-1, 1]
```

**Important**: Use `noise_mode='const'` during training to ensure deterministic
generation — random noise would make the reconstruction loss noisy and unstable.

### CNN Architecture

```python
import torch
import torch.nn as nn

class EmotionEncoder(nn.Module):
    """
    Maps 256×256 face images to 512-dim W vectors in StyleGAN2 latent space.
    Trained from scratch — no pretrained weights.
    Designed for single-identity, controlled-condition face images.
    """

    def __init__(self, w_dim=512):
        super().__init__()

        # Stage 1: Local feature extraction
        # Progressively builds receptive field while preserving fine-grained spatial info
        # Small kernels (3×3) capture subtle facial muscle activations
        self.features = nn.Sequential(
            # Block 1: 256×256 → 128×128, 32 channels
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 128×128 → 64×64, 64 channels
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64×64 → 32×32, 128 channels
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 32×32 → 16×16, 256 channels
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 16×16 → 8×8, 512 channels
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Output shape after features: (B, 512, 8, 8)

        # Stage 2: Spatial attention
        # Learns to focus on emotionally informative facial regions
        # (eyes, brows, mouth, nasolabial folds) rather than treating all
        # spatial locations equally
        self.attention = nn.Sequential(
            nn.Conv2d(512, 512, 1),   # 1×1 conv to compute attention scores
            nn.Softmax(dim=2),        # normalize over spatial locations (8×8=64 positions)
        )
        # Attention output: (B, 512, 64) — probability distribution over spatial locations

        # Stage 3: Regression head
        self.regressor = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, w_dim),
            # No final activation — W space is unbounded
        )

    def forward(self, x):
        # x: (B, 3, 256, 256), values in [-1, 1]

        feat = self.features(x)                          # (B, 512, 8, 8)

        # Spatial attention
        B, C, H, W = feat.shape
        feat_flat = feat.view(B, C, H * W)               # (B, 512, 64)
        attn = self.attention(feat_flat)                  # (B, 512, 64)
        attended = (feat_flat * attn).sum(dim=2)          # (B, 512) weighted sum

        w = self.regressor(attended)                      # (B, 512)
        return w
```

### Loss Functions

```python
import lpips

# Initialize perceptual loss (VGG-based LPIPS)
# This is the primary training signal
lpips_loss_fn = lpips.LPIPS(net='vgg').cuda()

def compute_losses(encoder, G, batch, lambda_temporal=0.1,
                   lambda_emotion=0.05, lambda_feeling_it=3.0):
    """
    Compute combined training loss for one batch.

    batch contains:
        images:        (B, 3, 256, 256) — normalized to [-1, 1]
        images_1024:   (B, 3, 1024, 1024) — full res for reconstruction loss
        session_ids:   (B,) — integer session identifier
        frame_numbers: (B,) — frame number within session
        emotion_labels:(B,) — integer emotion class, -1 if unknown
        feeling_it:    (B,) — float, 1.0 if confirmed genuine, 0.0 otherwise
        pool:          (B,) — 0=A, 1=B, 2=GAN
        is_consecutive:(B,) — bool, True if this frame is adjacent to previous in batch
    """

    images = batch['images'].cuda()
    w_pred = encoder(images)                              # (B, 512)

    # Generate reconstructed images through frozen generator
    reconstructed = generate_from_w(G, w_pred)           # (B, 3, 1024, 1024)

    # Resize reconstructed to 256×256 for loss computation
    # (encoder input resolution — avoids upsampling artifacts in loss)
    recon_256 = F.interpolate(reconstructed, size=(256, 256), mode='bilinear',
                               align_corners=False)

    # ── Loss 1: Perceptual reconstruction (all frames) ──────────────────────
    # LPIPS operates on [-1, 1] images
    feeling_it_weights = 1.0 + (lambda_feeling_it - 1.0) * batch['feeling_it'].cuda()
    # feeling_it frames get lambda_feeling_it× weight, others get 1×
    perceptual = (lpips_loss_fn(recon_256, images) * feeling_it_weights).mean()

    # ── Loss 2: Temporal smoothness (Pool A and Pool B only) ─────────────────
    # Penalize large W-space jumps between consecutive frames in same session
    # This ensures smooth latent trajectories through emotional states
    temporal = torch.tensor(0.0).cuda()
    pool_ab_mask = batch['pool'].cuda() < 2      # True for Pool A and B
    consecutive_mask = batch['is_consecutive'].cuda() & pool_ab_mask

    if consecutive_mask.sum() > 1:
        # Compare each frame's W vector to the previous frame's W vector
        # Only for frames flagged as consecutive within the same session
        w_curr = w_pred[1:][consecutive_mask[1:]]
        w_prev = w_pred[:-1][consecutive_mask[1:]]
        temporal = F.mse_loss(w_curr, w_prev)

    # ── Loss 3: Emotion consistency (Pool A and Pool B labeled frames) ────────
    # Pull W vectors for same emotion label closer together
    # Push W vectors for different emotion labels apart (contrastive)
    emotion = torch.tensor(0.0).cuda()
    labeled_mask = batch['emotion_labels'].cuda() >= 0
    if labeled_mask.sum() > 1:
        w_labeled = w_pred[labeled_mask]
        labels = batch['emotion_labels'][labeled_mask].cuda()
        fi_weights = feeling_it_weights[labeled_mask]

        # Pairwise distances in W space
        dists = torch.cdist(w_labeled, w_labeled, p=2)   # (N, N)
        same = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        diff = 1.0 - same

        # Weighted contrastive: same-label pairs should be close,
        # different-label pairs should be far (margin=2.0)
        margin = 2.0
        emotion = (same * dists * fi_weights.unsqueeze(1)).mean() + \
                  (diff * F.relu(margin - dists)).mean()

    total = perceptual + lambda_temporal * temporal + 0.05 * emotion
    return total, perceptual, temporal, emotion
```

### Training Loop

```python
# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
LAMBDA_TEMPORAL = 0.1
LAMBDA_EMOTION = 0.05
LAMBDA_FEELING_IT = 3.0   # upweight anchor frames by 3×

# Dataset split (respect session boundaries — no session spans train/val/test)
# Suggested split:
#   Train:      Pool GAN (all) + Pool A sessions 1,2,3 + Pool B (all)
#   Validation: Pool A session 4
#   Test:       ~125 held-out frames from Pool A (sampled uniformly)

optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
)

# Training loop sketch
for epoch in range(NUM_EPOCHS):
    encoder.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss, perceptual, temporal, emotion = compute_losses(
            encoder, G, batch,
            lambda_temporal=LAMBDA_TEMPORAL,
            lambda_emotion=LAMBDA_EMOTION,
            lambda_feeling_it=LAMBDA_FEELING_IT
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()

    # Validation: LPIPS only on held-out Pool A session
    encoder.eval()
    with torch.no_grad():
        val_lpips = evaluate_lpips(encoder, G, val_loader)

    print(f"Epoch {epoch}: loss={loss:.4f}, lpips={perceptual:.4f}, "
          f"temporal={temporal:.4f}, emotion={emotion:.4f}, "
          f"val_lpips={val_lpips:.4f}")
```

### DataLoader Construction

The DataLoader needs to:
1. Load frames in temporal order within each session (for consecutive masking)
2. Mix pools appropriately within each batch
3. Track which batch items are temporally consecutive

```python
class CrocodileDataset(torch.utils.data.Dataset):
    """
    Loads frames from cnn_training_manifest.csv.
    Returns images, metadata, and consecutive-frame flags.
    """
    def __init__(self, manifest_path, biodata_dir, transform=None):
        self.manifest = pd.read_csv(manifest_path)
        # Sort by session then frame_number to enable consecutive detection
        self.manifest = self.manifest.sort_values(['session_id', 'frame_number'])
        self.transform = transform

        # Precompute consecutive frame flags
        self.manifest['is_consecutive'] = (
            (self.manifest['session_id'] == self.manifest['session_id'].shift(1)) &
            (self.manifest['frame_number'] - self.manifest['frame_number'].shift(1) <= 35)
            # ≤35 frames apart at 30fps ≈ within ~1 second
        )

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        img = Image.open(row['frame_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Normalize to [-1, 1] for StyleGAN2 compatibility
        img = (img * 2.0) - 1.0

        return {
            'image': img,
            'session_id': row['session_id'],
            'frame_number': row['frame_number'],
            'emotion_label': row.get('emotion_label_id', -1),
            'feeling_it': float(row.get('feeling_it', 0)),
            'pool': {'A': 0, 'B': 1, 'GAN': 2}[row['pool']],
            'is_consecutive': bool(row['is_consecutive']),
        }
```

### Training Monitoring

Log the following metrics per epoch:
```
train_loss_total     - combined weighted loss
train_loss_lpips     - perceptual reconstruction component
train_loss_temporal  - temporal smoothness component
train_loss_emotion   - emotion consistency component
val_loss_lpips       - validation perceptual loss (primary quality signal)
```

Qualitative checks every 10 epochs:
- Generate a grid of 16 random frames and their reconstructions side-by-side
- Visually confirm faces are recognizably Dauphinais
- Check that W vectors for same-emotion frames are closer than different-emotion frames
  (compute pairwise cosine similarity matrix and visualize as heatmap)

---

## Stage 3: Encoder Validation

Before using encoder outputs for the biodata→W mapping, validate quality on a
small benchmark:

```
1. Select ~30 Pool A frames spanning at least 8 different emotion labels
   (include at least 10 feeling_it=True frames)

2. Run optimization-based inversion on same frames:
   - Start from mean W vector
   - Minimize LPIPS + L2 pixel loss
   - 1000-2000 optimization steps per frame
   - ~2-5 minutes per frame, feasible for 30 frames

3. Compare:
   - Visual quality: CNN reconstruction vs optimization reconstruction
   - W-space geometry: are CNN W vectors in the same region as optimization W vectors?
   - Emotion clustering: PCA/t-SNE of W vectors colored by emotion label

4. Acceptance threshold:
   - CNN LPIPS should be within 0.1 of optimization LPIPS
   - Emotion clusters should be visually separable in 2D PCA projection
```

---

## Stage 4: Biodata→W Training Set Assembly

Once encoder is validated, run inference on the biodata mapping set:

```python
def assemble_biodata_w_dataset(encoder, G, mapping_manifest, biodata_dir,
                                output_path):
    """
    Runs encoder inference on all Pool A biodata-mapping frames.
    Attaches biodata feature vectors to each predicted W vector.
    Outputs final training set for biodata→W regressor.
    """
    encoder.eval()
    records = []

    for _, row in mapping_manifest.iterrows():
        # Load and encode image
        img = load_and_preprocess(row['frame_path'])
        with torch.no_grad():
            w = encoder(img.unsqueeze(0).cuda()).cpu().numpy()[0]  # (512,)

        # Load corresponding biodata row
        biodata_file = os.path.join(biodata_dir, f"biodata_{row['session_id']}.csv")
        biodata_df = pd.read_csv(biodata_file)
        biodata_row = biodata_df.iloc[row['biodata_row_index']]  # (20 features)

        records.append({
            'session_id': row['session_id'],
            'frame_number': row['frame_number'],
            'timestamp_ms': row['timestamp_ms'],
            'emotion_label': row['emotion_label'],
            'feeling_it': row['feeling_it'],
            # W vector: columns w_000 through w_511
            **{f'w_{i:03d}': w[i] for i in range(512)},
            # Biodata features: all 20 columns
            **biodata_row.to_dict(),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} (biodata, W) pairs to {output_path}")
    return df
```

**Output**: `biodata_w_dataset.csv` with ~1,000-2,700 rows, each containing:
- 20 biodata features (heart, EDA, respiration)
- 512 W vector dimensions
- Metadata: session, timestamp, emotion label, feeling_it flag

This dataset is the input to Stage 5 (biodata→W regressor, implemented separately).

---

## Key Invariants to Maintain Throughout

```
1. SESSION BOUNDARIES ARE SACRED
   Never mix frames from different sessions in temporal loss computation.
   Always track session_id alongside every frame.

2. FRAME POSITIONS MUST BE PRESERVED
   Always store original frame_number (position in source video), not
   extracted frame index. This is the link to biodata.

3. W SPACE, NOT W+ SPACE
   The encoder predicts a single 512-dim W vector, broadcast to all layers.
   Do not predict per-layer W+ vectors — this would break smooth interpolation
   at runtime and make biodata→W mapping intractable (512 vs 9,216 dims).

4. GENERATOR IS ALWAYS FROZEN
   G.parameters() requires_grad = False at all times during encoder training.
   The generator is a fixed rendering function, not a trainable component.

5. FEELING_IT FRAMES ARE ANCHORS
   These are frames where Dauphinais confirmed genuine emotional experience.
   They represent the highest-quality ground truth in the entire dataset.
   Always upweight them in any loss computation and treat them as the primary
   quality signal during validation.

6. NORMALIZE IMAGES TO [-1, 1]
   StyleGAN2 expects inputs and outputs in [-1, 1] range.
   Apply this normalization consistently throughout — in the dataloader,
   in loss computation, and in reconstruction visualization.
```

---

## Dependencies

```
torch >= 2.0
torchvision
lpips                # pip install lpips
opencv-python        # pip install opencv-python
pandas
numpy
pillow
matplotlib           # for training visualizations
scikit-learn         # for PCA/t-SNE validation plots
tqdm                 # progress bars
```

StyleGAN2 should be loaded directly from your trained `.pkl` file using the
NVlabs StyleGAN2-ADA-PyTorch repository code. Ensure that repo is available
in your Python path.

---

## Suggested Implementation Order

```
1. Frame extraction script (Stage 1)
   → Verify output manifest CSVs are correct before proceeding
   → Spot-check ~10 frames visually to confirm 256×256 downsampling quality

2. StyleGAN2 loader + generate_from_w() utility
   → Test: generate 5 images from random W vectors, verify visual output

3. EmotionEncoder architecture + forward pass
   → Test: random image → encoder → W → generator → image (no training yet)

4. Dataset + DataLoader
   → Test: verify consecutive flags, session boundaries, emotion labels

5. Loss functions
   → Test each loss term independently with synthetic data

6. Training loop with logging
   → Train for 5 epochs first to verify convergence direction

7. Validation: optimization-based inversion benchmark (Stage 3)

8. Biodata→W dataset assembly (Stage 4)
```

---

## Notes for the Biodata→W Regressor (Stage 5, Future Work)

The output of Stage 4 (`biodata_w_dataset.csv`) will be used to train a regressor
mapping 20 biodata features → 512 W dimensions.

Key considerations for that stage:
- Use Ridge regression as baseline (proven in physiological alignment work)
- Apply temporal super-segment cross-validation (same as emotion classifier)
  to prevent data leakage from overlapping biodata windows
- Upweight `feeling_it=True` rows during regression fitting
- Consider dimensionality reduction on W vectors (PCA to ~50 dims) before
  regression to reduce the output space from 512 to something more tractable
- The physiological alignment step (mapping participant biodata to Dauphinais
  biodata space) must happen before the W regressor at runtime
