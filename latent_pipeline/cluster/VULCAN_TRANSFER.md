# Alliance Cluster Setup & Data Transfer

## 1. Globus Transfer Checklist

Upload these to your cluster project space (e.g. `~/projects/def-<PI>/crocodile/`):

### Code

| Local path | Cluster destination | Notes |
|---|---|---|
| This git repo (`crocodile/`) | `code/` | ~9 MB, code only |
| `~/Documents/workspace/stylegan_Autolume/` | `stylegan_Autolume/` | ~12 GB |

### Data & models

| Local path | Cluster destination | Size |
|---|---|---|
| `models/finalModel_Crocodile.pkl` | `code/models/finalModel_Crocodile.pkl` | 429 MB |
| `latent_pipeline/data/frames/` | `code/latent_pipeline/data/frames/` | ~791 MB |
| `latent_pipeline/data/metadata/` | `code/latent_pipeline/data/metadata/` | 1.4 MB |
| `latent_pipeline/outputs/best.pt` | `code/latent_pipeline/outputs/best.pt` | ~61 MB |

> **Note**: The synthetic dataset (`data/synthetic/`, ~26 GB) does NOT need to be
> transferred — we are resuming fine-tuning from the pre-trained `best.pt` checkpoint.
> Transfer it only if you need to re-run Phase 2A (`train_synthetic`) on the cluster.

### Directory structure on cluster
```
~/projects/def-<PI>/crocodile/
    code/                              <- git repo
        latent_pipeline/
            configs/vulcan.yaml        <- cluster config (edit CHANGEME fields)
            cluster/submit_train.sh    <- SLURM script (edit def-CHANGEME)
            data/frames/               <- uploaded via Globus
            data/metadata/             <- uploaded via Globus
            outputs/
                best.pt                <- uploaded from laptop (epoch 4, LPIPS 0.095)
        models/
            finalModel_Crocodile.pkl   <- uploaded via Globus
    stylegan_Autolume/                 <- uploaded via Globus
```

## 2. First-time setup (on cluster login node)

```bash
cd ~/projects/def-<PI>/crocodile/code

# Edit configs with your paths (replace all CHANGEME fields)
nano latent_pipeline/configs/vulcan.yaml
nano latent_pipeline/cluster/submit_train.sh

# Run setup (creates virtualenv, installs packages)
bash latent_pipeline/cluster/setup_vulcan.sh
```

## 3. Submit training

Continue fine-tuning from the uploaded laptop checkpoint:

```bash
cd ~/projects/def-<PI>/crocodile/code
sbatch latent_pipeline/cluster/submit_train.sh \
    --resume latent_pipeline/outputs/best.pt
```

Monitor:
```bash
squeue -u $USER                              # check job status
tail -f crocodile-encoder-*.out             # watch live output
cat latent_pipeline/outputs/training_log.json  # check losses
```

Resume after timeout or preemption:
```bash
sbatch latent_pipeline/cluster/submit_train.sh \
    --resume latent_pipeline/outputs/latest.pt
```

## 4. Download results

After training, download from cluster via Globus:
- `code/latent_pipeline/outputs/best.pt`
- `code/latent_pipeline/outputs/training_log.json`
- `code/latent_pipeline/outputs/recon_epoch_*.png`

## Performance comparison

| | Laptop (RTX 3060 6GB) | Cluster (L40s 48GB) |
|---|---|---|
| batch_size | 1 | 8 |
| grad_accum | 16 | 2 |
| effective batch | 16 | 16 |
| est. epoch time | ~50 min | ~6 min |
| epochs configured | 5 | 30 |
| VRAM used | ~5 GB | ~35 GB (est.) |

## Current training state (as of laptop run)

- Phase 2A (`train_synthetic`): complete — 20 epochs on 10k synthetic images
  - Best val_mse: 0.004552 → checkpoint: `outputs/train_synthetic/best.pt`
- Phase 2B (`train_frames`): 5 epochs on real frames, resuming from Phase 2A weights
  - Best val_lpips: **0.0950** (epoch 4) → checkpoint: `outputs/best.pt`
  - Baseline before fine-tuning: 0.2473 LPIPS (synthetic encoder only)
