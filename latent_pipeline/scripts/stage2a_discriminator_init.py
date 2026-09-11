#!/usr/bin/env python3
"""Phase 2A variant: pretrain a DiscriminatorEncoder (StyleGAN2's own
discriminator trunk, repurposed as an image -> W encoder) on synthetic
(image, W) pairs, staged to avoid a freshly-initialised head distorting the
pretrained trunk.

Background: EmotionEncoder (the existing stage2a_train_synthetic.py /
stage2b_train_frames.py encoder) is a from-scratch VGG trunk, and struggles
specifically on rare extreme-expression frames (closed eyes, bared teeth)
that are underrepresented in the biodata sessions relative to neutral frames
-- confirmed via a CNN-vs-slow-optimization-inversion comparison on 13 such
frames (session chat log, 2026-09-11): the frozen StyleGAN2 generator CAN
represent these expressions (optimization-based inversion recovers them
closely), the CNN encoder just doesn't. StyleGAN2's discriminator, trained
adversarially on the same distribution (including the Diverse/ pool's
deliberately extreme expressions), is a domain-matched pretrained trunk this
encoder currently has no equivalent of. See models/discriminator_encoder.py
for the transplant and PIPELINE.md's session notes for the full writeup.

Training is staged (LP-FT, Kumar et al. ICLR 2022 + FreezeD, Mo et al. 2020):

  Phase 1 (train_discriminator_init.phase1_epochs): the whole pretrained
  trunk is frozen; only the new entry adapter + W head train. A random head
  fine-tuned jointly with a pretrained trunk from step one distorts the
  trunk's features, worst on exactly the out-of-distribution portion of the
  data -- which for us is the rare extreme-expression frames this whole
  detour is about.

  Phase 2: pretrained groups are unfrozen one at a time, deepest (closest to
  the new head) first, every phase2_epochs_per_group epochs, each newly-
  unfrozen group added to the optimizer at a LR that decays with unfreeze
  order (lr_trunk_base * lr_decay_per_group ** i) -- deeper/task-specific
  layers get the most adaptation, shallower/generic ones the least.

Loss: MSE(encoder(image_resized), w_target), same as stage2a_train_synthetic.py
-- no StyleGAN2 generator needed in the loop (only loaded for visual checks).

The saved checkpoint is compatible with stage2b_train_frames.py's
--pretrained (encoder weights only) provided stage2b is run with
--encoder-arch discriminator_init so it builds the matching architecture.

Usage:
    python latent_pipeline/scripts/stage2a_discriminator_init.py \\
        --config latent_pipeline/configs/default.yaml

    # Resume after a timeout/preemption (or an interrupted local run):
    python latent_pipeline/scripts/stage2a_discriminator_init.py \\
        --config latent_pipeline/configs/default.yaml \\
        --resume latent_pipeline/outputs/train_discriminator_init/latest.pt
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, PIPELINE_DIR)
sys.path.insert(0, REPO_ROOT)

from models.discriminator_encoder import build_discriminator_encoder
from models.stylegan import load_stylegan
from scripts.stage2a_train_synthetic import SyntheticWDataset, save_visual_grid


def train_one_epoch(encoder, loader, optimizer, device):
    encoder.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for imgs, w_target in tqdm(loader, desc='  train', leave=False):
        imgs = imgs.to(device)
        w_target = w_target.to(device)

        w_pred = encoder(imgs)
        loss = F.mse_loss(w_pred, w_target)
        loss.backward()

        trainable_params = [p for p in encoder.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(encoder, loader, device):
    encoder.eval()
    total_loss = 0.0
    n_batches = 0
    for imgs, w_target in loader:
        imgs = imgs.to(device)
        w_target = w_target.to(device)
        w_pred = encoder(imgs)
        total_loss += F.mse_loss(w_pred, w_target).item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def apply_unfreeze_event(encoder, optimizer, tc, event, unfrozen_count):
    """Apply one 'unfreeze:<name>' schedule event (no-op for 'phase1_start'),
    returning the updated unfrozen_count. Factored out so --resume can replay
    the schedule up to the checkpointed epoch and reconstruct identical
    optimizer param_groups before loading saved optimizer state into them
    (torch's load_state_dict matches groups by position, so the replay must
    add them in the exact same order as the original run)."""
    if event == 'phase1_start':
        return unfrozen_count
    name = event.split(':', 1)[1]
    encoder.unfreeze_pretrained_group(name)
    lr = tc['lr_trunk_base'] * (tc['lr_decay_per_group'] ** unfrozen_count)
    optimizer.add_param_group({
        'params': encoder.pretrained_group_parameters(name),
        'lr': lr,
    })
    return unfrozen_count + 1


def build_phase_schedule(tc, group_names):
    """Return a list of (epoch, action) events: 'phase1_start' at epoch 0,
    then one 'unfreeze:<group_name>' event every phase2_epochs_per_group
    epochs, deepest group first."""
    schedule = {0: ['phase1_start']}
    epoch = tc['phase1_epochs']
    for name in group_names:
        schedule.setdefault(epoch, []).append(f'unfreeze:{name}')
        epoch += tc['phase2_epochs_per_group']
    total_epochs = epoch
    return schedule, total_epochs


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2A (discriminator-init variant): pretrain a '
                     'DiscriminatorEncoder on synthetic (image, W) pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', default='latent_pipeline/configs/default.yaml',
                        help='Path to pipeline config YAML')
    parser.add_argument('--resume', default=None,
                        help='Checkpoint path to resume from (restores epoch, '
                             'freeze/unfreeze state, and optimizer state)')
    args = parser.parse_args()

    # SLURM (and any non-TTY redirect) fully buffers stdout by default --
    # line-buffer so progress shows up in real time when tailing a log file.
    sys.stdout.reconfigure(line_buffering=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tc = config['train_discriminator_init']
    repo_root = config['paths']['repo_root']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = os.path.join(repo_root, tc['checkpoint_dir'])
    os.makedirs(output_dir, exist_ok=True)

    # StyleGAN2 generator (for visual grids only -- not used in training loss)
    G = load_stylegan(config, device)

    # Encoder: D's trunk transplanted in, pretrained groups start frozen
    encoder = build_discriminator_encoder(config, device)

    # Dataset (resolution is fixed by the transplanted block chain)
    synthetic_dir = os.path.join(repo_root, config['paths']['synthetic_dir'])
    full_dataset = SyntheticWDataset(synthetic_dir, target_size=tc['train_resolution'])

    val_n = int(len(full_dataset) * tc['val_split'])
    train_n = len(full_dataset) - val_n
    train_dataset, val_dataset = random_split(
        full_dataset, [train_n, val_n],
        generator=torch.Generator().manual_seed(0),
    )
    print(f"Train: {train_n}  Val: {val_n}")

    # num_workers=0 default: on the laptop, num_workers=4 hung mid-epoch with
    # 0% GPU utilization for hours (a DataLoader worker deadlock -- data/dataset.py
    # documents the same issue for the real-frame dataset). Not confirmed
    # whether this is laptop-specific; configs/rorqual.yaml raises it since
    # train_frames.num_workers=8 already works fine there for stage2b.
    num_workers = tc.get('num_workers', 0)
    train_loader = DataLoader(train_dataset, batch_size=tc['batch_size'],
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=tc['batch_size'],
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    # Phase schedule: epoch -> list of events ('phase1_start', 'unfreeze:<name>')
    group_names = encoder.pretrained_group_names()
    schedule, total_epochs = build_phase_schedule(tc, group_names)
    print(f"Phase schedule: phase1={tc['phase1_epochs']} epochs (new params only), "
          f"then {len(group_names)} groups x {tc['phase2_epochs_per_group']} epochs "
          f"each = {total_epochs} epochs total")
    print(f"Unfreeze order: {group_names}")

    optimizer = torch.optim.Adam(encoder.new_parameters(), lr=tc['lr_new'])
    unfrozen_count = 0
    start_epoch = 0
    best_val_loss = float('inf')

    log_path = os.path.join(output_dir, 'train_discriminator_init_log.json')
    log_entries = []

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        start_epoch = ckpt['epoch'] + 1
        # Replay schedule events up to (and including) the checkpointed epoch
        # so the encoder's freeze state and the optimizer's param_groups
        # exactly match what they were when the checkpoint was saved --
        # load_state_dict below matches groups by position, not name.
        for e in range(start_epoch):
            for event in schedule.get(e, []):
                unfrozen_count = apply_unfreeze_event(encoder, optimizer, tc, event, unfrozen_count)
        encoder.load_state_dict(ckpt['encoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch} "
              f"({unfrozen_count}/{len(group_names)} pretrained groups unfrozen)")
        if os.path.exists(log_path):
            with open(log_path) as f:
                log_entries = json.load(f)
            log_entries = [e for e in log_entries if e['epoch'] < start_epoch]

    for epoch in range(start_epoch, total_epochs):
        for event in schedule.get(epoch, []):
            if event == 'phase1_start':
                print(f"Epoch {epoch}: phase 1 -- pretrained trunk frozen, "
                      f"training new params only")
            else:
                name = event.split(':', 1)[1]
                unfrozen_count = apply_unfreeze_event(encoder, optimizer, tc, event, unfrozen_count)
                print(f"Epoch {epoch}: unfroze '{name}' at lr={optimizer.param_groups[-1]['lr']:.2e} "
                      f"({unfrozen_count}/{len(group_names)} pretrained groups unfrozen)")

        t0 = time.time()
        train_loss = train_one_epoch(encoder, train_loader, optimizer, device)
        val_loss = validate(encoder, val_loader, device)
        elapsed = time.time() - t0

        lrs = [g['lr'] for g in optimizer.param_groups]
        print(f"Epoch {epoch:3d} | train_mse={train_loss:.6f}  "
              f"val_mse={val_loss:.6f} | lrs={['%.1e' % lr for lr in lrs]} | {elapsed:.0f}s")

        entry = {'epoch': epoch, 'train_mse': train_loss, 'val_mse': val_loss,
                 'lrs': lrs, 'elapsed_s': elapsed,
                 'unfrozen_groups': unfrozen_count}
        log_entries.append(entry)

        ckpt = {
            'epoch': epoch,
            'encoder': encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'encoder_arch': 'discriminator_init',
            'best_val_loss': best_val_loss,
            'config': config,
        }
        torch.save(ckpt, os.path.join(output_dir, 'latest.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt['best_val_loss'] = best_val_loss
            torch.save(ckpt, os.path.join(output_dir, 'best.pt'))
            print(f"  -> New best val_mse: {best_val_loss:.6f}")

        visual_every = tc.get('visual_check_every', 5)
        if (epoch + 1) % visual_every == 0 or epoch == 0:
            grid_path = os.path.join(output_dir, f'recon_epoch_{epoch:03d}.png')
            save_visual_grid(encoder, G, val_dataset, device, grid_path)
            print(f"  -> Saved visual grid: {grid_path}")

        with open(log_path, 'w') as f:
            json.dump(log_entries, f, indent=2)

    print(f"\ntrain_discriminator_init complete. Best val_mse: {best_val_loss:.6f}")
    print(f"Checkpoint: {output_dir}/best.pt")
    print(f"\nNext: fine-tune on real frames with "
          f"stage2b_train_frames.py --encoder-arch discriminator_init "
          f"--pretrained {output_dir}/best.pt")


if __name__ == '__main__':
    main()
