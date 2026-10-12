"""CEASE: continual concept erasure with AIC and HOC.

Runs the continual protocol of Eq. (2): starting from the pretrained weights,
each of the N erasure requests is solved against the running weights and
applied on top of all previous edits, with the retain set held fixed.

Per step e = 1..N:
  1. Solve the closed-form update with AIC (``cease_edit.edit_model``).
  2. HOC (Eq. 12-14): if the accumulated historical update already perturbs the
     current target substantially, project the new update onto the orthogonal
     complement of the dominant left singular directions of Delta_hist.
  3. Commit the update to the live UNet and record the update-noise diagnostic
     of Eq. (4) on the erased targets and the retain set.

Flags:
  --aic        enable the Anchor Invariance Constraint
  --hoc        enable the Historical Orthogonality Constraint
  (neither)    reproduces the continually applied SPEED baseline

Outputs under --save_root:
  checkpoints/step_{e:03d}.pt   edited cross-attention weights after step e
  metrics.json                  per-step noise / interference diagnostics
  k_bank.pt                     fixed pre-edit token embeddings of erase/retain concepts
  delta_cum_final.pt            cumulative update after the last step
  run_config.json               the exact configuration and concept order
"""
import argparse
import json
import os
import random
import time

import pandas as pd
import torch
from diffusers import StableDiffusionPipeline

from cease_edit import LAYER_SCOPES, edit_model, last_subject_embedding, layer_in_scope
from concept_utils import load_concepts, normalize_concept
from src.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description='CEASE continual concept erasure')
    # Backbone and data
    parser.add_argument('--sd_ckpt', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--task', type=str, default='celebrity', choices=['celebrity', 'instance', 'style'],
                        help='celebrity: csv has a type column (erase/retain); '
                             'instance/style: first N concepts are erased, the next N are retained')
    parser.add_argument('--retain_path', type=str, default='data/100_celebrity.csv')
    parser.add_argument('--anchor', type=str, default='person',
                        help='shared anchor c_*; paper uses person / art / object')
    parser.add_argument('--n_concepts', type=int, default=100)
    parser.add_argument('--order_seed', type=int, default=-1,
                        help='if >= 0, permute the erasure order with this seed (order-robustness study)')
    # CEASE components
    parser.add_argument('--aic', action='store_true', default=False,
                        help='Anchor Invariance Constraint (AIC): add the anchor token to the invariance matrix K2')
    parser.add_argument('--hoc', action='store_true', default=False,
                        help='Historical Orthogonality Constraint (HOC): project each update away from historical directions')
    parser.add_argument('--hoc_ema', type=float, default=0.9,
                        help='EMA coefficient of the HOC interference statistic')
    parser.add_argument('--hoc_eta', type=float, default=1.5,
                        help='HOC trigger threshold = mean + eta * std of the interference statistic')
    parser.add_argument('--hoc_keep_frac', type=float, default=0.75,
                        help='upper bound on the fraction of output dimensions removed by HOC')
    # Closed-form solver hyperparameters (paper Appendix, Implementation Details)
    parser.add_argument('--params', type=str, default='V', choices=['V', 'K', 'KV'])
    parser.add_argument('--layer_scope', type=str, default='all', choices=LAYER_SCOPES,
                        help='cross-attention stage mask: all = All Stages (default), mid = Middle, fine = High, coarse = Low')
    parser.add_argument('--aug_num', type=int, default=10, help='augmented retain samples per concept')
    parser.add_argument('--threshold', type=float, default=1e-1, help='near-null projector eigenvalue threshold tau')
    parser.add_argument('--retain_scale', type=float, default=0.05)
    parser.add_argument('--lamb', type=float, default=0.0, help='invariance regularizer lambda')
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--disable_filter', dest='disable_filter', action='store_true',
                              help='disable the importance-based retain filter (default, multi-concept configuration)')
    filter_group.add_argument('--enable_filter', dest='disable_filter', action='store_false')
    parser.set_defaults(disable_filter=True)
    # Bookkeeping
    parser.add_argument('--save_root', type=str, default='logs/cease/celebrity')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save edited weights every N steps')
    parser.add_argument('--save_step_deltas', action='store_true', help='also save every per-step update')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda')
    seed_everything(args.seed)
    os.makedirs(os.path.join(args.save_root, 'checkpoints'), exist_ok=True)

    # ---- 1. Load the pipeline and the erase / retain concept lists ----
    pipeline = StableDiffusionPipeline.from_pretrained(args.sd_ckpt).to(device)
    df = pd.read_csv(args.retain_path)
    if 'concept' not in df.columns:
        raise ValueError(f'{args.retain_path} must contain a concept column')
    df['concept'] = df['concept'].map(normalize_concept)
    if args.task == 'celebrity':
        erase_concepts = df[df['type'] == 'erase']['concept'].drop_duplicates().tolist()[: args.n_concepts]
        retain_concepts = df[df['type'] == 'retain']['concept'].drop_duplicates().tolist()
    else:
        concepts = load_concepts(args.retain_path, limit=2 * args.n_concepts)
        erase_concepts = concepts[: args.n_concepts]
        retain_concepts = concepts[args.n_concepts: 2 * args.n_concepts]
    if args.order_seed >= 0:
        random.Random(args.order_seed).shuffle(erase_concepts)
        print(f'[order_seed={args.order_seed}] permuted erase order; first5={erase_concepts[:5]}')
    retain_texts = list(retain_concepts)
    print(f'{len(erase_concepts)} erase concepts, {len(retain_texts)} retain concepts '
          f'(task={args.task}, anchor={args.anchor!r}, aic={args.aic}, hoc={args.hoc}).')

    run_config = vars(args).copy()
    run_config.update({'erase_concepts': erase_concepts, 'retain_concepts': retain_concepts})
    with open(os.path.join(args.save_root, 'run_config.json'), 'w') as handle:
        json.dump(run_config, handle, indent=2)

    # ---- 2. Fixed pre-edit token embeddings used by HOC and by the diagnostics ----
    tokenizer, text_encoder = pipeline.tokenizer, pipeline.text_encoder
    with torch.no_grad():
        k_bank = {c: last_subject_embedding(c, tokenizer, text_encoder, device).squeeze(0).detach().clone().float()
                  for c in erase_concepts}
        k_retain_bank = {c: last_subject_embedding(c, tokenizer, text_encoder, device).squeeze(0).detach().clone().float()
                         for c in retain_concepts}
    torch.save({'erase': k_bank, 'retain': k_retain_bank}, os.path.join(args.save_root, 'k_bank.pt'))

    # ---- 3. Track the edited projections ----
    def matches(name):
        if args.params == 'V':
            return 'attn2.to_v' in name
        if args.params == 'K':
            return 'attn2.to_k' in name
        return 'attn2.to_k' in name or 'attn2.to_v' in name

    state = pipeline.unet.state_dict()
    layer_names = [n for n in state.keys() if matches(n) and layer_in_scope(n, args.layer_scope)]
    if not layer_names:
        raise ValueError(f'Layer scope {args.layer_scope!r} selected no {args.params} projections')
    print(f'Editing {len(layer_names)} layers ({args.params}, scope={args.layer_scope}).')

    prev_weights = {n: state[n].detach().clone().float() for n in layer_names}   # W^(e-1)
    delta_cum = {n: torch.zeros_like(prev_weights[n]) for n in layer_names}      # Delta_hist = sum_{j<e} Delta_j
    delta_per_step = {}                                                          # e -> {layer: Delta_e}
    hoc_mean = {n: 0.0 for n in layer_names}                                     # EMA of the HOC statistic
    hoc_sq = {n: 0.0 for n in layer_names}                                       # EMA of its square
    metrics = []

    # ---- 4. Continual erasure loop (Eq. 2) ----
    for e, target_concept in enumerate(erase_concepts, start=1):
        t0 = time.time()

        # (a) closed-form update against the running weights, with AIC if enabled
        edit_dict = edit_model(
            args=args,
            pipeline=pipeline,
            target_concepts=[target_concept],
            anchor_concepts=[args.anchor],
            retain_texts=retain_texts,
            device=device,
        )
        step_delta = {n: (edit_dict[n].detach().float() - prev_weights[n]).clone() for n in layer_names}

        # (b) HOC: project Delta_e away from the dominant output directions of Delta_hist
        if args.hoc and e > 1:
            k_e = k_bank[target_concept]
            for n in layer_names:
                D_hist = delta_cum[n]
                interference = (D_hist @ k_e).pow(2).sum().item()   # how much history already perturbs the target
                hoc_mean[n] = args.hoc_ema * hoc_mean[n] + (1 - args.hoc_ema) * interference
                hoc_sq[n] = args.hoc_ema * hoc_sq[n] + (1 - args.hoc_ema) * (interference ** 2)
                var = max(hoc_sq[n] - hoc_mean[n] ** 2, 0.0)
                trigger = hoc_mean[n] + args.hoc_eta * (var ** 0.5)
                if interference > trigger:
                    # Eq. 13: dominant left singular directions of Delta_hist via eigh(Delta_hist Delta_hist^T)
                    D = D_hist @ D_hist.T
                    eigvals, eigvecs = torch.linalg.eigh(D)
                    d_out = D.shape[0]
                    keep_max = max(1, int(args.hoc_keep_frac * d_out))
                    eigvals_desc, idx_desc = torch.sort(eigvals, descending=True)
                    pos_mask = eigvals_desc > eigvals_desc[0] * 1e-6
                    keep_k = min(int(pos_mask.sum().item()), keep_max)
                    U_hat = eigvecs[:, idx_desc[:keep_k]]
                    # Eq. 14: Delta_e <- (I - U_hat U_hat^T) Delta_e
                    Pi = torch.eye(d_out, device=device, dtype=D_hist.dtype) - U_hat @ U_hat.T
                    step_delta[n] = Pi @ step_delta[n]
                    edit_dict[n] = prev_weights[n] + step_delta[n]

        # (c) commit W^(e) = W^(e-1) + Delta_e
        for n in layer_names:
            delta_cum[n].add_(step_delta[n])
            prev_weights[n] = prev_weights[n] + step_delta[n]
        delta_per_step[e] = step_delta
        with torch.no_grad():
            for k, v in edit_dict.items():
                pipeline.unet.state_dict()[k].copy_(v)

        # (d) update-noise diagnostic (Eq. 4): ||Delta_hist c_i||^2 - ||Delta_i c_i||^2 for every erased target so far
        per_target_noise = {}
        for i, ci in enumerate(erase_concepts[:e], start=1):
            k_i = k_bank[ci]
            cum_sq, own_sq = 0.0, 0.0
            for n in layer_names:
                cum_sq += (delta_cum[n] @ k_i).pow(2).sum().item()
                own_sq += (delta_per_step[i][n] @ k_i).pow(2).sum().item()
            per_target_noise[ci] = {'cum_sq': cum_sq, 'own_sq': own_sq, 'noise': cum_sq - own_sq}
        retain_interf_sq = 0.0
        for kr in k_retain_bank.values():
            for n in layer_names:
                retain_interf_sq += (delta_cum[n] @ kr).pow(2).sum().item()
        retain_interf_sq /= max(len(k_retain_bank), 1)

        noise_avg = sum(d['noise'] for d in per_target_noise.values()) / len(per_target_noise)
        cur = per_target_noise[target_concept]
        step_metric = {
            'step': e,
            'target': target_concept,
            'seconds': time.time() - t0,
            'noise_E_avg_so_far': noise_avg,
            'cur_target_cum_sq': cur['cum_sq'],
            'cur_target_own_sq': cur['own_sq'],
            'cur_target_noise': cur['noise'],
            'retain_interference_avg_sq': retain_interf_sq,
        }
        metrics.append(step_metric)
        print(f'[step {e:03d}] target={target_concept!r:30s} noise_cur={cur["noise"]:.3e} '
              f'avg_noise={noise_avg:.3e} retain_intf={retain_interf_sq:.3e} ({step_metric["seconds"]:.1f}s)')

        if e % args.save_ckpt_every == 0 or e == len(erase_concepts):
            ckpt = {n: prev_weights[n].clone() for n in layer_names}
            torch.save(ckpt, os.path.join(args.save_root, 'checkpoints', f'step_{e:03d}.pt'))
        with open(os.path.join(args.save_root, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    torch.save({'delta_cum': delta_cum, 'layer_names': layer_names, 'erase_concepts': erase_concepts},
               os.path.join(args.save_root, 'delta_cum_final.pt'))
    if args.save_step_deltas:
        compact = {e: {n: d.half().cpu() for n, d in dd.items()} for e, dd in delta_per_step.items()}
        torch.save({'delta_per_step': compact, 'layer_names': layer_names, 'erase_concepts': erase_concepts},
                   os.path.join(args.save_root, 'step_deltas.pt'))
    print(f'Done. Metrics: {os.path.join(args.save_root, "metrics.json")}')


if __name__ == '__main__':
    main()
