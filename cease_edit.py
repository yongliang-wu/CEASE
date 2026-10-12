"""Closed-form erasure step with the Anchor Invariance Constraint (AIC).

This module implements one closed-form editing step of CEASE. It solves the
weighted least-squares problem of Eq. (1) for the cross-attention value
projection, right-projects the update through the near-null projector of the
augmented retain set (Appendix "Near-Null Projector"), and applies the
invariance correction on the matrix K2. The base solver follows the public
SPEED implementation.

AIC (Eq. 9) is enabled by ``args.aic``: the anchor token representation c_*
is appended to K2, so the solved update satisfies Delta @ c_* = 0 (up to
numerical precision when ``args.lamb == 0``).

The Historical Orthogonality Constraint (HOC) acts on the update returned by
``edit_model`` and is implemented in ``cease_sequential.py``.
"""
import re

import torch
from kmeans_pytorch import kmeans
from tqdm import tqdm


LAYER_SCOPES = ('all', 'mid', 'fine', 'coarse')


def layer_in_scope(layer_name, scope):
    """Return whether a cross-attention projection belongs to a UNet stage mask.

    ``all`` (All Stages) is the default used in the main experiments. The other
    scopes are used only in the layer-scope ablation (Fig. "Cross-attention
    layer-scope ablation"): ``mid`` = Middle, ``fine`` = High resolution,
    ``coarse`` = Low resolution.
    """
    if scope == 'all':
        return True
    prefixes = {
        'mid': ('mid_block.',),
        'fine': ('down_blocks.0.', 'up_blocks.3.'),
        'coarse': ('down_blocks.2.', 'mid_block.', 'up_blocks.1.'),
    }
    if scope not in prefixes:
        raise ValueError(f'Unknown layer scope {scope!r}; expected one of {LAYER_SCOPES}')
    return layer_name.startswith(prefixes[scope])


def get_token_id(prompt, tokenizer, return_ids_only=True):
    token_ids = tokenizer(
        prompt, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt',
    )
    return token_ids.input_ids if return_ids_only else token_ids


def last_subject_embedding(text, tokenizer, text_encoder, device):
    """Hidden state of the last subject token (the token before EOS), shape [1, d0]."""
    inputs = get_token_id(text, tokenizer, return_ids_only=False)
    embs = text_encoder(inputs.input_ids.to(device)).last_hidden_state[0]
    last_idx = inputs.attention_mask[0].sum().item() - 2
    return embs[[last_idx], :]


def generate_perturbed_embs(ret_embs, P, erase_weight, num_per_sample, mini_batch=8):
    """Retain-set augmentation along the least-significant right singular direction."""
    ret_embs = ret_embs.squeeze(1)
    out_embs, norm_list = [], []
    for i in range(0, ret_embs.size(0), mini_batch):
        mini_ret_embs = ret_embs[i:i + mini_batch]
        for _ in range(num_per_sample):
            noise = torch.randn_like(mini_ret_embs)
            perturbed_embs = mini_ret_embs + noise @ P
            out_embs.append(perturbed_embs)
            norm_list.append(torch.matmul(perturbed_embs, erase_weight.T).norm(dim=1))
    out_embs = torch.cat(out_embs, dim=0)
    norm_list = torch.cat(norm_list, dim=0)
    return out_embs[norm_list > norm_list.mean()].unsqueeze(1)  # [Num, 1, d0]


@torch.no_grad()
def edit_model(args, pipeline, target_concepts, anchor_concepts, retain_texts, chunk_size=128, device='cuda'):
    """Solve one closed-form erasure step on the current UNet weights.

    Args:
        args: namespace with ``params``, ``layer_scope``, ``aic``, ``aug_num``,
            ``disable_filter``, ``threshold``, ``retain_scale``, ``lamb``.
        pipeline: a diffusers StableDiffusionPipeline whose UNet holds the
            running weights W^(i-1).
        target_concepts: list of concepts to erase at this step.
        anchor_concepts: list of anchors, aligned with ``target_concepts``.
        retain_texts: list of retain prompts (the retain set R).

    Returns:
        dict mapping edited layer names to the updated weights W^(i-1) + Delta.
    """
    tokenizer, text_encoder = pipeline.tokenizer, pipeline.text_encoder
    emb_size = text_encoder.config.hidden_size
    I = torch.eye(emb_size, device=device)

    # ---- Select the cross-attention projections to edit ----
    if args.params == 'KV':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_k' in k or 'attn2.to_v' in k}
    elif args.params == 'V':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_v' in k}
    elif args.params == 'K':
        edit_dict = {k: v for k, v in pipeline.unet.state_dict().items() if 'attn2.to_k' in k}
    else:
        raise ValueError(f'Unknown parameter selection {args.params!r}; expected V, K, or KV')
    layer_scope = getattr(args, 'layer_scope', 'all')
    edit_dict = {k: v for k, v in edit_dict.items() if layer_in_scope(k, layer_scope)}
    if not edit_dict:
        raise ValueError(f'Layer scope {layer_scope!r} selected no {args.params} projections')

    # ---- Invariance matrix K2 (empty-prompt representation; plus the anchor under AIC) ----
    null_inputs = get_token_id('', tokenizer, return_ids_only=False)
    null_hidden = text_encoder(null_inputs.input_ids.to(device)).last_hidden_state[0]
    _, cluster_centers = kmeans(X=null_hidden[1:], num_clusters=3, distance='euclidean', device='cuda')
    K2_columns = [null_hidden[[0], :], cluster_centers.to(device)]
    if getattr(args, 'aic', False):
        # AIC (Eq. 9): K2 <- [K2, c_*] so that Delta @ c_* = 0.
        anchor_set = set(a for a in anchor_concepts if a is not None and a.strip() != '')
        for a in anchor_set:
            K2_columns.append(last_subject_embedding(a, tokenizer, text_encoder, device))
    K2 = torch.cat(K2_columns, dim=0).T
    I2 = torch.eye(len(K2.T), device=device)

    # ---- Target and anchor statistics (C_1, C_*) ----
    sum_anchor_target, sum_target_target = [], []
    for i in range(len(target_concepts)):
        target_embs = last_subject_embedding(target_concepts[i], tokenizer, text_encoder, device)
        anchor_embs = last_subject_embedding(anchor_concepts[i], tokenizer, text_encoder, device)
        sum_target_target.append(target_embs.T @ target_embs)
        sum_anchor_target.append(anchor_embs.T @ target_embs)
    sum_target_target = torch.stack(sum_target_target).mean(0)
    sum_anchor_target = torch.stack(sum_anchor_target).mean(0)

    # ---- Retain set (C_0): last-subject-token embeddings ----
    retain_texts = [
        text for text in retain_texts
        if not any(re.search(r'\b' + re.escape(concept.lower()) + r'\b', text.lower()) for concept in target_concepts)
    ]
    assert len(retain_texts) + len(target_concepts) == len(set(retain_texts + target_concepts))
    last_ret_embs = []
    for j in range(0, len(retain_texts), chunk_size):
        ret_inputs = get_token_id(retain_texts[j:j + chunk_size], tokenizer, return_ids_only=False)
        ret_embs = text_encoder(ret_inputs.input_ids.to(device)).last_hidden_state
        if retain_texts == ['']:
            last_ret_embs.append(ret_embs[:, 1:, :].permute(1, 0, 2))
        else:
            last_subject_indices = ret_inputs.attention_mask.sum(1) - 2
            last_ret_embs.append(ret_embs[torch.arange(ret_embs.size(0)), last_subject_indices].unsqueeze(1))
    last_ret_embs = torch.cat(last_ret_embs)
    last_ret_embs = last_ret_embs[torch.randperm(last_ret_embs.size(0))]

    # ---- Per-layer closed-form update ----
    for layer_name, layer_weight in tqdm(edit_dict.items(), desc='Model Editing'):
        erase_weight = layer_weight @ (sum_anchor_target - sum_target_target) @ (I + sum_target_target).inverse()
        _, _, V0 = torch.svd(layer_weight)
        P0_min = V0[:, -1:] @ V0[:, -1:].T

        if args.aug_num > 0 and not args.disable_filter:
            weight_norm_init = torch.matmul(last_ret_embs.squeeze(1), erase_weight.T).norm(dim=1)
            layer_ret_embs = last_ret_embs[weight_norm_init > weight_norm_init.mean()]
        else:
            layer_ret_embs = last_ret_embs

        sum_ret_ret, valid_num = [], 0
        for j in range(0, len(layer_ret_embs), chunk_size):
            chunk_ret_embs = layer_ret_embs[j:j + chunk_size]
            if args.aug_num > 0:
                chunk_ret_embs = torch.cat(
                    [chunk_ret_embs, generate_perturbed_embs(chunk_ret_embs, P0_min, erase_weight, num_per_sample=args.aug_num)],
                    dim=0,
                )
            valid_num += chunk_ret_embs.shape[0]
            sum_ret_ret.append((chunk_ret_embs.transpose(1, 2) @ chunk_ret_embs).sum(0))
        sum_ret_ret = torch.stack(sum_ret_ret, dim=0).sum(0) / valid_num

        # Near-null projector P of the retain Gram matrix (eigenvalues below the threshold).
        U, S, _ = torch.svd(sum_ret_ret)
        P = U[:, S < args.threshold] @ U[:, S < args.threshold].T
        M = (sum_target_target @ P + args.retain_scale * I).inverse()
        # Closed-form update with the K2 invariance correction; Delta @ K2 = 0 when lamb = 0.
        delta_weight = (
            layer_weight @ (sum_anchor_target - sum_target_target) @ P
            @ (I - M @ K2 @ (K2.T @ P @ M @ K2 + args.lamb * I2).inverse() @ K2.T @ P) @ M
        )
        edit_dict[layer_name] = layer_weight + delta_weight

    print(f'Edited {target_concepts} -> {anchor_concepts}')
    return edit_dict
