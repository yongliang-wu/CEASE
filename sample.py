"""Sample images from the original and the edited model under the paper protocol.

Sampling uses FP16, DPM-Solver with 20 steps, and classifier-free guidance 7.5.
Two prompt sources are supported, matching the benchmark protocol:

* ``--prompt_csv`` (celebrities): every row of ``data/100_celebrity.csv`` is one
  prompt with a fixed seed. The five prompts of a concept form one batch whose
  latents are drawn from the seed of the batch's first row; one image per row.
* ``--concepts_csv`` (styles, instances): the first ``--concept_limit`` concepts
  are combined with the five task templates in ``src/template.py``; a single
  seeded latent batch is shared across all prompts, ``--num_samples`` images each.

Examples:
  python sample.py --task celebrity --prompt_csv data/100_celebrity.csv \
      --mode original,edit --edit_ckpt logs/cease/celebrity/checkpoints/step_100.pt \
      --save_root outputs/celebrity

  python sample.py --task style --concepts_csv data/style.csv --concept_limit 200 \
      --num_samples 5 --batch_size 5 --mode original,edit \
      --edit_ckpt logs/cease/style/checkpoints/step_100.pt --save_root outputs/style
"""
import argparse
import copy
import os
import re
import warnings

import pandas as pd
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from tqdm import tqdm

from concept_utils import load_concepts, normalize_concept
from src.template import template_dict
from src.utils import get_textencoding, get_token, process_img, seed_everything

warnings.filterwarnings('ignore')


def load_edited_weights(unet, checkpoint_path):
    """Load the partial state dict written by ``cease_sequential.py`` into ``unet``."""
    state = torch.load(checkpoint_path, map_location='cpu')
    unknown = sorted(set(state) - set(unet.state_dict()))
    if unknown:
        raise KeyError(f'{len(unknown)} unexpected keys in {checkpoint_path}: {unknown[:5]}')
    dtype = next(unet.parameters()).dtype
    unet.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=False)
    return len(state)


@torch.no_grad()
def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, guidance_scale, desc=None):
    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps, desc=desc, leave=False):
        latent_model_input = scheduler.scale_model_input(torch.cat([latents] * 2), timestep)
        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample
        noise_uncond, noise_text = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
    return latents


def filename_for(prompt, index):
    return re.sub(r'[^\w\s]', '', prompt).replace(' ', '_') + f'_{index}.png'


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Sample images from the original and edited model')
    parser.add_argument('--sd_ckpt', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--edit_ckpt', type=str, default=None, help='checkpoint written by cease_sequential.py')
    parser.add_argument('--mode', type=str, default='original,edit', help='comma-separated subset of original, edit')
    parser.add_argument('--task', type=str, required=True, choices=list(template_dict.keys()))
    parser.add_argument('--prompt_csv', type=str, default=None,
                        help='csv with text, concept, seed columns (celebrity protocol)')
    parser.add_argument('--concepts_csv', type=str, default=None,
                        help='csv with a concept column, combined with the task templates (style/instance protocol)')
    parser.add_argument('--concept_start', type=int, default=0)
    parser.add_argument('--concept_limit', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=5, help='images per prompt (concepts_csv protocol)')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--total_timesteps', type=int, default=20)
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--seed', type=int, default=0, help='global seed (concepts_csv protocol)')
    parser.add_argument('--save_root', type=str, required=True)
    args = parser.parse_args()

    modes = [m.strip() for m in args.mode.split(',') if m.strip()]
    if not modes or set(modes) - {'original', 'edit'}:
        parser.error('--mode must be a non-empty subset of original, edit')
    if 'edit' in modes and args.edit_ckpt is None:
        parser.error('--edit_ckpt is required for mode edit')
    if (args.prompt_csv is None) == (args.concepts_csv is None):
        parser.error('provide exactly one of --prompt_csv or --concepts_csv')
    if args.num_samples % args.batch_size != 0:
        parser.error('--num_samples must be divisible by --batch_size')

    pipe = DiffusionPipeline.from_pretrained(args.sd_ckpt, safety_checker=None, torch_dtype=torch.float16).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    unets = {}
    if 'original' in modes:
        unets['original'] = pipe.unet
    if 'edit' in modes:
        unets['edit'] = copy.deepcopy(pipe.unet)
        print(f'Loaded {load_edited_weights(unets["edit"], args.edit_ckpt)} edited tensors from {args.edit_ckpt}')
    uncond = get_textencoding(get_token('', pipe.tokenizer), pipe.text_encoder)

    def run_batch(prompts, latent, concept, indices):
        embedding = get_textencoding(get_token(prompts, pipe.tokenizer), pipe.text_encoder)
        n = latent.shape[0]
        if embedding.shape[0] == 1:
            embedding = embedding.expand(n, -1, -1)
        text_embeddings = torch.cat([uncond.expand(n, -1, -1), embedding], dim=0)
        for mode, unet in unets.items():
            out_dir = os.path.join(args.save_root, concept, mode)
            os.makedirs(out_dir, exist_ok=True)
            latents = diffusion(unet, pipe.scheduler, latent, text_embeddings,
                                args.total_timesteps, args.guidance_scale, desc=f'{concept} | {mode}')
            for z, prompt, index in zip(latents, prompts if len(prompts) > 1 else prompts * n, indices):
                image = process_img(pipe.vae.decode(z.unsqueeze(0) / pipe.vae.config.scaling_factor, return_dict=False)[0])
                image.save(os.path.join(out_dir, filename_for(prompt, index)))

    if args.prompt_csv:
        # Celebrity protocol: one batch per concept, latents seeded by the first row's seed, one image per row.
        df = pd.read_csv(args.prompt_csv)
        df['concept'] = df['concept'].map(normalize_concept)
        for concept in df['concept'].drop_duplicates():
            rows = df[df['concept'] == concept]
            generator = torch.Generator('cpu').manual_seed(int(rows['seed'].iloc[0]))
            latent = torch.randn((len(rows), 4, 64, 64), generator=generator).to(pipe.device, dtype=pipe.dtype)
            run_batch(rows['text'].tolist(), latent, concept, [int(i) for i in rows['id']])
    else:
        # Style / instance protocol: five templates per concept, a shared seeded latent batch, num_samples images each.
        concepts = load_concepts(args.concepts_csv, start=args.concept_start, limit=args.concept_limit)
        templates = template_dict[args.task]
        seed_everything(args.seed, deterministic=True)
        bs = args.batch_size
        for batch_idx in range(args.num_samples // bs):
            latent = torch.randn(bs, 4, 64, 64).to(pipe.device, dtype=pipe.dtype)
            for concept in concepts:
                for template in templates:
                    prompt = template.format(concept)
                    run_batch([prompt], latent, concept, [idx + bs * batch_idx for idx in range(bs)])


if __name__ == '__main__':
    main()
