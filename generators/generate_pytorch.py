import os
from datetime import datetime

import torch

from models_pytorch import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def cast_list(el):
    return el if isinstance(el, list) else [el]


def timestamped_filename(prefix='generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'


def _generate_images(
        results_dir='./results',
        models_dir='./models',
        name='faces',
        load_from=-1,
        image_size=128,
        network_capacity=16,
        fmap_max=512,
        transparent=False,
        batch_size=16,
        gradient_accumulate_every=6,
        learning_rate=2e-4,
        lr_mlp=0.1,
        ttur_mult=1.5,
        rel_disc_loss=False,
        num_workers=None,
        save_every=1000,
        evaluate_every=1000,
        num_generate=1,
        num_image_tiles=8,
        trunc_psi=0.75,
        mixed_prob=0.9,
        fp16=False,
        no_pl_reg=False,
        cl_reg=False,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        no_const=False,
        aug_prob=0.,
        aug_types=['translation', 'cutout'],
        top_k_training=False,
        generator_top_k_gamma=0.99,
        generator_top_k_frac=0.5,
        dual_contrast_loss=False,
        dataset_aug_prob=0.,
        calculate_fid_every=None,
        calculate_fid_num_images=12800,
        clear_fid_cache=False,
        log=False
):
    model_args = dict(
        name=name,
        results_dir=results_dir,
        models_dir=models_dir,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        image_size=image_size,
        network_capacity=network_capacity,
        fmap_max=fmap_max,
        transparent=transparent,
        lr=learning_rate,
        lr_mlp=lr_mlp,
        ttur_mult=ttur_mult,
        rel_disc_loss=rel_disc_loss,
        num_workers=num_workers,
        save_every=save_every,
        evaluate_every=evaluate_every,
        num_image_tiles=num_image_tiles,
        trunc_psi=trunc_psi,
        fp16=fp16,
        no_pl_reg=no_pl_reg,
        cl_reg=cl_reg,
        fq_layers=fq_layers,
        fq_dict_size=fq_dict_size,
        attn_layers=attn_layers,
        no_const=no_const,
        aug_prob=aug_prob,
        aug_types=cast_list(aug_types),
        top_k_training=top_k_training,
        generator_top_k_gamma=generator_top_k_gamma,
        generator_top_k_frac=generator_top_k_frac,
        dual_contrast_loss=dual_contrast_loss,
        dataset_aug_prob=dataset_aug_prob,
        calculate_fid_every=calculate_fid_every,
        calculate_fid_num_images=calculate_fid_num_images,
        clear_fid_cache=clear_fid_cache,
        mixed_prob=mixed_prob,
        log=log
    )

    model = Trainer(**model_args)
    model.load(load_from)
    samples_name = timestamped_filename()
    for num in range(num_generate):
        model.evaluate(f'{samples_name}-{num}', num_image_tiles)
    print(f'sample images generated at {results_dir}/{name}/{samples_name}')


def main():
    _generate_images()


if __name__ == '__main__':
    main()
