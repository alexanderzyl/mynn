"""
This file defines the core research contribution
"""
import pdb

import matplotlib

from generators.models_pytorch import Trainer

matplotlib.use('Agg')
import math
import torch

from torch import nn
from pixel2style.models.encoders.psp_encoders import GradualStyleEncoder
# from pixel2style.models.stylegan2.model import Generator

from pixel2style.configs.paths_config import model_paths


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(device)


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.set_opts(opts)
        self.latent_avg = None
        # compute number of style inputs based on the output resolution
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self._init_encoder()
        self._init_decoder()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        # self.load_weights()

    def _init_decoder(
            self,
            results_dir='./results',
            models_dir='../generators/models',
            name='faces',
            load_from=-1,
            image_size=128,
            network_capacity=16,
            transparent=False,
            batch_size=16,
            ttur_mult=1.5,
            rel_disc_loss=False,
            num_workers=None,
            save_every=1000,
            evaluate_every=1000,
            trunc_psi=0.75,
            mixed_prob=0.9,
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
            clear_fid_cache=False):
        model_args = dict(
            name=name,
            results_dir=results_dir,
            models_dir=models_dir,
            batch_size=batch_size,
            image_size=image_size,
            network_capacity=network_capacity,
            transparent=transparent,
            ttur_mult=ttur_mult,
            rel_disc_loss=rel_disc_loss,
            num_workers=num_workers,
            save_every=save_every,
            evaluate_every=evaluate_every,
            trunc_psi=trunc_psi,
            no_pl_reg=no_pl_reg,
            cl_reg=cl_reg,
            fq_layers=fq_layers,
            fq_dict_size=fq_dict_size,
            attn_layers=attn_layers,
            no_const=no_const,
            aug_prob=aug_prob,
            aug_types=aug_types,
            top_k_training=top_k_training,
            generator_top_k_gamma=generator_top_k_gamma,
            generator_top_k_frac=generator_top_k_frac,
            dual_contrast_loss=dual_contrast_loss,
            dataset_aug_prob=dataset_aug_prob,
            calculate_fid_every=calculate_fid_every,
            calculate_fid_num_images=calculate_fid_num_images,
            clear_fid_cache=clear_fid_cache,
            mixed_prob=mixed_prob)
        full_style_gan = Trainer(**model_args)
        full_style_gan.load(load_from)
        full_style_gan.GAN.eval()
        self.decoder = full_style_gan.GAN.G
        self.style = full_style_gan.GAN.S

    # generator = Generator(self.opts.output_size, 512, 8)
    # return generator

    def _init_encoder(self):
        encoder = GradualStyleEncoder(50, 'ir_se', self.opts)
        self.encoder = encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            self.__load_latent_avg(ckpt)
        else:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            # if input to encoder is not an RGB image, do not load the input layer weights
            if self.opts.label_nc != 0:
                encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
            self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            if self.opts.learn_in_w:
                self.__load_latent_avg(ckpt, repeat=1)
            else:
                self.__load_latent_avg(ckpt, repeat=self.opts.n_styles)

    def forward(self, x, resize=True, latent_mask=None, inject_latent=None, return_latents=False, alpha=None):
        latents = self.encoder(x)
        # normalize with respect to the center of an average face
        if self.opts.start_from_latent_avg:
            if self.opts.learn_in_w:
                latents = latents + self.latent_avg.repeat(latents.shape[0], 1)
            else:
                latents = latents + self.latent_avg.repeat(latents.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        latents[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * latents[:, i]
                    else:
                        latents[:, i] = inject_latent[:, i]
                else:
                    latents[:, i] = 0
        noise = image_noise(self.opts.batch_size, self.opts.output_size, device=self.opts.device)
        images = self.decoder(latents, noise)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, latents
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
