from datetime import datetime

import numpy as np

import torch
import torchvision


def timestamped_filename(prefix='generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'


def noise(n, latent_dim):
    return torch.randn(n, latent_dim).cpu()


def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]


def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cpu()


class ImageGenerator:
    def __init__(self, model, num, num_image_tiles, save=True):
        self.save = save
        self.model = model
        self.num = num
        self.num_image_tiles = num_image_tiles
        self.num_rows = num_image_tiles

        self.ext = self.model.image_extension
        self.latent_dim = self.model.GAN.G.latent_dim
        self.image_size = self.model.GAN.G.image_size
        self.num_layers = self.model.GAN.G.num_layers

    @torch.no_grad()
    def evaluate(self):
        self.model.GAN.eval()
        latents, gen_noise = self.latents_and_noise()

        self.generate_regular(gen_noise, latents)

        self.generate_moving_averages(gen_noise, latents)

        self.generate_mixing_regularities(gen_noise)

    @torch.no_grad()
    def generate_mixing_regularities(self, gen_noise):
        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.LongTensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cpu()
            return torch.index_select(a, dim, order_index)

        nn = noise(self.num_rows, self.latent_dim)
        tmp1 = tile(nn, 0, self.num_rows)
        tmp2 = nn.repeat(self.num_rows, 1)
        tt = int(self.num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, self.num_layers - tt)]
        generated_images = self.model.generate_truncated(self.model.GAN.SE, self.model.GAN.GE, mixed_latents, gen_noise,
                                                         trunc_psi=self.model.trunc_psi)

        if self.save:
            torchvision.utils.save_image(generated_images, str(
                self.model.results_dir / self.model.name / f'{str(self.num)}-mr.{self.ext}'),
                                         nrow=self.num_rows)
        return generated_images

    @torch.no_grad()
    def generate_moving_averages(self, gen_noise, latents):
        generated_images = self.model.generate_truncated(self.model.GAN.SE, self.model.GAN.GE, latents, gen_noise,
                                                         trunc_psi=self.model.trunc_psi)

        if self.save:
            torchvision.utils.save_image(generated_images, str(
                self.model.results_dir / self.model.name / f'{str(self.num)}-ema.{self.ext}'),
                                         nrow=self.num_rows)

        return generated_images

    @torch.no_grad()
    def generate_regular(self, gen_noise, latents):
        generated_images = self.model.generate_truncated(self.model.GAN.S, self.model.GAN.G, latents, gen_noise,
                                                         trunc_psi=self.model.trunc_psi)

        if self.save:
            torchvision.utils.save_image(generated_images, str(
                self.model.results_dir / self.model.name / f'{str(self.num)}.{self.ext}'),
                                         nrow=self.num_rows)
        return generated_images

    def latents_and_noise(self):
        # latents and noise
        latents = noise_list(self.num_rows ** 2, self.num_layers, self.latent_dim)
        gen_noise = image_noise(self.num_rows ** 2, self.image_size)
        return latents, gen_noise


def evaluate(model, num_image_tiles=8):
    samples_name = timestamped_filename()
    image_generator = ImageGenerator(model, f'{samples_name}-0', num_image_tiles)
    image_generator.evaluate()
    print(f'sample images generated: {samples_name}')
