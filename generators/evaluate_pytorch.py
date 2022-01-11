from datetime import datetime

import numpy as np

import torch
import torchvision


def timestamped_filename(prefix='generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


class ImageGenerator:
    def __init__(self, model, sample_name='sample', num_image_tiles=8, trunc_psi=0.75, save=True, device='cpu'):
        self.device = device
        self.trunc_psi = trunc_psi
        self.save = save
        self.model = model
        self.sample_name = sample_name
        self.num_image_tiles = num_image_tiles
        self.num_rows = num_image_tiles

        self.ext = self.model.image_extension
        self.latent_dim = self.model.GAN.G.latent_dim
        self.image_size = self.model.GAN.G.image_size
        self.num_layers = self.model.GAN.G.num_layers
        self.model.GAN.eval()

    @torch.no_grad()
    def evaluate_all(self):
        # latents and noise
        latents = self.gen_latents()
        gen_noise = self.gen_noise()

        w_styles_s = self.map_styles(self.model.GAN.S, latents)
        self.generate_regular(w_styles_s, gen_noise)

        w_styles_se = self.map_styles(self.model.GAN.SE, latents)
        self.generate_moving_averages(gen_noise, w_styles_se)

        self.generate_mixing_regularities(gen_noise)

    @torch.no_grad()
    def generate_regular(self, w_styles, gen_noise):
        generated_images = self.generate_images(self.model.GAN.G, w_styles, gen_noise)

        if self.save:
            torchvision.utils.save_image(generated_images, str(
                self.model.results_dir / self.model.name / f'{str(self.sample_name)}.{self.ext}'),
                                         nrow=self.num_rows)
        return generated_images

    @torch.no_grad()
    def generate_moving_averages(self, gen_noise, w_styles):
        generated_images = self.generate_images(self.model.GAN.GE, w_styles, gen_noise)

        if self.save:
            torchvision.utils.save_image(generated_images, str(
                self.model.results_dir / self.model.name / f'{str(self.sample_name)}-ema.{self.ext}'),
                                         nrow=self.num_rows)

        return generated_images

    @torch.no_grad()
    def generate_mixing_regularities(self, gen_noise):
        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.LongTensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(self.device)
            return torch.index_select(a, dim, order_index)

        nn = torch.randn(self.num_rows, self.latent_dim).to(self.device)
        tmp1 = tile(nn, 0, self.num_rows)
        tmp2 = nn.repeat(self.num_rows, 1)
        tt = int(self.num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, self.num_layers - tt)]
        w_styles = self.map_styles(self.model.GAN.SE, mixed_latents)
        generated_images = self.generate_images(self.model.GAN.GE, w_styles, gen_noise)

        if self.save:
            torchvision.utils.save_image(generated_images, str(
                self.model.results_dir / self.model.name / f'{str(self.sample_name)}-mr.{self.ext}'),
                                         nrow=self.num_rows)
        return generated_images

    @torch.no_grad()
    def generate_images(self, generator, w_styles, noi):
        generated_images = evaluate_in_chunks(self.model.batch_size, generator, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def map_styles(self, style_mapper, style):
        w = map(lambda t: (style_mapper(t[0]), t[1]), style)
        w_truncated = self.model.truncate_style_defs(w, trunc_psi=self.trunc_psi)
        w_styles = styles_def_to_tensor(w_truncated)
        return w_styles

    def gen_noise(self):
        return self.__image_noise(self.num_rows ** 2, self.image_size)

    def __image_noise(self, n, im_size):
        return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).to(self.device)

    def gen_latents(self):
        return [(torch.randn(self.num_rows ** 2, self.latent_dim).to(self.device), self.num_layers)]


def evaluate(model, device, num_image_tiles=8):
    samples_name = timestamped_filename()
    image_generator = ImageGenerator(model, f'{samples_name}-0', num_image_tiles, device=device)
    image_generator.evaluate_all()
    print(f'sample images generated: {samples_name}')
