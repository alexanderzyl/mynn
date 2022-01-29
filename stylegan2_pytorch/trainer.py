import json
import math
import os
from functools import partial
from math import log2, floor
from pathlib import Path
from random import random
from shutil import rmtree

import aim
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import stylegan2_pytorch.stylegan2
from stylegan2_pytorch import StyleGAN2, NanException
from stylegan2_pytorch.datasets import Dataset
from stylegan2_pytorch.losses import gen_hinge_loss, hinge_loss, dual_contrastive_loss
from stylegan2_pytorch.stylegan2 import EMA
from stylegan2_pytorch.utils import cast_list, APEX_AVAILABLE, default, cycle, loss_backwards, mixed_list, noise_list, \
    image_noise, latent_to_w, styles_def_to_tensor, \
    gradient_accumulate_contexts, gradient_penalty, raise_if_nan, \
    calc_pl_lengths, is_empty, noise, evaluate_in_chunks, slerp, exists
from stylegan2_pytorch.version import __version__


class Trainer:
    def __init__(
            self,
            name='default',
            results_dir='results',
            models_dir='models',
            base_dir='./',
            image_size=128,
            network_capacity=16,
            fmap_max=512,
            transparent=False,
            batch_size=4,
            mixed_prob=0.9,
            gradient_accumulate_every=1,
            lr=2e-4,
            lr_mlp=0.1,
            ttur_mult=2,
            rel_disc_loss=False,
            num_workers=None,
            save_every=1000,
            evaluate_every=1000,
            num_image_tiles=8,
            trunc_psi=0.6,
            fp16=False,
            cl_reg=False,
            no_pl_reg=False,
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
            device='cpu',
            log=False,
            *args,
            **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        self.dual_contrast_loss = dual_contrast_loss

        assert not cl_reg, 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.device = device

        self.logger = aim.Session(experiment=name) if log else None

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr=self.lr, lr_mlp=self.lr_mlp, ttur_mult=self.ttur_mult, image_size=self.image_size,
                             network_capacity=self.network_capacity, fmap_max=self.fmap_max,
                             transparent=self.transparent, fq_layers=self.fq_layers, fq_dict_size=self.fq_dict_size,
                             attn_layers=self.attn_layers, fp16=self.fp16, cl_reg=self.cl_reg, no_const=self.no_const,
                             device=self.device, *args, **kwargs)

        if exists(self.logger):
            self.logger.set_params(self.hparams)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp,
                'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size,
                'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder):
        self.dataset = Dataset(folder, self.image_size, transparent=self.transparent, aug_prob=self.dataset_aug_prob)
        num_workers = num_workers = default(self.num_workers, 0)
        sampler = None
        dataloader = data.DataLoader(self.dataset, num_workers=num_workers,
                                     batch_size=math.ceil(self.batch_size), sampler=sampler,
                                     shuffle=True, drop_last=True, pin_memory=True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).to(self.device)
        total_gen_loss = torch.tensor(0.).to(self.device)

        batch_size = math.ceil(self.batch_size)

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob = self.aug_prob
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        S = self.GAN.S
        G = self.GAN.G
        D = self.GAN.D
        D_aug = self.GAN.D_aug

        backwards = partial(loss_backwards, self.fp16)

        if exists(self.GAN.D_cl):
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
                    style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.device)
                    noise = image_noise(batch_size, image_size, device=self.device)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(w_styles, noise)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                image_batch = next(self.loader).to(self.device)
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt, loss_id=0)

            self.GAN.D_opt.step()

        # setup losses

        if not self.dual_contrast_loss:
            D_loss_fn = hinge_loss
            G_loss_fn = gen_hinge_loss
            G_requires_reals = False
        else:
            D_loss_fn = dual_contrastive_loss
            G_loss_fn = dual_contrastive_loss
            G_requires_reals = True

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every):
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(batch_size, image_size, device=self.device)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = G(w_styles, noise)
            fake_output, fake_q_loss = D_aug(generated_images.clone().detach(), detach=True, **aug_kwargs)

            image_batch = next(self.loader).to(self.device)
            image_batch.requires_grad_()
            real_output, real_q_loss = D_aug(image_batch, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            if self.rel_disc_loss:
                real_output_loss = real_output_loss - fake_output.mean()
                fake_output_loss = fake_output_loss - real_output.mean()

            divergence = D_loss_fn(real_output_loss, fake_output_loss)
            disc_loss = divergence

            if self.has_fq:
                quantize_loss = (fake_q_loss + real_q_loss).mean()
                self.q_loss = float(quantize_loss.detach().item())

                disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id=1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every):
            style = get_latents_fn(batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(batch_size, image_size, device=self.device)

            w_space = latent_to_w(S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = G(w_styles, noise)
            fake_output, _ = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            real_output = None
            if G_requires_reals:
                image_batch = next(self.loader).to(self.device)
                real_output, _ = D_aug(image_batch, detach=True, **aug_kwargs)
                real_output = real_output.detach()

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            loss = G_loss_fn(fake_output_loss, real_output)
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, loss_id=2)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.track(self.g_loss, 'G')

        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.steps % 10 == 0 and self.steps > 20000:
            stylegan2_pytorch.stylegan2.EMA()

        if self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results
        if self.steps % self.save_every == 0:
            self.save(self.checkpoint_num)

        if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
            self.evaluate(floor(self.steps / self.evaluate_every))

        if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
            num_batches = math.ceil(self.calculate_fid_num_images / self.batch_size)
            fid = self.calculate_fid(num_batches)
            self.last_fid = fid

            with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                f.write(f'{self.steps},{fid}\n')

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num=0, trunc=1.0):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = self.num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.device)
        n = image_noise(num_rows ** 2, image_size, device=self.device)

        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'),
                                     nrow=num_rows)

        # moving averages

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'),
                                     nrow=num_rows)

        # mixing regularities

        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(self.device)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim, device=self.device)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi=self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'),
                                     nrow=num_rows)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(image, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.device)
            noise = image_noise(self.batch_size, image_size, device=self.device)

            # moving averages
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise,
                                                       trunc_psi=self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image,
                                             str(fake_path / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi=0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.device)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)

        av_torch = torch.from_numpy(self.av).to(self.device)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi=0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi=trunc_psi)
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi=0.75, num_image_tiles=8):
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi=trunc_psi)
        w_styles = styles_def_to_tensor(w_truncated)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num=0, num_image_tiles=8, trunc=1.0, num_steps=100, save_frames=False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim, device=self.device)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.device)
        n = image_noise(num_rows ** 2, image_size, device=self.device)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi=self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow=num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())

            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)

            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:],
                       duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name=name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__
        }

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name), map_location=self.device)

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print(
                'unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])


class ModelLoader:
    def __init__(self, *, base_dir, name='default', load_from=-1):
        self.model = Trainer(name=name, base_dir=base_dir)
        self.model.load(load_from)

    def noise_to_styles(self, noise, trunc_psi=None):
        noise = noise.cuda()
        w = self.model.GAN.SE(noise)
        if exists(trunc_psi):
            w = self.model.truncate_style(w)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device=0)

        images = self.model.GAN.GE(w_tensors, noise)
        images.clamp_(0., 1.)
        return images