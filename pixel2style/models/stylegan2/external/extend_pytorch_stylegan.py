import torch


def noise(n, latent_dim):
    return torch.randn(n, latent_dim).cpu()


def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]


def gen_latents(num_rows, num_layers, latent_dim):
    return noise_list(num_rows ** 2, num_layers, latent_dim)


def mean_latent(generator, style, n_latent, device):
    latents = torch.randn(n_latent, generator.latent_dim, device=device)
    means = style(latents).mean(0, keepdim=True)
    return means
