from torch import nn
from torch.optim import Adam

from stylegan2_pytorch.utils import exists
from stylegan2_pytorch.styles import StyleVectorizer
from stylegan2_pytorch.discriminator import Discriminator, AugWrapper
from stylegan2_pytorch.generator import Generator


class StyleGAN2(nn.Module):
    def __init__(self, image_size, latent_dim=512, fmap_max=512, style_depth=8, network_capacity=16, transparent=False,
                 fp16=False, cl_reg=False, steps=1, lr=1e-4, ttur_mult=2, fq_layers=[], fq_dict_size=256,
                 attn_layers=[], no_const=False, lr_mlp=0.1, device='cpu'):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers,
                           no_const=no_const, fmap_max=fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers=fq_layers, fq_dict_size=fq_dict_size,
                               attn_layers=attn_layers, transparent=transparent, fmap_max=fmap_max)

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers,
                            no_const=no_const)

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # experimental contrastive loss discriminator regularization
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr=self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr=self.lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.to(device)

        # startup apex mixed precision
        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize(
                [self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool