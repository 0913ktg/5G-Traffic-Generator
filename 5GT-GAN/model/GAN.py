import torch
from pytorch_lightning import LightningModule
import torch.nn as nn
from torch.optim import Adam
from model.submodels.Discriminator import Discriminator
from model.submodels.Generator import Generator


class GAN(LightningModule):
    def __init__(self, seq_len, latent_dim, cond_dim, hidden, n_layers, sample_size, lr=2e-4):
        super(GAN, self).__init__()
        self.G = Generator(latent_dim + cond_dim, hidden=hidden, num_layers=n_layers)
        self.D = Discriminator(input_size=cond_dim + 1, n_layers=8, n_channel=10, kernel_size=8, seq_len=seq_len,
                               dropout=0)

        self.lr = lr

        self.d_optim = Adam(self.D.parameters(), lr=self.lr)
        self.g_optim = Adam(self.G.parameters(), lr=self.lr)

        self.seq_len = seq_len
        self.sample_size = sample_size
        self.latent_dim = latent_dim

        self.bce_loss = nn.BCELoss()
        self.fm_loss = nn.MSELoss()

        self.fixed_z = self.sample_Z(sample_size, seq_len, latent_dim).to(self.device)

        self.save_hyperparameters()

    def forward(self, z, c):
        return self.G(z, c, self.device)

    def feature_loss(self, fmap_r, fmap_g):
        losses = [self.fm_loss(dg, dr) for dr, dg in zip(fmap_r, fmap_g)]
        loss = sum(losses)

        return loss * 2

    def training_step(self, batch, batch_idx, optimizer_idx):
        y, c = batch
        z = self.sample_Z(batch_size=len(y), seq_length=self.seq_len, latent_dim=self.latent_dim).to(self.device)

        if optimizer_idx == 0:
            y_h = self(z, c)

            fake_d, features_f = self.D(y_h, c)

            _, features_r = self.D(y.unsqueeze(-1), c)

            g_loss = self.bce_loss(fake_d, torch.ones_like(fake_d).to(self.device))
            f_loss = self.feature_loss(features_r, features_f)

            loss = g_loss + f_loss

            self.log('gen_loss', loss.clone())
            self.log('adv_g_loss', g_loss.clone())
            self.log('feature_loss', f_loss.clone())

            return {'loss': loss, 'g_loss': g_loss.detach(), 'f_loss': f_loss.detach()}
        else:
            fake = self(z, c)

            fake_d, _ = self.D(fake, c)
            real_d, _ = self.D(y.unsqueeze(-1), c)

            loss_real = self.bce_loss(real_d, torch.ones_like(real_d).to(self.device) * 0.9)
            loss_fake = self.bce_loss(fake_d, torch.zeros_like(fake_d).to(self.device))

            d_loss = loss_real + loss_fake

            self.log('disc_loss', d_loss.clone())

            return d_loss

    def create_condition(self, condition, len_data):
        label = torch.tensor([[condition] * self.seq_len] * len_data)

        return label.to(self.device)

    def configure_optimizers(self):
        return [self.g_optim, self.d_optim], []

    def sample_Z(self, batch_size, seq_length, latent_dim):
        sample = torch.randn(batch_size, seq_length, latent_dim).to(self.device)

        return sample
