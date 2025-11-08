import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, x_dim, c_dim, z_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(x_dim + c_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )

        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

    def forward(self, x, c):
        h = torch.cat((x, c), dim=1)
        h = self.network(h)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(self, z_dim, c_dim, x_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(z_dim + c_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, x_dim)
        )

    def forward(self, z, c):
        h = torch.cat((z, c), dim=1)
        return self.network(h)


class cVAE(nn.Module):
    def __init__(self, x_dim, z_dim, c_dim):
        super().__init__()

        self.encoder = Encoder(x_dim, c_dim, z_dim)
        self.decoder = Decoder(z_dim, c_dim, x_dim)

    def reparameterize(self, z_mu, z_logvar):
        eps = torch.randn_like(z_mu)
        return z_mu + torch.exp(0.5 * z_logvar) * eps * 0.01

    def forward(self, x, c):
        z_mu, z_logvar = self.encoder(x, c)
        z = self.reparameterize(z_mu, z_logvar)
        x_recon = self.decoder(z, c)
        return x_recon, z_mu, z_logvar

    def compute_loss(self, x_recon, x, z_mu, z_logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        return recon_loss + kl_loss

    def save(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_path, "model_params.pt"))

    def load(self, checkpoint_path, device='cpu'):
        self.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
