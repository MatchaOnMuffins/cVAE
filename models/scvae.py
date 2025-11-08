import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class scVAE(nn.Module):
    def __init__(self, x_dim, z_dim, c_dim):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.enc_fc1 = nn.Linear(x_dim + c_dim, 512)
        self.enc_bn1 = nn.BatchNorm1d(512)
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_bn2 = nn.BatchNorm1d(256)

        self.fc_mu = nn.Linear(256, z_dim)
        self.fc_logvar = nn.Linear(256, z_dim)

        self.dec_fc1 = nn.Linear(z_dim + c_dim, 256)
        self.dec_bn1 = nn.BatchNorm1d(256)
        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_bn2 = nn.BatchNorm1d(512)
        self.dec_fc3 = nn.Linear(512, x_dim)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=0.2)

    def encode(self, x, c):
        h = torch.cat((x, c), dim=1)

        h = self.enc_fc1(h)
        h = self.enc_bn1(h)
        h = self.leaky_relu(h)
        h = self.dropout(h)

        h = self.enc_fc2(h)
        h = self.enc_bn2(h)
        h = self.leaky_relu(h)
        h = self.dropout(h)

        z_mu = self.fc_mu(h)
        z_logvar = self.fc_logvar(h)

        return z_mu, z_logvar

    def reparameterize(self, z_mu, z_logvar):
        eps = torch.randn_like(z_mu)
        z = z_mu + torch.exp(0.5 * z_logvar) * eps * 0.01
        return z

    def decode(self, z, c):
        h = torch.cat((z, c), dim=1)

        h = self.dec_fc1(h)
        h = self.dec_bn1(h)
        h = self.leaky_relu(h)
        h = self.dropout(h)

        h = self.dec_fc2(h)
        h = self.dec_bn2(h)
        h = self.leaky_relu(h)
        h = self.dropout(h)

        x_recon = self.dec_fc3(h)

        return x_recon

    def forward(self, x, c):
        z_mu, z_logvar = self.encode(x, c)
        z = self.reparameterize(z_mu, z_logvar)
        x_recon = self.decode(z, c)
        return x_recon, z_mu, z_logvar

    def compute_loss(self, x_recon, x, z_mu, z_logvar):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        return recon_loss + kl_loss

    def save(self, model_path):
        os.makedirs(model_path, exist_ok=True)
        save_path = os.path.join(model_path, "model_params.pt")
        torch.save(self.state_dict(), save_path)

    def load(self, checkpoint_path, device='cpu'):
        self.load_state_dict(torch.load(checkpoint_path, map_location=device))
