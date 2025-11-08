import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(
        self,
        x_dim: int,
        c_dim: int,
        z_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        layers = []
        input_dim = x_dim + c_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(negative_slope),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim

        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat((x, c), dim=1)
        h = self.network(h)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    def __init__(
        self,
        z_dim: int,
        c_dim: int,
        x_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        layers = []
        input_dim = z_dim + c_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(negative_slope),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], x_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = torch.cat((z, c), dim=1)
        return self.network(h)


class cVAE(nn.Module):
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        c_dim: int,
        encoder_hidden_dims: list[int] = [512, 256],
        decoder_hidden_dims: list[int] = [256, 512],
        dropout: float = 0.2,
        negative_slope: float = 0.2,
    ):
        super().__init__()

        self.encoder = Encoder(
            x_dim, c_dim, z_dim, encoder_hidden_dims, dropout, negative_slope
        )
        self.decoder = Decoder(
            z_dim, c_dim, x_dim, decoder_hidden_dims, dropout, negative_slope
        )

    def reparameterize(
        self, z_mu: torch.Tensor, z_logvar: torch.Tensor
    ) -> torch.Tensor:
        eps = torch.randn_like(z_mu)
        return z_mu + torch.exp(0.5 * z_logvar) * eps * 0.01

    def forward(
        self, x: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mu, z_logvar = self.encoder(x, c)
        z = self.reparameterize(z_mu, z_logvar)
        x_recon = self.decoder(z, c)
        return x_recon, z_mu, z_logvar

    def compute_loss(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        z_mu: torch.Tensor,
        z_logvar: torch.Tensor,
    ) -> torch.Tensor:
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
        return recon_loss + kl_loss

    def save(self, model_path: str) -> None:
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(model_path, "model_params.pt"))

    def load(self, checkpoint_path: str, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
