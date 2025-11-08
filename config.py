import torch
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_path: str = "data/raw/dataset.h5ad"
    train_split: str = "train"
    valid_split: str = "valid"
    z_dim: int = 32
    n_epochs: int = 30
    batch_size: int = 128
    encoder_hidden_dims: list[int] = [512, 256]
    decoder_hidden_dims: list[int] = [256, 512]
    dropout: float = 0.2
    negative_slope: float = 0.2
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path: str = "./checkpoints"
    loss_curve_path: str = "training_curves.png"
    latent_space_path: str = "latent_space.png"
    viz_drugs: list | None = None

    def __post_init__(self):
        if self.viz_drugs is None:
            self.viz_drugs = ["DMSO_TF", "Quinestrol"]


def get_config():
    return TrainingConfig()
