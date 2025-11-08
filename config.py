import torch
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_path: str = "HW5.h5ad.gz"
    train_split: str = "train"
    valid_split: str = "valid"
    z_dim: int = 32
    n_epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_save_path: str = "./scVAE_model"
    loss_curve_path: str = "training_curves.png"
    latent_space_path: str = "latent_space.png"
    viz_drugs: list = None

    def __post_init__(self):
        if self.viz_drugs is None:
            self.viz_drugs = ['DMSO_TF', 'Quinestrol']


def get_config():
    return TrainingConfig()
