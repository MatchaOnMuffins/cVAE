from models import cVAE
from data import load_data, prepare_datasets
from training import Trainer
from visualization import plot_training_curves, plot_latent_space
from config import get_config


def main():
    config = get_config()

    adata = load_data(config.data_path)
    train_dataset, valid_dataset, x_dim, c_dim = prepare_datasets(
        adata, train_split=config.train_split, valid_split=config.valid_split
    )

    model = cVAE(x_dim=x_dim, c_dim=c_dim, z_dim=config.z_dim)
    trainer = Trainer(model, device=config.device, lr=config.learning_rate)

    train_losses, valid_losses = trainer.train(
        train_dataset,
        valid_dataset,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
    )

    trainer.save_checkpoint(config.model_save_path)
    plot_training_curves(train_losses, valid_losses, save_path=config.loss_curve_path)
    plot_latent_space(
        model,
        adata,
        device=config.device,
        drugs=config.viz_drugs,
        save_path=config.latent_space_path,
    )


if __name__ == "__main__":
    main()
