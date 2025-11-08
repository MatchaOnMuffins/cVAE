import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import umap


def plot_training_curves(train_losses, valid_losses, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(valid_losses, label='Val Loss', linewidth=2, linestyle='--')

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_latent_space(model, adata, device='cuda', drugs=None, save_path=None):
    if drugs is None:
        drugs = ['DMSO_TF', 'Quinestrol']

    model.eval()
    model.to(device)

    mask = adata.obs['drug_name'].isin(drugs)
    subset = adata[mask]

    X = subset.X.A if hasattr(subset.X, "A") else subset.X
    X = torch.tensor(X, dtype=torch.float32).to(device)

    drug_dict = adata.uns['drug_embeddings']
    C = np.stack([drug_dict[d] for d in subset.obs['drug_name']])
    C = torch.tensor(C, dtype=torch.float32).to(device)

    with torch.no_grad():
        z_mu, z_logvar = model.encode(X, C)
        z = model.reparameterize(z_mu, z_logvar).cpu().numpy()

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(z)

    drug_labels, _ = pd.factorize(subset.obs['drug_name'].values)
    cell_labels, _ = pd.factorize(subset.obs['cell_line_name'].values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(embedding[:, 0], embedding[:, 1], c=drug_labels, cmap='viridis', s=5, alpha=0.8)
    axes[0].set_title("Colored by Drug Treatment", fontsize=12)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    axes[1].scatter(embedding[:, 0], embedding[:, 1], c=cell_labels, cmap='viridis', s=5, alpha=0.8)
    axes[1].set_title("Colored by Cell Type", fontsize=12)
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
