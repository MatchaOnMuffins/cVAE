import numpy as np
import torch
from torch.utils.data import TensorDataset
import scanpy as sc


def load_data(data_path):
    return sc.read_h5ad(data_path)


def prepare_datasets(adata, train_split='train', valid_split='valid'):
    train_mask = adata.obs['split'] == train_split
    valid_mask = adata.obs['split'] == valid_split

    X_train = adata.X[train_mask].A if hasattr(adata.X, "A") else adata.X[train_mask]
    X_valid = adata.X[valid_mask].A if hasattr(adata.X, "A") else adata.X[valid_mask]

    drug_dict = adata.uns['drug_embeddings']
    C_train = np.stack([drug_dict[d] for d in adata.obs.loc[train_mask, 'drug_name']])
    C_valid = np.stack([drug_dict[d] for d in adata.obs.loc[valid_mask, 'drug_name']])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    C_train_t = torch.tensor(C_train, dtype=torch.float32)
    X_valid_t = torch.tensor(X_valid, dtype=torch.float32)
    C_valid_t = torch.tensor(C_valid, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, C_train_t)
    valid_dataset = TensorDataset(X_valid_t, C_valid_t)

    x_dim = X_train_t.shape[1]
    c_dim = C_train_t.shape[1]

    return train_dataset, valid_dataset, x_dim, c_dim
