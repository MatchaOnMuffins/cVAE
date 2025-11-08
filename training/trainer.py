import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    def __init__(self, model, device='cuda', lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.train_losses = []
        self.valid_losses = []

    def train_epoch(self, train_loader, epoch, n_epochs):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs} [Train]", leave=False)
        for x_batch, c_batch in pbar:
            x_batch = x_batch.to(self.device)
            c_batch = c_batch.to(self.device)

            x_recon, z_mu, z_logvar = self.model(x_batch, c_batch)
            loss = self.model.compute_loss(x_recon, x_batch, z_mu, z_logvar)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate_epoch(self, valid_loader, epoch, n_epochs):
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(valid_loader, desc=f"Epoch {epoch}/{n_epochs} [Valid]", leave=False)

        with torch.no_grad():
            for x_batch, c_batch in pbar:
                x_batch = x_batch.to(self.device)
                c_batch = c_batch.to(self.device)

                x_recon, z_mu, z_logvar = self.model(x_batch, c_batch)
                loss = self.model.compute_loss(x_recon, x_batch, z_mu, z_logvar)

                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(valid_loader)
        return avg_loss

    def train(self, train_dataset, valid_dataset, n_epochs=30, batch_size=128):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch, n_epochs)
            self.train_losses.append(train_loss)

            valid_loss = self.validate_epoch(valid_loader, epoch, n_epochs)
            self.valid_losses.append(valid_loss)

            tqdm.write(f"Epoch {epoch:03d}: Train = {train_loss:.4f}, Val = {valid_loss:.4f}")

        return self.train_losses, self.valid_losses

    def save_checkpoint(self, save_path):
        self.model.save(save_path)

    def load_checkpoint(self, checkpoint_path):
        self.model.load(checkpoint_path, device=self.device)
