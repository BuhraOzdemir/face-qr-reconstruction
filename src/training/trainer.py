import torch
from tqdm import tqdm

from src.training.visualizer import save_comparison


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch}/{total_epochs}] Train",
            leave=False,
        )

        for embeddings, images in pbar:
            embeddings = embeddings.to(self.device)
            images = images.to(self.device)

            predictions = self.model(embeddings)
            loss = self.criterion(predictions, images)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch [{epoch}/{total_epochs}]   Val",
            leave=False,
        )

        for embeddings, images in pbar:
            embeddings = embeddings.to(self.device)
            images = images.to(self.device)

            predictions = self.model(embeddings)
            loss = self.criterion(predictions, images)

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.6f}")

        return total_loss / len(self.val_loader)

    @torch.no_grad()
    def save_sample(self, epoch: int):
        self.model.eval()

        embeddings, images = next(iter(self.val_loader))
        embeddings = embeddings.to(self.device)
        images = images.to(self.device)

        generated = self.model(embeddings)

        save_comparison(
            generated[0],
            images[0],
            epoch,
        )
