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
        identity_loss=None,
        identity_loss_weight: float = 0.2,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        # Optional identity loss — used only during training
        self.identity_loss = identity_loss
        self.identity_loss_weight = identity_loss_weight

        # Apply identity loss every N batches to reduce GPU overhead
        self.identity_loss_freq = 10

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        images: torch.Tensor,
        use_identity: bool = True,
    ) -> torch.Tensor:
        """
        predictions : (B, 3, H, W) in [-1, 1]  — Tanh output
        images      : (B, 3, H, W) in [-1, 1]  — loader output (normalized)

        total_loss = L1+SSIM  [+ identity_loss_weight * identity_loss]
        Identity loss yalnızca use_identity=True olduğunda hesaplanır.
        """
        loss = self.criterion(predictions, images)

        if use_identity and self.identity_loss is not None:
            # Identity loss [0,1] bekliyor — her iki tensörü dönüştür
            pred_01 = (predictions + 1.0) * 0.5
            gt_01   = (images + 1.0) * 0.5
            id_loss = self.identity_loss(pred_01, gt_01)
            loss = loss + self.identity_loss_weight * id_loss

        return loss

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch}/{total_epochs}] Train",
            leave=False,
        )

        for batch_idx, (embeddings, images) in enumerate(pbar):
            embeddings = embeddings.to(self.device)
            images = images.to(self.device)

            predictions = self.model(embeddings)

            # Identity loss sadece her identity_loss_freq batch'te hesaplanır
            use_identity = (
                self.identity_loss is not None
                and batch_idx % self.identity_loss_freq == 0
            )
            loss = self._compute_loss(predictions, images, use_identity=use_identity)

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
            # Use only reconstruction loss for validation (faster + stable metric)
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
