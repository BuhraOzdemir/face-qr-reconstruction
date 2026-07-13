import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.loader import FaceDataset
from src.model.face_reconstruction import FaceReconstructionModel
from src.training.losses import ReconstructionLoss
from src.training.identity_loss import IdentityLoss
from src.training.trainer import Trainer
from src.training.logger import TrainingLogger
from src.config import CHECKPOINT_DIR, SSIM_LOSS_WEIGHT, IDENTITY_LOSS_WEIGHT


# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

# Early stopping
EARLY_STOP_PATIENCE = 10

# ReduceLROnPlateau
LR_SCHEDULER_PATIENCE = 4
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN_LR = 1e-6
# ─────────────────────────────────────────────────────────────────────────────


def build_loader(split: str, shuffle: bool) -> DataLoader:
    dataset = FaceDataset(split=split)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 50)
    print(f"Device           : {device}")
    print(f"Epochs           : {EPOCHS}")
    print(f"Batch size       : {BATCH_SIZE}")
    print(f"LR               : {LEARNING_RATE}")
    print(f"SSIM weight      : {SSIM_LOSS_WEIGHT}")
    print(f"Identity weight  : {IDENTITY_LOSS_WEIGHT}")
    print("=" * 50)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader = build_loader("train", shuffle=True)
    val_loader = build_loader("val", shuffle=False)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")

    # ── Model / loss / optimizer ──────────────────────────────────────────────
    model = FaceReconstructionModel().to(device)
    criterion = ReconstructionLoss(ssim_weight=SSIM_LOSS_WEIGHT)

    # Identity loss: frozen InsightFace model, training-only
    print("Identity Loss yükleniyor...")
    id_loss = IdentityLoss(device=device)
    print(f"Identity Loss hazır  (weight={IDENTITY_LOSS_WEIGHT})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        identity_loss=id_loss,
        identity_loss_weight=IDENTITY_LOSS_WEIGHT,
    )

    logger = TrainingLogger()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")

    for epoch in epoch_bar:
        train_loss = trainer.train_epoch(epoch, EPOCHS)
        val_loss = trainer.validate_epoch(epoch, EPOCHS)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix(
            train=f"{train_loss:.6f}",
            val=f"{val_loss:.6f}",
            lr=f"{current_lr:.2e}",
        )
        print(
            f"Epoch [{epoch:>3}/{EPOCHS}] "
            f"Train: {train_loss:.6f}  "
            f"Val: {val_loss:.6f}  "
            f"LR: {current_lr:.2e}"
        )

        # ── Checkpoint every epoch ────────────────────────────────────────────
        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR / f"model_epoch_{epoch:03d}.pth",
        )

        # ── Best model ────────────────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best_model.pth",
            )
            print(f"  ✓ best_model.pth güncellendi  (val_loss={best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1

        # ── Sample visualization ──────────────────────────────────────────────
        trainer.save_sample(epoch)

        # ── Logging ───────────────────────────────────────────────────────────
        logger.log(epoch, train_loss, val_loss)

        # ── Early stopping ────────────────────────────────────────────────────
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(
                f"\nEarly stopping: {EARLY_STOP_PATIENCE} epoch boyunca "
                f"val_loss iyileşmedi. Eğitim durduruluyor."
            )
            break

    print(f"\nEğitim tamamlandı. En iyi val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
