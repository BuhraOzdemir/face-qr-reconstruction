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
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_WORKERS = 2

EARLY_STOP_PATIENCE = 10

LR_SCHEDULER_PATIENCE = 4
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN_LR = 1e-6

# Resume checkpoint — her epoch sonunda üzerine yazılır
RESUME_CHECKPOINT = CHECKPOINT_DIR / "checkpoint_latest.pth"
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


def save_checkpoint(
    epoch: int,
    model,
    optimizer,
    scheduler,
    best_val_loss: float,
    epochs_without_improvement: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        },
        RESUME_CHECKPOINT,
    )


def load_checkpoint(model, optimizer, scheduler, device):
    """
    Eğer Drive'da checkpoint_latest.pth varsa yükler ve
    (start_epoch, best_val_loss, epochs_without_improvement) döner.
    Yoksa sıfırdan başlar.
    """
    if not RESUME_CHECKPOINT.exists():
        print("Checkpoint bulunamadı — eğitim baştan başlıyor.")
        return 1, float("inf"), 0

    print(f"Checkpoint bulundu: {RESUME_CHECKPOINT}")
    ckpt = torch.load(RESUME_CHECKPOINT, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    epochs_without_improvement = ckpt["epochs_without_improvement"]

    print(
        f"  → Epoch {ckpt['epoch']}'dan devam ediliyor  "
        f"(best_val_loss={best_val_loss:.6f})"
    )
    return start_epoch, best_val_loss, epochs_without_improvement


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

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch, best_val_loss, epochs_without_improvement = load_checkpoint(
        model, optimizer, scheduler, device
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

    logger = TrainingLogger(resume=(start_epoch > 1))

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch_bar = tqdm(
        range(start_epoch, EPOCHS + 1),
        desc="Training",
        unit="epoch",
        initial=start_epoch - 1,
        total=EPOCHS,
    )

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

        # ── Resume checkpoint (Drive'a her epoch kaydedilir) ──────────────────
        save_checkpoint(
            epoch, model, optimizer, scheduler,
            best_val_loss, epochs_without_improvement,
        )

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
