import torch
from torch.utils.data import DataLoader

from src.dataset.loader import FaceDataset
from src.model.face_reconstruction import FaceReconstructionModel
from src.training.losses import ReconstructionLoss
from src.training.trainer import Trainer
from src.config import CHECKPOINT_DIR
from src.training.logger import TrainingLogger


def main():

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print(f"Device : {device}")
    print("=" * 50)

    # Dataset
    dataset = FaceDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    # Model
    model = FaceReconstructionModel().to(device)

    # Loss
    criterion = ReconstructionLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )
    
    # Logger
    logger = TrainingLogger()

    epochs = 10

    for epoch in range(epochs):

        loss = trainer.train_epoch()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss:.6f}"
        )

        torch.save(
        model.state_dict(),
        CHECKPOINT_DIR / f"model_epoch_{epoch+1:03d}.pth",
        )
        trainer.save_sample(epoch)
        logger.log(epoch + 1, loss)

if __name__ == "__main__":
    main()