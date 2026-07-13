import csv

from src.config import LOG_DIR


class TrainingLogger:

    def __init__(self):
        self.file = LOG_DIR / "train_log.csv"

        with open(self.file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    def log(self, epoch: int, train_loss: float, val_loss: float = None):
        with open(self.file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}" if val_loss is not None else "",
            ])
