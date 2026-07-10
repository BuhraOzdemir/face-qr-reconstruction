import csv

from src.config import LOG_DIR


class TrainingLogger:

    def __init__(self):

        self.file = LOG_DIR / "train_log.csv"

        with open(self.file, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    "epoch",
                    "loss",
                ]
            )

    def log(self, epoch, loss):

        with open(self.file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    epoch,
                    loss,
                ]
            )