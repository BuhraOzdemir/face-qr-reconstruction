import torch
from src.training.visualizer import save_comparison

class Trainer:

    def __init__(
        self,
        model,
        dataloader,
        criterion,
        optimizer,
        device,
    ):

        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0

        for embeddings, images in self.dataloader:

            embeddings = embeddings.to(self.device)
            images = images.to(self.device)

            predictions = self.model(embeddings)

            loss = self.criterion(predictions, images)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.dataloader)
    @torch.no_grad()
    def save_sample(self, epoch):

        self.model.eval()

        embeddings, images = next(iter(self.dataloader))

        embeddings = embeddings.to(self.device)
        images = images.to(self.device)

        generated = self.model(embeddings)

        save_comparison(
            generated[0],
            images[0],
            epoch + 1,
        )

        self.model.train()
    