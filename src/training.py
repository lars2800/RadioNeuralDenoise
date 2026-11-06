import logging; LOGGER = logging.getLogger("training")

from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import tqdm

from dataset import CustomAudioDataset
from model import AudioDenoiser
from loss import DenoisingLoss
import config

class Training:
    def __init__(self) -> None:
        self.model = AudioDenoiser().to(config.DEVICE)
        self.criterion = DenoisingLoss()
        self.optimizer =  AdamW( self.model.parameters(), lr = config.LEARNING_RATE, weight_decay = config.WEIGHT_DECAY )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( self.optimizer, mode="min", factor=0.5, patience=3 )
        self.scaler = torch.amp.GradScaler( enabled = (config.USE_AMP and config.DEVICE.type == "cuda") )

        self.create_loaders()

    def create_loaders(self) -> None:
        training_dataset   = CustomAudioDataset(folder = config.TRAINING_DATA_FOLDER)
        validation_dataset = CustomAudioDataset(folder = config.VALIDATION_DATA_FOLDER)

        LOGGER.info(f"Training files: {len(training_dataset.collected_files)} | Validation files: {len(validation_dataset.collected_files)}")

        self.train_loader = DataLoader(
            training_dataset,
            batch_size = config.BATCH_SIZE,
            shuffle = True,
            num_workers = config.NUM_WORKERS,
            pin_memory = config.PIN_MEMORY,
            persistent_workers = (config.NUM_WORKERS > 0),
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size = config.BATCH_SIZE,
            shuffle = False,
            num_workers = config.NUM_WORKERS,
            pin_memory = config.PIN_MEMORY,
            persistent_workers = (config.NUM_WORKERS > 0),
        )

    def train(self) -> None:

        # Load if needed
        start_epoch = 0
        if config.CONTINUE_FROM_LAST_CHECKPOINT:
            start_epoch = self.load_model()

        # Start raining
        self.best_validation_loss = float("inf")
        for epoch in range(start_epoch, config.EPOCHS):
            self.run_training_epoch(epoch)
            self.run_validation_epoch(epoch)

        # Save
        self.save_model(config.EPOCHS)

    def save_model(self, epoch: int) -> None:
        checkpoint = {
            "epoch": epoch,
            "best_validation_loss": self.best_validation_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }

        config.MODEL_FOLDER.mkdir(exist_ok=True)
        torch.save(checkpoint, config.MODEL_FOLDER / f"model_{epoch}.pth")

    def load_model(self) -> int:

        try:
            models = [(x,x.name) for x in config.MODEL_FOLDER.iterdir() if x.is_file()]
            if ( len(models) == 0 ): raise FileNotFoundError
            models.sort(key=lambda x: x[1])
            checkpoint = torch.load( models[-1][0].absolute(), map_location=config.DEVICE, weights_only=True )

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            self.best_validation_loss = checkpoint["best_validation_loss"]

            start_epoch = checkpoint["epoch"] + 1

            LOGGER.info(f"Resuming training from epoch {start_epoch}")

            return start_epoch

        except FileNotFoundError:
            LOGGER.info("No checkpoint found. Starting training from scratch.")
            self.best_validation_loss = float("inf")
            return 0

    def run_training_epoch(self,epoch:int) -> None:

        self.model.train()

        total_loss = 0.0
        audio_total = 0.0
        spec_total = 0.0

        for noisy_audio, clean_audio in tqdm.tqdm( self.train_loader, desc="training epoch: " ):

            # Move training data to device
            noisy_audio = noisy_audio.to(config.DEVICE, non_blocking=True)
            clean_audio = clean_audio.to(config.DEVICE, non_blocking=True)

            # Reset optimizer and do a forward pass
            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast( config.DEVICE.type, enabled = (config.USE_AMP and config.DEVICE.type == "cuda") ):
                predicted_audio, predicted_spec = self.model(noisy_audio)
                loss, audio_loss, spec_loss = self.criterion(predicted_audio, predicted_spec, clean_audio, None )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Backward pass
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.GRAD_CLIP_NORM)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Save loss
            total_loss += loss.item()
            audio_total += audio_loss.item()
            spec_total += spec_loss.item()

        num_batches = len(self.train_loader)

        LOGGER.info( f"Training loss: {total_loss / num_batches:.6f} | Audio (SI-SNR): {audio_total / num_batches:.6f} | Spectrogram: {spec_total / num_batches:.6f}" )

    def run_validation_epoch(self,epoch:int) -> None:

        self.model.eval()
        validation_loss = 0.0

        with torch.inference_mode():
            for noisy_audio, clean_audio in tqdm.tqdm( self.validation_loader, desc="validation epoch: " ): # CHANGED: simple pair

                # Move training data to device
                noisy_audio = noisy_audio.to(config.DEVICE)
                clean_audio = clean_audio.to(config.DEVICE)

                # Forward pass
                predicted_audio, predicted_spec = self.model(noisy_audio)

                # Calculate the loss
                loss, _, _ = self.criterion( predicted_audio, predicted_spec, clean_audio, None )
                validation_loss += loss.item()

        validation_loss /= len(self.validation_loader)
        self.scheduler.step(validation_loss)

        LOGGER.info(f"Validation epoch loss: {validation_loss:.6f}")

        # Save if better then last save
        LOGGER.info(f"Saving model")
        self.best_validation_loss = validation_loss
        self.save_model(epoch)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Training().train()