from pathlib import Path
from torch.utils.data import Dataset, get_worker_info
import torch
import numpy as np

import config
import helpers

class CustomAudioDataset(Dataset):
    def __init__(self, folder:Path|None = None,training: bool = True,) -> None:
        self.rng = np.random.default_rng()
        self._rng_worker_id = None

        self.training = training

        self.folder = folder if ( folder is not None ) else config.TRAINING_DATA_FOLDER

        self.collect_files()

    def collect_files(self) -> None:

        self.collected_files = sorted(
            path
            for path in self.folder.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".wav"
        )

        if not self.collected_files:
            raise RuntimeError(f"No WAV files found in {self.folder.resolve()}" )

    def get_rng(self):
        worker = get_worker_info()

        if worker is None:
            return self.rng

        if self._rng_worker_id != worker.id:
            worker_seed = torch.initial_seed() % (2**32)
            self.rng = np.random.default_rng(worker_seed)
            self._rng_worker_id = worker.id

        return self.rng
    
    def __len__(self):
        return len(self.collected_files) * config.CROPS_PER_FILE

    def __getitem__(self, idx):
        file_idx = idx // config.CROPS_PER_FILE

        path = self.collected_files[file_idx]
        rng = self.get_rng()

        # Load audio and create 'ground' truth data
        audio = helpers.load_audio(path)
        audio = helpers.ssb_bandpass(audio)

        # Pad data ( should not be needed, but just to be sure )
        if len(audio) < config.SLIDING_WINDOW_SIZE:
            audio = np.pad(audio, (0, config.SLIDING_WINDOW_SIZE - len(audio)) )

        # Randomly select a segment
        max_start = len(audio) - config.SLIDING_WINDOW_SIZE
        start = rng.integers(0, max_start + 1)
        audio = audio[start:start + config.SLIDING_WINDOW_SIZE]

        # Normalize the absolute level of the window
        audio = helpers.normalize_rms(audio)

        # Add noise
        snr_db = float(rng.integers(config.SNR_RANGE[0], config.SNR_RANGE[1] + 1))
        audio_noise = helpers.noise( audio, rng, snr_db )

        return (
            torch.from_numpy(audio_noise.copy()).unsqueeze(0),
            torch.from_numpy(audio.copy()).unsqueeze(0)
        )