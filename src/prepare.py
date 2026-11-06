import logging; LOGGER = logging.getLogger("data-prep")
from pathlib import Path
import soundfile as sf
from tqdm import tqdm
import numpy as np
import urllib.request
import tarfile

import config

class DataPrep:
    def __init__(self) -> None:
        pass

    def verify_data(self) -> None:
        LOGGER.info("Verifying data")

        # Main folder
        if ( not config.LIBRI_SPEECH_FOLDER.exists() ):
            LOGGER.critical(f"{config.LIBRI_SPEECH_FOLDER.absolute()} does not exsist! Did you download the training data?"); raise FileNotFoundError

        # Dev-clean folder
        if ( not (config.LIBRI_SPEECH_FOLDER_DEV_CLEAN_FOLDER).exists() ):
            LOGGER.critical(f"{config.LIBRI_SPEECH_FOLDER_DEV_CLEAN_FOLDER.absolute()} does not exsist!\n Did you download the dev-clean version from teh training data?"); raise FileNotFoundError

    def collect_files(self) -> None:
        LOGGER.info("Collecting files")

        self.collected_files:dict[str, list[Path]] = {}

        for folder in [ x for x in config.LIBRI_SPEECH_FOLDER_DEV_CLEAN_FOLDER.iterdir() if x.is_dir() ]:
            speaker:str = folder.name

            for sub_folder in [ x for x in folder.iterdir() if x.is_dir() ]:
                for flac_file in [ x for x in sub_folder.iterdir() if x.is_file() and x.name.endswith(".flac") ]:
                    self.collected_files.setdefault(speaker, []).append(flac_file)

    def split_speakers(self) -> None:
        LOGGER.info("Splitting speakers into training / validation")

        speakers:list[str] = sorted(self.collected_files.keys())
        rng = np.random.default_rng(42)
        rng.shuffle(speakers)

        validation_count:int = max(1, int(len(speakers) * config.VALIDATION_RATIO))
        self.validation_speakers:set[str] = set(speakers[:validation_count])

        LOGGER.info(f"{len(speakers)} speakers total -> {validation_count} for validation, {len(speakers) - validation_count} for training")

    def convert_files(self) -> None:
        LOGGER.info("Converting files")

        config.TRAINING_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        config.VALIDATION_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

        stale_files:list[Path] = [ x for x in config.PREPARED_DATA_FOLDER.iterdir() if x.is_file() and x.suffix.lower() == ".wav" ]
        if stale_files:
            LOGGER.warning(f"{len(stale_files)} .wav files sit directly in {config.PREPARED_DATA_FOLDER} (old flat layout). They are no longer used -- you can delete them.")

        training_index:int = 0
        validation_index:int = 0

        for speaker, flac_files in self.collected_files.items():
            for flac_file in tqdm( flac_files, desc=f"speaker {speaker}" ):
                audio, sample_rate = sf.read( flac_file.resolve() )

                if ( speaker in self.validation_speakers ):
                    output_path:Path = config.VALIDATION_DATA_FOLDER / f"{validation_index}.wav"
                    validation_index = validation_index + 1
                else:
                    output_path = config.TRAINING_DATA_FOLDER / f"{training_index}.wav"
                    training_index = training_index + 1

                sf.write(output_path.resolve(), audio, sample_rate)

        LOGGER.info(f"Wrote {training_index} training files to {config.TRAINING_DATA_FOLDER} and {validation_index} validation files to {config.VALIDATION_DATA_FOLDER}")

    def download_dataset(self) -> None:
        LOGGER.info("Checking LibriSpeech dataset...")

        config.LIBRI_SPEECH_FOLDER.mkdir(parents=True, exist_ok=True)

        # Skip download if dataset already exists
        if config.LIBRI_SPEECH_FOLDER_DEV_CLEAN_FOLDER.exists():
            LOGGER.info("Dataset already downloaded. Skipping download.")
            return

        archive_path = config.LIBRI_SPEECH_FOLDER / "dev-clean.tar.gz"

        LOGGER.info("Downloading LibriSpeech dev-clean dataset...")

        urllib.request.urlretrieve(
            config.DATASET_URL,
            archive_path
        )

        LOGGER.info("Download complete. Extracting dataset...")

        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(
                path=config.LIBRI_SPEECH_FOLDER,
                filter="data"
            )
                
        # Fix LibriSpeech's nested directory structure
        nested_folder = config.LIBRI_SPEECH_FOLDER / "LibriSpeech"

        if nested_folder.exists():
            nested_dev_clean = nested_folder / "dev-clean"

            if nested_dev_clean.exists():
                import shutil

                shutil.move(
                    str(nested_dev_clean),
                    str(config.LIBRI_SPEECH_FOLDER / "dev-clean")
                )

                shutil.rmtree(nested_folder)

        LOGGER.info("Dataset extraction complete.")

        # Remove archive to save disk space
        archive_path.unlink()

        LOGGER.info("Removed downloaded archive.")

    def run(self) -> None:
        self.download_dataset()
        self.verify_data()
        self.collect_files()
        self.split_speakers()
        self.convert_files()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    DataPrep().run()