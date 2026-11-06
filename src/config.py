from pathlib import Path

# Constants

KHZ = 1e3

# General

SAMPLE_RATE:int = int( 8.192*KHZ )
SLIDING_WINDOW_SIZE = 4096
SSB_BANDPASS = ( 0.3*KHZ, 2.7*KHZ ) # 300Hz - 2700Hz

STFT_FFT_SIZE = 512
STFT_HOP_SIZE = 128

CROPS_PER_FILE = 8

# Data preperation

DATASET_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
LIBRI_SPEECH_FOLDER:Path = Path("./data/LibriSpeech/")
LIBRI_SPEECH_FOLDER_DEV_CLEAN_FOLDER:Path = LIBRI_SPEECH_FOLDER / "dev-clean"
PREPARED_DATA_FOLDER:Path = Path("./data/prepared/")

TRAINING_DATA_FOLDER:Path = PREPARED_DATA_FOLDER / "training"
VALIDATION_DATA_FOLDER:Path = PREPARED_DATA_FOLDER / "validation"

# Training

CONTINUE_FROM_LAST_CHECKPOINT = True
SNR_RANGE = (-10, 10)
EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4
VALIDATION_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
USE_AMP = False # automatic mixed precision (only active on CUDA)
AUDIO_WEIGHT = 0.5
MODEL_FOLDER:Path = Path("./data/model/")

# Inference


# Torch

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if ( DEVICE.type == "cpu" ):
    torch.set_num_threads( 16 )
    torch.set_num_interop_threads(1)

PIN_MEMORY = ( DEVICE.type == "cuda" )