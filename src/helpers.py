import config
import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt
from math import gcd
from pathlib import Path
import soundfile as sf

# Pre-compute second order sections for filters
SSB_BANDPASS_SOS  = butter(4, config.SSB_BANDPASS, btype="bandpass", fs = config.SAMPLE_RATE, output="sos", )
NOISE_LOWPASS_SOS = butter(2, 1200, btype="lowpass", fs = config.SAMPLE_RATE, output="sos", )

# Filters
def ssb_bandpass(x:np.ndarray) -> np.ndarray:
    return sosfiltfilt(SSB_BANDPASS_SOS,x).astype(np.float32)

def noise(x:np.ndarray,rng:np.random.Generator,snr_db:float) -> np.ndarray:
    n:int = len(x)

    # Create noise
    white_noise:np.ndarray = rng.normal( 0.0, 1.0, size=n ).astype(np.float32)
    colored_noise:np.ndarray = sosfiltfilt(NOISE_LOWPASS_SOS, white_noise)
    total_noise:np.ndarray = 0.5 * white_noise + 0.5 * colored_noise

    # Calculate power levels and scale to snr
    signal_power = np.mean(x**2)
    noise_power = np.mean(total_noise**2)

    # Rescale
    if ( min(signal_power,noise_power) < 1e-12 ):
        total_noise = np.zeros_like(total_noise, dtype=np.float32)

    else:
        target_noise_power = ( signal_power / (10.0 ** (snr_db / 10.0)) )
        total_noise = total_noise * np.sqrt( target_noise_power / noise_power )

    # Add noise and return
    summed = total_noise + x
    return summed.astype(np.float32)

def load_audio(path:Path) -> np.ndarray:
    audio, sample_rate = sf.read( path, dtype="float32" )

    # Stereo -> Mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.ndim != 1:
        raise ValueError( f"Unexpected audio shape in {path}: {audio.shape}")

    # Fix broken shit
    if len(audio) == 0:
        audio = np.zeros(1, dtype=np.float32)

    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0, )

    # Resample
    if sample_rate != config.SAMPLE_RATE:
        g = gcd(sample_rate, config.SAMPLE_RATE)

        audio = resample_poly(
            audio,
            config.SAMPLE_RATE // g,
            sample_rate // g,
        )

    return audio

def normalize_rms(audio:np.ndarray, target_rms:float = 0.1) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2)))

    if ( rms < 1e-8 ):
        return audio.astype(np.float32)

    return ( audio * (target_rms / rms) ).astype(np.float32)
