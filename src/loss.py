import torch
import torch.nn as nn
import torch.nn.functional as F

import config

#
# log-magnitude L1 on the predicted complex STFT + SI-SNR on the reconstructed waveform. SI-SNR is scale-invariant and the standard metric for speech denoising.
#

class DenoisingLoss(nn.Module):
    def __init__(self, fft_size=None, hop_size=None):
        super().__init__()

        self.spectrum_loss  = nn.L1Loss()
        self.si_snr_epsilon = 1e-8

        #TFT parameters, matching the model (used to build the target
        # complex STFT from the clean waveform when none is supplied)
        self.fft_size = fft_size if fft_size is not None else config.STFT_FFT_SIZE
        self.hop_size = hop_size if hop_size is not None else config.STFT_HOP_SIZE
        self.window = None
    
    def reference_stft(self, audio:torch.Tensor) -> torch.Tensor:
        if ( self.window is None or self.window.device != audio.device ):
            self.window = torch.hann_window(self.fft_size, device=audio.device)

        return torch.stft(
            audio.squeeze(1),
            n_fft = self.fft_size,
            hop_length = self.hop_size,
            window = self.window,
            return_complex = True,
        )

    def si_snr(self, estimated:torch.Tensor, reference:torch.Tensor) -> torch.Tensor:
        estimated = estimated - estimated.mean(dim=-1, keepdim=True)
        reference = reference - reference.mean(dim=-1, keepdim=True)

        # Project the estimate onto the reference (remove scaling/alignment freedom)
        projection = ( torch.sum(estimated * reference, dim=-1, keepdim=True) / (torch.sum(reference * reference, dim=-1, keepdim=True) + self.si_snr_epsilon) ) * reference
        residual = estimated - projection

        return 10.0 * torch.log10(
            (torch.sum(projection**2, dim=-1) + self.si_snr_epsilon) /
            (torch.sum(residual**2,   dim=-1) + self.si_snr_epsilon)
        ).mean()

    def forward( self, predicted_audio, predicted_spec, target_audio, target_spec ):
        # signature note: predicted_spec / target_spec are now COMPLEX STFTs
        # produced inside the model, not precomputed power spectrograms.
        # if no target spectrogram is passed, build it from the clean waveform.

        if ( target_spec is None ):
            target_spec = self.reference_stft(target_audio)

        # Log-magnitude L1 (log-compression so all bins contribute to the loss)
        loss_spec = F.l1_loss(
            torch.log1p(predicted_spec.abs().clamp_min(0.0)),
            torch.log1p(target_spec.abs().clamp_min(0.0)),
        )

        # SI-SNR on the reconstructed waveform (negated: we minimize loss)
        loss_audio = -self.si_snr( predicted_audio, target_audio )
        total_loss = loss_spec + config.AUDIO_WEIGHT * loss_audio

        return total_loss, loss_audio, loss_spec