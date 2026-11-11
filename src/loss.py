import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingLoss(nn.Module):

    def __init__(
        self,
        fft_sizes=(256, 512, 1024, 2048),
        hop_sizes=(64, 128, 256, 512),
        lambda_wave=1.0,
        lambda_sisdr=0.1,
        lambda_mrstft=0.5,
        lambda_complex=0.5,
        eps=1e-8,
    ):
        super().__init__()

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes

        self.lambda_wave = lambda_wave
        self.lambda_sisdr = lambda_sisdr
        self.lambda_mrstft = lambda_mrstft
        self.lambda_complex = lambda_complex

        self.eps = eps

    # -----------------------------------------
    # Waveform L1
    # -----------------------------------------

    def waveform_loss(self, pred, target):
        return F.l1_loss(pred, target)

    # -----------------------------------------
    # SI-SDR
    # -----------------------------------------

    def sisdr_loss(self, pred, target):

        pred = pred.squeeze(1)
        target = target.squeeze(1)

        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)

        target_energy = (
            target.pow(2).sum(dim=-1, keepdim=True)
            + self.eps
        )

        scale = (
            (pred * target).sum(dim=-1, keepdim=True)
            / target_energy
        )

        projection = scale * target
        noise = pred - projection

        ratio = (
            projection.pow(2).sum(dim=-1)
            /
            (noise.pow(2).sum(dim=-1) + self.eps)
        )

        sisdr = 10.0 * torch.log10(ratio + self.eps)

        return -sisdr.mean()

    # -----------------------------------------
    # STFT helper
    # -----------------------------------------

    def stft(self, audio, fft_size, hop_size):

        window = torch.hann_window(
            fft_size,
            device=audio.device,
            dtype=audio.dtype,
        )

        return torch.stft(
            audio,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window=window,
            return_complex=True,
            center=True,
        )

    # -----------------------------------------
    # Multi-resolution STFT
    # -----------------------------------------

    def mrstft_loss(self, pred, target):

        total = 0.0

        for fft_size, hop_size in zip(
            self.fft_sizes,
            self.hop_sizes,
        ):

            pred_spec = self.stft(pred, fft_size, hop_size)
            target_spec = self.stft(target, fft_size, hop_size)

            pred_mag = pred_spec.abs()
            target_mag = target_spec.abs()

            # Spectral convergence
            convergence = (
                torch.linalg.vector_norm(target_mag - pred_mag)
                /
                (torch.linalg.vector_norm(target_mag) + self.eps)
            )

            # Log magnitude distance
            log_mag = F.l1_loss(
                torch.log(pred_mag + self.eps),
                torch.log(target_mag + self.eps),
            )

            total = total + convergence + log_mag

        return total / len(self.fft_sizes)

    # -----------------------------------------
    # Complex STFT loss
    # -----------------------------------------

    def complex_loss(self, pred_spec, target_spec):

        difference = (pred_spec - target_spec).abs()

        return (
            difference.mean()
            /
            (target_spec.abs().mean() + self.eps)
        )

    # -----------------------------------------
    # Total loss
    # -----------------------------------------

    def forward(
        self,
        predicted_audio,
        predicted_spec,
        clean_audio,
        clean_spec,
    ):

        predicted_audio = predicted_audio.squeeze(1)
        clean_audio = clean_audio.squeeze(1)

        # Waveform losses
        wave_loss = self.waveform_loss(
            predicted_audio,
            clean_audio,
        )

        sisdr = self.sisdr_loss(
            predicted_audio.unsqueeze(1),
            clean_audio.unsqueeze(1),
        )

        audio_loss = (
            self.lambda_wave * wave_loss
            + self.lambda_sisdr * sisdr
        )

        # Spectral losses
        mrstft = self.mrstft_loss(
            predicted_audio,
            clean_audio,
        )

        complex_loss = self.complex_loss(
            predicted_spec,
            clean_spec,
        )

        spec_loss = (
            self.lambda_mrstft * mrstft
            + self.lambda_complex * complex_loss
        )

        total_loss = audio_loss + spec_loss

        return total_loss, audio_loss, spec_loss
