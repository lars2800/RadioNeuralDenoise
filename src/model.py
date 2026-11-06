#
# The model is a single mask-based denoiser in the complex STFT domain,
# the standard approach (DCCRN / DeepFilterNet / CRN style):
#
# noisy audio -> STFT (complex) -> UNet over (freq, time) -> complex mask -> mask * noisy_STFT -> ISTFT -> denoised audio
#
# The model only has to predict a per-bin gain/mask instead of synthesizing the
# waveform, and multiplying the noisy *complex* spectrum by a complex mask also
# corrects phase. STFT/ISTFT are differentiable, so everything stays trainable.
#
# Auther note: I have no clue how this one works, this is the work of an llm
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class AudioDenoiser(nn.Module):
    def __init__(self, base_channels=32, fft_size=None, hop_size=None):
        super().__init__()

        self.fft_size = fft_size if fft_size is not None else config.STFT_FFT_SIZE
        self.hop_size = hop_size if hop_size is not None else config.STFT_HOP_SIZE
        self.register_buffer( "window", torch.hann_window(self.fft_size), persistent=False, )

        # input has 2 channels (real + imaginary of the noisy STFT)
        self.enc1 = ConvBlock2D(2, base_channels)
        self.enc2 = ConvBlock2D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock2D(base_channels * 2, base_channels * 4)

        self.bottleneck = ConvBlock2D(base_channels * 4, base_channels * 8)

        # dilated convolutions at the bottleneck to enlarge the receptive
        # field without more pooling (broadband radio noise needs context)
        self.dilated = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=2, dilation=2),
            nn.GroupNorm(8, base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=4, dilation=4),
            nn.GroupNorm(8, base_channels * 8),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.dec3 = ConvBlock2D(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = ConvBlock2D(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = ConvBlock2D(base_channels * 2 + base_channels, base_channels)

        #2 output channels = real & imaginary parts of the complex mask
        self.mask_out = nn.Conv2d(base_channels, 2, 1)

        nn.init.zeros_(self.mask_out.weight)
        nn.init.zeros_(self.mask_out.bias) # pyright: ignore[reportArgumentType]

        with torch.no_grad():
            self.mask_out.bias[0] = 1.5 # pyright: ignore[reportOptionalSubscript]

    def stft(self, x:torch.Tensor) -> torch.Tensor:
        # complex STFT of the (B, 1, T) waveform -> (B, F, T) complex
        if ( self.window is None or self.window.device != x.device ):
            self.window = torch.hann_window(self.fft_size, device=x.device)

        return torch.stft(
            x.squeeze(1),
            n_fft = self.fft_size,
            hop_length = self.hop_size,
            window = self.window,
            return_complex = True,
        )

    def istft(self, spec:torch.Tensor, length:int) -> torch.Tensor:
        # inverse STFT back to (B, 1, T)
        return torch.istft(
            spec,
            n_fft = self.fft_size,
            hop_length = self.hop_size,
            window = self.window,
            length = length,
        )

    def forward(self, audio:torch.Tensor):
        length = audio.shape[-1]

        # Complex STFT of the noisy input
        noisy_spec = self.stft(audio) # (B, F, T) complex

        # Pack complex spectrum into a 2-channel "image" for the UNet
        x = torch.view_as_real(noisy_spec).permute(0, 3, 1, 2) # (B, 2, F, T)

        # UNet encoder / bottleneck / decoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        b = self.bottleneck(F.max_pool2d(e3, 2))
        b = self.dilated(b)

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Complex ratio mask, bounded with tanh for training stability
        mask = torch.tanh(self.mask_out(d1))# (B, 2, F, T)
        mask = torch.view_as_complex( mask.permute(0, 2, 3, 1).contiguous() )# (B, F, T) complex

        # Apply the mask to the noisy complex spectrum (corrects phase as well)
        clean_spec = (1.0 + mask) * noisy_spec

        # Back to waveform
        clean_audio = self.istft(clean_spec, length) # (B, T)

        return clean_audio.unsqueeze(1), clean_spec