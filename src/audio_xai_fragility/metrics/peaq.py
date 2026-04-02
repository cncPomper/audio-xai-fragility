"""Perceptual Evaluation of Audio Quality (PEAQ)-inspired metric.

This module implements a compact, differentiable, and dependency-light approximation
of the PEAQ signal path:

1. Outer/middle-ear transfer weighting
2. Time-frequency analysis
3. Internal-ear critical-band processing
4. Perceptual post-processing
5. Cognitive aggregation into a final quality score

The implementation is intentionally lightweight and suitable for quick objective
quality comparisons in ML workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PEAQResult:
    """Container for PEAQ output values.

    Attributes:
        odg: Objective Difference Grade in ``[-4, 0]`` where ``0`` is best.
        distortion_index: Non-negative weighted distortion index.
        mov: Model output variables used for the final aggregation.
    """

    odg: float
    distortion_index: float
    mov: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation."""
        return {
            "odg": self.odg,
            "distortion_index": self.distortion_index,
            "mov": self.mov,
        }


def _to_mono_tensor(signal: torch.Tensor | list[float] | tuple[float, ...]) -> torch.Tensor:
    tensor = torch.as_tensor(signal, dtype=torch.float32)
    if tensor.ndim == 1:
        return tensor
    if tensor.ndim == 2 and 1 in tensor.shape:
        return tensor.reshape(-1)
    raise ValueError("Signal must be 1D mono audio or a single-channel 2D tensor.")


def _hz_to_bark(frequency_hz: torch.Tensor) -> torch.Tensor:
    return 13.0 * torch.atan(0.00076 * frequency_hz) + 3.5 * torch.atan((frequency_hz / 7500.0) ** 2)


def _build_bark_filterbank(
    n_fft: int,
    sample_rate: int,
    n_bands: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    frequencies = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1, device=device, dtype=dtype)
    bark = _hz_to_bark(frequencies)
    bark_max = float(_hz_to_bark(torch.tensor(sample_rate / 2.0, device=device, dtype=dtype)))
    centers = torch.linspace(0.0, bark_max, n_bands, device=device, dtype=dtype)
    bandwidth = bark_max / max(1, n_bands - 1)

    filters = []
    for center in centers:
        distance = (bark - center).abs() / max(bandwidth, 1e-6)
        triangle = torch.clamp(1.0 - distance, min=0.0)
        filters.append(triangle)

    filterbank = torch.stack(filters, dim=0)
    normalization = filterbank.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return filterbank / normalization


def _a_weighting_linear(frequency_hz: torch.Tensor) -> torch.Tensor:
    f2 = frequency_hz.square().clamp_min(1e-12)
    ra_num = (12200.0**2) * f2.square()
    ra_den = (f2 + 20.6**2) * torch.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12200.0**2)
    ra = (ra_num / ra_den.clamp_min(1e-24)).clamp_min(1e-24)
    a_db = 20.0 * torch.log10(ra) + 2.0
    return torch.pow(10.0, a_db / 20.0)


def _frame(signal: torch.Tensor, frame_size: int, hop_size: int) -> torch.Tensor:
    if signal.numel() < frame_size:
        padding = frame_size - signal.numel()
        signal = torch.nn.functional.pad(signal, (0, padding))
    return signal.unfold(0, frame_size, hop_size)


def peaq(
    reference: torch.Tensor | list[float] | tuple[float, ...],
    test: torch.Tensor | list[float] | tuple[float, ...],
    sample_rate: int,
    frame_size: int = 2048,
    hop_size: int = 512,
    n_bands: int = 49,
) -> PEAQResult:
    """Compute a compact PEAQ-like quality estimate.

    Args:
        reference: Clean reference mono signal.
        test: Degraded/test mono signal.
        sample_rate: Audio sample rate in Hz.
        frame_size: STFT frame size.
        hop_size: STFT hop size.
        n_bands: Number of Bark-like critical bands.

    Returns:
        ``PEAQResult`` containing the aggregated quality indicators.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive.")
    if n_bands < 8:
        raise ValueError("n_bands must be at least 8.")

    ref = _to_mono_tensor(reference)
    deg = _to_mono_tensor(test)
    if ref.shape != deg.shape:
        raise ValueError("reference and test must have the same shape.")

    device = ref.device
    dtype = ref.dtype
    deg = deg.to(device=device, dtype=dtype)

    window = torch.hann_window(frame_size, device=device, dtype=dtype)
    ref_frames = _frame(ref, frame_size, hop_size) * window
    deg_frames = _frame(deg, frame_size, hop_size) * window

    ref_spec = torch.fft.rfft(ref_frames, n=frame_size, dim=-1)
    deg_spec = torch.fft.rfft(deg_frames, n=frame_size, dim=-1)

    ref_power = ref_spec.abs().square()
    deg_power = deg_spec.abs().square()

    freqs = torch.linspace(0.0, sample_rate / 2.0, frame_size // 2 + 1, device=device, dtype=dtype)
    outer_ear = _a_weighting_linear(freqs).clamp_min(1e-4)
    ref_weighted = ref_power * outer_ear
    deg_weighted = deg_power * outer_ear

    bark_bank = _build_bark_filterbank(frame_size, sample_rate, n_bands, device, dtype)
    ref_excitation = ref_weighted @ bark_bank.T
    deg_excitation = deg_weighted @ bark_bank.T

    ref_percept = torch.log1p(ref_excitation)
    deg_percept = torch.log1p(deg_excitation)

    temporal_kernel = torch.tensor([0.2, 0.6, 0.2], device=device, dtype=dtype).view(1, 1, -1)

    def smooth(x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(0, 1).unsqueeze(0)
        x_padded = torch.nn.functional.pad(x_t, (1, 1), mode="replicate")
        depthwise_kernel = temporal_kernel.repeat(x_t.shape[1], 1, 1)
        y = torch.nn.functional.conv1d(x_padded, depthwise_kernel, groups=x_t.shape[1])
        return y.squeeze(0).transpose(0, 1)

    ref_smooth = smooth(ref_percept)
    deg_smooth = smooth(deg_percept)
    error = deg_smooth - ref_smooth

    loudness_error = error.abs().mean()
    symmetric_disturbance = torch.sqrt((error.square().mean()) + 1e-12)
    asymmetric_disturbance = torch.relu(error).mean()

    ref_modulation = (ref_smooth[1:] - ref_smooth[:-1]).abs().mean()
    deg_modulation = (deg_smooth[1:] - deg_smooth[:-1]).abs().mean()
    modulation_diff = (deg_modulation - ref_modulation).abs()

    spectral_ratio = (deg_excitation + 1e-8) / (ref_excitation + 1e-8)
    noise_to_mask_ratio = torch.relu(spectral_ratio - 1.0).mean()

    mov = {
        "loudness_error": float(loudness_error.item()),
        "symmetric_disturbance": float(symmetric_disturbance.item()),
        "asymmetric_disturbance": float(asymmetric_disturbance.item()),
        "modulation_diff": float(modulation_diff.item()),
        "noise_to_mask_ratio": float(noise_to_mask_ratio.item()),
    }

    distortion_index = (
        2.2 * loudness_error
        + 1.7 * symmetric_disturbance
        + 1.2 * asymmetric_disturbance
        + 1.0 * modulation_diff
        + 0.8 * noise_to_mask_ratio
    )

    odg = torch.clamp(-4.0 * (1.0 - torch.exp(-1.8 * distortion_index)), min=-4.0, max=0.0)

    return PEAQResult(
        odg=float(odg.item()),
        distortion_index=float(distortion_index.item()),
        mov=mov,
    )
