"""Tests for `audio_xai_fragility` package."""

import pytest
import torch

import audio_xai_fragility
from audio_xai_fragility.metrics.peaq import peaq


def test_import():
    """Verify the package can be imported."""
    assert audio_xai_fragility


def test_peaq_identical_signals_are_high_quality():
    sample_rate = 16_000
    duration = 1.0
    time = torch.linspace(0.0, duration, int(sample_rate * duration), dtype=torch.float32)
    reference = 0.5 * torch.sin(2.0 * torch.pi * 440.0 * time)

    result = peaq(reference, reference.clone(), sample_rate=sample_rate)

    assert -4.0 <= result.odg <= 0.0
    assert result.odg > -0.1
    assert result.distortion_index < 0.1


def test_peaq_detects_noise_degradation():
    sample_rate = 16_000
    duration = 1.0
    time = torch.linspace(0.0, duration, int(sample_rate * duration), dtype=torch.float32)
    reference = 0.5 * torch.sin(2.0 * torch.pi * 440.0 * time)
    noisy = reference + 0.08 * torch.randn_like(reference)

    clean_result = peaq(reference, reference, sample_rate=sample_rate)
    noisy_result = peaq(reference, noisy, sample_rate=sample_rate)

    assert noisy_result.distortion_index > clean_result.distortion_index
    assert noisy_result.odg < clean_result.odg


def test_peaq_rejects_shape_mismatch():
    sample_rate = 16_000
    reference = torch.zeros(16_000)
    test = torch.zeros(8_000)

    with pytest.raises(ValueError, match="same shape"):
        peaq(reference, test, sample_rate=sample_rate)
