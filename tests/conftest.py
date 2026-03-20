"""Shared fixtures for LF Music Mapper test suite."""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def silence_1024():
    """1024-frame stereo silence buffer."""
    return np.zeros((1024, 2), dtype=np.float32)


@pytest.fixture
def sine_1024():
    """1024-frame stereo 440Hz sine wave at 48kHz."""
    t = np.arange(1024, dtype=np.float32) / 48000.0
    mono = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    return np.column_stack([mono, mono])


@pytest.fixture
def impulse_1024():
    """1024-frame stereo impulse (sample 0 = 1.0, rest = 0)."""
    buf = np.zeros((1024, 2), dtype=np.float32)
    buf[0, :] = 1.0
    return buf


@pytest.fixture
def noise_1024():
    """1024-frame stereo white noise."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((1024, 2)).astype(np.float32) * 0.5
