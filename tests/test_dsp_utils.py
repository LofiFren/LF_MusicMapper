"""Tests for DSP utility functions in effects.py."""

import math
import numpy as np
import pytest
from effects import beat_to_samples, make_biquad_sos, LFO, BEAT_FRACTIONS


# ── beat_to_samples ──

class TestBeatToSamples:
    def test_quarter_note_120bpm(self):
        # 1 beat at 120 BPM = 0.5 sec = 24000 samples at 48kHz
        assert beat_to_samples(1.0, 120.0, 48000) == 24000

    def test_eighth_note_120bpm(self):
        assert beat_to_samples(0.5, 120.0, 48000) == 12000

    def test_sixteenth_note_120bpm(self):
        assert beat_to_samples(0.25, 120.0, 48000) == 6000

    def test_whole_note_120bpm(self):
        assert beat_to_samples(2.0, 120.0, 48000) == 48000

    def test_minimum_is_1(self):
        # Very small values should still return at least 1
        assert beat_to_samples(0.0001, 999.0, 48000) >= 1

    def test_bpm_floor_at_1(self):
        # BPM <= 0 should not crash (max(bpm, 1.0))
        result = beat_to_samples(1.0, 0.0, 48000)
        assert result > 0

    def test_different_sample_rates(self):
        s48 = beat_to_samples(1.0, 120.0, 48000)
        s44 = beat_to_samples(1.0, 120.0, 44100)
        assert s48 > s44  # higher sample rate = more samples

    def test_all_beat_fractions(self):
        for label, frac in BEAT_FRACTIONS:
            result = beat_to_samples(frac, 120.0, 48000)
            assert result >= 1, f"Beat fraction {label} returned {result}"


# ── make_biquad_sos ──

class TestMakeBiquadSos:
    def test_lpf_shape(self):
        sos = make_biquad_sos('lpf', 1000, 0.707, 48000)
        assert sos.shape == (1, 6)
        assert sos.dtype == np.float64

    def test_hpf_shape(self):
        sos = make_biquad_sos('hpf', 1000, 0.707, 48000)
        assert sos.shape == (1, 6)

    def test_bpf_shape(self):
        sos = make_biquad_sos('bpf', 1000, 0.707, 48000)
        assert sos.shape == (1, 6)

    def test_notch_shape(self):
        sos = make_biquad_sos('notch', 1000, 0.707, 48000)
        assert sos.shape == (1, 6)

    def test_unknown_type_passthrough(self):
        sos = make_biquad_sos('unknown', 1000, 0.707, 48000)
        # Should return identity: [1, 0, 0, 1, 0, 0]
        np.testing.assert_array_equal(sos, [[1, 0, 0, 1, 0, 0]])

    def test_freq_clamping_low(self):
        # Frequency below 20 Hz should be clamped
        sos = make_biquad_sos('lpf', 5.0, 0.707, 48000)
        assert sos.shape == (1, 6)
        assert np.all(np.isfinite(sos))

    def test_freq_clamping_high(self):
        # Frequency above Nyquist should be clamped
        sos = make_biquad_sos('lpf', 30000, 0.707, 48000)
        assert np.all(np.isfinite(sos))

    def test_q_clamping(self):
        # Very low Q should be clamped to 0.1
        sos = make_biquad_sos('lpf', 1000, 0.01, 48000)
        assert np.all(np.isfinite(sos))

    def test_a0_normalized(self):
        # a0 coefficient should always be 1.0 (normalized)
        sos = make_biquad_sos('lpf', 1000, 0.707, 48000)
        assert sos[0, 3] == 1.0

    def test_lpf_attenuates_high_freq(self):
        from scipy.signal import sosfilt
        sos = make_biquad_sos('lpf', 200, 0.707, 48000)
        # Generate 5kHz tone
        t = np.arange(4096) / 48000.0
        signal = np.sin(2 * np.pi * 5000 * t)
        filtered = sosfilt(sos, signal)
        # High freq should be attenuated
        assert np.max(np.abs(filtered[512:])) < 0.5


# ── LFO ──

class TestLFO:
    def test_sine_output_range(self):
        lfo = LFO(rate=1.0, sample_rate=48000)
        out = lfo.generate(48000, shape='sine')
        assert out.dtype == np.float32
        assert np.max(out) <= 1.01  # small tolerance for float
        assert np.min(out) >= -1.01

    def test_triangle_output_range(self):
        lfo = LFO(rate=1.0, sample_rate=48000)
        out = lfo.generate(48000, shape='triangle')
        assert np.max(out) <= 1.01
        assert np.min(out) >= -1.01

    def test_square_output_values(self):
        lfo = LFO(rate=1.0, sample_rate=48000)
        out = lfo.generate(48000, shape='square')
        # Square wave should only be +1 or -1
        unique = np.unique(out)
        assert len(unique) == 2
        assert -1.0 in unique
        assert 1.0 in unique

    def test_sine_frequency(self):
        lfo = LFO(rate=10.0, sample_rate=48000)
        out = lfo.generate(48000, shape='sine')
        # Count zero crossings ~ 2 per cycle, so ~20 for 10 Hz over 1 sec
        crossings = np.sum(np.diff(np.sign(out)) != 0)
        assert 18 <= crossings <= 22

    def test_reset_restarts_phase(self):
        lfo = LFO(rate=1.0, sample_rate=48000)
        out1 = lfo.generate(100, shape='sine')
        lfo.reset()
        out2 = lfo.generate(100, shape='sine')
        np.testing.assert_array_almost_equal(out1, out2)

    def test_generate_length(self):
        lfo = LFO(rate=1.0, sample_rate=48000)
        out = lfo.generate(512)
        assert len(out) == 512

    def test_very_low_rate_clamped(self):
        lfo = LFO(rate=0.0, sample_rate=48000)
        out = lfo.generate(100)
        assert len(out) == 100
        assert np.all(np.isfinite(out))
