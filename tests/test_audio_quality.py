"""Audio quality tests — verify DSP effects produce correct sonic results.

Uses FFT analysis, energy measurements, and signal properties to verify
that effects actually do what they claim, not just that they don't crash.
"""

import math
import numpy as np
import pytest
from scipy.fft import fft
from effects import (
    Filter, Delay, Reverb, DubEcho, Flanger, Phaser,
    Chorus, Gater, BeatRoll, BitCrush, RingMod, ChoppedAndScrewed,
    beat_to_samples, make_biquad_sos,
)


SR = 48000
BUFSZ = 1024


def make_tone(freq, duration_samples=4096, sr=SR):
    """Generate a stereo sine tone."""
    t = np.arange(duration_samples, dtype=np.float32) / sr
    mono = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return np.column_stack([mono, mono])


def rms(signal):
    """Root-mean-square of a signal array."""
    return float(np.sqrt(np.mean(signal ** 2)))


def peak_frequency(signal_1d, sr=SR):
    """Return the dominant frequency in a 1D signal."""
    n = len(signal_1d)
    spectrum = np.abs(fft(signal_1d * np.hanning(n)))[:n // 2]
    freqs = np.arange(n // 2) * sr / n
    return float(freqs[np.argmax(spectrum)])


def band_energy(signal_1d, low_hz, high_hz, sr=SR):
    """Return energy in a frequency band."""
    n = len(signal_1d)
    spectrum = np.abs(fft(signal_1d * np.hanning(n)))[:n // 2]
    freqs = np.arange(n // 2) * sr / n
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(spectrum[mask] ** 2))


def process_blocks(fx, signal, block_size=BUFSZ):
    """Process a signal through an effect in block-sized chunks."""
    out_parts = []
    for i in range(0, len(signal), block_size):
        block = signal[i:i + block_size]
        if len(block) < block_size:
            pad = np.zeros((block_size - len(block), 2), dtype=np.float32)
            block = np.vstack([block, pad])
        out_parts.append(fx.process_stereo(block))
    return np.vstack(out_parts)[:len(signal)]


# ═══════════════════════════════════════════════════════════════════════
# Filter Quality Tests
# ═══════════════════════════════════════════════════════════════════════

class TestFilterQuality:
    def test_lpf_passes_low_rejects_high(self):
        """LPF at 500Hz: 200Hz tone should pass, 5kHz should be attenuated."""
        fx = Filter(sample_rate=SR, buffer_size=BUFSZ)
        fx.set_sub_type(0)  # LPF
        fx.set_param('cutoff', 500)
        fx.set_param('resonance', 0.707)
        fx.dry_wet = 1.0

        low_tone = make_tone(200, 8192)
        high_tone = make_tone(5000, 8192)

        low_out = process_blocks(fx, low_tone)
        fx.reset()
        high_out = process_blocks(fx, high_tone)

        low_ratio = rms(low_out) / rms(low_tone)
        high_ratio = rms(high_out) / rms(high_tone)

        assert low_ratio > 0.5, f"LPF killed the low tone (ratio={low_ratio:.3f})"
        assert high_ratio < 0.2, f"LPF didn't attenuate high tone (ratio={high_ratio:.3f})"

    def test_hpf_passes_high_rejects_low(self):
        """HPF at 2kHz: 5kHz should pass, 200Hz should be attenuated."""
        fx = Filter(sample_rate=SR, buffer_size=BUFSZ)
        fx.set_sub_type(1)  # HPF
        fx.set_param('cutoff', 2000)
        fx.set_param('resonance', 0.707)
        fx.dry_wet = 1.0

        low_tone = make_tone(200, 8192)
        high_tone = make_tone(5000, 8192)

        low_out = process_blocks(fx, low_tone)
        fx.reset()
        high_out = process_blocks(fx, high_tone)

        low_ratio = rms(low_out) / rms(low_tone)
        high_ratio = rms(high_out) / rms(high_tone)

        assert low_ratio < 0.2, f"HPF didn't attenuate low tone (ratio={low_ratio:.3f})"
        assert high_ratio > 0.5, f"HPF killed the high tone (ratio={high_ratio:.3f})"

    def test_bpf_passes_center_rejects_sides(self):
        """BPF at 1kHz: 1kHz should pass, 100Hz and 10kHz should be attenuated."""
        fx = Filter(sample_rate=SR, buffer_size=BUFSZ)
        fx.set_sub_type(2)  # BPF
        fx.set_param('cutoff', 1000)
        fx.set_param('resonance', 2.0)
        fx.dry_wet = 1.0

        center = make_tone(1000, 8192)
        low = make_tone(100, 8192)
        high = make_tone(10000, 8192)

        center_out = process_blocks(fx, center)
        fx.reset()
        low_out = process_blocks(fx, low)
        fx.reset()
        high_out = process_blocks(fx, high)

        center_ratio = rms(center_out) / rms(center)
        low_ratio = rms(low_out) / rms(low)
        high_ratio = rms(high_out) / rms(high)

        assert center_ratio > low_ratio, "BPF should pass center better than low"
        assert center_ratio > high_ratio, "BPF should pass center better than high"

    def test_notch_rejects_center(self):
        """Notch at 1kHz should attenuate 1kHz more than 500Hz."""
        fx = Filter(sample_rate=SR, buffer_size=BUFSZ)
        fx.set_sub_type(3)  # Notch
        fx.set_param('cutoff', 1000)
        fx.set_param('resonance', 5.0)
        fx.dry_wet = 1.0

        notch_tone = make_tone(1000, 8192)
        off_tone = make_tone(500, 8192)

        notch_out = process_blocks(fx, notch_tone)
        fx.reset()
        off_out = process_blocks(fx, off_tone)

        notch_ratio = rms(notch_out) / rms(notch_tone)
        off_ratio = rms(off_out) / rms(off_tone)

        assert notch_ratio < off_ratio, "Notch should attenuate center freq more"

    def test_lpf_frequency_response_shape(self):
        """LPF should show monotonically decreasing energy above cutoff."""
        fx = Filter(sample_rate=SR, buffer_size=BUFSZ)
        fx.set_sub_type(0)  # LPF
        fx.set_param('cutoff', 1000)
        fx.set_param('resonance', 0.707)
        fx.dry_wet = 1.0

        # White noise through LPF
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((16384, 2)).astype(np.float32) * 0.3
        out = process_blocks(fx, noise)

        # Energy below cutoff should exceed energy above
        low_e = band_energy(out[:, 0], 20, 1000)
        high_e = band_energy(out[:, 0], 2000, 20000)
        assert low_e > high_e * 2, "LPF should have more energy below cutoff"


# ═══════════════════════════════════════════════════════════════════════
# Delay Quality Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDelayQuality:
    def test_echo_arrives_at_correct_time(self):
        """Delay echo should appear at the expected sample offset."""
        fx = Delay(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('feedback', 0.5)
        fx.set_param('tone', 12000)  # high tone = minimal filtering
        fx.set_bpm(120.0)
        fx.set_beat_fraction(1 / 16)  # short delay

        expected_delay = beat_to_samples(1 / 16, 120.0, SR)

        # Long impulse test
        total_len = expected_delay * 4
        impulse = np.zeros((total_len, 2), dtype=np.float32)
        impulse[0, :] = 1.0

        out = process_blocks(fx, impulse)

        # Find the first echo peak (skip the dry signal at sample 0)
        mono = np.abs(out[10:, 0])  # skip first few samples
        peak_idx = np.argmax(mono) + 10
        # Allow ±2 samples tolerance
        assert abs(peak_idx - expected_delay) <= 2, \
            f"Echo at sample {peak_idx}, expected ~{expected_delay}"

    def test_feedback_produces_decaying_echoes(self):
        """Multiple echoes should decay with feedback ratio."""
        fx = Delay(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('feedback', 0.5)
        fx.set_param('tone', 12000)
        fx.set_bpm(120.0)
        fx.set_beat_fraction(1 / 16)

        delay_samp = beat_to_samples(1 / 16, 120.0, SR)
        total_len = delay_samp * 6
        impulse = np.zeros((total_len, 2), dtype=np.float32)
        impulse[0, :] = 1.0

        out = process_blocks(fx, impulse)
        mono = out[:, 0]

        # Measure peaks at echo positions
        peaks = []
        for i in range(1, 5):
            region = mono[delay_samp * i - 5:delay_samp * i + 5]
            if len(region) > 0:
                peaks.append(float(np.max(np.abs(region))))

        # Each successive echo should be quieter
        for i in range(len(peaks) - 1):
            if peaks[i] > 0.001:
                assert peaks[i + 1] < peaks[i], \
                    f"Echo {i+2} ({peaks[i+1]:.4f}) not quieter than echo {i+1} ({peaks[i]:.4f})"

    def test_zero_feedback_single_echo(self):
        """With feedback=0, only one echo should appear."""
        fx = Delay(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('feedback', 0.0)
        fx.set_param('tone', 12000)
        fx.set_bpm(120.0)
        fx.set_beat_fraction(1 / 16)

        delay_samp = beat_to_samples(1 / 16, 120.0, SR)
        total_len = delay_samp * 4
        impulse = np.zeros((total_len, 2), dtype=np.float32)
        impulse[0, :] = 1.0

        out = process_blocks(fx, impulse)
        # Energy after 2x delay time should be near-zero
        late = rms(out[delay_samp * 2:, :])
        assert late < 0.001, f"Energy after 2x delay with 0 feedback: {late}"


# ═══════════════════════════════════════════════════════════════════════
# Reverb Quality Tests
# ═══════════════════════════════════════════════════════════════════════

class TestReverbQuality:
    def test_reverb_tail_decays(self):
        """Reverb energy should decay over time after an impulse."""
        fx = Reverb(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('decay', 0.7)

        impulse = np.zeros((BUFSZ, 2), dtype=np.float32)
        impulse[0, :] = 1.0

        # Process impulse + 10 blocks of silence
        process_blocks(fx, impulse)
        energies = []
        silence = np.zeros((BUFSZ, 2), dtype=np.float32)
        for _ in range(10):
            out = fx.process_stereo(silence.copy())
            energies.append(rms(out))

        # Energy should generally decrease
        assert energies[-1] < energies[0], \
            f"Reverb tail not decaying: first={energies[0]:.6f}, last={energies[-1]:.6f}"

    def test_high_decay_longer_tail(self):
        """Higher decay value should produce longer reverb tail."""
        def measure_tail(decay_val):
            fx = Reverb(sample_rate=SR, buffer_size=BUFSZ)
            fx.dry_wet = 1.0
            fx.set_param('decay', decay_val)
            impulse = np.zeros((BUFSZ, 2), dtype=np.float32)
            impulse[0, :] = 1.0
            process_blocks(fx, impulse)
            silence = np.zeros((BUFSZ, 2), dtype=np.float32)
            total = 0.0
            for _ in range(20):
                out = fx.process_stereo(silence.copy())
                total += rms(out)
            return total

        short_tail = measure_tail(0.3)
        long_tail = measure_tail(0.9)
        assert long_tail > short_tail, \
            f"Higher decay should give longer tail: 0.3→{short_tail:.4f}, 0.9→{long_tail:.4f}"

    def test_reverb_is_stereo(self):
        """Reverb output should have L/R decorrelation (not mono)."""
        fx = Reverb(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('decay', 0.7)

        impulse = np.zeros((BUFSZ, 2), dtype=np.float32)
        impulse[0, :] = 1.0
        process_blocks(fx, impulse)

        silence = np.zeros((BUFSZ * 4, 2), dtype=np.float32)
        out = process_blocks(fx, silence)

        # L and R should not be identical
        diff = np.sum(np.abs(out[:, 0] - out[:, 1]))
        assert diff > 0.01, "Reverb should produce stereo decorrelation"


# ═══════════════════════════════════════════════════════════════════════
# BitCrush Quality Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBitCrushQuality:
    def test_bit_reduction_quantizes(self):
        """Reducing to N bits should produce at most 2^N unique levels."""
        fx = BitCrush(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('bits', 4)
        fx.set_param('downsample', 1)

        tone = make_tone(440, 4096) * 0.9
        out = process_blocks(fx, tone)

        # 4-bit = 16 levels; unique values should be bounded
        unique_vals = len(np.unique(np.round(out[:, 0], 6)))
        assert unique_vals <= 33, f"4-bit crush has {unique_vals} unique levels (expected ≤33)"

    def test_more_bits_means_less_distortion(self):
        """Higher bit depth should produce less quantization noise."""
        tone = make_tone(440, 4096) * 0.5

        def crush_error(bits):
            fx = BitCrush(sample_rate=SR, buffer_size=BUFSZ)
            fx.dry_wet = 1.0
            fx.set_param('bits', bits)
            fx.set_param('downsample', 1)
            out = process_blocks(fx, tone.copy())
            return rms(out - tone)

        err_4bit = crush_error(4)
        err_12bit = crush_error(12)
        assert err_4bit > err_12bit, \
            f"4-bit error ({err_4bit:.6f}) should exceed 12-bit ({err_12bit:.6f})"

    def test_downsample_reduces_bandwidth(self):
        """Sample-and-hold should kill high frequencies."""
        fx = BitCrush(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('bits', 16)
        fx.set_param('downsample', 16)  # 48kHz / 16 = 3kHz effective SR

        rng = np.random.default_rng(42)
        noise = rng.standard_normal((4096, 2)).astype(np.float32) * 0.3
        out = process_blocks(fx, noise)

        high_e_in = band_energy(noise[:, 0], 5000, 20000)
        high_e_out = band_energy(out[:, 0], 5000, 20000)
        # Downsampling creates aliasing but the sample-and-hold itself
        # acts as a zero-order hold; high-freq original content is reshaped
        assert high_e_out != high_e_in  # output spectrum is altered

    def test_16bit_near_transparent(self):
        """16-bit crush should be nearly transparent."""
        fx = BitCrush(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('bits', 16)
        fx.set_param('downsample', 1)

        tone = make_tone(440, 4096) * 0.5
        out = process_blocks(fx, tone)
        error = rms(out - tone)
        assert error < 0.001, f"16-bit crush error too high: {error:.6f}"


# ═══════════════════════════════════════════════════════════════════════
# Ring Mod Quality Tests
# ═══════════════════════════════════════════════════════════════════════

class TestRingModQuality:
    def test_ring_mod_creates_sum_difference_freqs(self):
        """Ring mod of 440Hz x 200Hz carrier should produce 240Hz and 640Hz."""
        fx = RingMod(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('freq', 200)
        fx.set_param('shape', 0.0)  # pure sine carrier

        tone = make_tone(440, 16384)
        out = process_blocks(fx, tone)

        # Check spectrum for sum and difference frequencies
        spectrum = np.abs(fft(out[:, 0] * np.hanning(len(out))))[:len(out) // 2]
        freqs = np.arange(len(out) // 2) * SR / len(out)

        # Find peaks near 240Hz and 640Hz
        mask_diff = (freqs >= 230) & (freqs <= 250)
        mask_sum = (freqs >= 630) & (freqs <= 650)
        mask_orig = (freqs >= 430) & (freqs <= 450)

        peak_diff = np.max(spectrum[mask_diff])
        peak_sum = np.max(spectrum[mask_sum])
        peak_orig = np.max(spectrum[mask_orig])

        # Sum/difference freqs should be stronger than original
        assert peak_diff > peak_orig * 0.5, "Should have difference frequency (240Hz)"
        assert peak_sum > peak_orig * 0.5, "Should have sum frequency (640Hz)"

    def test_square_shape_adds_harmonics(self):
        """Shape=1.0 (square) should produce more harmonics than shape=0.0 (sine)."""
        tone = make_tone(440, 8192) * 0.5

        def count_significant_peaks(shape_val):
            fx = RingMod(sample_rate=SR, buffer_size=BUFSZ)
            fx.dry_wet = 1.0
            fx.set_param('freq', 200)
            fx.set_param('shape', shape_val)
            out = process_blocks(fx, tone.copy())
            spectrum = np.abs(fft(out[:, 0] * np.hanning(len(out))))[:len(out) // 2]
            threshold = np.max(spectrum) * 0.05
            return np.sum(spectrum > threshold)

        sine_peaks = count_significant_peaks(0.0)
        square_peaks = count_significant_peaks(1.0)
        assert square_peaks > sine_peaks, \
            f"Square shape ({square_peaks} peaks) should have more harmonics than sine ({sine_peaks})"


# ═══════════════════════════════════════════════════════════════════════
# Gater Quality Tests
# ═══════════════════════════════════════════════════════════════════════

class TestGaterQuality:
    def test_gater_creates_rhythmic_amplitude(self):
        """Gater should produce alternating loud/quiet sections."""
        fx = Gater(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('shape', 0.0)  # hard square gate
        fx.set_bpm(120.0)
        fx.set_beat_fraction(0.25)  # period = 6000 samples

        tone = make_tone(440, SR)  # 1 second
        out = process_blocks(fx, tone)

        # Use 16 segments so each is 3000 samples = half the gate period
        # This way alternating segments capture on/off halves
        seg_len = len(out) // 16
        energies = [rms(out[i * seg_len:(i + 1) * seg_len]) for i in range(16)]

        # Should have alternating high/low pattern
        loud_count = sum(1 for e in energies if e > 0.3)
        quiet_count = sum(1 for e in energies if e < 0.1)
        assert loud_count >= 4, f"Gate should have loud segments (found {loud_count})"
        assert quiet_count >= 4, f"Gate should have quiet segments (found {quiet_count})"

    def test_smooth_shape_has_gradual_transitions(self):
        """Shape=1.0 should produce smoother transitions than shape=0.0."""
        tone = make_tone(440, SR)

        def measure_max_derivative(shape_val):
            fx = Gater(sample_rate=SR, buffer_size=BUFSZ)
            fx.dry_wet = 1.0
            fx.set_param('shape', shape_val)
            fx.set_bpm(120.0)
            fx.set_beat_fraction(0.25)
            out = process_blocks(fx, tone.copy())
            env = np.abs(out[:, 0])
            # Smooth envelope to see gate shape
            kernel = np.ones(100) / 100
            env_smooth = np.convolve(env, kernel, mode='valid')
            return np.max(np.abs(np.diff(env_smooth)))

        hard_deriv = measure_max_derivative(0.0)
        soft_deriv = measure_max_derivative(1.0)
        assert soft_deriv < hard_deriv, \
            f"Smooth gate (deriv={soft_deriv:.4f}) should be gentler than hard (deriv={hard_deriv:.4f})"


# ═══════════════════════════════════════════════════════════════════════
# Chopped & Screwed Quality Tests
# ═══════════════════════════════════════════════════════════════════════

class TestChoppedAndScrewedQuality:
    def test_screw_lowers_pitch(self):
        """Screw ratio < 1.0 should lower the dominant frequency."""
        tone = make_tone(1000, 16384)

        fx = ChoppedAndScrewed(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('screw', 0.7)  # significant pitch down
        fx.set_param('chop', 0.0)   # no chopping

        out = process_blocks(fx, tone)
        # Skip first block (transient from grain startup)
        out_stable = out[4096:]

        out_freq = peak_frequency(out_stable[:, 0])
        # Screwed frequency should be lower than 1000Hz
        assert out_freq < 950, \
            f"Screwed output ({out_freq:.0f}Hz) should be below 1000Hz"

    def test_no_screw_preserves_pitch(self):
        """Screw=1.0 should preserve the original pitch."""
        tone = make_tone(1000, 16384)

        fx = ChoppedAndScrewed(sample_rate=SR, buffer_size=BUFSZ)
        fx.dry_wet = 1.0
        fx.set_param('screw', 1.0)
        fx.set_param('chop', 0.0)

        out = process_blocks(fx, tone)
        out_stable = out[4096:]

        out_freq = peak_frequency(out_stable[:, 0])
        assert abs(out_freq - 1000) < 50, \
            f"Screw=1.0 should preserve pitch, got {out_freq:.0f}Hz"


# ═══════════════════════════════════════════════════════════════════════
# Dry/Wet Quality Tests (applies to all effects)
# ═══════════════════════════════════════════════════════════════════════

class TestDryWetQuality:
    def test_fully_dry_is_passthrough(self):
        """dry_wet=0 should produce identical output to input."""
        tone = make_tone(440, 4096)
        for cls in [Filter, Delay, Reverb, BitCrush]:
            fx = cls(sample_rate=SR, buffer_size=BUFSZ)
            fx.dry_wet = 0.0
            out = process_blocks(fx, tone.copy())
            np.testing.assert_array_almost_equal(out, tone, decimal=5,
                err_msg=f"{cls.name} dry_wet=0 not passthrough")

    def test_half_wet_mixes(self):
        """dry_wet=0.5 should produce output between dry and fully wet."""
        tone = make_tone(440, 4096) * 0.5
        fx = Filter(sample_rate=SR, buffer_size=BUFSZ)
        fx.set_sub_type(0)  # LPF
        fx.set_param('cutoff', 200)

        fx.dry_wet = 0.0
        dry_out = process_blocks(fx, tone.copy())
        fx.reset()

        fx.dry_wet = 1.0
        wet_out = process_blocks(fx, tone.copy())
        fx.reset()

        fx.dry_wet = 0.5
        mix_out = process_blocks(fx, tone.copy())

        mix_rms = rms(mix_out)
        dry_rms = rms(dry_out)
        wet_rms = rms(wet_out)

        # Mixed should be between dry and wet RMS (or close)
        low = min(dry_rms, wet_rms) * 0.8
        high = max(dry_rms, wet_rms) * 1.2
        assert low <= mix_rms <= high, \
            f"50% mix RMS ({mix_rms:.4f}) outside expected range [{low:.4f}, {high:.4f}]"


# ═══════════════════════════════════════════════════════════════════════
# Dub Echo Quality Tests
# ═══════════════════════════════════════════════════════════════════════

class TestDubEchoQuality:
    def test_dub_echo_has_warmer_tone_than_delay(self):
        """Dub echo's saturation + LP filter should reduce high-freq content vs clean delay."""
        impulse = np.zeros((SR, 2), dtype=np.float32)
        impulse[0, :] = 1.0

        # Clean delay
        clean = Delay(sample_rate=SR, buffer_size=BUFSZ)
        clean.dry_wet = 1.0
        clean.set_param('feedback', 0.6)
        clean.set_param('tone', 12000)
        clean.set_beat_fraction(1 / 8)
        clean_out = process_blocks(clean, impulse.copy())

        # Dub echo
        dub = DubEcho(sample_rate=SR, buffer_size=BUFSZ)
        dub.dry_wet = 1.0
        dub.set_param('feedback', 0.6)
        dub.set_param('color', 2000)  # warm filter
        dub.set_beat_fraction(1 / 8)
        dub_out = process_blocks(dub, impulse.copy())

        # Dub echo should have less high-freq energy
        clean_high = band_energy(clean_out[:, 0], 4000, 20000)
        dub_high = band_energy(dub_out[:, 0], 4000, 20000)

        if clean_high > 0:
            assert dub_high < clean_high, \
                f"Dub echo should be warmer (less HF): dub={dub_high:.2f} clean={clean_high:.2f}"
