"""Tests for apply_alignment_delay (per-track dry/wet alignment)."""

import sys
from unittest.mock import MagicMock

import numpy as np

# Mock pyaudio before importing mapper
mock_pa = MagicMock()
mock_pa.PyAudio.return_value.get_device_count.return_value = 0
sys.modules.setdefault('pyaudio', mock_pa)

from mapper import apply_alignment_delay, AudioManager


MAX_D = 9600  # 200ms at 48kHz


def make_tails(channels=2, max_d=MAX_D):
    return np.zeros((channels, max_d), dtype=np.float32)


class TestApplyAlignmentDelay:
    def test_zero_delay_passthrough(self):
        tails = make_tails()
        frames = np.random.default_rng(1).standard_normal((1024, 2)).astype(np.float32)
        out = apply_alignment_delay(frames.copy(), tails, [0, 0])
        np.testing.assert_array_equal(out, frames)

    def test_impulse_delayed_by_n_samples_within_block(self):
        tails = make_tails()
        frames = np.zeros((1024, 2), dtype=np.float32)
        frames[0, 0] = 1.0
        out = apply_alignment_delay(frames, tails, [100, 0])
        assert out[100, 0] == 1.0
        assert out[0, 0] == 0.0
        assert np.count_nonzero(out[:, 0]) == 1

    def test_delay_carries_across_blocks(self):
        tails = make_tails()
        block1 = np.zeros((1024, 2), dtype=np.float32)
        block1[1000, 0] = 1.0  # impulse near end of block — must emerge next block
        out1 = apply_alignment_delay(block1, tails, [200, 0])
        assert np.count_nonzero(out1[:, 0]) == 0
        block2 = np.zeros((1024, 2), dtype=np.float32)
        out2 = apply_alignment_delay(block2, tails, [200, 0])
        assert out2[1000 + 200 - 1024, 0] == 1.0

    def test_per_channel_independence(self):
        tails = make_tails()
        frames = np.zeros((512, 2), dtype=np.float32)
        frames[0, :] = 1.0
        out = apply_alignment_delay(frames, tails, [50, 0])
        assert out[50, 0] == 1.0   # delayed channel
        assert out[0, 1] == 1.0    # undelayed channel untouched

    def test_delay_clamped_to_tail_length(self):
        tails = make_tails(max_d=100)
        frames = np.zeros((512, 2), dtype=np.float32)
        frames[0, 0] = 1.0
        out = apply_alignment_delay(frames, tails, [10_000, 0])
        assert out[100, 0] == 1.0  # clamped to max_d

    def test_more_frame_channels_than_tails_is_safe(self):
        tails = make_tails(channels=2)
        frames = np.random.default_rng(2).standard_normal((256, 4)).astype(np.float32)
        out = apply_alignment_delay(frames.copy(), tails, [0, 0, 0, 0])
        assert out.shape == (256, 4)
        np.testing.assert_array_equal(out[:, 2:], frames[:, 2:])

    def test_continuous_signal_no_discontinuity(self):
        # A delayed sine must stay a clean sine across block boundaries
        tails = make_tails(channels=1)
        sr, freq, d = 48000, 440.0, 333
        t = np.arange(4096) / sr
        sig = np.sin(2 * np.pi * freq * t).astype(np.float32)
        out = np.concatenate([
            apply_alignment_delay(sig[i:i + 1024].reshape(-1, 1).copy(), tails, [d])
            for i in range(0, 4096, 1024)
        ]).ravel()
        expected = np.concatenate([np.zeros(d, dtype=np.float32), sig[:-d]])
        np.testing.assert_allclose(out, expected, atol=1e-6)


class TestSetTrackDelay:
    def make_am(self):
        am = AudioManager.__new__(AudioManager)
        am.track_delay_ms = [[0] * 4 for _ in range(3)]
        return am

    def test_set_and_clamp(self):
        am = self.make_am()
        am.set_track_delay(1, 2, 50)
        assert am.track_delay_ms[1][2] == 50
        am.set_track_delay(1, 2, 9999)
        assert am.track_delay_ms[1][2] == AudioManager.MAX_ALIGN_MS
        am.set_track_delay(1, 2, -5)
        assert am.track_delay_ms[1][2] == 0

    def test_out_of_range_indices_ignored(self):
        am = self.make_am()
        am.set_track_delay(7, 0, 50)
        am.set_track_delay(0, 9, 50)
        assert all(all(v == 0 for v in row) for row in am.track_delay_ms)
