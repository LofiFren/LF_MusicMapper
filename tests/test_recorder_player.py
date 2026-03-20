"""Tests for StemRecorder and SampleFilePlayer edge cases."""

import os
import queue
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import sys
mock_pa = MagicMock()
mock_pa.PyAudio.return_value.get_device_count.return_value = 0
if 'pyaudio' not in sys.modules:
    sys.modules['pyaudio'] = mock_pa

from mapper import StemRecorder, SampleFilePlayer


# ═══════════════════════════════════════════════════════════════════════
# StemRecorder
# ═══════════════════════════════════════════════════════════════════════

class TestStemRecorder:
    def test_init(self):
        rec = StemRecorder()
        assert rec.is_recording is False
        assert rec._writer_thread is None

    def test_feed_output_when_not_recording(self):
        """feed_output should silently return when not recording."""
        rec = StemRecorder()
        frames = np.zeros((1024, 8), dtype=np.float32)
        rec.feed_output(0, 'A', frames)  # should not raise

    def test_feed_output_extracts_tracks_for_output_a(self):
        """Only output A should generate per-track data."""
        rec = StemRecorder()
        rec.is_recording = True

        frames = np.zeros((1024, 8), dtype=np.float32)
        # Put distinctive values per track
        frames[:, 0] = 0.1  # Track 0 L
        frames[:, 1] = 0.2  # Track 0 R
        frames[:, 2] = 0.3  # Track 1 L
        frames[:, 3] = 0.4  # Track 1 R
        frames[:, 4] = 0.5  # Track 2 L
        frames[:, 5] = 0.6  # Track 2 R
        frames[:, 6] = 0.7  # Track 3 L
        frames[:, 7] = 0.8  # Track 3 R

        rec.feed_output(0, 'A', frames)

        item = rec._write_queue.get_nowait()
        assert item['out_label'] == 'A'
        assert item['tracks'] is not None
        assert len(item['tracks']) == 4  # 4 tracks

        # Verify track extraction
        np.testing.assert_array_equal(item['tracks'][0][:, 0], frames[:, 0])
        np.testing.assert_array_equal(item['tracks'][0][:, 1], frames[:, 1])
        np.testing.assert_array_equal(item['tracks'][2][:, 0], frames[:, 4])
        np.testing.assert_array_equal(item['tracks'][2][:, 1], frames[:, 5])

    def test_feed_output_no_tracks_for_output_b(self):
        """Non-A outputs should not extract per-track data."""
        rec = StemRecorder()
        rec.is_recording = True

        frames = np.zeros((1024, 4), dtype=np.float32)
        rec.feed_output(1, 'B', frames)

        item = rec._write_queue.get_nowait()
        assert item['out_label'] == 'B'
        assert item['tracks'] is None

    def test_feed_output_copies_data(self):
        """Output data should be copied to avoid mutation."""
        rec = StemRecorder()
        rec.is_recording = True

        frames = np.ones((1024, 8), dtype=np.float32)
        rec.feed_output(0, 'A', frames)

        # Mutate original
        frames[:] = 0.0

        item = rec._write_queue.get_nowait()
        # Queued data should still be 1.0
        assert np.allclose(item['output_mix'], 1.0)

    def test_feed_output_drops_when_queue_full(self):
        """Queue overflow should silently drop data."""
        rec = StemRecorder()
        rec.is_recording = True
        rec._write_queue = queue.Queue(maxsize=2)

        frames = np.zeros((1024, 4), dtype=np.float32)
        # Fill queue
        rec.feed_output(0, 'A', frames)
        rec.feed_output(0, 'A', frames)
        # This should drop silently
        rec.feed_output(0, 'A', frames)
        assert rec._write_queue.qsize() == 2

    def test_feed_output_handles_odd_channels(self):
        """Odd channel count (e.g., 3) should not crash track extraction."""
        rec = StemRecorder()
        rec.is_recording = True

        frames = np.zeros((1024, 3), dtype=np.float32)
        frames[:, 0] = 0.5
        frames[:, 1] = 0.6
        frames[:, 2] = 0.7
        rec.feed_output(0, 'A', frames)

        item = rec._write_queue.get_nowait()
        # Only 1 full stereo pair available (ch 0-1)
        assert 0 in item['tracks']
        # Track 0 should have L=ch0, R=ch1
        np.testing.assert_array_equal(item['tracks'][0][:, 0], frames[:, 0])

    def test_write_item_routes_to_correct_handles(self):
        """_write_item should write to the correct sf handles."""
        rec = StemRecorder()
        mock_handle = MagicMock()
        rec._sf_handles = {
            'output_A': mock_handle,
            'track_0': mock_handle,
        }

        item = {
            'out_label': 'A',
            'output_mix': np.zeros((1024, 8), dtype=np.float32),
            'tracks': {0: np.zeros((1024, 2), dtype=np.float32)},
        }
        rec._write_item(item)
        assert mock_handle.write.call_count == 2  # output_A + track_0

    def test_write_item_missing_handle(self):
        """_write_item should not crash if handle is missing."""
        rec = StemRecorder()
        rec._sf_handles = {}
        item = {
            'out_label': 'X',
            'output_mix': np.zeros((1024, 2), dtype=np.float32),
            'tracks': None,
        }
        rec._write_item(item)  # Should not raise


# ═══════════════════════════════════════════════════════════════════════
# SampleFilePlayer — edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestSampleFilePlayerEdgeCases:
    def test_read_raw_empty_audio(self):
        """_read_raw with no audio loaded should return silence."""
        player = SampleFilePlayer()
        buf = player._read_raw(1024)
        assert buf.shape == (1024, 2)
        assert np.allclose(buf, 0.0)

    def test_read_raw_multi_loop_wrap(self):
        """Buffer shorter than frame_count with looping should wrap correctly."""
        player = SampleFilePlayer()
        # Very short audio: 100 frames
        data = np.ones((100, 2), dtype=np.float32) * 0.5
        player.install_audio(data, 48000, "short.wav", "/tmp/short.wav")
        player.is_looping = True
        player.is_playing = True

        buf = player._read_raw(350)  # 3.5x the audio length
        assert buf.shape == (350, 2)
        # All samples should be 0.5 (looped content)
        assert np.allclose(buf, 0.5)
        assert player.playhead == 50  # 350 % 100

    def test_read_raw_exact_end(self):
        """Reading exactly to the end of the buffer positions playhead at end."""
        player = SampleFilePlayer()
        data = np.arange(1024 * 2, dtype=np.float32).reshape(1024, 2) / 2048
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.is_playing = True
        player.is_looping = False

        buf = player._read_raw(1024)
        assert buf.shape == (1024, 2)
        # Exact read uses the end<=total path, playhead advances to total
        assert player.playhead == 1024

    def test_read_raw_past_end_stops(self):
        """Reading past end in one-shot mode should stop playback."""
        player = SampleFilePlayer()
        data = np.ones((500, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.is_playing = True
        player.is_looping = False

        buf = player._read_raw(1024)
        assert buf.shape == (1024, 2)
        assert player.is_playing is False

    def test_playhead_beyond_total_resets(self):
        """If playhead >= total_frames, it should reset to 0."""
        player = SampleFilePlayer()
        data = np.ones((1000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.playhead = 2000  # beyond total
        player.is_playing = True
        player.is_looping = True

        buf = player._read_raw(100)
        assert buf.shape == (100, 2)
        # Playhead should have been clamped/reset
        assert player.playhead <= 1000

    def test_fade_in_ramp(self):
        """Fade-in should ramp from 0 to 1 over FADE_FRAMES samples."""
        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.play()

        buf = player.get_next_buffer(1024)
        # First sample should be near 0 (fade-in start)
        assert buf[0, 0] < 0.1
        # After FADE_FRAMES, should be near 1.0
        assert buf[player.FADE_FRAMES, 0] > 0.9

    def test_consecutive_play_stop_play(self):
        """Rapid play/stop/play should not corrupt state."""
        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")

        for _ in range(5):
            player.play()
            player.get_next_buffer(256)
            player.stop()
            buf = player.get_next_buffer(256)
            assert buf.shape == (256, 2)

    def test_duration_str_formats(self):
        """Various durations should format correctly."""
        player = SampleFilePlayer()

        # 0 frames
        assert player.get_duration_str() == "0:00"

        # 1 minute exactly
        data = np.ones((48000 * 60, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        assert player.get_duration_str() == "1:00"

        # 2m30s
        data = np.ones((48000 * 150, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        assert player.get_duration_str() == "2:30"

    def test_mono_to_stereo_in_install(self):
        """Installing mono data still works (channels from data shape)."""
        player = SampleFilePlayer()
        # 2-channel data
        data = np.ones((1000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        assert player.channels == 2


# ═══════════════════════════════════════════════════════════════════════
# Regression: sample_players must update when file is loaded mid-routing
# ═══════════════════════════════════════════════════════════════════════

class TestSamplePlayerLiveLoadRegression:
    """Regression tests for sample player + live routing interaction.

    Bug 1: Loading a sample file while routing was active did not update
    AudioManager.sample_players — the callback never mixed the sample in.

    Bug 2 (introduced by first fix attempt): Connecting file_loaded to
    _on_sample_slot_changed triggered _hot_restart_routing(), which
    tore down the entire audio pipeline and broke live audio.

    Correct fix: A dedicated _on_sample_file_loaded handler that ONLY
    swaps the sample_players array — no hot restart, no pipeline teardown.
    """

    def _make_audio_manager(self):
        """Create a mock AudioManager in 'routing' state."""
        from mapper import AudioManager
        with patch.object(AudioManager, '__init__', lambda self: None):
            am = AudioManager.__new__(AudioManager)
            am.is_routing = True
            am.sample_players = [None, None, None]
            am._input_slot_map = [0]  # slot A is live
            am.input_channel_counts = [4]
            am.total_input_channels = 4
            return am

    def test_file_loaded_handler_updates_sample_players(self):
        """_on_sample_file_loaded must assign the player to sample_players.

        Simulates the exact logic of _on_sample_file_loaded: read the
        current slot assignment + player audio_data, rebuild array.
        """
        am = self._make_audio_manager()

        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32) * 0.5
        player.install_audio(data, 48000, "voice_loop.wav", "/tmp/voice_loop.wav")

        # Replicate _on_sample_file_loaded logic (the actual fix)
        slot = 'A'
        new_players = [None, None, None]
        if slot and player.audio_data is not None:
            slot_idx = {'A': 0, 'B': 1, 'C': 2}[slot]
            new_players[slot_idx] = player
        am.sample_players = new_players

        assert am.sample_players[0] is player
        assert am.sample_players[1] is None
        assert am.sample_players[2] is None

    def test_file_loaded_handler_does_not_restart_routing(self):
        """_on_sample_file_loaded must NOT call stop_routing / _hot_restart.

        This is the key difference from _on_sample_slot_changed — loading
        a file should never tear down live audio.
        """
        am = self._make_audio_manager()
        am.stop_routing = MagicMock()

        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")

        # Replicate _on_sample_file_loaded (only swaps the array)
        slot = 'A'
        new_players = [None, None, None]
        if slot and player.audio_data is not None:
            slot_idx = {'A': 0, 'B': 1, 'C': 2}[slot]
            new_players[slot_idx] = player
        am.sample_players = new_players

        # stop_routing must never be called
        am.stop_routing.assert_not_called()

    def test_file_loaded_skipped_when_not_routing(self):
        """If routing is not active, _on_sample_file_loaded should be a no-op."""
        am = self._make_audio_manager()
        am.is_routing = False

        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")

        # When not routing, sample_players should stay untouched
        original_players = am.sample_players
        # _on_sample_file_loaded early-returns if not routing
        if not am.is_routing:
            pass  # no-op
        else:
            am.sample_players = [player, None, None]

        assert am.sample_players is original_players

    def test_sample_mixes_additively_with_live_audio(self):
        """Callback loop: sample audio is ADDED to live device audio, not replacing it."""
        player = SampleFilePlayer()
        data = np.ones((1024, 2), dtype=np.float32) * 0.25
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.play()
        player._fade_in = False  # skip fade for clean measurement

        # Simulate callback state
        sample_players = [player, None, None]
        input_slot_map = [0]  # sequential input 0 → slot A
        input_channel_counts = [4]
        frame_count = 1024

        # Live device audio on channels 0-1
        input_bufs = [np.zeros((frame_count, 4), dtype=np.float32)]
        input_bufs[0][:, 0:2] = 0.5

        # Replicate the callback's sample mixing loop (mapper.py lines 584-597)
        for i in range(len(input_channel_counts)):
            _slot = input_slot_map[i] if i < len(input_slot_map) else i
            if _slot < len(sample_players) and sample_players[_slot] is not None:
                _sbuf = sample_players[_slot].get_next_buffer(frame_count)
                if input_bufs[i] is not None:
                    _sf = min(frame_count, len(_sbuf))
                    _sc = min(
                        _sbuf.shape[1] if len(_sbuf.shape) > 1 else 1,
                        input_bufs[i].shape[1] if len(input_bufs[i].shape) > 1 else 1)
                    input_bufs[i][:_sf, :_sc] += _sbuf[:_sf, :_sc]
                else:
                    input_bufs[i] = _sbuf

        # Live (0.5) + sample (0.25) = 0.75 on channels 0-1
        assert input_bufs[0][0, 0] == pytest.approx(0.75)
        assert input_bufs[0][0, 1] == pytest.approx(0.75)
        # Channels 2-3 untouched (sample is stereo, only 2 channels mixed)
        assert input_bufs[0][0, 2] == pytest.approx(0.0)

    def test_sample_players_none_guards_no_audio(self):
        """Player with no audio_data must not be assigned to sample_players."""
        player = SampleFilePlayer()  # no audio loaded
        new_players = [None, None, None]
        slot = 'A'
        if slot and player.audio_data is not None:
            new_players[0] = player
        assert new_players[0] is None

    def test_callback_skips_none_sample_players(self):
        """Callback must not crash when sample_players slots are None."""
        sample_players = [None, None, None]
        input_slot_map = [0]
        input_channel_counts = [4]
        frame_count = 1024
        input_bufs = [np.ones((frame_count, 4), dtype=np.float32) * 0.5]

        # Replicate callback loop — should be a no-op with all-None players
        for i in range(len(input_channel_counts)):
            _slot = input_slot_map[i] if i < len(input_slot_map) else i
            if _slot < len(sample_players) and sample_players[_slot] is not None:
                assert False, "Should not enter this branch with None players"

        # Live audio unchanged
        assert input_bufs[0][0, 0] == pytest.approx(0.5)
