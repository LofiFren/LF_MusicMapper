"""Tests for AudioManager mapping logic and related functions."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# Mock pyaudio before importing mapper
import sys
mock_pa = MagicMock()
mock_pa.PyAudio.return_value.get_device_count.return_value = 0
sys.modules['pyaudio'] = mock_pa

from mapper import AudioManager, SampleFilePlayer


# ── AudioManager.set_track_mapping ──

class TestSetTrackMapping:
    @pytest.fixture
    def am(self):
        with patch.object(AudioManager, '__init__', lambda self: None):
            mgr = AudioManager.__new__(AudioManager)
            mgr.num_output_channels = 8
            mgr.track_mappings = [[]]
            mgr.total_input_channels = 8
            return mgr

    def test_simple_dict_entries(self, am):
        mapping = [
            [{'index': 0, 'gain': 1.0}],
            [{'index': 1, 'gain': 0.5}],
        ]
        am.set_track_mapping(mapping, 0)
        result = am.track_mappings[0]
        assert result[0] == [{'index': 0, 'gain': 1.0}]
        assert result[1] == [{'index': 1, 'gain': 0.5}]

    def test_int_entries_converted(self, am):
        mapping = [0, 1, 2, 3]
        am.set_track_mapping(mapping, 0)
        result = am.track_mappings[0]
        assert result[0] == [{'index': 0, 'gain': 1.0}]
        assert result[3] == [{'index': 3, 'gain': 1.0}]

    def test_gain_clamped_to_2(self, am):
        mapping = [[{'index': 0, 'gain': 5.0}]]
        am.set_track_mapping(mapping, 0)
        assert am.track_mappings[0][0][0]['gain'] == 2.0

    def test_gain_clamped_to_0(self, am):
        mapping = [[{'index': 0, 'gain': -1.0}]]
        am.set_track_mapping(mapping, 0)
        assert am.track_mappings[0][0][0]['gain'] == 0.0

    def test_max_3_sources_per_channel(self, am):
        mapping = [[
            {'index': 0, 'gain': 1.0},
            {'index': 1, 'gain': 1.0},
            {'index': 2, 'gain': 1.0},
            {'index': 3, 'gain': 1.0},  # 4th source — should be dropped
        ]]
        am.set_track_mapping(mapping, 0)
        assert len(am.track_mappings[0][0]) == 3

    def test_padding_to_num_output_channels(self, am):
        mapping = [[{'index': 0, 'gain': 1.0}]]
        am.set_track_mapping(mapping, 0)
        assert len(am.track_mappings[0]) == 8

    def test_negative_index_skipped(self, am):
        mapping = [[{'index': -1, 'gain': 1.0}]]
        am.set_track_mapping(mapping, 0)
        assert am.track_mappings[0][0] == []

    def test_multiple_outputs(self, am):
        am.set_track_mapping([[{'index': 0, 'gain': 1.0}]], 0)
        am.set_track_mapping([[{'index': 4, 'gain': 0.8}]], 1)
        assert am.track_mappings[0][0] == [{'index': 0, 'gain': 1.0}]
        assert am.track_mappings[1][0] == [{'index': 4, 'gain': 0.8}]

    def test_empty_mapping(self, am):
        am.set_track_mapping([], 0)
        assert len(am.track_mappings[0]) == 8
        for entry in am.track_mappings[0]:
            assert entry == []

    def test_nested_list_of_dicts(self, am):
        mapping = [
            [{'index': 0, 'gain': 0.5}, {'index': 4, 'gain': 0.5}],
        ]
        am.set_track_mapping(mapping, 0)
        result = am.track_mappings[0][0]
        assert len(result) == 2
        assert result[0]['index'] == 0
        assert result[1]['index'] == 4


# ── AudioManager.update_default_mapping ──

class TestUpdateDefaultMapping:
    @pytest.fixture
    def am(self):
        with patch.object(AudioManager, '__init__', lambda self: None):
            mgr = AudioManager.__new__(AudioManager)
            mgr.num_output_channels = 8
            mgr.track_mappings = [[]]
            mgr.total_input_channels = 2
            return mgr

    def test_basic_mapping(self, am):
        am.update_default_mapping(4)
        assert am.total_input_channels == 4
        mapping = am.track_mappings[0]
        assert len(mapping) == 8
        # Channels wrap modulo total_input_channels
        for i in range(8):
            assert mapping[i] == [{'index': i % 4, 'gain': 1.0}]

    def test_zero_channels_becomes_1(self, am):
        am.update_default_mapping(0)
        assert am.total_input_channels == 1

    def test_negative_channels_becomes_1(self, am):
        am.update_default_mapping(-5)
        assert am.total_input_channels == 1


# ── SampleFilePlayer ──

class TestSampleFilePlayer:
    def test_init(self):
        player = SampleFilePlayer()
        assert player.is_playing is False
        assert player.is_looping is True
        assert player.playhead == 0

    def test_install_audio(self):
        player = SampleFilePlayer()
        data = np.random.randn(48000, 2).astype(np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        assert player.total_frames == 48000
        assert player.channels == 2
        assert player.filename == "test.wav"

    def test_get_next_buffer_silence_when_not_playing(self):
        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        buf = player.get_next_buffer(1024)
        assert np.allclose(buf, 0.0)

    def test_get_next_buffer_returns_audio(self):
        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32) * 0.5
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.play()
        # Skip fade-in artifact
        player._fade_in = False
        buf = player.get_next_buffer(1024)
        assert np.allclose(buf, 0.5)

    def test_playhead_advances(self):
        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.play()
        player._fade_in = False
        player.get_next_buffer(1024)
        assert player.playhead == 1024

    def test_loop_wraps(self):
        player = SampleFilePlayer()
        # Short audio: 2048 frames
        data = np.ones((2048, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.is_looping = True
        player.play()
        player._fade_in = False
        # Read 3 x 1024 = 3072 frames from 2048-frame buffer
        for _ in range(3):
            buf = player.get_next_buffer(1024)
            assert buf.shape == (1024, 2)
        # Playhead should have wrapped
        assert player.playhead == 1024  # 3072 % 2048

    def test_one_shot_stops_at_end(self):
        player = SampleFilePlayer()
        data = np.ones((512, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.is_looping = False
        player.play()
        player._fade_in = False
        buf = player.get_next_buffer(1024)
        assert buf.shape == (1024, 2)
        assert player.is_playing is False

    def test_toggle_loop(self):
        player = SampleFilePlayer()
        assert player.is_looping is True
        player.toggle_loop()
        assert player.is_looping is False
        player.toggle_loop()
        assert player.is_looping is True

    def test_stop_resets_playhead(self):
        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.play()
        player._fade_in = False
        player.get_next_buffer(1024)
        player.stop()
        assert player.playhead == 0
        assert player.is_playing is False

    def test_position_pct(self):
        player = SampleFilePlayer()
        data = np.ones((1000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.playhead = 500
        assert player.get_position_pct() == pytest.approx(0.5)

    def test_position_pct_empty(self):
        player = SampleFilePlayer()
        assert player.get_position_pct() == 0.0

    def test_duration_str(self):
        player = SampleFilePlayer()
        data = np.ones((96000, 2), dtype=np.float32)  # 2 seconds at 48kHz
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        assert player.get_duration_str() == "0:02"

    def test_fade_out_on_pause(self):
        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32) * 0.5
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.play()
        player._fade_in = False
        player.pause()
        buf = player.get_next_buffer(1024)
        # Fade-out: first samples should be non-zero, last should be zero
        assert buf[0, 0] != 0.0
        assert buf[-1, 0] == 0.0

    def test_silence_when_loading(self):
        player = SampleFilePlayer()
        data = np.ones((48000, 2), dtype=np.float32)
        player.install_audio(data, 48000, "test.wav", "/tmp/test.wav")
        player.play()
        player._loading = True
        buf = player.get_next_buffer(1024)
        assert np.allclose(buf, 0.0)
