"""Tests for PresetManager in mapper.py."""

import json
import os
import pytest
from unittest.mock import patch

import sys
from unittest.mock import MagicMock
mock_pa = MagicMock()
mock_pa.PyAudio.return_value.get_device_count.return_value = 0
if 'pyaudio' not in sys.modules:
    sys.modules['pyaudio'] = mock_pa

from mapper import PresetManager


class TestPresetManager:
    @pytest.fixture
    def pm(self, tmp_path):
        """PresetManager with temp directories."""
        pm = PresetManager.__new__(PresetManager)
        pm.PRESETS_DIR = str(tmp_path / 'presets')
        pm.LAST_SESSION_FILE = str(tmp_path / '_last_session.json')
        os.makedirs(pm.PRESETS_DIR, exist_ok=True)
        return pm

    def test_save_and_load_preset(self, pm):
        config = {'inputs': ['Volt 476'], 'outputs': ['S-4']}
        filepath = pm.save_preset('My Setup', config)
        loaded = pm.load_preset(filepath)
        assert loaded['name'] == 'My Setup'
        assert loaded['version'] == 1
        assert loaded['inputs'] == ['Volt 476']
        assert 'created' in loaded

    def test_list_presets(self, pm):
        pm.save_preset('Setup A', {'data': 1})
        pm.save_preset('Setup B', {'data': 2})
        presets = pm.list_presets()
        names = [name for name, _ in presets]
        assert 'Setup A' in names
        assert 'Setup B' in names

    def test_list_presets_empty(self, pm):
        assert pm.list_presets() == []

    def test_delete_preset(self, pm):
        filepath = pm.save_preset('Temp', {'data': 1})
        assert os.path.isfile(filepath)
        pm.delete_preset(filepath)
        assert not os.path.isfile(filepath)

    def test_delete_nonexistent(self, pm):
        pm.delete_preset('/nonexistent/path.json')  # Should not raise

    def test_save_last_session(self, pm):
        config = {'inputs': ['Volt'], 'outputs': ['S-4']}
        pm.save_last_session(config)
        assert os.path.isfile(pm.LAST_SESSION_FILE)
        with open(pm.LAST_SESSION_FILE) as f:
            data = json.load(f)
        assert data['name'] == '_last_session'
        assert data['version'] == 1

    def test_load_last_session(self, pm):
        config = {'inputs': ['Volt']}
        pm.save_last_session(config)
        loaded = pm.load_last_session()
        assert loaded['inputs'] == ['Volt']

    def test_load_last_session_missing(self, pm):
        assert pm.load_last_session() is None

    def test_preset_name_sanitization(self, pm):
        filepath = pm.save_preset('My/Bad:Name*Here', {'data': 1})
        assert os.path.isfile(filepath)
        # Filename should not contain special chars
        basename = os.path.basename(filepath)
        for ch in '/:*':
            assert ch not in basename

    def test_load_corrupt_last_session(self, pm):
        with open(pm.LAST_SESSION_FILE, 'w') as f:
            f.write('not valid json{{{')
        assert pm.load_last_session() is None

    def test_overwrite_preset(self, pm):
        pm.save_preset('Same Name', {'version_data': 1})
        pm.save_preset('Same Name', {'version_data': 2})
        presets = pm.list_presets()
        # Should only have one file with that name
        matching = [p for name, p in presets if name == 'Same Name']
        assert len(matching) == 1
        loaded = pm.load_preset(matching[0])
        assert loaded['version_data'] == 2
