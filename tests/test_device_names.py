"""Tests for short_device_name display-name shortening."""

import sys
from unittest.mock import MagicMock

# Mock pyaudio before importing mapper
mock_pa = MagicMock()
mock_pa.PyAudio.return_value.get_device_count.return_value = 0
sys.modules.setdefault('pyaudio', mock_pa)

from mapper import short_device_name


class TestShortDeviceName:
    def test_plain_name_unchanged(self):
        assert short_device_name("Volt 476") == "Volt 476"

    def test_strips_noise_words(self):
        assert short_device_name("Volt 476 USB Audio Device") == "Volt 476"

    def test_strips_noise_words_case_insensitive(self):
        assert short_device_name("Scarlett 2i2 usb AUDIO") == "Scarlett 2i2"

    def test_strips_parentheticals(self):
        assert short_device_name("Digitakt (Elektron)") == "Digitakt"

    def test_truncates_long_names_with_ellipsis(self):
        result = short_device_name("Some Extremely Long Hardware Name", max_len=14)
        assert len(result) <= 14
        assert result.endswith("…")

    def test_respects_custom_max_len(self):
        assert short_device_name("Analog Heat +FX", max_len=16) == "Analog Heat +FX"

    def test_all_noise_words_falls_back_to_original(self):
        # A name made only of noise words must not collapse to empty
        assert short_device_name("USB Audio Device") == "USB Audio Device"

    def test_empty_and_none(self):
        assert short_device_name("") == ""
        assert short_device_name(None) == "None"

    def test_no_trailing_space_before_ellipsis(self):
        result = short_device_name("Analog Heat +FX MKII Special", max_len=13)
        assert "  " not in result
        assert not result[:-1].endswith(" ")

    def test_keeps_meaningful_words_within_limit(self):
        assert short_device_name("BIG Ag") == "BIG Ag"
        assert short_device_name("S-4") == "S-4"
