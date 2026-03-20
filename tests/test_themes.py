"""Tests for themes.py — palette consistency and QSS generation."""

import pytest
from themes import THEMES, generate_qss


# Required palette keys for generate_qss
REQUIRED_KEYS = [
    'bg_primary', 'bg_secondary', 'bg_tertiary', 'bg_input',
    'text_primary', 'text_secondary', 'border', 'accent',
    'accent_hover', 'hover', 'accent_button', 'accent_button_hover',
    'preset_bg', 'preset_hover',
]


class TestThemeConsistency:
    def test_at_least_one_theme(self):
        assert len(THEMES) >= 1

    @pytest.mark.parametrize("theme_name", list(THEMES.keys()))
    def test_palette_has_required_keys(self, theme_name):
        palette = THEMES[theme_name]['palette']
        for key in REQUIRED_KEYS:
            assert key in palette, f"Theme '{theme_name}' missing palette key '{key}'"

    @pytest.mark.parametrize("theme_name", list(THEMES.keys()))
    def test_palette_values_are_strings(self, theme_name):
        palette = THEMES[theme_name]['palette']
        for key in REQUIRED_KEYS:
            assert isinstance(palette[key], str), \
                f"Theme '{theme_name}' key '{key}' is not a string"

    @pytest.mark.parametrize("theme_name", list(THEMES.keys()))
    def test_has_meter_zones(self, theme_name):
        assert 'meter_zones' in THEMES[theme_name]
        zones = THEMES[theme_name]['meter_zones']
        assert len(zones) >= 1

    @pytest.mark.parametrize("theme_name", list(THEMES.keys()))
    def test_has_spectrogram_stops(self, theme_name):
        assert 'spectrogram_stops' in THEMES[theme_name]
        stops = THEMES[theme_name]['spectrogram_stops']
        assert len(stops) >= 2
        # First stop should be at 0.0, last at 1.0
        assert stops[0][0] == pytest.approx(0.0)
        assert stops[-1][0] == pytest.approx(1.0)

    @pytest.mark.parametrize("theme_name", list(THEMES.keys()))
    def test_spectrogram_stop_values_valid(self, theme_name):
        stops = THEMES[theme_name]['spectrogram_stops']
        for t, (r, g, b) in stops:
            assert 0.0 <= t <= 1.0
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255


class TestGenerateQss:
    @pytest.mark.parametrize("theme_name", list(THEMES.keys()))
    def test_generates_valid_qss(self, theme_name):
        palette = THEMES[theme_name]['palette']
        qss = generate_qss(palette)
        assert isinstance(qss, str)
        assert len(qss) > 100
        # Should contain CSS-like selectors
        assert 'QMainWindow' in qss
        assert 'QPushButton' in qss
        assert 'QComboBox' in qss

    @pytest.mark.parametrize("theme_name", list(THEMES.keys()))
    def test_qss_uses_palette_colors(self, theme_name):
        palette = THEMES[theme_name]['palette']
        qss = generate_qss(palette)
        # Accent color should appear in the stylesheet
        assert palette['accent'] in qss
        assert palette['bg_primary'] in qss
