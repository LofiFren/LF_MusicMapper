"""
themes.py - Built-in color themes for LF Music Mapper.

Provides theme definitions (palettes, meter zones, spectrogram color stops),
a QSS stylesheet generator, and a spectrogram color LUT builder.
"""

from PyQt5 import QtGui

# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------
# Each theme is a dict with:
#   palette           – color strings keyed by role
#   meter_zones       – list of (threshold, (top_r,g,b), (bottom_r,g,b))
#   meter_peak_zones  – list of (threshold, (r,g,b)) for peak indicators
#   spectrogram_stops – list of (t, (r,g,b)) for LUT interpolation
#   spectrogram_bg    – (r,g,b) background behind the 3D waterfall

THEMES = {
    # ------------------------------------------------------------------
    # Dark  –  VS Code-style grey, the original default
    # ------------------------------------------------------------------
    'Dark': {
        'palette': {
            'bg_primary':       '#2D2D30',
            'bg_secondary':     '#252526',
            'bg_tertiary':      '#1E1E1E',
            'bg_input':         '#333337',
            'text_primary':     '#F0F0F0',
            'text_secondary':   '#CCCCCC',
            'text_dim':         '#BBBBBB',
            'text_bright':      '#EEEEEE',
            'border':           '#3F3F46',
            'accent':           '#007ACC',
            'accent_hover':     '#1C97EA',
            'accent_button':    '#2C5AA0',
            'accent_button_hover': '#3771C8',
            'hover':            '#3E3E40',
            'preset_bg':        '#424245',
            'preset_hover':     '#4D4D52',
            'danger':           '#B33A3A',
            'signal_active_bg': '#1E3A29',
            'signal_low_bg':    '#3A3A1E',
            'separator':        '#444444',
            'meter_bg':         '#1C1C1E',
            'track_colors': ['#4B9DE0', '#50C878', '#E6A23C', '#E77F7F'],
        },
        'meter_zones': [
            (0.40, (50, 205, 50),  (30, 160, 30)),
            (0.70, (180, 210, 40), (170, 170, 20)),
            (0.85, (255, 215, 0),  (240, 170, 0)),
            (0.95, (255, 140, 0),  (220, 80, 0)),
            (1.01, (255, 30, 30),  (180, 0, 0)),
        ],
        'meter_peak_zones': [
            (0.40, (120, 255, 120)),
            (0.70, (230, 230, 100)),
            (0.85, (255, 200, 0)),
            (0.95, (255, 140, 0)),
            (1.01, (255, 50, 50)),
        ],
        'spectrogram_stops': [
            (0.00, (5, 5, 20)),
            (0.02, (5, 5, 20)),
            (0.15, (5, 185, 235)),
            (0.35, (15, 240, 50)),
            (0.55, (255, 255, 10)),
            (0.75, (255, 80, 15)),
            (0.92, (255, 40, 95)),
            (1.00, (255, 255, 255)),
        ],
        'spectrogram_bg': (8, 8, 15),
    },

    # ------------------------------------------------------------------
    # Midnight  –  Deep navy, electric blue accents, studio night vibe
    # ------------------------------------------------------------------
    'Midnight': {
        'palette': {
            'bg_primary':       '#0E1525',
            'bg_secondary':     '#0A0E1A',
            'bg_tertiary':      '#060A14',
            'bg_input':         '#141E33',
            'text_primary':     '#D0D8E8',
            'text_secondary':   '#8899BB',
            'text_dim':         '#6677AA',
            'text_bright':      '#E0E8F8',
            'border':           '#1E2D4A',
            'accent':           '#00BFFF',
            'accent_hover':     '#33CCFF',
            'accent_button':    '#0066AA',
            'accent_button_hover': '#0088CC',
            'hover':            '#182840',
            'preset_bg':        '#1A2540',
            'preset_hover':     '#223355',
            'danger':           '#CC3355',
            'signal_active_bg': '#0A2233',
            'signal_low_bg':    '#1A2222',
            'separator':        '#1E2D4A',
            'meter_bg':         '#080C16',
            'track_colors': ['#00BFFF', '#00E5A0', '#FFB030', '#FF6B8A'],
        },
        'meter_zones': [
            (0.40, (0, 160, 220),  (0, 100, 180)),
            (0.70, (0, 200, 160),  (0, 160, 120)),
            (0.85, (255, 200, 50), (200, 160, 30)),
            (0.95, (255, 120, 50), (200, 80, 20)),
            (1.01, (255, 40, 80),  (180, 20, 50)),
        ],
        'meter_peak_zones': [
            (0.40, (80, 200, 255)),
            (0.70, (80, 230, 180)),
            (0.85, (255, 220, 100)),
            (0.95, (255, 150, 80)),
            (1.01, (255, 60, 100)),
        ],
        'spectrogram_stops': [
            (0.00, (2, 3, 12)),
            (0.02, (2, 3, 12)),
            (0.15, (0, 60, 180)),
            (0.30, (0, 160, 255)),
            (0.50, (0, 220, 180)),
            (0.70, (100, 255, 100)),
            (0.85, (255, 200, 50)),
            (0.95, (255, 100, 60)),
            (1.00, (255, 240, 255)),
        ],
        'spectrogram_bg': (4, 6, 16),
    },

    # ------------------------------------------------------------------
    # Neon  –  True black, cyan+magenta+lime, cyberpunk
    # ------------------------------------------------------------------
    'Neon': {
        'palette': {
            'bg_primary':       '#0A0A0A',
            'bg_secondary':     '#111111',
            'bg_tertiary':      '#050505',
            'bg_input':         '#1A1A1A',
            'text_primary':     '#E0E0E0',
            'text_secondary':   '#AAAAAA',
            'text_dim':         '#888888',
            'text_bright':      '#FFFFFF',
            'border':           '#333333',
            'accent':           '#00FFCC',
            'accent_hover':     '#33FFD4',
            'accent_button':    '#CC00FF',
            'accent_button_hover': '#DD44FF',
            'hover':            '#222222',
            'preset_bg':        '#1E1E1E',
            'preset_hover':     '#2A2A2A',
            'danger':           '#FF0055',
            'signal_active_bg': '#001A14',
            'signal_low_bg':    '#1A1A00',
            'separator':        '#333333',
            'meter_bg':         '#080808',
            'track_colors': ['#00FFCC', '#FF00AA', '#AAFF00', '#FFAA00'],
        },
        'meter_zones': [
            (0.40, (0, 255, 200),  (0, 180, 140)),
            (0.70, (170, 255, 0),  (120, 200, 0)),
            (0.85, (255, 255, 0),  (200, 200, 0)),
            (0.95, (255, 0, 170),  (200, 0, 120)),
            (1.01, (255, 0, 85),   (200, 0, 60)),
        ],
        'meter_peak_zones': [
            (0.40, (100, 255, 220)),
            (0.70, (200, 255, 80)),
            (0.85, (255, 255, 100)),
            (0.95, (255, 80, 200)),
            (1.01, (255, 30, 100)),
        ],
        'spectrogram_stops': [
            (0.00, (2, 2, 2)),
            (0.02, (2, 2, 2)),
            (0.12, (0, 40, 80)),
            (0.25, (0, 180, 255)),
            (0.40, (0, 255, 180)),
            (0.55, (160, 255, 0)),
            (0.70, (255, 255, 0)),
            (0.82, (255, 0, 170)),
            (0.93, (255, 0, 80)),
            (1.00, (255, 255, 255)),
        ],
        'spectrogram_bg': (2, 2, 4),
    },

    # ------------------------------------------------------------------
    # Ember  –  Warm dark grey, amber/copper accents, analog warmth
    # ------------------------------------------------------------------
    'Ember': {
        'palette': {
            'bg_primary':       '#1E1A14',
            'bg_secondary':     '#181410',
            'bg_tertiary':      '#12100C',
            'bg_input':         '#2A2418',
            'text_primary':     '#E8DCC8',
            'text_secondary':   '#B8A888',
            'text_dim':         '#998866',
            'text_bright':      '#F0E8D8',
            'border':           '#3A3020',
            'accent':           '#FF8C00',
            'accent_hover':     '#FFA030',
            'accent_button':    '#AA5500',
            'accent_button_hover': '#CC6600',
            'hover':            '#2A2418',
            'preset_bg':        '#302820',
            'preset_hover':     '#3A3228',
            'danger':           '#CC3300',
            'signal_active_bg': '#1E2A14',
            'signal_low_bg':    '#2A2414',
            'separator':        '#3A3020',
            'meter_bg':         '#141210',
            'track_colors': ['#FF8C00', '#CC6633', '#FFBB44', '#FF5533'],
        },
        'meter_zones': [
            (0.40, (200, 140, 40), (160, 100, 20)),
            (0.70, (220, 180, 40), (180, 140, 20)),
            (0.85, (255, 200, 50), (220, 160, 30)),
            (0.95, (255, 130, 30), (220, 80, 10)),
            (1.01, (255, 50, 20),  (180, 20, 0)),
        ],
        'meter_peak_zones': [
            (0.40, (230, 180, 80)),
            (0.70, (240, 210, 80)),
            (0.85, (255, 220, 100)),
            (0.95, (255, 160, 60)),
            (1.01, (255, 70, 40)),
        ],
        'spectrogram_stops': [
            (0.00, (8, 4, 2)),
            (0.02, (8, 4, 2)),
            (0.12, (40, 10, 5)),
            (0.25, (120, 30, 5)),
            (0.40, (200, 80, 10)),
            (0.55, (255, 140, 20)),
            (0.70, (255, 200, 50)),
            (0.85, (255, 240, 120)),
            (1.00, (255, 255, 240)),
        ],
        'spectrogram_bg': (6, 4, 2),
    },

    # ------------------------------------------------------------------
    # Arctic  –  Light grey background, dark text, cool blue accents
    # ------------------------------------------------------------------
    'Arctic': {
        'palette': {
            'bg_primary':       '#E8EAED',
            'bg_secondary':     '#F2F3F5',
            'bg_tertiary':      '#D8DCE0',
            'bg_input':         '#FFFFFF',
            'text_primary':     '#1A1A2E',
            'text_secondary':   '#444466',
            'text_dim':         '#666688',
            'text_bright':      '#0A0A1E',
            'border':           '#C0C4CC',
            'accent':           '#3A7BD5',
            'accent_hover':     '#5090E0',
            'accent_button':    '#2860B0',
            'accent_button_hover': '#3570CC',
            'hover':            '#D4D8DD',
            'preset_bg':        '#DDE0E4',
            'preset_hover':     '#CCD0D6',
            'danger':           '#D03030',
            'signal_active_bg': '#D0EED0',
            'signal_low_bg':    '#EEE8C0',
            'separator':        '#C0C4CC',
            'meter_bg':         '#D0D4D8',
            'track_colors': ['#3A7BD5', '#2AAA55', '#DD8800', '#D04040'],
        },
        'meter_zones': [
            (0.40, (40, 180, 80),  (30, 140, 60)),
            (0.70, (160, 190, 40), (140, 160, 30)),
            (0.85, (220, 190, 0),  (200, 160, 0)),
            (0.95, (230, 120, 0),  (200, 80, 0)),
            (1.01, (220, 30, 30),  (180, 10, 10)),
        ],
        'meter_peak_zones': [
            (0.40, (60, 200, 100)),
            (0.70, (180, 210, 60)),
            (0.85, (240, 210, 30)),
            (0.95, (240, 140, 20)),
            (1.01, (240, 50, 50)),
        ],
        'spectrogram_stops': [
            (0.00, (200, 205, 215)),
            (0.02, (200, 205, 215)),
            (0.15, (80, 140, 220)),
            (0.30, (40, 180, 200)),
            (0.50, (50, 200, 100)),
            (0.70, (220, 200, 40)),
            (0.85, (230, 120, 30)),
            (0.95, (210, 50, 40)),
            (1.00, (160, 20, 20)),
        ],
        'spectrogram_bg': (210, 215, 225),
    },

    # ------------------------------------------------------------------
    # Monochrome  –  Grey-scale, white/silver accents, minimal
    # ------------------------------------------------------------------
    'Monochrome': {
        'palette': {
            'bg_primary':       '#1E1E1E',
            'bg_secondary':     '#181818',
            'bg_tertiary':      '#121212',
            'bg_input':         '#2A2A2A',
            'text_primary':     '#E0E0E0',
            'text_secondary':   '#AAAAAA',
            'text_dim':         '#888888',
            'text_bright':      '#FFFFFF',
            'border':           '#3A3A3A',
            'accent':           '#C0C0C0',
            'accent_hover':     '#DDDDDD',
            'accent_button':    '#555555',
            'accent_button_hover': '#6A6A6A',
            'hover':            '#2A2A2A',
            'preset_bg':        '#303030',
            'preset_hover':     '#3A3A3A',
            'danger':           '#AA3333',
            'signal_active_bg': '#1A2A1A',
            'signal_low_bg':    '#2A2A1A',
            'separator':        '#3A3A3A',
            'meter_bg':         '#141414',
            'track_colors': ['#C0C0C0', '#909090', '#E0E0E0', '#707070'],
        },
        'meter_zones': [
            (0.40, (160, 160, 160), (120, 120, 120)),
            (0.70, (190, 190, 190), (150, 150, 150)),
            (0.85, (210, 210, 210), (180, 180, 180)),
            (0.95, (230, 230, 230), (200, 200, 200)),
            (1.01, (255, 255, 255), (220, 220, 220)),
        ],
        'meter_peak_zones': [
            (0.40, (180, 180, 180)),
            (0.70, (200, 200, 200)),
            (0.85, (220, 220, 220)),
            (0.95, (240, 240, 240)),
            (1.01, (255, 255, 255)),
        ],
        'spectrogram_stops': [
            (0.00, (8, 8, 8)),
            (0.02, (8, 8, 8)),
            (0.20, (50, 50, 50)),
            (0.40, (100, 100, 100)),
            (0.60, (150, 150, 150)),
            (0.80, (200, 200, 200)),
            (1.00, (255, 255, 255)),
        ],
        'spectrogram_bg': (6, 6, 6),
    },
}


# ---------------------------------------------------------------------------
# QSS generator
# ---------------------------------------------------------------------------

def generate_qss(palette):
    """Generate the full application QSS stylesheet from a palette dict."""
    p = palette
    return f"""
        QMainWindow, QWidget {{
            background-color: {p['bg_primary']};
            color: {p['text_primary']};
        }}
        QGroupBox {{
            border: 1px solid {p['border']};
            border-radius: 8px;
            margin-top: 1.5ex;
            font-weight: bold;
            padding: 10px;
            background-color: {p['bg_secondary']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 8px;
            background-color: {p['bg_primary']};
        }}
        QComboBox {{
            background-color: {p['bg_input']};
            border: 1px solid {p['border']};
            border-radius: 5px;
            padding: 6px;
            color: {p['text_primary']};
            min-height: 25px;
        }}
        QComboBox:hover {{
            border-color: {p['accent']};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {p['bg_input']};
            color: {p['text_primary']};
            border: 1px solid {p['border']};
            selection-background-color: {p['accent']};
        }}
        QPushButton {{
            background-color: {p['bg_input']};
            border: 1px solid {p['border']};
            border-radius: 5px;
            padding: 6px;
            color: {p['text_primary']};
            min-height: 25px;
        }}
        QPushButton:hover {{
            background-color: {p['hover']};
            border-color: {p['accent']};
        }}
        QPushButton:pressed {{
            background-color: {p['accent']};
        }}
        QLabel {{
            padding: 2px;
        }}
        QStatusBar {{
            background-color: {p['bg_tertiary']};
            color: {p['text_secondary']};
        }}
        QCheckBox {{
            spacing: 5px;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border-radius: 3px;
        }}
        QTabWidget::pane {{
            border: 1px solid {p['border']};
            border-radius: 5px;
            background: {p['bg_secondary']};
        }}
        QTabBar::tab {{
            background: {p['bg_primary']};
            border: 1px solid {p['border']};
            border-bottom-color: {p['bg_secondary']};
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 6px 10px;
            margin-right: 2px;
        }}
        QTabBar::tab:selected, QTabBar::tab:hover {{
            background: {p['bg_input']};
        }}
        QTabBar::tab:selected {{
            border-bottom-color: {p['bg_secondary']};
        }}
        QSlider::groove:horizontal {{
            border: 1px solid {p['border']};
            height: 8px;
            background: {p['bg_primary']};
            margin: 2px 0;
            border-radius: 4px;
        }}
        QSlider::handle:horizontal {{
            background: {p['accent']};
            border: 1px solid {p['accent']};
            width: 18px;
            height: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {p['accent_hover']};
        }}

        /* Apply button style */
        #applyButton {{
            background-color: {p['accent_button']};
            color: white;
            font-weight: bold;
            min-height: 35px;
            border-radius: 5px;
        }}
        #applyButton:hover {{
            background-color: {p['accent_button_hover']};
        }}

        /* Preset button styles */
        QPushButton[preset="true"] {{
            background-color: {p['preset_bg']};
            border-width: 1px;
        }}
        QPushButton[preset="true"]:hover {{
            background-color: {p['preset_hover']};
        }}

        /* QDial (mixer strip volume knob) */
        QDial {{
            background-color: {p['bg_secondary']};
        }}

        /* Menu bar */
        QMenuBar {{
            background-color: {p['bg_tertiary']};
            color: {p['text_primary']};
            border-bottom: 1px solid {p['border']};
        }}
        QMenuBar::item:selected {{
            background-color: {p['hover']};
        }}
        QMenu {{
            background-color: {p['bg_input']};
            color: {p['text_primary']};
            border: 1px solid {p['border']};
        }}
        QMenu::item:selected {{
            background-color: {p['accent']};
        }}
    """


# ---------------------------------------------------------------------------
# Spectrogram LUT builder
# ---------------------------------------------------------------------------

def build_spectrogram_lut(stops, steps=256):
    """Interpolate spectrogram color stops into a *steps*-entry QColor list.

    Args:
        stops: list of ``(t, (r, g, b))`` where *t* is 0.0..1.0.
        steps: number of LUT entries (default 256).

    Returns:
        list[QColor] of length *steps*.
    """
    lut = []
    for i in range(steps):
        t = i / (steps - 1)
        # Find the surrounding stops
        lower = stops[0]
        upper = stops[-1]
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                lower = stops[j]
                upper = stops[j + 1]
                break
        # Interpolate
        span = upper[0] - lower[0]
        if span <= 0:
            frac = 0.0
        else:
            frac = (t - lower[0]) / span
        r = int(lower[1][0] + (upper[1][0] - lower[1][0]) * frac)
        g = int(lower[1][1] + (upper[1][1] - lower[1][1]) * frac)
        b = int(lower[1][2] + (upper[1][2] - lower[1][2]) * frac)
        lut.append(QtGui.QColor(
            max(0, min(255, r)),
            max(0, min(255, g)),
            max(0, min(255, b)),
        ))
    return lut
