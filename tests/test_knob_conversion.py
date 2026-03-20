"""Tests for knob/value conversion functions in effects.py.

These functions live inside `if QtWidgets is not None:` so they require
PyQt5/PySide2 to be importable. If Qt is unavailable, tests are skipped.
"""

import math
import pytest

try:
    from effects import _knob_to_value, _value_to_knob, _fmt_value
    HAS_QT = True
except ImportError:
    HAS_QT = False

pytestmark = pytest.mark.skipif(not HAS_QT, reason="Qt not available")


# Sample parameter definitions matching effects.py
LINEAR_PDEF = {'key': 'feedback', 'min': 0.0, 'max': 0.95, 'default': 0.4, 'log': False}
LOG_PDEF = {'key': 'cutoff', 'min': 20, 'max': 20000, 'default': 1000, 'log': True}
PERCENT_PDEF = {'key': 'depth', 'min': 0.0, 'max': 1.0, 'default': 0.6, 'log': False}
SMALL_PDEF = {'key': 'decay', 'min': 0.1, 'max': 0.99, 'default': 0.6, 'log': False}


class TestKnobToValue:
    def test_linear_min(self):
        val = _knob_to_value(0, LINEAR_PDEF)
        assert val == pytest.approx(0.0)

    def test_linear_max(self):
        val = _knob_to_value(1000, LINEAR_PDEF)
        assert val == pytest.approx(0.95)

    def test_linear_mid(self):
        val = _knob_to_value(500, LINEAR_PDEF)
        assert val == pytest.approx(0.475)

    def test_log_min(self):
        val = _knob_to_value(0, LOG_PDEF)
        assert val == pytest.approx(20.0)

    def test_log_max(self):
        val = _knob_to_value(1000, LOG_PDEF)
        assert val == pytest.approx(20000.0)

    def test_log_midpoint_is_geometric_mean(self):
        val = _knob_to_value(500, LOG_PDEF)
        geometric_mean = math.sqrt(20.0 * 20000.0)
        assert val == pytest.approx(geometric_mean, rel=0.01)


class TestValueToKnob:
    def test_linear_roundtrip(self):
        for dial in [0, 250, 500, 750, 1000]:
            val = _knob_to_value(dial, LINEAR_PDEF)
            recovered = _value_to_knob(val, LINEAR_PDEF)
            assert recovered == pytest.approx(dial, abs=1)

    def test_log_roundtrip(self):
        for dial in [0, 250, 500, 750, 1000]:
            val = _knob_to_value(dial, LOG_PDEF)
            recovered = _value_to_knob(val, LOG_PDEF)
            assert recovered == pytest.approx(dial, abs=1)

    def test_clamp_low(self):
        result = _value_to_knob(-10.0, LINEAR_PDEF)
        assert result >= 0

    def test_clamp_high(self):
        result = _value_to_knob(100.0, LINEAR_PDEF)
        assert result <= 1000


class TestFmtValue:
    def test_large_log_value_uses_k(self):
        pdef = {'key': 'cutoff', 'min': 20, 'max': 20000, 'log': True}
        assert _fmt_value(5000, pdef) == "5.0k"
        assert _fmt_value(10000, pdef) == "10.0k"

    def test_small_log_value_uses_int(self):
        pdef = {'key': 'cutoff', 'min': 20, 'max': 20000, 'log': True}
        assert _fmt_value(500, pdef) == "500"

    def test_percentage(self):
        pdef = {'key': 'depth', 'min': 0.0, 'max': 1.0, 'log': False}
        assert _fmt_value(0.5, pdef) == "50%"
        assert _fmt_value(1.0, pdef) == "100%"

    def test_small_range(self):
        pdef = {'key': 'decay', 'min': 0.1, 'max': 2.0, 'log': False}
        assert _fmt_value(0.6, pdef) == "0.60"
