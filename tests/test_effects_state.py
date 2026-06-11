"""Tests for EffectsEngine state serialization (preset persistence) and DSP load tracking."""

import json

import numpy as np
import pytest

from effects import (
    EFFECT_CLASSES, EffectsEngine, Delay, Reverb, Gater,
)


def make_engine(**kw):
    return EffectsEngine(sample_rate=48000, buffer_size=1024, **kw)


class TestGetState:
    def test_empty_engine_state(self):
        eng = make_engine()
        state = eng.get_state()
        assert state['enabled'] is False
        assert state['bpm'] == eng.bpm
        assert state['track_fx'] == [[None] * 4] * 3
        assert state['bus_fx'] == [None] * 3

    def test_state_is_json_serializable(self):
        eng = make_engine()
        eng.set_track_effect(0, 0, Delay)
        eng.set_bus_effect(1, Reverb)
        json.dumps(eng.get_state())  # must not raise

    def test_effect_state_contains_type_and_params(self):
        eng = make_engine()
        fx = eng.set_track_effect(0, 2, Delay)
        fx.set_param('dry_wet', 0.8)
        state = eng.get_state()
        fx_state = state['track_fx'][0][2]
        assert fx_state['type'] == Delay.name
        assert fx_state['dry_wet'] == pytest.approx(0.8)
        assert set(fx_state['params']) == {pd['key'] for pd in Delay.param_defs}


class TestSetState:
    def test_round_trip_preserves_everything(self):
        src = make_engine()
        src.enabled = True
        src.set_bpm(141.5)
        delay = src.set_track_effect(0, 1, Delay)
        delay.set_param('dry_wet', 0.65)
        delay.set_beat_fraction(0.5)
        for pd in Delay.param_defs:
            delay.set_param(pd['key'], pd['max'])
        reverb = src.set_bus_effect(2, Reverb)
        if Reverb.sub_types:
            reverb.set_sub_type(len(Reverb.sub_types) - 1)

        dst = make_engine()
        dst.set_state(src.get_state())

        assert dst.enabled is True
        assert dst.bpm == pytest.approx(141.5)
        fx = dst.get_track_effect(0, 1)
        assert isinstance(fx, Delay)
        assert fx.dry_wet == pytest.approx(0.65)
        assert fx._beat_frac == pytest.approx(0.5)
        assert fx._bpm == pytest.approx(141.5)
        for pd in Delay.param_defs:
            assert fx.get_param(pd['key']) == pytest.approx(pd['max'])
        bus = dst.get_bus_effect(2)
        assert isinstance(bus, Reverb)
        if Reverb.sub_types:
            assert bus._sub_type == len(Reverb.sub_types) - 1

    def test_round_trip_through_json(self):
        src = make_engine()
        src.set_track_effect(0, 0, Gater)
        blob = json.dumps(src.get_state())
        dst = make_engine()
        dst.set_state(json.loads(blob))
        assert isinstance(dst.get_track_effect(0, 0), Gater)

    def test_set_state_clears_existing_slots(self):
        dst = make_engine()
        dst.set_track_effect(1, 3, Delay)
        dst.set_state(make_engine().get_state())
        assert dst.get_track_effect(1, 3) is None

    def test_unknown_effect_type_skipped(self):
        dst = make_engine()
        state = make_engine().get_state()
        state['track_fx'][0][0] = {'type': 'Nonexistent', 'params': {}}
        dst.set_state(state)
        assert dst.get_track_effect(0, 0) is None

    def test_unknown_param_keys_ignored(self):
        dst = make_engine()
        state = make_engine().get_state()
        state['track_fx'][0][0] = {
            'type': Delay.name,
            'params': {'no_such_param': 123.0},
        }
        dst.set_state(state)
        assert isinstance(dst.get_track_effect(0, 0), Delay)

    @pytest.mark.parametrize('garbage', [None, 42, "x", [], {}, {'track_fx': "bad"}])
    def test_garbage_input_does_not_crash(self, garbage):
        dst = make_engine()
        dst.set_state(garbage)

    def test_processes_after_restore(self, sine_1024):
        dst = make_engine()
        src = make_engine()
        src.enabled = True
        src.set_track_effect(0, 0, Delay)
        dst.set_state(src.get_state())
        out = dst.process(0, np.tile(sine_1024, (1, 4)).astype(np.float32), 8)
        assert out.shape == (1024, 8)
        assert np.all(np.isfinite(out))


class TestHasAnyEffect:
    def test_empty(self):
        assert make_engine().has_any_effect() is False

    def test_track_effect(self):
        eng = make_engine()
        eng.set_track_effect(2, 3, Delay)
        assert eng.has_any_effect() is True

    def test_bus_effect(self):
        eng = make_engine()
        eng.set_bus_effect(0, Reverb)
        assert eng.has_any_effect() is True


class TestDspLoad:
    def test_load_zero_when_disabled(self, sine_1024):
        eng = make_engine()
        eng.set_track_effect(0, 0, Delay)
        eng.process(0, np.tile(sine_1024, (1, 4)).astype(np.float32), 8)
        assert eng.get_load_pct(0) == 0.0

    def test_load_tracked_when_enabled(self, sine_1024):
        eng = make_engine()
        eng.enabled = True
        eng.set_track_effect(0, 0, Reverb)
        frames = np.tile(sine_1024, (1, 4)).astype(np.float32)
        for _ in range(5):
            eng.process(0, frames.copy(), 8)
        assert eng.get_load_pct(0) > 0.0

    def test_load_pct_bounds_and_bad_index(self):
        eng = make_engine()
        assert eng.get_load_pct(-1) == 0.0
        assert eng.get_load_pct(99) == 0.0
