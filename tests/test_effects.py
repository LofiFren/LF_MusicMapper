"""Tests for effect classes in effects.py."""

import math
import numpy as np
import pytest
from effects import (
    EffectBase, Filter, Delay, Reverb, DubEcho, Flanger, Phaser,
    Chorus, Gater, BeatRoll, BitCrush, RingMod, ChoppedAndScrewed,
    EffectsEngine, EFFECT_CLASSES,
)


# ── EffectBase ──

class TestEffectBase:
    def test_defaults(self):
        fx = EffectBase()
        assert fx.sample_rate == 48000
        assert fx.enabled is True
        assert fx.dry_wet == 0.5
        assert fx.freeze is False

    def test_set_param_dry_wet(self):
        fx = EffectBase()
        fx.set_param('dry_wet', 0.75)
        assert fx.dry_wet == 0.75

    def test_dry_wet_clamp(self):
        fx = EffectBase()
        fx.set_param('dry_wet', 1.5)
        assert fx.dry_wet == 1.0
        fx.set_param('dry_wet', -0.5)
        assert fx.dry_wet == 0.0

    def test_passthrough_when_disabled(self, sine_1024):
        fx = EffectBase()
        fx.enabled = False
        out = fx.process_stereo(sine_1024)
        np.testing.assert_array_equal(out, sine_1024)

    def test_passthrough_process_impl(self, sine_1024):
        fx = EffectBase()
        fx.dry_wet = 1.0
        out = fx.process_stereo(sine_1024)
        np.testing.assert_array_almost_equal(out, sine_1024)

    def test_set_bpm(self):
        fx = EffectBase()
        fx.set_bpm(140.0)
        assert fx._bpm == 140.0

    def test_bpm_clamp(self):
        fx = EffectBase()
        fx.set_bpm(0.0)
        assert fx._bpm == 1.0

    def test_set_sub_type(self):
        fx = EffectBase()
        fx.set_sub_type(2)
        assert fx._sub_type == 2

    def test_set_beat_fraction(self):
        fx = EffectBase()
        fx.set_beat_fraction(0.5)
        assert fx._beat_frac == 0.5

    def test_reset_clears_freeze(self):
        fx = EffectBase()
        fx.freeze = True
        fx._freeze_buf = np.zeros((100, 2))
        fx.reset()
        assert fx.freeze is False
        assert fx._freeze_buf is None

    def test_dry_wet_mix(self, sine_1024):
        fx = EffectBase()
        fx.dry_wet = 0.0  # fully dry
        out = fx.process_stereo(sine_1024)
        np.testing.assert_array_almost_equal(out, sine_1024)

    def test_freeze_loops_buffer(self, sine_1024):
        fx = EffectBase()
        fx.dry_wet = 1.0
        # Process once to capture
        fx.freeze = True
        out1 = fx.process_stereo(sine_1024.copy())
        # Now frozen — should loop captured buffer
        out2 = fx.process_stereo(sine_1024.copy())
        assert fx._freeze_buf is not None


# ── All registered effects: basic smoke tests ──

class TestAllEffectsSmoke:
    """Verify every registered effect can be instantiated, process audio,
    and reset without crashing."""

    @pytest.mark.parametrize("effect_cls", EFFECT_CLASSES,
                             ids=[c.name for c in EFFECT_CLASSES])
    def test_instantiate(self, effect_cls):
        fx = effect_cls(sample_rate=48000, buffer_size=1024)
        assert fx.name
        assert fx.sample_rate == 48000

    @pytest.mark.parametrize("effect_cls", EFFECT_CLASSES,
                             ids=[c.name for c in EFFECT_CLASSES])
    def test_process_silence(self, effect_cls, silence_1024):
        fx = effect_cls(sample_rate=48000, buffer_size=1024)
        fx.dry_wet = 1.0
        out = fx.process_stereo(silence_1024.copy())
        assert out.shape == (1024, 2)
        assert out.dtype == np.float32

    @pytest.mark.parametrize("effect_cls", EFFECT_CLASSES,
                             ids=[c.name for c in EFFECT_CLASSES])
    def test_process_sine(self, effect_cls, sine_1024):
        fx = effect_cls(sample_rate=48000, buffer_size=1024)
        fx.dry_wet = 1.0
        out = fx.process_stereo(sine_1024.copy())
        assert out.shape == (1024, 2)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("effect_cls", EFFECT_CLASSES,
                             ids=[c.name for c in EFFECT_CLASSES])
    def test_process_noise(self, effect_cls, noise_1024):
        fx = effect_cls(sample_rate=48000, buffer_size=1024)
        fx.dry_wet = 0.5
        out = fx.process_stereo(noise_1024.copy())
        assert out.shape == (1024, 2)
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("effect_cls", EFFECT_CLASSES,
                             ids=[c.name for c in EFFECT_CLASSES])
    def test_reset(self, effect_cls):
        fx = effect_cls(sample_rate=48000, buffer_size=1024)
        fx.reset()
        assert fx.freeze is False

    @pytest.mark.parametrize("effect_cls", EFFECT_CLASSES,
                             ids=[c.name for c in EFFECT_CLASSES])
    def test_multiple_blocks(self, effect_cls, sine_1024):
        """Process multiple blocks to check for state corruption."""
        fx = effect_cls(sample_rate=48000, buffer_size=1024)
        fx.dry_wet = 0.5
        for _ in range(10):
            out = fx.process_stereo(sine_1024.copy())
            assert out.shape == (1024, 2)
            assert np.all(np.isfinite(out)), f"{effect_cls.name} produced NaN/Inf"

    @pytest.mark.parametrize("effect_cls", EFFECT_CLASSES,
                             ids=[c.name for c in EFFECT_CLASSES])
    def test_param_defaults_valid(self, effect_cls):
        fx = effect_cls(sample_rate=48000, buffer_size=1024)
        for pdef in fx.param_defs:
            val = fx.get_param(pdef['key'])
            assert pdef['min'] <= val <= pdef['max'], \
                f"{effect_cls.name}.{pdef['key']} default {val} out of range"


# ── Filter-specific tests ──

class TestFilter:
    def test_lpf_attenuates_high(self, noise_1024):
        fx = Filter(sample_rate=48000, buffer_size=1024)
        fx.set_sub_type(0)  # LPF
        fx.set_param('cutoff', 200)
        fx.set_param('resonance', 0.707)
        fx.dry_wet = 1.0
        # Process several blocks to let filter settle
        for _ in range(5):
            out = fx.process_stereo(noise_1024.copy())
        # Energy should be lower than input (LPF removes high freq content)
        assert np.mean(out**2) < np.mean(noise_1024**2)

    def test_sub_type_change(self):
        fx = Filter()
        for i in range(4):  # LPF, HPF, BPF, Notch
            fx.set_sub_type(i)
            assert fx._sub_type == i

    def test_recalc_produces_finite_sos(self):
        fx = Filter()
        fx.set_param('cutoff', 5000)
        fx.set_param('resonance', 5.0)
        assert np.all(np.isfinite(fx._sos))


# ── Delay-specific tests ──

class TestDelay:
    def test_delay_adds_echo(self, impulse_1024):
        fx = Delay(sample_rate=48000, buffer_size=1024)
        fx.dry_wet = 1.0
        fx.set_param('feedback', 0.5)
        fx.set_bpm(120.0)
        # Use 1/16 note for short delay (~62ms = ~3000 samples)
        fx.set_beat_fraction(1/16)
        # Process impulse then several silence blocks
        fx.process_stereo(impulse_1024.copy())
        silence = np.zeros((1024, 2), dtype=np.float32)
        total_energy = 0.0
        for _ in range(10):
            out = fx.process_stereo(silence.copy())
            total_energy += np.sum(out**2)
        assert total_energy > 0

    def test_tone_recalc(self):
        fx = Delay()
        fx.set_param('tone', 1000)
        assert 0.0 < fx._tone_coeff < 1.0


# ── Reverb-specific tests ──

class TestReverb:
    def test_room_hall_plate_subtypes(self, impulse_1024):
        for sub_type in range(3):
            fx = Reverb(sample_rate=48000, buffer_size=1024)
            fx.set_sub_type(sub_type)
            fx.dry_wet = 1.0
            out = fx.process_stereo(impulse_1024.copy())
            assert out.shape == (1024, 2)
            assert np.all(np.isfinite(out))

    def test_reverb_produces_tail(self, impulse_1024):
        fx = Reverb(sample_rate=48000, buffer_size=1024)
        fx.dry_wet = 1.0
        fx.set_param('decay', 0.8)
        # Process impulse
        fx.process_stereo(impulse_1024.copy())
        # Process silence — reverb tail should still have energy
        silence = np.zeros((1024, 2), dtype=np.float32)
        out = fx.process_stereo(silence)
        assert np.max(np.abs(out)) > 1e-6

    def test_delay_lines_initialized(self):
        fx = Reverb(sample_rate=48000)
        assert len(fx._bufs) == 4
        assert len(fx._delays) == 4


# ── EffectsEngine ──

class TestEffectsEngine:
    def test_init(self):
        engine = EffectsEngine()
        assert engine.enabled is False
        assert engine.num_outputs == 3
        assert engine.num_tracks == 4

    def test_set_track_effect(self):
        engine = EffectsEngine()
        fx = engine.set_track_effect(0, 0, Filter)
        assert fx is not None
        assert isinstance(fx, Filter)
        assert engine.get_track_effect(0, 0) is fx

    def test_clear_track_effect(self):
        engine = EffectsEngine()
        engine.set_track_effect(0, 0, Filter)
        engine.set_track_effect(0, 0, None)
        assert engine.get_track_effect(0, 0) is None

    def test_set_bus_effect(self):
        engine = EffectsEngine()
        fx = engine.set_bus_effect(0, Reverb)
        assert isinstance(fx, Reverb)
        assert engine.get_bus_effect(0) is fx

    def test_clear_bus_effect(self):
        engine = EffectsEngine()
        engine.set_bus_effect(0, Reverb)
        engine.set_bus_effect(0, None)
        assert engine.get_bus_effect(0) is None

    def test_set_bpm_propagates(self):
        engine = EffectsEngine()
        engine.set_track_effect(0, 0, Delay)
        engine.set_bus_effect(1, DubEcho)
        engine.set_bpm(140.0)
        assert engine.bpm == 140.0
        assert engine.get_track_effect(0, 0)._bpm == 140.0
        assert engine.get_bus_effect(1)._bpm == 140.0

    def test_process_disabled(self, sine_1024):
        engine = EffectsEngine()
        engine.enabled = False
        frames = np.column_stack([sine_1024, sine_1024, sine_1024, sine_1024])
        out = engine.process(0, frames.copy(), 8)
        np.testing.assert_array_equal(out, frames)

    def test_process_with_track_effect(self, sine_1024):
        engine = EffectsEngine()
        engine.enabled = True
        engine.set_track_effect(0, 0, Filter)
        # 8-channel frame (4 stereo tracks)
        frames = np.zeros((1024, 8), dtype=np.float32)
        frames[:, 0:2] = sine_1024
        out = engine.process(0, frames.copy(), 8)
        assert out.shape == (1024, 8)
        assert np.all(np.isfinite(out))

    def test_process_with_bus_effect(self, sine_1024):
        engine = EffectsEngine()
        engine.enabled = True
        engine.set_bus_effect(0, Reverb)
        frames = np.zeros((1024, 8), dtype=np.float32)
        frames[:, 0:2] = sine_1024
        out = engine.process(0, frames.copy(), 8)
        assert out.shape == (1024, 8)
        assert np.all(np.isfinite(out))

    def test_reset_all(self):
        engine = EffectsEngine()
        engine.set_track_effect(0, 0, Filter)
        engine.set_bus_effect(0, Reverb)
        engine.reset_all()  # Should not raise

    def test_get_effect_out_of_range(self):
        engine = EffectsEngine()
        assert engine.get_track_effect(99, 99) is None
        assert engine.get_bus_effect(99) is None

    def test_noise_gate(self, silence_1024):
        engine = EffectsEngine()
        engine.enabled = True
        engine.set_track_effect(0, 0, Reverb)
        frames = np.zeros((1024, 8), dtype=np.float32)
        # Tiny values that should be flushed by noise gate
        frames[:, :] = 1e-9
        out = engine.process(0, frames.copy(), 8)
        # Values below 1e-8 should be zeroed
        assert np.sum(np.abs(out) > 0) < frames.size  # at least some zeroed
