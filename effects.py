"""
effects.py — Real-time Effects Engine for LF Music Mapper

Separate but integrated DSP effects processing, inspired by:
- Pioneer DJM-V10 (Beat FX / Sound Color FX)
- Eventide H9 (studio-grade algorithms)
- NI Traktor Pro (software DJ effects)
- DJ Screw (Chopped & Screwed, Houston TX 1990s)

All processing is numpy-vectorized for real-time safety in PortAudio callbacks.
"""

import numpy as np
import math
import time as _time
from scipy.signal import sosfilt, sosfilt_zi, lfilter

try:
    from PySide2 import QtCore, QtWidgets, QtGui
    Signal = QtCore.Signal
except ImportError:
    try:
        from PyQt5 import QtCore, QtWidgets, QtGui
        Signal = QtCore.pyqtSignal
    except ImportError:
        QtCore = QtWidgets = QtGui = None
        Signal = None

if QtWidgets is not None:
    try:
        from widgets import MiniKnob
    except ImportError:  # standalone use without widgets.py
        MiniKnob = QtWidgets.QDial
else:
    MiniKnob = None


# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

BEAT_FRACTIONS = [
    ('1/16', 1/16), ('1/8', 1/8), ('3/16', 3/16), ('1/4', 1/4),
    ('3/8', 3/8), ('1/2', 1/2), ('3/4', 3/4), ('1', 1.0), ('2', 2.0),
]

DEFAULT_BPM = 120.0
MAX_DELAY_SEC = 3.0

# Populated by @_register — order determines combo box order
EFFECT_CLASSES = []


def _register(cls):
    """Decorator: add effect class to the global registry."""
    EFFECT_CLASSES.append(cls)
    return cls


# ═══════════════════════════════════════════════════════════════════════
# DSP Utilities
# ═══════════════════════════════════════════════════════════════════════

def beat_to_samples(beats, bpm, sr):
    """Convert a beat-fraction value at *bpm* to a sample count."""
    return max(1, int(beats * 60.0 / max(bpm, 1.0) * sr))


def make_biquad_sos(ftype, freq, q, sr):
    """Compute biquad SOS coefficients (Audio EQ Cookbook).

    Returns ndarray shape (1, 6) for scipy.signal.sosfilt.
    """
    freq = max(20.0, min(freq, sr * 0.49))
    q = max(0.1, q)
    w0 = 2.0 * math.pi * freq / sr
    cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    if ftype == 'lpf':
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = b0
    elif ftype == 'hpf':
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = b0
    elif ftype == 'bpf':
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
    elif ftype == 'notch':
        b0 = 1.0
        b1 = -2.0 * cos_w0
        b2 = 1.0
    else:
        return np.array([[1, 0, 0, 1, 0, 0]], dtype=np.float64)

    a0 = 1 + alpha
    a1_c = -2.0 * cos_w0
    a2_c = 1 - alpha
    return np.array([[b0/a0, b1/a0, b2/a0, 1.0, a1_c/a0, a2_c/a0]], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════
# LFO  — low-frequency oscillator for modulation effects
# ═══════════════════════════════════════════════════════════════════════

class LFO:
    def __init__(self, rate=1.0, sample_rate=48000):
        self.rate = rate
        self.sr = sample_rate
        self.phase = 0.0

    def generate(self, n, shape='sine'):
        """Return *n* samples of the oscillator waveform in [-1, 1]."""
        rate = max(self.rate, 0.001)
        t = (self.phase + np.arange(n, dtype=np.float64)) * rate / self.sr
        self.phase = (self.phase + n) % (self.sr / rate)
        if shape == 'triangle':
            return (2.0 * np.abs(2.0 * (t % 1.0) - 1.0) - 1.0).astype(np.float32)
        if shape == 'square':
            return np.where(t % 1.0 < 0.5, 1.0, -1.0).astype(np.float32)
        return np.sin(2.0 * np.pi * t).astype(np.float32)

    def reset(self):
        self.phase = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Base class for all effects
# ═══════════════════════════════════════════════════════════════════════

class EffectBase:
    """Base class for real-time audio effects.

    Subclasses override ``_process_impl(frames)`` where *frames* is an
    ``(n, 2)`` float32 array.  The base class handles dry/wet mixing,
    freeze, and parameter bookkeeping.
    """
    name = "Off"
    # Each entry: {'key', 'label', 'min', 'max', 'default', 'log': bool}
    param_defs = []
    sub_types = []          # e.g. ['LPF', 'HPF', 'BPF', 'Notch']
    has_beat_sync = False

    def __init__(self, sample_rate=48000, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.enabled = True
        self.dry_wet = 0.5
        self.freeze = False
        self._freeze_buf = None
        self._freeze_pos = 0
        self._sub_type = 0
        self._beat_frac = 0.25   # quarter note default
        self._bpm = DEFAULT_BPM
        # Set defaults
        self._params = {}
        for pd in self.param_defs:
            self._params[pd['key']] = pd['default']

    # ── Parameter interface (called from UI / main thread) ──

    def set_param(self, key, value):
        if key == 'dry_wet':
            self.dry_wet = max(0.0, min(1.0, float(value)))
        else:
            self._params[key] = value
            self._on_param_change(key, value)

    def get_param(self, key):
        if key == 'dry_wet':
            return self.dry_wet
        return self._params.get(key, 0.0)

    def _on_param_change(self, key, value):
        """Override to recalculate coefficients when a parameter changes."""
        pass

    def set_sub_type(self, idx):
        self._sub_type = idx
        self._on_sub_type_change()

    def _on_sub_type_change(self):
        pass

    def set_beat_fraction(self, frac):
        self._beat_frac = frac

    def set_bpm(self, bpm):
        self._bpm = max(1.0, bpm)

    def set_sample_rate(self, sr):
        self.sample_rate = sr

    def reset(self):
        """Clear DSP state (delay lines, etc.) — called on routing start."""
        self.freeze = False
        self._freeze_buf = None
        self._freeze_pos = 0

    # ── Processing (called from audio callback thread) ──

    def process_stereo(self, frames):
        """Process an (n, 2) float32 stereo buffer.  Returns (n, 2)."""
        if not self.enabled:
            return frames
        # Freeze: loop captured buffer
        if self.freeze and self._freeze_buf is not None:
            n = len(frames)
            buf_len = len(self._freeze_buf)
            if buf_len == 0:
                return frames
            idx = (np.arange(n) + self._freeze_pos) % buf_len
            self._freeze_pos = (self._freeze_pos + n) % buf_len
            return self._freeze_buf[idx]

        wet = self._process_impl(frames)

        dw = self.dry_wet
        result = frames * (1.0 - dw) + wet * dw

        # Capture on first frame after freeze toggled on
        if self.freeze and self._freeze_buf is None:
            self._freeze_buf = result.copy()
            self._freeze_pos = 0
        elif not self.freeze:
            self._freeze_buf = None

        return result

    def _process_impl(self, frames):
        return frames


# ═══════════════════════════════════════════════════════════════════════
# Effect implementations
# ═══════════════════════════════════════════════════════════════════════

# ── 1. Filter (HP/LP/BP/Notch with resonance) ────────────────────────

@_register
class Filter(EffectBase):
    name = "Filter"
    param_defs = [
        {'key': 'cutoff', 'label': 'Cutoff', 'min': 20, 'max': 20000,
         'default': 1000, 'log': True},
        {'key': 'resonance', 'label': 'Reso', 'min': 0.5, 'max': 15.0,
         'default': 0.707, 'log': True},
    ]
    sub_types = ['LPF', 'HPF', 'BPF', 'Notch']

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        self._sos = make_biquad_sos('lpf', 1000, 0.707, sample_rate)
        self._zi = [sosfilt_zi(self._sos) * 0.0 for _ in range(2)]

    def _on_param_change(self, key, value):
        self._recalc()

    def _on_sub_type_change(self):
        self._recalc()

    def _recalc(self):
        ftypes = ['lpf', 'hpf', 'bpf', 'notch']
        ft = ftypes[self._sub_type % len(ftypes)]
        self._sos = make_biquad_sos(
            ft, self._params['cutoff'], self._params['resonance'],
            self.sample_rate)
        # Preserve zi length but zero to avoid transients on big jumps
        self._zi = [sosfilt_zi(self._sos) * 0.0 for _ in range(2)]

    def _process_impl(self, frames):
        out = np.empty_like(frames)
        for ch in range(2):
            out[:, ch], self._zi[ch] = sosfilt(
                self._sos, frames[:, ch], zi=self._zi[ch])
        return out

    def reset(self):
        super().reset()
        self._zi = [sosfilt_zi(self._sos) * 0.0 for _ in range(2)]


# ── 2. Delay (beat-synced with feedback) ─────────────────────────────

@_register
class Delay(EffectBase):
    name = "Delay"
    param_defs = [
        {'key': 'feedback', 'label': 'Feedback', 'min': 0.0, 'max': 0.95,
         'default': 0.4, 'log': False},
        {'key': 'tone', 'label': 'Tone', 'min': 200, 'max': 12000,
         'default': 5000, 'log': True},
    ]
    has_beat_sync = True

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        max_samp = int(MAX_DELAY_SEC * sample_rate)
        self._buf = np.zeros((max_samp, 2), dtype=np.float32)
        self._buf_len = max_samp
        self._write_pos = 0
        # Tone filter (1-pole LPF on feedback path)
        self._tone_coeff = 0.5
        self._tone_state = np.zeros(2, dtype=np.float32)
        self._recalc_tone()

    def _recalc_tone(self):
        freq = self._params.get('tone', 5000)
        freq = max(200, min(freq, self.sample_rate * 0.49))
        x = math.exp(-2.0 * math.pi * freq / self.sample_rate)
        self._tone_coeff = x

    def _on_param_change(self, key, value):
        if key == 'tone':
            self._recalc_tone()

    def _process_impl(self, frames):
        n = len(frames)
        delay_samp = beat_to_samples(self._beat_frac, self._bpm, self.sample_rate)
        delay_samp = min(delay_samp, self._buf_len - 1)
        feedback = self._params['feedback']
        tone_c = self._tone_coeff

        # Read indices
        read_pos = (self._write_pos - delay_samp + np.arange(n)) % self._buf_len
        delayed = self._buf[read_pos]  # (n, 2)

        # Tone filter on feedback (vectorized 1-pole LPF)
        b_filt = np.array([1.0 - tone_c])
        a_filt = np.array([1.0, -tone_c])
        fb = np.empty_like(delayed)
        for ch in range(2):
            fb[:, ch], zf = lfilter(b_filt, a_filt, delayed[:, ch],
                                     zi=np.array([self._tone_state[ch]]))
            self._tone_state[ch] = zf[0]

        # Write: input + filtered feedback (with denormal flush)
        write_data = frames + fb * feedback
        write_data[np.abs(write_data) < 1e-8] = 0.0
        write_idx = (self._write_pos + np.arange(n)) % self._buf_len
        self._buf[write_idx] = write_data

        self._write_pos = (self._write_pos + n) % self._buf_len
        return delayed

    def reset(self):
        super().reset()
        self._buf[:] = 0
        self._write_pos = 0
        self._tone_state[:] = 0


# ── 3. Reverb (4-line FDN with damping) ──────────────────────────────

@_register
class Reverb(EffectBase):
    name = "Reverb"
    param_defs = [
        {'key': 'decay', 'label': 'Decay', 'min': 0.1, 'max': 0.99,
         'default': 0.6, 'log': False},
        {'key': 'damping', 'label': 'Damp', 'min': 0.0, 'max': 0.9,
         'default': 0.4, 'log': False},
    ]
    sub_types = ['Room', 'Hall', 'Plate']

    # Prime delay-line lengths (at 48 kHz) — scaled for other rates
    _BASE_DELAYS = {
        0: [241, 307, 373, 439],     # Room
        1: [617, 743, 911, 1061],    # Hall
        2: [149, 181, 223, 263],     # Plate
    }
    # Hadamard feedback matrix (4x4, energy-preserving)
    _HMTX = 0.5 * np.array([
        [1,  1,  1,  1],
        [1, -1,  1, -1],
        [1,  1, -1, -1],
        [1, -1, -1,  1],
    ], dtype=np.float32)

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        self._init_delay_lines()

    def _init_delay_lines(self):
        base = self._BASE_DELAYS.get(self._sub_type, self._BASE_DELAYS[0])
        scale = self.sample_rate / 48000.0
        self._delays = [max(1, int(d * scale)) for d in base]
        max_d = max(self._delays) + self.buffer_size + 1
        # 4 delay lines, mono (stereo decorrelation on output)
        self._bufs = [np.zeros(max_d, dtype=np.float32) for _ in range(4)]
        self._buf_lens = [max_d] * 4
        self._write_pos = 0
        # Damping state (1-pole LPF per line, float64 for precision)
        self._damp_z = np.zeros(4, dtype=np.float64)

    def _on_sub_type_change(self):
        self._init_delay_lines()

    def _process_impl(self, frames):
        n = len(frames)
        decay = self._params['decay']
        damp = self._params['damping']

        # Mono input
        mono_in = (frames[:, 0] + frames[:, 1]) * 0.5

        # Read taps from 4 delay lines
        taps = np.empty((4, n), dtype=np.float32)
        for i in range(4):
            rd = (self._write_pos - self._delays[i] + np.arange(n)) % self._buf_lens[i]
            taps[i] = self._bufs[i][rd]

        # Feedback matrix
        mixed = self._HMTX @ taps  # (4, n)

        # Damping (1-pole LPF) + decay
        b_d = np.array([1.0 - damp], dtype=np.float64)
        a_d = np.array([1.0, -damp], dtype=np.float64)
        for i in range(4):
            filtered, zf = lfilter(b_d, a_d, mixed[i].astype(np.float64),
                                    zi=np.array([self._damp_z[i]], dtype=np.float64))
            self._damp_z[i] = zf[0]
            mixed[i] = filtered.astype(np.float32) * decay

        # Flush denormals in feedback to prevent HF noise accumulation
        for i in range(4):
            mixed[i][np.abs(mixed[i]) < 1e-8] = 0.0

        # Write: distribute input + feedback
        wr = (self._write_pos + np.arange(n)) % self._buf_lens[0]
        for i in range(4):
            self._bufs[i][wr] = mono_in * 0.25 + mixed[i]

        self._write_pos = (self._write_pos + n) % self._buf_lens[0]

        # Stereo decorrelation from 4 taps
        out_l = (taps[0] + taps[1] - taps[2] - taps[3]) * 0.25
        out_r = (taps[0] - taps[1] + taps[2] - taps[3]) * 0.25
        return np.column_stack([out_l, out_r])

    def reset(self):
        super().reset()
        for b in self._bufs:
            b[:] = 0
        self._write_pos = 0
        self._damp_z[:] = 0


# ── 4. Dub Echo (warm delay with filtered, saturated feedback) ───────

@_register
class DubEcho(EffectBase):
    name = "Dub Echo"
    param_defs = [
        {'key': 'feedback', 'label': 'Feedback', 'min': 0.0, 'max': 0.95,
         'default': 0.55, 'log': False},
        {'key': 'color', 'label': 'Color', 'min': 200, 'max': 8000,
         'default': 2000, 'log': True},
    ]
    has_beat_sync = True

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        max_samp = int(MAX_DELAY_SEC * sample_rate)
        self._buf = np.zeros((max_samp, 2), dtype=np.float32)
        self._buf_len = max_samp
        self._write_pos = 0
        self._lp_z = np.zeros(2, dtype=np.float32)
        self._recalc_color()

    def _recalc_color(self):
        freq = self._params.get('color', 2000)
        freq = max(200, min(freq, self.sample_rate * 0.49))
        self._lp_coeff = math.exp(-2.0 * math.pi * freq / self.sample_rate)

    def _on_param_change(self, key, value):
        if key == 'color':
            self._recalc_color()

    def _process_impl(self, frames):
        n = len(frames)
        delay_samp = beat_to_samples(self._beat_frac, self._bpm, self.sample_rate)
        delay_samp = min(delay_samp, self._buf_len - 1)
        feedback = self._params['feedback']
        lp_c = self._lp_coeff

        read_idx = (self._write_pos - delay_samp + np.arange(n)) % self._buf_len
        delayed = self._buf[read_idx]  # (n, 2)

        # LP filter on feedback
        b_f = np.array([1.0 - lp_c])
        a_f = np.array([1.0, -lp_c])
        fb = np.empty_like(delayed)
        for ch in range(2):
            fb[:, ch], zf = lfilter(b_f, a_f, delayed[:, ch],
                                     zi=np.array([self._lp_z[ch]]))
            self._lp_z[ch] = zf[0]

        # Soft saturation in feedback path (warm dub character)
        fb = np.tanh(fb * 1.2) * feedback

        write_data = frames + fb
        write_data[np.abs(write_data) < 1e-8] = 0.0
        write_idx = (self._write_pos + np.arange(n)) % self._buf_len
        self._buf[write_idx] = write_data

        self._write_pos = (self._write_pos + n) % self._buf_len
        return delayed

    def reset(self):
        super().reset()
        self._buf[:] = 0
        self._write_pos = 0
        self._lp_z[:] = 0


# ── 5. Flanger ───────────────────────────────────────────────────────

@_register
class Flanger(EffectBase):
    name = "Flanger"
    param_defs = [
        {'key': 'rate', 'label': 'Rate', 'min': 0.05, 'max': 5.0,
         'default': 0.5, 'log': True},
        {'key': 'depth', 'label': 'Depth', 'min': 0.0, 'max': 1.0,
         'default': 0.6, 'log': False},
    ]

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        max_delay = int(0.012 * sample_rate)  # 12 ms max
        self._buf = np.zeros((max_delay, 2), dtype=np.float32)
        self._buf_len = max_delay
        self._write_pos = 0
        self._lfo = LFO(rate=0.5, sample_rate=sample_rate)
        self._feedback = 0.6

    def _on_param_change(self, key, value):
        if key == 'rate':
            self._lfo.rate = value

    def _process_impl(self, frames):
        n = len(frames)
        depth = self._params['depth']
        lfo_out = self._lfo.generate(n)  # [-1, 1]

        # Modulated delay in samples (0.5ms to 10ms)
        base = 0.0005 * self.sample_rate
        sweep = 0.0095 * self.sample_rate * depth
        delays = base + (lfo_out * 0.5 + 0.5) * sweep  # always positive

        # Write to delay buffer
        wr = (self._write_pos + np.arange(n)) % self._buf_len

        # Read with fractional interpolation
        read_f = (self._write_pos + np.arange(n, dtype=np.float64) - delays) % self._buf_len
        idx0 = read_f.astype(np.int32) % self._buf_len
        idx1 = (idx0 + 1) % self._buf_len
        frac = (read_f - np.floor(read_f)).astype(np.float32)

        out = np.empty_like(frames)
        for ch in range(2):
            delayed = self._buf[idx0, ch] * (1.0 - frac) + self._buf[idx1, ch] * frac
            out[:, ch] = delayed
            fb_write = frames[:, ch] + delayed * self._feedback
            fb_write[np.abs(fb_write) < 1e-8] = 0.0
            self._buf[wr, ch] = fb_write

        self._write_pos = (self._write_pos + n) % self._buf_len
        return out

    def reset(self):
        super().reset()
        self._buf[:] = 0
        self._write_pos = 0
        self._lfo.reset()


# ── 6. Phaser ─────────────────────────────────────────────────────────

@_register
class Phaser(EffectBase):
    name = "Phaser"
    param_defs = [
        {'key': 'rate', 'label': 'Rate', 'min': 0.05, 'max': 5.0,
         'default': 0.4, 'log': True},
        {'key': 'depth', 'label': 'Depth', 'min': 0.0, 'max': 1.0,
         'default': 0.7, 'log': False},
    ]
    _NUM_STAGES = 6  # 6 all-pass stages

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        self._lfo = LFO(rate=0.4, sample_rate=sample_rate)
        # sosfilt state: (n_stages, 2) per channel — 2 delay elements per SOS section
        self._ap_zi = [np.zeros((self._NUM_STAGES, 2), dtype=np.float64)
                       for _ in range(2)]
        self._feedback = 0.5

    def _on_param_change(self, key, value):
        if key == 'rate':
            self._lfo.rate = value

    def _process_impl(self, frames):
        n = len(frames)
        depth = self._params['depth']
        lfo = self._lfo.generate(n)  # [-1, 1]

        # Sweep all-pass coefficient between 0.1 and 0.9
        coeff_arr = 0.1 + (lfo * 0.5 + 0.5) * 0.8 * depth  # (n,)

        # Process in chunks — within each chunk, the LFO coefficient
        # is held constant so we can use C-optimized sosfilt.
        chunk_size = 64
        out = np.empty_like(frames)
        for ch in range(2):
            x = frames[:, ch].copy()
            for ci in range(0, n, chunk_size):
                ce = min(ci + chunk_size, n)
                c = float(coeff_arr[ci + (ce - ci) // 2])
                # Build SOS for all all-pass stages at this coefficient
                # All-pass: H(z) = (c + z^-1) / (1 + c*z^-1)
                sos = np.tile(
                    np.array([[c, 1.0, 0.0, 1.0, c, 0.0]]),
                    (self._NUM_STAGES, 1))
                chunk, zf = sosfilt(sos, x[ci:ce].astype(np.float64),
                                    zi=self._ap_zi[ch].copy())
                x[ci:ce] = chunk.astype(np.float32)
                self._ap_zi[ch] = zf
            out[:, ch] = x

        return out

    def reset(self):
        super().reset()
        for zi in self._ap_zi:
            zi[:] = 0
        self._lfo.reset()


# ── 7. Chorus ─────────────────────────────────────────────────────────

@_register
class Chorus(EffectBase):
    name = "Chorus"
    param_defs = [
        {'key': 'rate', 'label': 'Rate', 'min': 0.1, 'max': 3.0,
         'default': 0.8, 'log': True},
        {'key': 'depth', 'label': 'Depth', 'min': 0.0, 'max': 1.0,
         'default': 0.5, 'log': False},
    ]

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        max_delay = int(0.030 * sample_rate)  # 30 ms
        self._buf = np.zeros((max_delay, 2), dtype=np.float32)
        self._buf_len = max_delay
        self._write_pos = 0
        # 3 voices with slightly different rates for richness
        self._lfos = [
            LFO(rate=0.8, sample_rate=sample_rate),
            LFO(rate=0.93, sample_rate=sample_rate),
            LFO(rate=1.07, sample_rate=sample_rate),
        ]
        # Offset LFO phases for spread
        self._lfos[1].phase = sample_rate / 3
        self._lfos[2].phase = 2 * sample_rate / 3

    def _on_param_change(self, key, value):
        if key == 'rate':
            self._lfos[0].rate = value
            self._lfos[1].rate = value * 1.16
            self._lfos[2].rate = value * 1.34

    def _process_impl(self, frames):
        n = len(frames)
        depth = self._params['depth']

        # Write to buffer
        wr = (self._write_pos + np.arange(n)) % self._buf_len
        for ch in range(2):
            self._buf[wr, ch] = frames[:, ch]

        # Sum 3 modulated delay voices
        out = np.zeros_like(frames)
        base_delay = 0.007 * self.sample_rate  # 7ms center
        for lfo in self._lfos:
            mod = lfo.generate(n)
            delays = base_delay + mod * depth * 0.010 * self.sample_rate
            read_f = (self._write_pos + np.arange(n, dtype=np.float64) - delays) % self._buf_len
            idx0 = read_f.astype(np.int32) % self._buf_len
            idx1 = (idx0 + 1) % self._buf_len
            frac = (read_f - np.floor(read_f)).astype(np.float32)
            for ch in range(2):
                out[:, ch] += (self._buf[idx0, ch] * (1.0 - frac)
                               + self._buf[idx1, ch] * frac)

        out /= 3.0  # normalize voices
        self._write_pos = (self._write_pos + n) % self._buf_len
        return out

    def reset(self):
        super().reset()
        self._buf[:] = 0
        self._write_pos = 0
        for lfo in self._lfos:
            lfo.reset()


# ── 8. Gater (beat-synced amplitude gate) ─────────────────────────────

@_register
class Gater(EffectBase):
    name = "Gater"
    param_defs = [
        {'key': 'shape', 'label': 'Shape', 'min': 0.0, 'max': 1.0,
         'default': 0.5, 'log': False},
        {'key': 'swing', 'label': 'Swing', 'min': 0.0, 'max': 0.9,
         'default': 0.0, 'log': False},
    ]
    has_beat_sync = True

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        self._phase = 0.0

    def _process_impl(self, frames):
        n = len(frames)
        period = beat_to_samples(self._beat_frac, self._bpm, self.sample_rate)
        shape = self._params['shape']  # 0 = hard square, 1 = smooth sine
        swing = self._params['swing']  # 0 = straight, 0.9 = heavy shuffle

        # Gate envelope — work in double-period units for swing
        t = (self._phase + np.arange(n, dtype=np.float64)) / period
        self._phase = (self._phase + n) % (period * 2)
        pair_frac = t % 2.0  # position within a pair of beats [0, 2)

        # Swing: first beat spans [0, 1+swing), second spans [1+swing, 2)
        # Both are mapped back to [0, 1) for the gate envelope
        boundary = 1.0 + swing
        phase_frac = np.where(
            pair_frac < boundary,
            pair_frac / boundary,
            (pair_frac - boundary) / (2.0 - boundary),
        ).astype(np.float32)

        if shape < 0.01:
            gate = (phase_frac < 0.5).astype(np.float32)
        else:
            # Raised-cosine transitions (fully vectorized)
            rise = max(0.01, 0.25 * shape)
            gate = np.zeros(n, dtype=np.float32)
            # Ramp up: 0 → rise
            m = phase_frac < rise
            gate[m] = 0.5 * (1.0 - np.cos(np.pi * phase_frac[m] / rise))
            # Fully on: rise → 0.5
            gate[(phase_frac >= rise) & (phase_frac < 0.5)] = 1.0
            # Ramp down: 0.5 → 0.5+rise
            m = (phase_frac >= 0.5) & (phase_frac < 0.5 + rise)
            gate[m] = 0.5 * (1.0 + np.cos(np.pi * (phase_frac[m] - 0.5) / rise))
            # Off: 0.5+rise → 1.0 (already zero)

        gate_2d = np.column_stack([gate, gate])
        return frames * gate_2d

    def reset(self):
        super().reset()
        self._phase = 0.0


# ── 9. Beat Roll (buffer repeat at beat fraction) ────────────────────

@_register
class BeatRoll(EffectBase):
    name = "Beat Roll"
    param_defs = [
        {'key': 'decay', 'label': 'Decay', 'min': 0.0, 'max': 1.0,
         'default': 0.0, 'log': False},
        {'key': 'pitch', 'label': 'Pitch', 'min': 0.5, 'max': 2.0,
         'default': 1.0, 'log': False},
    ]
    has_beat_sync = True

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        max_samp = int(2.0 * sample_rate)  # 2 sec max
        self._capture = np.zeros((max_samp, 2), dtype=np.float32)
        self._cap_len = 0
        self._cap_write = 0
        self._play_pos = 0.0
        self._rolling = False
        self._roll_rep = 0

    def _process_impl(self, frames):
        n = len(frames)
        roll_len = beat_to_samples(self._beat_frac, self._bpm, self.sample_rate)
        roll_len = min(roll_len, len(self._capture))

        if not self._rolling:
            # Continuous capture
            if roll_len <= len(self._capture):
                cap_end = min(self._cap_write + n, roll_len)
                take = cap_end - self._cap_write
                if take > 0:
                    self._capture[self._cap_write:cap_end] = frames[:take]
                self._cap_write = cap_end % roll_len
                self._cap_len = max(self._cap_len, cap_end)
            return frames

        # Rolling: repeat captured buffer
        if self._cap_len == 0:
            return frames
        seg_len = min(self._cap_len, roll_len)
        pitch = self._params.get('pitch', 1.0)
        decay = self._params.get('decay', 0.0)

        idx_f = (self._play_pos + np.arange(n, dtype=np.float64) * pitch) % seg_len
        idx0 = idx_f.astype(np.int32) % seg_len
        idx1 = (idx0 + 1) % seg_len
        frac = (idx_f - np.floor(idx_f)).astype(np.float32)

        out = (self._capture[idx0] * (1.0 - frac[:, None])
               + self._capture[idx1] * frac[:, None])

        # Apply decay across repeats
        if decay > 0.01:
            rep_phase = idx_f / seg_len
            reps = (self._play_pos / seg_len)
            atten = (1.0 - decay) ** reps
            out *= atten

        self._play_pos = (self._play_pos + n * pitch) % (seg_len * 256)
        return out

    def set_rolling(self, active):
        if active and not self._rolling:
            self._play_pos = 0.0
            self._roll_rep = 0
        self._rolling = active

    def reset(self):
        super().reset()
        self._capture[:] = 0
        self._cap_len = 0
        self._cap_write = 0
        self._play_pos = 0.0
        self._rolling = False


# ── 10. Bit Crush ────────────────────────────────────────────────────

@_register
class BitCrush(EffectBase):
    name = "Bit Crush"
    param_defs = [
        {'key': 'bits', 'label': 'Bits', 'min': 1, 'max': 16,
         'default': 8, 'log': False},
        {'key': 'downsample', 'label': 'Dnsamp', 'min': 1, 'max': 64,
         'default': 1, 'log': False},
    ]

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        self._hold = np.zeros(2, dtype=np.float32)

    def _process_impl(self, frames):
        n = len(frames)
        bits = max(1, int(self._params['bits']))
        ds = max(1, int(self._params['downsample']))

        # Bit reduction
        levels = 2 ** bits
        out = np.round(frames * levels) / levels

        # Sample-rate reduction (sample-and-hold)
        if ds > 1:
            for i in range(n):
                if i % ds == 0:
                    self._hold[:] = out[i]
                else:
                    out[i] = self._hold

        return out

    def reset(self):
        super().reset()
        self._hold[:] = 0


# ── 11. Ring Mod ─────────────────────────────────────────────────────

@_register
class RingMod(EffectBase):
    name = "Ring Mod"
    param_defs = [
        {'key': 'freq', 'label': 'Freq', 'min': 20, 'max': 5000,
         'default': 200, 'log': True},
        {'key': 'shape', 'label': 'Shape', 'min': 0.0, 'max': 1.0,
         'default': 0.0, 'log': False},
    ]

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        self._osc_phase = 0.0

    def _process_impl(self, frames):
        n = len(frames)
        freq = self._params['freq']
        shape = self._params['shape']  # 0 = sine, 1 = square

        t = (self._osc_phase + np.arange(n, dtype=np.float64)) / self.sample_rate
        self._osc_phase = (self._osc_phase + n) % self.sample_rate

        sine = np.sin(2.0 * np.pi * freq * t).astype(np.float32)
        if shape > 0.01:
            square = np.where(sine >= 0, 1.0, -1.0).astype(np.float32)
            carrier = sine * (1.0 - shape) + square * shape
        else:
            carrier = sine

        return frames * carrier[:, None]

    def reset(self):
        super().reset()
        self._osc_phase = 0.0


# ── 12. Chopped & Screwed (DJ Screw, Houston TX 1990s) ───────────────

@_register
class ChoppedAndScrewed(EffectBase):
    """Pitch-shift down (screw) + beat-synced stutter (chop).

    Inspired by DJ Screw's technique of slowing records to create a
    syrupy, hypnotic sound, with crossfader chops for rhythmic stutter.
    Sub-types:
      Classic  — standard screw with periodic chop
      Stutter  — aggressive, frequent chopping
      Backspin — reversed audio at chop points
    """
    name = "Chopped & Screwed"
    param_defs = [
        {'key': 'screw', 'label': 'Screw', 'min': 0.5, 'max': 1.0,
         'default': 0.85, 'log': False},
        {'key': 'chop', 'label': 'Chop', 'min': 0.0, 'max': 1.0,
         'default': 0.0, 'log': False},
    ]
    sub_types = ['Classic', 'Stutter', 'Backspin']
    has_beat_sync = True

    # Grain size for pitch shifter
    _GRAIN = 2048

    def __init__(self, sample_rate=48000, buffer_size=1024):
        super().__init__(sample_rate, buffer_size)
        grain = self._GRAIN
        buf_len = grain * 4

        # ── Pitch shifter state (per channel) ──
        self._pbuf = np.zeros((buf_len, 2), dtype=np.float32)
        self._pbuf_len = buf_len
        self._pwrite = 0
        # Two grains per channel, offset by half grain
        self._read_a = np.array([0.0, 0.0], dtype=np.float64)
        self._read_b = np.array([float(grain // 2)] * 2, dtype=np.float64)
        self._phase_a = np.array([0, 0], dtype=np.int64)
        self._phase_b = np.array([grain // 2] * 2, dtype=np.int64)

        # ── Chop state ──
        max_chop = int(2.0 * sample_rate)
        self._chop_buf = np.zeros((max_chop, 2), dtype=np.float32)
        self._chop_cap_pos = 0     # circular write into capture buffer
        self._chop_len = 0         # current chop segment length (samples)
        self._chop_active = False
        self._chop_play_pos = 0
        self._chop_remaining = 0
        self._chop_counter = 0     # samples until next trigger decision
        self._chop_trigger_idx = 0

        # ── Wobble LFO (tape flutter) ──
        self._wobble_lfo = LFO(rate=2.5, sample_rate=sample_rate)

    def _process_impl(self, frames):
        n = len(frames)
        screw = self._params['screw']  # 0.5–1.0
        chop_depth = self._params['chop']  # 0.0–1.0

        # ── Step 1: Pitch shift (screw) ──
        screwed = self._pitch_shift(frames, screw, n)

        # ── Step 2: Chop / stutter ──
        if chop_depth > 0.01:
            screwed = self._chop_process(screwed, chop_depth, n)

        return screwed

    def _pitch_shift(self, frames, ratio, n):
        """Granular dual-grain pitch shifter (vectorized)."""
        grain = self._GRAIN
        buf_len = self._pbuf_len

        # Subtle wow/flutter when screwed
        wobble = 0.0
        if ratio < 0.99:
            wobble_lfo = self._wobble_lfo.generate(n)
            wobble = wobble_lfo * 0.003 * (1.0 - ratio)

        actual_ratio = ratio + wobble if isinstance(wobble, np.ndarray) else ratio

        # Write input to circular pitch buffer
        wr = (self._pwrite + np.arange(n)) % buf_len
        self._pbuf[wr] = frames

        out = np.zeros_like(frames)
        hann_denom = float(grain)

        for ch in range(2):
            # Grain A
            offsets_a = np.arange(n, dtype=np.float64)
            if isinstance(actual_ratio, np.ndarray):
                read_a = (self._read_a[ch] + np.cumsum(actual_ratio)) % buf_len
            else:
                read_a = (self._read_a[ch] + offsets_a * actual_ratio) % buf_len
            phase_a = (self._phase_a[ch] + np.arange(n)) % grain
            win_a = (0.5 * (1.0 - np.cos(2.0 * np.pi * phase_a / hann_denom))).astype(np.float32)

            idx_a = read_a.astype(np.int64) % buf_len
            samp_a = self._pbuf[idx_a, ch]

            # Grain B
            if isinstance(actual_ratio, np.ndarray):
                read_b = (self._read_b[ch] + np.cumsum(actual_ratio)) % buf_len
            else:
                read_b = (self._read_b[ch] + offsets_a * actual_ratio) % buf_len
            phase_b = (self._phase_b[ch] + np.arange(n)) % grain
            win_b = (0.5 * (1.0 - np.cos(2.0 * np.pi * phase_b / hann_denom))).astype(np.float32)

            idx_b = read_b.astype(np.int64) % buf_len
            samp_b = self._pbuf[idx_b, ch]

            out[:, ch] = samp_a * win_a + samp_b * win_b

            # Update state
            if isinstance(actual_ratio, np.ndarray):
                adv = np.sum(actual_ratio)
            else:
                adv = n * actual_ratio
            new_read_a = (self._read_a[ch] + adv) % buf_len
            new_read_b = (self._read_b[ch] + adv) % buf_len
            new_phase_a = (self._phase_a[ch] + n) % grain
            new_phase_b = (self._phase_b[ch] + n) % grain

            # Grain reset: re-sync read pointer to near write head
            reset_pos = float((self._pwrite + n) % buf_len) - grain
            if reset_pos < 0:
                reset_pos += buf_len
            if new_phase_a < self._phase_a[ch]:  # wrapped
                new_read_a = reset_pos
            if new_phase_b < self._phase_b[ch]:  # wrapped
                new_read_b = reset_pos

            self._read_a[ch] = new_read_a
            self._read_b[ch] = new_read_b
            self._phase_a[ch] = new_phase_a
            self._phase_b[ch] = new_phase_b

        self._pwrite = (self._pwrite + n) % buf_len
        return out

    def _chop_process(self, frames, depth, n):
        """Beat-synced stutter/chop on already-screwed audio."""
        chop_len = beat_to_samples(self._beat_frac, self._bpm, self.sample_rate)
        chop_len = min(chop_len, len(self._chop_buf))
        if chop_len < 2:
            return frames

        # Determine chop behavior from sub-type
        # Classic: chop every 4 triggers, Stutter: every 1-2, Backspin: every 4 reversed
        if self._sub_type == 1:    # Stutter
            trigger_period = max(1, int(2 * (1.0 - depth) + 1))
            num_repeats = 4
        elif self._sub_type == 2:  # Backspin
            trigger_period = max(1, int(4 * (1.0 - depth) + 1))
            num_repeats = 2
        else:                      # Classic
            trigger_period = max(1, int(4 * (1.0 - depth) + 1))
            num_repeats = 3

        # Continuous capture
        cap_end = self._chop_cap_pos + n
        if cap_end <= chop_len:
            self._chop_buf[self._chop_cap_pos:cap_end] = frames
        else:
            first = chop_len - self._chop_cap_pos
            if first > 0:
                self._chop_buf[self._chop_cap_pos:chop_len] = frames[:first]
            remainder = n - first
            if remainder > 0:
                wrap = min(remainder, chop_len)
                self._chop_buf[:wrap] = frames[first:first + wrap]
        self._chop_cap_pos = cap_end % chop_len

        # Trigger logic
        self._chop_counter -= n
        if self._chop_counter <= 0:
            chop_interval = chop_len  # trigger decision every chop_len samples
            self._chop_counter = chop_interval
            self._chop_trigger_idx += 1
            if self._chop_trigger_idx % trigger_period == 0:
                self._chop_active = True
                self._chop_play_pos = 0
                self._chop_remaining = chop_len * num_repeats
                self._chop_len = chop_len

        if not self._chop_active:
            return frames

        # Playback
        out = frames.copy()
        play_n = min(n, self._chop_remaining)
        seg = self._chop_len
        indices = (np.arange(play_n) + self._chop_play_pos) % seg

        if self._sub_type == 2:  # Backspin: reverse
            indices = seg - 1 - indices

        chop_audio = self._chop_buf[indices]

        # Crossfade at boundaries (first/last 64 samples)
        xfade = min(64, play_n // 4)
        if xfade > 1 and self._chop_play_pos == 0:
            ramp = np.linspace(0, 1, xfade, dtype=np.float32)[:, None]
            chop_audio[:xfade] = (out[:xfade] * (1 - ramp)
                                  + chop_audio[:xfade] * ramp)

        out[:play_n] = chop_audio
        self._chop_play_pos += play_n
        self._chop_remaining -= play_n
        if self._chop_remaining <= 0:
            self._chop_active = False

        return out

    def reset(self):
        super().reset()
        self._pbuf[:] = 0
        self._pwrite = 0
        self._read_a[:] = 0
        self._read_b[:] = [self._GRAIN // 2] * 2
        self._phase_a[:] = 0
        self._phase_b[:] = [self._GRAIN // 2] * 2
        self._chop_buf[:] = 0
        self._chop_active = False
        self._chop_counter = 0
        self._chop_trigger_idx = 0
        self._wobble_lfo.reset()


# ═══════════════════════════════════════════════════════════════════════
# Effects Engine — manages per-track and per-output effect slots
# ═══════════════════════════════════════════════════════════════════════

class EffectsEngine:
    """Central manager for all effect instances across outputs.

    Thread-safety: parameter writes (from main/UI thread) are atomic
    Python attribute assignments protected by the GIL.  The audio
    callback only reads parameters — same pattern as track_mappings.
    """
    def __init__(self, sample_rate=48000, buffer_size=1024,
                 num_outputs=3, num_tracks=4):
        self.enabled = False
        self.bpm = DEFAULT_BPM
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.num_outputs = num_outputs
        self.num_tracks = num_tracks

        # Per-output, per-track effect instances (or None)
        # _track_fx[out_idx][track_idx] = EffectBase | None
        self._track_fx = [[None] * num_tracks for _ in range(num_outputs)]
        # Per-output bus effect
        self._bus_fx = [None] * num_outputs

        # Smoothed per-output process() duration in ms (written from the
        # audio callback thread, read by the UI — GIL-atomic floats)
        self._load_ms = [0.0] * num_outputs

    # ── Configuration (main thread) ──

    def set_track_effect(self, out_idx, track_idx, effect_cls):
        """Set the effect for a track slot.  Pass None to clear."""
        if effect_cls is None:
            self._track_fx[out_idx][track_idx] = None
            return None
        fx = effect_cls(sample_rate=self.sample_rate,
                        buffer_size=self.buffer_size)
        fx.set_bpm(self.bpm)
        self._track_fx[out_idx][track_idx] = fx
        return fx

    def get_track_effect(self, out_idx, track_idx):
        if out_idx < len(self._track_fx) and track_idx < len(self._track_fx[out_idx]):
            return self._track_fx[out_idx][track_idx]
        return None

    def set_bus_effect(self, out_idx, effect_cls):
        if effect_cls is None:
            self._bus_fx[out_idx] = None
            return None
        fx = effect_cls(sample_rate=self.sample_rate,
                        buffer_size=self.buffer_size)
        fx.set_bpm(self.bpm)
        self._bus_fx[out_idx] = fx
        return fx

    def get_bus_effect(self, out_idx):
        if out_idx < len(self._bus_fx):
            return self._bus_fx[out_idx]
        return None

    def set_bpm(self, bpm):
        self.bpm = max(1.0, bpm)
        for out_slots in self._track_fx:
            for fx in out_slots:
                if fx:
                    fx.set_bpm(self.bpm)
        for fx in self._bus_fx:
            if fx:
                fx.set_bpm(self.bpm)

    def set_sample_rate(self, sr):
        self.sample_rate = sr
        # Existing effect instances keep their rate — they'll be
        # recreated when routing restarts.

    def reset_all(self):
        """Clear DSP state on all effects (called on routing start)."""
        for out_slots in self._track_fx:
            for fx in out_slots:
                if fx:
                    fx.reset()
        for fx in self._bus_fx:
            if fx:
                fx.reset()

    # ── Serialization (presets / session restore) ──

    @staticmethod
    def _fx_state(fx):
        """Serializable dict for one effect instance (None passes through)."""
        if fx is None:
            return None
        return {
            'type': fx.name,
            'dry_wet': fx.dry_wet,
            'sub_type': fx._sub_type,
            'beat_frac': fx._beat_frac,
            'params': {pd['key']: fx.get_param(pd['key'])
                       for pd in fx.param_defs},
        }

    def _fx_from_state(self, state):
        """Recreate an effect instance from _fx_state output (or None)."""
        if not isinstance(state, dict):
            return None
        cls = next((c for c in EFFECT_CLASSES if c.name == state.get('type')), None)
        if cls is None:
            return None
        fx = cls(sample_rate=self.sample_rate, buffer_size=self.buffer_size)
        fx.set_bpm(self.bpm)
        try:
            fx.set_param('dry_wet', float(state.get('dry_wet', 0.5)))
            sub = int(state.get('sub_type', 0))
            if fx.sub_types and 0 <= sub < len(fx.sub_types):
                fx.set_sub_type(sub)
            fx.set_beat_fraction(float(state.get('beat_frac', 0.25)))
            valid_keys = {pd['key'] for pd in fx.param_defs}
            for key, val in (state.get('params') or {}).items():
                if key in valid_keys:
                    fx.set_param(key, float(val))
        except (TypeError, ValueError):
            pass
        return fx

    def get_state(self):
        """Snapshot of the whole rack for presets: bpm, enabled, all slots."""
        return {
            'enabled': self.enabled,
            'bpm': self.bpm,
            'track_fx': [[self._fx_state(fx) for fx in out_slots]
                         for out_slots in self._track_fx],
            'bus_fx': [self._fx_state(fx) for fx in self._bus_fx],
        }

    def set_state(self, state):
        """Restore a get_state() snapshot. Unknown effect types are skipped."""
        if not isinstance(state, dict):
            return
        try:
            self.bpm = max(1.0, float(state.get('bpm') or self.bpm))
        except (TypeError, ValueError):
            pass
        self.enabled = bool(state.get('enabled', False))
        track_fx = state.get('track_fx') or []
        for oi in range(self.num_outputs):
            slots = track_fx[oi] if oi < len(track_fx) else []
            for ti in range(self.num_tracks):
                st = slots[ti] if ti < len(slots) else None
                self._track_fx[oi][ti] = self._fx_from_state(st)
        bus_fx = state.get('bus_fx') or []
        for oi in range(self.num_outputs):
            self._bus_fx[oi] = self._fx_from_state(
                bus_fx[oi] if oi < len(bus_fx) else None)

    def has_any_effect(self):
        """True if any track or bus slot holds an effect."""
        return (any(fx for slots in self._track_fx for fx in slots)
                or any(self._bus_fx))

    # ── Processing (audio callback thread) ──

    def process(self, out_idx, frames, num_channels):
        """Apply track + bus effects to output_frames.

        Called from the output callback between mapping and soft-clip.
        """
        if not self.enabled:
            if out_idx < len(self._load_ms):
                self._load_ms[out_idx] = 0.0
            return frames
        _t0 = _time.perf_counter()

        # Per-track effects (one effect per stereo pair)
        if out_idx < len(self._track_fx):
            for t in range(min(self.num_tracks, num_channels // 2)):
                fx = self._track_fx[out_idx][t]
                if fx and fx.enabled:
                    l, r = t * 2, t * 2 + 1
                    if r < num_channels:
                        pair = frames[:, l:r + 1].copy()
                        try:
                            pair = fx.process_stereo(pair)
                        except Exception:
                            pass
                        frames[:, l:r + 1] = pair

        # Bus effect — send/return style, processed ONCE per block.
        # Sum all tracks to stereo, process through the bus effect,
        # then add the wet signal back to all tracks.  This avoids
        # calling the effect N times per block which would corrupt
        # stateful effects (delay/reverb write positions advance N×).
        if out_idx < len(self._bus_fx):
            bus = self._bus_fx[out_idx]
            if bus and bus.enabled:
                n_pairs = max(1, num_channels // 2)
                n_frames = frames.shape[0]
                # Sum all tracks to a stereo bus send
                bus_send = np.zeros((n_frames, 2), dtype=np.float32)
                for t in range(n_pairs):
                    l = t * 2
                    r = min(l + 1, num_channels - 1)
                    bus_send[:, 0] += frames[:, l]
                    bus_send[:, 1] += frames[:, r]
                if n_pairs > 1:
                    bus_send *= (1.0 / n_pairs)
                # Process once — _process_impl returns wet-only
                try:
                    wet = bus._process_impl(bus_send)
                except Exception:
                    wet = np.zeros_like(bus_send)
                # Handle freeze
                if bus.freeze:
                    if bus._freeze_buf is not None:
                        buf_len = len(bus._freeze_buf)
                        if buf_len > 0:
                            idx = (np.arange(n_frames) + bus._freeze_pos) % buf_len
                            bus._freeze_pos = (bus._freeze_pos + n_frames) % buf_len
                            wet = bus._freeze_buf[idx]
                    elif wet is not None:
                        bus._freeze_buf = wet.copy()
                        bus._freeze_pos = 0
                elif bus._freeze_buf is not None:
                    bus._freeze_buf = None
                # Add wet signal to all tracks (send/return blend)
                dw = bus.dry_wet
                for t in range(n_pairs):
                    l = t * 2
                    r = min(l + 1, num_channels - 1)
                    frames[:, l] += wet[:, 0] * dw
                    frames[:, r] += wet[:, 1] * dw

        # Noise gate — flush values below -160dBFS to prevent
        # float32 quantization noise from accumulating in
        # feedback paths and producing high-frequency artifacts
        frames[np.abs(frames) < 1e-8] = 0.0

        # Smoothed DSP load (EMA over ~10 callbacks)
        if out_idx < len(self._load_ms):
            dt_ms = (_time.perf_counter() - _t0) * 1000.0
            self._load_ms[out_idx] = self._load_ms[out_idx] * 0.9 + dt_ms * 0.1

        return frames

    def get_load_pct(self, out_idx):
        """Effects DSP load as a percentage of the callback time budget."""
        if not (0 <= out_idx < len(self._load_ms)) or self.sample_rate <= 0:
            return 0.0
        budget_ms = self.buffer_size / float(self.sample_rate) * 1000.0
        if budget_ms <= 0:
            return 0.0
        return min(999.0, self._load_ms[out_idx] / budget_ms * 100.0)


# ═══════════════════════════════════════════════════════════════════════
# UI Widgets
# ═══════════════════════════════════════════════════════════════════════

if QtWidgets is not None:

    def _knob_to_value(dial_val, pdef):
        """Map QDial integer (0-1000) to parameter value."""
        t = dial_val / 1000.0
        if pdef.get('log') and pdef['min'] > 0:
            log_min = math.log(pdef['min'])
            log_max = math.log(pdef['max'])
            return math.exp(log_min + t * (log_max - log_min))
        return pdef['min'] + t * (pdef['max'] - pdef['min'])

    def _value_to_knob(value, pdef):
        """Map parameter value to QDial integer (0-1000)."""
        if pdef.get('log') and pdef['min'] > 0:
            log_min = math.log(pdef['min'])
            log_max = math.log(pdef['max'])
            if log_max == log_min:
                return 500
            t = (math.log(max(pdef['min'], value)) - log_min) / (log_max - log_min)
        else:
            rng = pdef['max'] - pdef['min']
            t = (value - pdef['min']) / rng if rng else 0.5
        return int(max(0, min(1000, t * 1000)))

    def _fmt_value(value, pdef):
        """Format a parameter value for display."""
        if pdef.get('log') and pdef['max'] >= 1000:
            if value >= 1000:
                return f"{value/1000:.1f}k"
            return f"{int(value)}"
        if pdef['max'] <= 1.0:
            return f"{int(value * 100)}%"
        if pdef['max'] <= 2.0:
            return f"{value:.2f}"
        return f"{value:.1f}" if value < 100 else f"{int(value)}"


    class EffectSlotWidget(QtWidgets.QWidget):
        """UI for one effect slot: type selector + parameter knobs."""
        effect_changed = Signal()

        def __init__(self, engine, out_idx, track_idx, is_bus=False,
                     parent=None):
            super().__init__(parent)
            self.engine = engine
            self.out_idx = out_idx
            self.track_idx = track_idx
            self.is_bus = is_bus
            self._fx = None
            self._updating = False

            # Card look (styled via QSS #fxSlot rule)
            self.setObjectName("fxSlot")
            self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(6, 6, 6, 6)
            layout.setSpacing(3)

            # Header label
            if is_bus:
                header = QtWidgets.QLabel("Bus")
            else:
                header = QtWidgets.QLabel(f"Track {track_idx + 1}")
            header.setAlignment(QtCore.Qt.AlignCenter)
            header.setStyleSheet("font-weight: bold; font-size: 11px;")
            layout.addWidget(header)
            self._header = header

            # Effect type selector
            self.type_combo = QtWidgets.QComboBox()
            self.type_combo.addItem("Off")
            for cls in EFFECT_CLASSES:
                self.type_combo.addItem(cls.name)
            self.type_combo.currentIndexChanged.connect(self._on_type_changed)
            self.type_combo.setMaximumHeight(28)
            layout.addWidget(self.type_combo)

            # Sub-type selector (hidden when not applicable)
            self.sub_combo = QtWidgets.QComboBox()
            self.sub_combo.currentIndexChanged.connect(self._on_sub_changed)
            self.sub_combo.setMaximumHeight(24)
            self.sub_combo.hide()
            layout.addWidget(self.sub_combo)

            # Dry/Wet knob
            dw_row = QtWidgets.QVBoxLayout()
            dw_row.setSpacing(1)
            self.dw_dial = MiniKnob()
            self.dw_dial.setRange(0, 1000)
            self.dw_dial.setValue(500)
            if hasattr(self.dw_dial, 'set_default_value'):
                self.dw_dial.set_default_value(500)
            self.dw_dial.setFixedSize(48, 48)
            self.dw_dial.valueChanged.connect(self._on_dw_changed)
            dw_lbl = QtWidgets.QLabel("D/W")
            dw_lbl.setAlignment(QtCore.Qt.AlignCenter)
            dw_lbl.setStyleSheet("font-size: 10px;")
            self.dw_val_lbl = QtWidgets.QLabel("50%")
            self.dw_val_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self.dw_val_lbl.setStyleSheet("font-size: 10px;")
            dw_row.addWidget(self.dw_dial, 0, QtCore.Qt.AlignCenter)
            dw_row.addWidget(dw_lbl)
            dw_row.addWidget(self.dw_val_lbl)
            layout.addLayout(dw_row)

            # Parameter knobs (2)
            self._param_dials = []
            self._param_labels = []
            self._param_val_labels = []
            for i in range(2):
                row = QtWidgets.QVBoxLayout()
                row.setSpacing(1)
                dial = MiniKnob()
                dial.setRange(0, 1000)
                dial.setValue(500)
                dial.setFixedSize(48, 48)
                dial.valueChanged.connect(lambda v, idx=i: self._on_param_changed(idx, v))
                plbl = QtWidgets.QLabel("---")
                plbl.setAlignment(QtCore.Qt.AlignCenter)
                plbl.setStyleSheet("font-size: 10px;")
                vlbl = QtWidgets.QLabel("")
                vlbl.setAlignment(QtCore.Qt.AlignCenter)
                vlbl.setStyleSheet("font-size: 10px;")
                row.addWidget(dial, 0, QtCore.Qt.AlignCenter)
                row.addWidget(plbl)
                row.addWidget(vlbl)
                layout.addLayout(row)
                self._param_dials.append(dial)
                self._param_labels.append(plbl)
                self._param_val_labels.append(vlbl)

            # Beat fraction selector
            self.beat_combo = QtWidgets.QComboBox()
            for label, _ in BEAT_FRACTIONS:
                self.beat_combo.addItem(label)
            self.beat_combo.setCurrentIndex(3)  # 1/4 default
            self.beat_combo.currentIndexChanged.connect(self._on_beat_changed)
            self.beat_combo.setMaximumHeight(24)
            self.beat_combo.hide()
            layout.addWidget(self.beat_combo)

            # Freeze button
            self.freeze_btn = QtWidgets.QPushButton("FRZ")
            self.freeze_btn.setCheckable(True)
            self.freeze_btn.setMaximumHeight(26)
            self.freeze_btn.clicked.connect(self._on_freeze)
            layout.addWidget(self.freeze_btn)

            layout.addStretch()
            self.setMinimumWidth(90)
            self.setMaximumWidth(140)

        def _get_effect(self):
            if self.is_bus:
                return self.engine.get_bus_effect(self.out_idx)
            return self.engine.get_track_effect(self.out_idx, self.track_idx)

        def _set_effect(self, cls):
            if self.is_bus:
                return self.engine.set_bus_effect(self.out_idx, cls)
            return self.engine.set_track_effect(
                self.out_idx, self.track_idx, cls)

        def _on_type_changed(self, idx):
            if self._updating:
                return
            if idx == 0:
                self._set_effect(None)
                self._fx = None
                self._update_controls()
                self.effect_changed.emit()
                return
            cls = EFFECT_CLASSES[idx - 1]
            self._fx = self._set_effect(cls)
            self._update_controls()
            self.effect_changed.emit()

        def _update_controls(self):
            """Refresh knob labels, ranges, sub-type list for current effect."""
            self._updating = True
            fx = self._fx

            if fx is None:
                self.sub_combo.hide()
                self.beat_combo.hide()
                for i in range(2):
                    self._param_labels[i].setText("---")
                    self._param_val_labels[i].setText("")
                    self._param_dials[i].setEnabled(False)
                self.dw_dial.setEnabled(False)
                self.freeze_btn.setEnabled(False)
                self.freeze_btn.setChecked(False)
                self.dw_val_lbl.setText("")
                self._updating = False
                return

            self.dw_dial.setEnabled(True)
            self.dw_dial.setValue(int(fx.dry_wet * 1000))
            self.dw_val_lbl.setText(f"{int(fx.dry_wet * 100)}%")
            self.freeze_btn.setEnabled(True)
            self.freeze_btn.setChecked(bool(fx.freeze))

            # Sub-types
            if fx.sub_types:
                self.sub_combo.clear()
                for st in fx.sub_types:
                    self.sub_combo.addItem(st)
                self.sub_combo.setCurrentIndex(fx._sub_type)
                self.sub_combo.show()
            else:
                self.sub_combo.hide()

            # Beat sync — select the closest fraction to the effect's state
            if fx.has_beat_sync:
                closest = min(range(len(BEAT_FRACTIONS)),
                              key=lambda i: abs(BEAT_FRACTIONS[i][1] - fx._beat_frac))
                self.beat_combo.setCurrentIndex(closest)
                self.beat_combo.show()
            else:
                self.beat_combo.hide()

            # Parameter knobs
            for i in range(2):
                if i < len(fx.param_defs):
                    pd = fx.param_defs[i]
                    self._param_labels[i].setText(pd['label'])
                    self._param_dials[i].setEnabled(True)
                    val = fx.get_param(pd['key'])
                    self._param_dials[i].setValue(_value_to_knob(val, pd))
                    if hasattr(self._param_dials[i], 'set_default_value'):
                        self._param_dials[i].set_default_value(
                            _value_to_knob(pd['default'], pd))
                    self._param_val_labels[i].setText(_fmt_value(val, pd))
                else:
                    self._param_labels[i].setText("---")
                    self._param_val_labels[i].setText("")
                    self._param_dials[i].setEnabled(False)

            self._updating = False

        def _on_sub_changed(self, idx):
            if self._updating or not self._fx:
                return
            self._fx.set_sub_type(idx)

        def _on_dw_changed(self, val):
            if self._updating or not self._fx:
                return
            dw = val / 1000.0
            self._fx.set_param('dry_wet', dw)
            self.dw_val_lbl.setText(f"{int(dw * 100)}%")

        def _on_param_changed(self, param_idx, dial_val):
            if self._updating or not self._fx:
                return
            if param_idx >= len(self._fx.param_defs):
                return
            pd = self._fx.param_defs[param_idx]
            value = _knob_to_value(dial_val, pd)
            self._fx.set_param(pd['key'], value)
            self._param_val_labels[param_idx].setText(_fmt_value(value, pd))

        def _on_beat_changed(self, idx):
            if self._updating or not self._fx:
                return
            if 0 <= idx < len(BEAT_FRACTIONS):
                self._fx.set_beat_fraction(BEAT_FRACTIONS[idx][1])

        def _on_freeze(self, checked):
            if self._fx:
                self._fx.freeze = checked

        def update_header_color(self, color_str):
            self._header.setStyleSheet(
                f"font-weight: bold; font-size: 11px; color: {color_str};")
            # Match the knobs to the track color
            for dial in (self.dw_dial, *self._param_dials):
                if hasattr(dial, 'set_accent_color'):
                    dial.set_accent_color(color_str)

        def sync_from_engine(self):
            """Re-read effect state (e.g. after output tab switch)."""
            fx = self._get_effect()
            self._fx = fx
            self._updating = True
            if fx is None:
                self.type_combo.setCurrentIndex(0)
            else:
                for i, cls in enumerate(EFFECT_CLASSES):
                    if isinstance(fx, cls):
                        self.type_combo.setCurrentIndex(i + 1)
                        break
            self._updating = False
            self._update_controls()


    class EffectsRackWidget(QtWidgets.QGroupBox):
        """Full effects rack: BPM control, 4 track slots + 1 bus slot.

        Standalone: collapsible — the slot strip hides behind a disclosure
        header so the rack stays out of the way until effects are in use.
        Embedded (embedded=True, e.g. inside a section tab): always
        expanded, disclosure header hidden — the tab provides the title.
        """

        rack_changed = Signal()  # emitted when any slot's effect changes

        def __init__(self, engine, parent=None, embedded=False):
            # No group-box title — the disclosure button IS the header
            super().__init__("", parent)
            self.engine = engine
            self._embedded = embedded

            root = QtWidgets.QVBoxLayout(self)
            root.setSpacing(4)
            root.setContentsMargins(6, 6, 6, 6)

            # ── Header: disclosure + summary + BPM + Tap + DSP + Bypass ──
            top = QtWidgets.QHBoxLayout()
            top.setSpacing(6)

            self.collapse_btn = QtWidgets.QToolButton()
            self.collapse_btn.setText("Effects")
            self.collapse_btn.setArrowType(QtCore.Qt.RightArrow)
            self.collapse_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            self.collapse_btn.setAutoRaise(True)
            self.collapse_btn.setStyleSheet(
                "QToolButton { font-weight: bold; border: none; padding: 4px; }")
            self.collapse_btn.setCursor(QtCore.Qt.PointingHandCursor)
            self.collapse_btn.setToolTip("Click to show/hide the effects rack")
            self.collapse_btn.clicked.connect(self._on_collapse)
            top.addWidget(self.collapse_btn)

            # Summary shown while collapsed, e.g. "2 active"
            self.summary_lbl = QtWidgets.QLabel("")
            self.summary_lbl.setStyleSheet("font-size: 11px; color: #888888;")
            top.addWidget(self.summary_lbl)

            top.addSpacing(10)

            bpm_lbl = QtWidgets.QLabel("BPM")
            bpm_lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
            top.addWidget(bpm_lbl)

            self.bpm_spin = QtWidgets.QDoubleSpinBox()
            self.bpm_spin.setRange(40.0, 300.0)
            self.bpm_spin.setValue(engine.bpm)
            self.bpm_spin.setSingleStep(0.5)
            self.bpm_spin.setDecimals(1)
            self.bpm_spin.setMaximumWidth(80)
            self.bpm_spin.valueChanged.connect(self._on_bpm)
            top.addWidget(self.bpm_spin)

            self.tap_btn = QtWidgets.QPushButton("TAP")
            self.tap_btn.setMaximumWidth(50)
            self.tap_btn.setMaximumHeight(28)
            self.tap_btn.clicked.connect(self._on_tap)
            top.addWidget(self.tap_btn)
            self._tap_times = []

            top.addStretch()

            # DSP load readout (% of the audio callback time budget)
            self.dsp_lbl = QtWidgets.QLabel("")
            self.dsp_lbl.setToolTip(
                "Effects DSP load — share of the audio buffer time spent "
                "in effects processing. Keep below ~50%.")
            self.dsp_lbl.setStyleSheet("font-size: 10px; color: #888888;")
            top.addWidget(self.dsp_lbl)

            self.bypass_btn = QtWidgets.QPushButton("FX ON")
            self.bypass_btn.setCheckable(True)
            self.bypass_btn.setChecked(engine.enabled)
            self.bypass_btn.setMaximumWidth(70)
            self.bypass_btn.setMaximumHeight(28)
            self.bypass_btn.clicked.connect(self._on_bypass)
            self._style_bypass_btn()
            top.addWidget(self.bypass_btn)

            root.addLayout(top)

            # ── Effect slot strips (collapsible body) ──
            self._body = QtWidgets.QWidget()
            strips = QtWidgets.QHBoxLayout(self._body)
            strips.setSpacing(6)
            strips.setContentsMargins(0, 0, 0, 0)

            self.track_slots = []
            for t in range(engine.num_tracks):
                slot = EffectSlotWidget(engine, 0, t, is_bus=False)
                slot.effect_changed.connect(self._update_summary)
                self.track_slots.append(slot)
                strips.addWidget(slot)

            self.bus_slot = EffectSlotWidget(engine, 0, 0, is_bus=True)
            self.bus_slot.effect_changed.connect(self._update_summary)
            strips.addWidget(self.bus_slot)

            root.addWidget(self._body)

            self._current_out = 0
            self._dsp_timer = QtCore.QTimer(self)
            self._dsp_timer.timeout.connect(self._update_dsp_label)
            self._dsp_timer.start(500)

            if self._embedded:
                # The section tab is the header — no disclosure needed
                self.collapse_btn.hide()
                self.summary_lbl.hide()
                self.set_collapsed(False)
            else:
                # Start collapsed; expands automatically when a preset or
                # the last session restores active effects
                self.set_collapsed(True)

        # ── Collapse ──

        def set_collapsed(self, collapsed):
            if self._embedded:
                collapsed = False
            self._body.setVisible(not collapsed)
            self.collapse_btn.setArrowType(
                QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow)
            self._update_summary()

        def is_collapsed(self):
            return not self._body.isVisible()

        def _on_collapse(self):
            self.set_collapsed(not self.is_collapsed())

        def active_count(self):
            """Number of slots (all outputs) holding an effect."""
            count = sum(1 for slots in self.engine._track_fx
                        for fx in slots if fx)
            return count + sum(1 for fx in self.engine._bus_fx if fx)

        def _update_summary(self):
            """Refresh the 'N active' hint shown next to the header."""
            count = self.active_count()
            if self.is_collapsed():
                self.summary_lbl.setText(
                    f"{count} active" if count else "click to add effects")
            else:
                self.summary_lbl.setText("")
            self.rack_changed.emit()

        def refresh_from_engine(self):
            """Re-sync the whole rack after engine state was replaced
            (preset load / session restore)."""
            self.bpm_spin.blockSignals(True)
            self.bpm_spin.setValue(self.engine.bpm)
            self.bpm_spin.blockSignals(False)
            self.bypass_btn.setChecked(self.engine.enabled)
            self._style_bypass_btn()
            self.set_output_index(self._current_out)
            self.set_collapsed(not self.engine.has_any_effect())

        def _update_dsp_label(self):
            if not self.engine.enabled or not self.engine.has_any_effect():
                self.dsp_lbl.setText("")
                return
            pct = self.engine.get_load_pct(self._current_out)
            if pct >= 80:
                color = "#E05555"
            elif pct >= 50:
                color = "#E6A23C"
            else:
                color = "#888888"
            self.dsp_lbl.setStyleSheet(f"font-size: 10px; color: {color};")
            self.dsp_lbl.setText(f"DSP {pct:.0f}%")

        def _on_bpm(self, val):
            self.engine.set_bpm(val)

        def _on_tap(self):
            now = _time.time()
            self._tap_times.append(now)
            # Keep last 8 taps, discard if gap > 3 sec
            self._tap_times = [t for t in self._tap_times
                               if now - t < 3.0]
            if len(self._tap_times) >= 3:
                intervals = [self._tap_times[i+1] - self._tap_times[i]
                             for i in range(len(self._tap_times) - 1)]
                avg = sum(intervals) / len(intervals)
                if avg > 0:
                    bpm = 60.0 / avg
                    bpm = max(40.0, min(300.0, bpm))
                    self.bpm_spin.setValue(round(bpm, 1))

        def _on_bypass(self, checked):
            self.engine.enabled = checked
            self._style_bypass_btn()
            # Turning FX on reveals the rack — no hidden state
            if checked and self.is_collapsed():
                self.set_collapsed(False)

        def _style_bypass_btn(self):
            # Checked state styling comes from the theme QSS
            self.bypass_btn.setText("FX ON" if self.engine.enabled else "FX OFF")

        def set_output_index(self, out_idx):
            """Switch which output's effects are shown."""
            self._current_out = out_idx
            for slot in self.track_slots:
                slot.out_idx = out_idx
                slot.sync_from_engine()
            self.bus_slot.out_idx = out_idx
            self.bus_slot.sync_from_engine()

        def update_track_colors(self, colors):
            """Apply theme track colors to slot headers."""
            for i, slot in enumerate(self.track_slots):
                if i < len(colors):
                    slot.update_header_color(colors[i])
            self.bus_slot.update_header_color('#CCCCCC')
