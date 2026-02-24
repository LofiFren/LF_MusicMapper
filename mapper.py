#!/usr/bin/env python3
"""
LF Music Mapper - Routes audio inputs to specific output tracks with enhanced visualizations
"""

import sys
import numpy as np
import pyaudio
import time
from threading import Thread
from collections import deque
from scipy.fft import fft  # For spectrum analysis
from themes import THEMES, generate_qss, build_spectrogram_lut

# macOS-only: programmatic CoreAudio aggregate device creation for split
# USB devices (e.g. S-4 enumerated as separate input/output entries).
_HAS_COREAUDIO_AGG = False
if sys.platform == 'darwin':
    try:
        from coreaudio_aggregate import (
            get_uids_for_device_name, create_aggregate_device,
            destroy_aggregate_device, AGG_DEVICE_NAME,
        )
        _HAS_COREAUDIO_AGG = True
    except Exception:
        pass

DEBUG = False

def dbg(*args, **kwargs):
    """Print only when DEBUG is enabled."""
    if DEBUG:
        print(*args, **kwargs)

# First try PySide2, then try PyQt5 as fallback
try:
    from PySide2 import QtCore, QtWidgets, QtGui
    Signal = QtCore.Signal
    dbg("Using PySide2")
except ImportError:
    try:
        from PyQt5 import QtCore, QtWidgets, QtGui
        Signal = QtCore.pyqtSignal
        dbg("Using PyQt5")
    except ImportError:
        print("ERROR: Neither PySide2 nor PyQt5 is installed.")
        print("Please install one with:")
        print("pip install PySide2  # or")
        print("pip install PyQt5")
        sys.exit(1)


class AudioManager:
    """Manages audio devices and routing with track mapping capability"""
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.input_streams = []  # List of input streams (up to 3 devices)
        self.output_streams = []  # List of output streams (up to 3 devices)
        self.is_routing = False
        self.buffer_size = 1024
        self.audio_data = np.zeros((self.buffer_size, 8), dtype=np.float32)
        self.channel_levels = [0.0] * 8
        self.debug_counter = 0
        self.input_raw_data = []  # Raw numpy arrays from each input device
        self.input_channel_counts = []  # Channel count per input device
        self.total_input_channels = 2  # Total channels across all input devices
        self.num_output_channels = 8  # Default (for backward compat / viz)
        self.output_channel_counts = []  # Per-output channel counts
        self.viz_output_idx = 0  # Which output drives visualization

        self._aggregate_device_ids = []  # CoreAudio aggregates (macOS only)

        # Audio diagnostics — always-on, lightweight, logs to file
        self._diag_file = None
        self._diag_underruns = 0       # PortAudio reported input/output underflow
        self._diag_overruns = 0        # PortAudio reported overflow
        self._diag_no_data = 0         # callback fired but no input data available
        self._diag_exceptions = 0      # exceptions caught in callback
        self._diag_peak = 0.0          # peak output level since last report
        self._diag_cb_count = 0        # total callback invocations
        self._diag_slow_cb = 0         # callbacks that took > 50% of buffer time
        self._diag_last_report = 0.0   # timestamp of last diagnostic dump
        self._diag_cb_max_ms = 0.0     # slowest callback since last report
        self._diag_frame_mismatch = 0  # input frames didn't match buffer_size

        # Per-output track mappings
        self.track_mappings = [[]]  # List of mappings, one per output device
        # Create a dynamic mapping for default output
        default_mapping = []
        for i in range(self.num_output_channels):
            default_mapping.append([{'index': i % self.total_input_channels, 'gain': 1.0}])
        self.track_mappings = [default_mapping]
        
    def update_default_mapping(self, total_input_channels):
        """Update the default mapping based on total available input channels across all devices"""
        if total_input_channels <= 0:
            total_input_channels = 1

        self.total_input_channels = total_input_channels
        new_mapping = []
        for i in range(self.num_output_channels):
            new_mapping.append([{'index': i % total_input_channels, 'gain': 1.0}])
        if self.track_mappings:
            self.track_mappings[0] = new_mapping
        else:
            self.track_mappings = [new_mapping]
        dbg(f"Updated default mapping for {total_input_channels} input channels")
        
    def reinitialize(self):
        """Reinitialize PyAudio to detect newly connected USB devices"""
        if not self.is_routing:
            try:
                self.pa.terminate()
            except Exception:
                pass
            self.pa = pyaudio.PyAudio()
            dbg("PyAudio reinitialized - rescanning devices")

    def get_devices(self):
        """Get list of audio devices"""
        devices = []
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            devices.append({
                'index': i,
                'name': info['name'],
                'inputs': info['maxInputChannels'],
                'outputs': info['maxOutputChannels'],
                'rate': info['defaultSampleRate']
            })
        dbg(f"Found {len(devices)} audio device(s):")
        for d in devices:
            dbg(f"  [{d['index']}] {d['name']} (in:{d['inputs']} out:{d['outputs']} rate:{d['rate']})")
        return devices
    
    def set_track_mapping(self, mapping, output_idx=0):
        """Set channel mapping from inputs to output tracks (with gains)"""
        normalized = []

        def normalize_entry(entry):
            items = []
            if isinstance(entry, dict):
                if 'index' in entry:
                    items.append(entry)
            elif isinstance(entry, (list, tuple)):
                for sub in entry:
                    if isinstance(sub, dict) and 'index' in sub:
                        items.append(sub)
                    elif isinstance(sub, int) and sub >= 0:
                        items.append({'index': sub, 'gain': 1.0})
            elif isinstance(entry, int) and entry >= 0:
                items.append({'index': entry, 'gain': 1.0})
            result = []
            for item in items:
                idx = int(item.get('index', -1))
                if idx < 0:
                    continue
                gain = float(item.get('gain', 1.0))
                gain = max(0.0, min(2.0, gain))
                result.append({'index': idx, 'gain': gain})
                if len(result) >= 3:
                    break
            return result

        for entry in mapping:
            normalized.append(normalize_entry(entry))

        if len(normalized) < self.num_output_channels:
            normalized.extend([[] for _ in range(self.num_output_channels - len(normalized))])

        while len(self.track_mappings) <= output_idx:
            self.track_mappings.append([])
        self.track_mappings[output_idx] = normalized
        dbg(f"Track mapping set for output {output_idx}: {normalized}")
    
    def start_routing(self, input_devices, output_devices_or_idx, output_channels=8, sample_rate=48000):
        """Start routing audio from multiple input devices to output device(s).

        input_devices:          list of {'index': int, 'channels': int} dicts
        output_devices_or_idx:  list of {'index': int, 'channels': int, 'label': str}
                                OR a single int (legacy: output device index)
        output_channels:        used only with legacy int form (default 8)
        sample_rate:            desired sample rate in Hz (default 48000)
        """
        if self.is_routing:
            self.stop_routing()

        # Normalize to list-of-dicts format
        if isinstance(output_devices_or_idx, int):
            output_devices = [{'index': output_devices_or_idx,
                               'channels': output_channels, 'label': 'A'}]
        else:
            output_devices = output_devices_or_idx

        try:
            # Use first output's channel count as the default for viz/meters
            self.num_output_channels = output_devices[0]['channels']
            self.output_channel_counts = [d['channels'] for d in output_devices]

            dbg(f"\n----- AUDIO DEVICE DETAILS -----")
            for i, dev in enumerate(input_devices):
                info = self.pa.get_device_info_by_index(dev['index'])
                label = chr(ord('A') + i)
                dbg(f"Input {label}: {info['name']} ({dev['channels']} ch)")
            for i, dev in enumerate(output_devices):
                info = self.pa.get_device_info_by_index(dev['index'])
                max_ch = int(info['maxOutputChannels'])
                if dev['channels'] > max_ch:
                    dbg(f"Warning: Output {dev['label']} limited to {max_ch} channels")
                    dev['channels'] = max_ch
                    self.output_channel_counts[i] = max_ch
                dbg(f"Output {dev['label']}: {info['name']} ({dev['channels']} ch)")

            rate = int(sample_rate)

            self.input_channel_counts = [d['channels'] for d in input_devices]
            self.total_input_channels = sum(self.input_channel_counts)

            dbg(f"\n*** {len(input_devices)} input(s), {self.total_input_channels} in-ch | "
                f"{len(output_devices)} output(s) ***")

            # Dump per-output mappings
            for oi, od in enumerate(output_devices):
                mapping = self.track_mappings[oi] if oi < len(self.track_mappings) else []
                tracks = od['channels'] // 2
                dbg(f"\nOutput {od['label']} mapping ({od['channels']} ch, {tracks} tracks):")
                for out_ch, entry in enumerate(mapping):
                    if out_ch >= od['channels']:
                        break
                    t = out_ch // 2 + 1
                    lr = "L" if out_ch % 2 == 0 else "R"
                    if entry:
                        sources = ", ".join(
                            f"ch{e.get('index')}@{e.get('gain',1.0)*100:.0f}%"
                            for e in (entry if isinstance(entry, list) else [entry]))
                        dbg(f"  Track {t}.{lr} (out_ch {out_ch}) <- {sources}")
                    else:
                        dbg(f"  Track {t}.{lr} (out_ch {out_ch}) <- [silence]")

            return self._start_multi_input_stream(input_devices, output_devices, rate)

        except Exception as e:
            print(f"Error in audio routing setup: {e}")
            return False

    def _start_multi_input_stream(self, input_devices, output_devices, rate):
        """Open input streams for each device, mix via per-output callbacks."""
        try:
            num_inputs = len(input_devices)
            num_outputs = len(output_devices)

            self.debug_counter = 0
            self.input_raw_data = [None] * num_inputs
            # Per-output ring buffers — input callbacks broadcast to ALL sets,
            # each output callback pops from its own set independently.
            # maxlen=4 gives ~80ms of jitter absorption at 1024/48kHz.
            self._output_ringbufs = [
                [deque(maxlen=4) for _ in range(num_inputs)]
                for _ in range(num_outputs)
            ]
            self._prev_outputs = [None] * num_outputs  # per-output fade-out
            self._diag_stale_reads = 0
            self._diag_input_peaks = [0.0] * num_inputs  # per-input peak tracking

            # Open diagnostics log file (only when DEBUG enabled)
            if DEBUG:
                import os, datetime
                log_dir = os.path.dirname(os.path.abspath(__file__))
                log_path = os.path.join(log_dir, "audio_diag.log")
                self._diag_file = open(log_path, "w")
                self._diag_file.write(f"=== LF MusicMapper Audio Diagnostics ===\n")
                self._diag_file.write(f"Started: {datetime.datetime.now().isoformat()}\n")
                self._diag_file.write(f"Sample rate: {rate} Hz | Buffer: {self.buffer_size} frames "
                                      f"({self.buffer_size/rate*1000:.1f}ms)\n")
                for i, dev in enumerate(input_devices):
                    info = self.pa.get_device_info_by_index(dev['index'])
                    self._diag_file.write(f"Input {chr(ord('A')+i)}: {info['name']} ({dev['channels']}ch)\n")
                for i, dev in enumerate(output_devices):
                    info = self.pa.get_device_info_by_index(dev['index'])
                    self._diag_file.write(f"Output {dev['label']}: {info['name']} ({dev['channels']}ch)\n")
                self._diag_file.write(f"{'='*60}\n")
                self._diag_file.flush()
                print(f"Audio diagnostics logging to: {log_path}")
            self._diag_last_report = time.time()
            self._diag_underruns = 0
            self._diag_overruns = 0
            self._diag_no_data = 0
            self._diag_exceptions = 0
            self._diag_peak = 0.0
            self._diag_cb_count = 0
            self._diag_slow_cb = 0
            self._diag_cb_max_ms = 0.0
            self._diag_frame_mismatch = 0
            self._diag_buffer_time_ms = self.buffer_size / rate * 1000

            # Per-output duplex detection — each output independently checks
            # if an input device shares the same physical device.  An input can
            # only be duplex-paired with one output (first match wins).
            # Two cases:
            #   1. Same PortAudio index (single-entry device like Volt 476)
            #   2. Split device — same name but different indices (e.g. macOS
            #      enumerates the S-4 as separate input-only and output-only
            #      entries; opening them independently causes CoreAudio to
            #      never activate the USB return path → input is all zeros).
            # _duplex_pairs: {out_idx: (in_dev_idx, in_channels,
            #                           stream_channels, input_pa_idx)}
            _duplex_pairs = {}
            _claimed_inputs = set()
            for oi, odev in enumerate(output_devices):
                out_name = self.pa.get_device_info_by_index(odev['index'])['name']
                for di, idev in enumerate(input_devices):
                    if di in _claimed_inputs:
                        continue
                    in_name = self.pa.get_device_info_by_index(idev['index'])['name']
                    if idev['index'] == odev['index'] or in_name == out_name:
                        out_info = self.pa.get_device_info_by_index(odev['index'])
                        stream_ch = max(
                            idev['channels'],
                            int(out_info['maxOutputChannels']),
                            odev['channels'],
                        )
                        _duplex_pairs[oi] = (di, idev['channels'], stream_ch, idev['index'])
                        _claimed_inputs.add(di)
                        if idev['index'] == odev['index']:
                            dbg(f"Full-duplex: Input {chr(ord('A') + di)} shares "
                                f"device index with Output {odev['label']} "
                                f"(channels={stream_ch})")
                        else:
                            dbg(f"Full-duplex (split device): Input "
                                f"{chr(ord('A') + di)} '{in_name}' "
                                f"(idx={idev['index']}) paired with Output "
                                f"{odev['label']} '{out_name}' "
                                f"(idx={odev['index']}), "
                                f"stream channels={stream_ch}")
                        break

            # For split devices on macOS, create CoreAudio aggregate devices
            # to properly co-activate both USB data paths.  PortAudio's own
            # internal aggregate (via different input/output device indices)
            # doesn't reliably activate the USB return on some interfaces.
            _need_pa_reinit = False
            if _HAS_COREAUDIO_AGG:
                # Save device names before any potential PyAudio re-init
                _saved_in_names = {}
                for _si, _sd in enumerate(input_devices):
                    _saved_in_names[_si] = self.pa.get_device_info_by_index(
                        _sd['index'])['name']
                _saved_out_names = {}
                for _si, _sd in enumerate(output_devices):
                    _saved_out_names[_si] = self.pa.get_device_info_by_index(
                        _sd['index'])['name']

                for oi in list(_duplex_pairs.keys()):
                    di, in_ch, stream_ch, in_pa_idx = _duplex_pairs[oi]
                    if in_pa_idx == output_devices[oi]['index']:
                        continue  # same index — no split, no aggregate needed
                    out_name = _saved_out_names[oi]
                    dbg(f"Creating CoreAudio aggregate for Output "
                        f"{output_devices[oi]['label']} split device...")

                    uids = get_uids_for_device_name(out_name)
                    dbg(f"  CoreAudio UIDs for '{out_name}': {uids}")

                    if len(uids) >= 2:
                        master_uid = None
                        all_uids = []
                        for _uid, _, _has_out in uids:
                            all_uids.append(_uid)
                            if _has_out:
                                master_uid = _uid

                        if master_uid and len(all_uids) >= 2:
                            agg_id = create_aggregate_device(
                                all_uids, master_uid=master_uid)
                            if agg_id:
                                self._aggregate_device_ids.append(agg_id)
                                _need_pa_reinit = True
                                dbg(f"  Aggregate created "
                                    f"(AudioDeviceID={agg_id})")
                            else:
                                dbg("  WARNING: CoreAudio aggregate "
                                    "creation failed")
                        else:
                            dbg(f"  WARNING: could not identify output UID "
                                f"from {uids}")
                    else:
                        dbg(f"  WARNING: expected 2+ CoreAudio devices for "
                            f"'{out_name}', found {len(uids)}")

            # Re-init PyAudio once if any aggregates were created
            if _need_pa_reinit:
                self.pa.terminate()
                self.pa = pyaudio.PyAudio()

                # Re-resolve all device indices by name
                for _si, _sd in enumerate(input_devices):
                    _target = _saved_in_names[_si]
                    for _pi in range(self.pa.get_device_count()):
                        _pinfo = self.pa.get_device_info_by_index(_pi)
                        if (_pinfo['name'] == _target
                                and _pinfo['maxInputChannels']
                                >= _sd['channels']):
                            _sd['index'] = _pi
                            break

                for _si, _sd in enumerate(output_devices):
                    _target = _saved_out_names[_si]
                    for _pi in range(self.pa.get_device_count()):
                        _pinfo = self.pa.get_device_info_by_index(_pi)
                        if (_pinfo['name'] == _target
                                and _pinfo['maxOutputChannels']
                                >= _sd['channels']):
                            _sd['index'] = _pi
                            break

                # Find aggregate devices and update duplex pairs
                for oi in list(_duplex_pairs.keys()):
                    di, in_ch, stream_ch, in_pa_idx = _duplex_pairs[oi]
                    if in_pa_idx == output_devices[oi]['index']:
                        continue  # was same-index, not a split device
                    # Look for the aggregate in the refreshed device list
                    _agg_pa_idx = None
                    for _pi in range(self.pa.get_device_count()):
                        _pinfo = self.pa.get_device_info_by_index(_pi)
                        if AGG_DEVICE_NAME in _pinfo['name']:
                            _agg_pa_idx = _pi
                            break

                    if _agg_pa_idx is not None:
                        _agg_info = self.pa.get_device_info_by_index(
                            _agg_pa_idx)
                        dbg(f"  Aggregate at PA idx {_agg_pa_idx}: "
                            f"in={_agg_info['maxInputChannels']} "
                            f"out={_agg_info['maxOutputChannels']}")
                        # Use separate streams on aggregate instead of
                        # full-duplex (avoids all-zero input on some)
                        output_devices[oi]['index'] = _agg_pa_idx
                        input_devices[di]['index'] = _agg_pa_idx
                        input_devices[di]['channels'] = int(
                            _agg_info['maxInputChannels'])
                        del _duplex_pairs[oi]  # clear duplex flag
                    else:
                        dbg("  WARNING: aggregate not found in "
                            "PyAudio after re-init")

            # Create a callback closure for each input device.
            # Each input broadcasts its buffer to ALL output ring buffer sets
            # so each output callback can independently consume data.
            def make_input_callback(dev_idx, num_channels):
                def callback(in_data, frame_count, time_info, status):
                    if status:
                        if status & pyaudio.paInputUnderflow or status & pyaudio.paInputOverflow:
                            self._diag_overruns += 1
                    data = np.frombuffer(in_data, dtype=np.float32)
                    if len(data) > 0 and num_channels > 0:
                        try:
                            buf = data.reshape(-1, num_channels).copy()
                            # Broadcast to every output's ring buffer set
                            for out_bufs in self._output_ringbufs:
                                out_bufs[dev_idx].append(buf)
                            self.input_raw_data[dev_idx] = buf  # latest ref for viz
                            # Per-input peak tracking
                            peak = float(np.abs(buf).max())
                            if peak > self._diag_input_peaks[dev_idx]:
                                self._diag_input_peaks[dev_idx] = peak
                        except Exception:
                            pass
                    return (None, pyaudio.paContinue)
                return callback

            # Crossfade ramp — precomputed for underrun fade-out
            _xfade_out = np.linspace(1.0, 0.0, self.buffer_size, dtype=np.float32).reshape(-1, 1)

            # Collect which input indices are claimed by any duplex pair
            _duplex_input_idxs = {info[0] for info in _duplex_pairs.values()}

            # Output callback factory — one closure per output device.
            # Each captures its own out_idx, channel count, and ring buffer set.
            def make_output_callback(out_idx, out_channels):
                # Duplex info for this output (if any)
                _dup = _duplex_pairs.get(out_idx)  # (in_dev_idx, in_ch, stream_ch, in_pa_idx) or None
                _stream_channels = _dup[2] if _dup else out_channels

                def output_callback(in_data, frame_count, time_info, status):
                    cb_start = time.time()
                    try:
                        self._diag_cb_count += 1

                        # Track PortAudio status flags
                        if status:
                            if status & pyaudio.paOutputUnderflow:
                                self._diag_underruns += 1
                            if status & pyaudio.paOutputOverflow:
                                self._diag_overruns += 1

                        # Full-duplex: capture input from the shared device
                        _dup_buf_local = None
                        if _dup is not None and in_data is not None:
                            dup_dev_idx, dup_in_ch, dup_stream_ch, _ = _dup
                            try:
                                dup_data = np.frombuffer(in_data, dtype=np.float32)
                                if len(dup_data) > 0:
                                    dup_buf = dup_data.reshape(-1, dup_stream_ch).copy()
                                    if dup_in_ch < dup_stream_ch:
                                        dup_buf = dup_buf[:, :dup_in_ch]
                                    _dup_buf_local = (dup_dev_idx, dup_buf)
                                    # Broadcast to OTHER outputs' ring buffer sets
                                    # (skip our own — we use _dup_buf_local directly)
                                    for oi_rb, obufs in enumerate(self._output_ringbufs):
                                        if oi_rb != out_idx:
                                            obufs[dup_dev_idx].append(dup_buf)
                                    self.input_raw_data[dup_dev_idx] = dup_buf
                                    dup_peak = float(np.abs(dup_buf).max())
                                    if dup_peak > self._diag_input_peaks[dup_dev_idx]:
                                        self._diag_input_peaks[dup_dev_idx] = dup_peak
                            except Exception:
                                pass
                        elif _dup is not None and in_data is None:
                            self._diag_no_data += 1

                        if frame_count != self.buffer_size:
                            self._diag_frame_mismatch += 1

                        # Pop fresh data from THIS output's ring buffer set
                        has_data = False
                        input_bufs = [None] * len(self.input_channel_counts)
                        for i in range(len(self.input_channel_counts)):
                            rb = self._output_ringbufs[out_idx][i]
                            if rb:
                                input_bufs[i] = rb.popleft()
                                has_data = True
                            else:
                                self._diag_stale_reads += 1

                        # Inject duplex-captured input directly (guaranteed fresh,
                        # not dependent on ring buffer timing)
                        if _dup_buf_local is not None:
                            di, buf = _dup_buf_local
                            input_bufs[di] = buf
                            has_data = True

                        if not has_data:
                            self._diag_no_data += 1
                            # Fade out previous output to avoid hard cut to silence
                            prev = self._prev_outputs[out_idx]
                            if prev is not None:
                                xf_len = min(frame_count, len(_xfade_out))
                                faded = prev[:xf_len] * _xfade_out[:xf_len]
                                self._prev_outputs[out_idx] = None
                                return (faded.flatten().tobytes(), pyaudio.paContinue)
                            silence = np.zeros(frame_count * _stream_channels, dtype=np.float32)
                            return (silence.tobytes(), pyaudio.paContinue)

                        # Build combined input array (frames x total_channels)
                        combined = np.zeros((frame_count, self.total_input_channels), dtype=np.float32)
                        ch_offset = 0
                        for i in range(len(self.input_channel_counts)):
                            num_ch = self.input_channel_counts[i]
                            raw = input_bufs[i]
                            if raw is not None:
                                use_frames = min(frame_count, len(raw))
                                use_ch = min(num_ch, raw.shape[1]) if len(raw.shape) > 1 else num_ch
                                combined[:use_frames, ch_offset:ch_offset + use_ch] = raw[:use_frames, :use_ch]
                            ch_offset += num_ch

                        # Apply THIS output's track mapping
                        mapping = self.track_mappings[out_idx] if out_idx < len(self.track_mappings) else []
                        output_frames = np.zeros((frame_count, out_channels), dtype=np.float32)
                        for out_ch in range(out_channels):
                            mapping_entry = []
                            if out_ch < len(mapping):
                                mapping_entry = mapping[out_ch]
                            if isinstance(mapping_entry, dict):
                                mapping_entry = [mapping_entry]
                            elif not isinstance(mapping_entry, (list, tuple)):
                                mapping_entry = []

                            valid_entries = []
                            for entry in mapping_entry:
                                idx = entry.get('index') if isinstance(entry, dict) else None
                                gain = entry.get('gain', 1.0) if isinstance(entry, dict) else 1.0
                                if isinstance(idx, int) and 0 <= idx < self.total_input_channels:
                                    valid_entries.append({'index': idx, 'gain': max(0.0, float(gain))})

                            if valid_entries:
                                mixed = np.zeros(frame_count, dtype=np.float32)
                                for item in valid_entries:
                                    mixed += combined[:, item['index']] * item['gain']
                                # Normalize by source count (not total gain) to
                                # prevent clipping when mixing multiple inputs
                                # while preserving per-source volume control.
                                if len(valid_entries) > 1:
                                    mixed /= len(valid_entries)
                                output_frames[:, out_ch] = mixed

                        # Soft-clip output to prevent popping from hot signals
                        output_frames = np.tanh(output_frames)

                        # Pad output for full-duplex stream (more device channels
                        # than routed output channels)
                        if _stream_channels > out_channels:
                            padded = np.zeros((frame_count, _stream_channels), dtype=np.float32)
                            padded[:, :out_channels] = output_frames
                            output_frames = padded

                        # Save for fade-out on future underrun
                        self._prev_outputs[out_idx] = output_frames.copy()

                        # Update visualization only from the viz output
                        if out_idx == self.viz_output_idx:
                            peak = float(np.abs(output_frames).max())
                            if peak > self._diag_peak:
                                self._diag_peak = peak

                            self.channel_levels = [
                                np.abs(output_frames[:, ch]).mean() if ch < out_channels else 0.0
                                for ch in range(8)
                            ]
                            sample_count = min(self.buffer_size, len(output_frames))
                            for ch in range(min(8, out_channels)):
                                if sample_count > 0:
                                    self.audio_data[:sample_count, ch] = output_frames[:sample_count, ch]
                                    if sample_count < self.buffer_size:
                                        self.audio_data[sample_count:, ch] = 0

                        result = (output_frames.flatten().tobytes(), pyaudio.paContinue)

                        # Track callback duration
                        cb_ms = (time.time() - cb_start) * 1000
                        if cb_ms > self._diag_cb_max_ms:
                            self._diag_cb_max_ms = cb_ms
                        if cb_ms > self._diag_buffer_time_ms * 0.5:
                            self._diag_slow_cb += 1

                        # Diagnostics every 2s (only from output 0 to avoid dupes)
                        if out_idx == 0:
                            now = time.time()
                            if now - self._diag_last_report >= 2.0 and self._diag_file:
                                elapsed = now - self._diag_last_report
                                cbs = self._diag_cb_count
                                cb_rate = cbs / elapsed if elapsed > 0 else 0
                                input_peak_str = " ".join(
                                    f"{chr(ord('A')+i)}={p:.4f}"
                                    for i, p in enumerate(self._diag_input_peaks)
                                )
                                self._diag_file.write(
                                    f"[{now:.1f}] callbacks={cbs} ({cb_rate:.0f}/s) | "
                                    f"underruns={self._diag_underruns} overruns={self._diag_overruns} | "
                                    f"no_data={self._diag_no_data} stale={self._diag_stale_reads} "
                                    f"frame_mismatch={self._diag_frame_mismatch} | "
                                    f"slow_cb(>{self._diag_buffer_time_ms*0.5:.1f}ms)={self._diag_slow_cb} "
                                    f"max_cb={self._diag_cb_max_ms:.2f}ms | "
                                    f"out_peak={self._diag_peak:.4f} | "
                                    f"in_peaks: {input_peak_str} | "
                                    f"exceptions={self._diag_exceptions}\n"
                                )
                                self._diag_file.flush()
                                self._diag_last_report = now
                                self._diag_cb_count = 0
                                self._diag_underruns = 0
                                self._diag_overruns = 0
                                self._diag_no_data = 0
                                self._diag_stale_reads = 0
                                self._diag_slow_cb = 0
                                self._diag_cb_max_ms = 0.0
                                self._diag_peak = 0.0
                                self._diag_frame_mismatch = 0
                                self._diag_exceptions = 0
                                self._diag_input_peaks = [0.0] * len(self._diag_input_peaks)

                        return result

                    except Exception as e:
                        self._diag_exceptions += 1
                        if self._diag_file:
                            self._diag_file.write(f"!!! EXCEPTION in output_callback[{out_idx}]: {e}\n")
                            self._diag_file.flush()
                        silence = np.zeros(frame_count * _stream_channels, dtype=np.float32)
                        return (silence.tobytes(), pyaudio.paContinue)

                return output_callback

            # Open one input stream per device (skip duplex-paired inputs)
            self.input_streams = []
            for dev_idx, dev in enumerate(input_devices):
                if dev_idx in _duplex_input_idxs:
                    dbg(f"Skipping separate input stream for "
                        f"{chr(ord('A') + dev_idx)} (full-duplex)")
                    self.input_streams.append(None)  # placeholder
                    continue
                label = chr(ord('A') + dev_idx)
                dbg(f"Opening input stream {label} with {dev['channels']} channels at {rate}Hz, buffer={self.buffer_size}")
                stream = self.pa.open(
                    format=pyaudio.paFloat32,
                    channels=dev['channels'],
                    rate=rate,
                    input=True,
                    input_device_index=dev['index'],
                    frames_per_buffer=self.buffer_size,
                    stream_callback=make_input_callback(dev_idx, dev['channels'])
                )
                self.input_streams.append(stream)

            # Open output streams — one per output device.
            # Full-duplex when an input shares the device.
            self.output_streams = []
            for oi, odev in enumerate(output_devices):
                cb = make_output_callback(oi, odev['channels'])
                if oi in _duplex_pairs:
                    di, in_ch, stream_ch, in_pa_idx = _duplex_pairs[oi]
                    dbg(f"Opening FULL-DUPLEX stream for Output {odev['label']}: "
                        f"in_idx={in_pa_idx} out_idx={odev['index']} "
                        f"channels={stream_ch} rate={rate}Hz")
                    stream = self.pa.open(
                        format=pyaudio.paFloat32,
                        channels=stream_ch,
                        rate=rate,
                        input=True,
                        output=True,
                        input_device_index=in_pa_idx,
                        output_device_index=odev['index'],
                        frames_per_buffer=self.buffer_size,
                        stream_callback=cb
                    )
                else:
                    dbg(f"Opening output stream {odev['label']} with "
                        f"{odev['channels']} channels at {rate}Hz, "
                        f"buffer={self.buffer_size}")
                    stream = self.pa.open(
                        format=pyaudio.paFloat32,
                        channels=odev['channels'],
                        rate=rate,
                        output=True,
                        output_device_index=odev['index'],
                        frames_per_buffer=self.buffer_size,
                        stream_callback=cb
                    )
                self.output_streams.append(stream)

            self.is_routing = True
            dbg(f"\nStarted routing with {len(input_devices)} input(s), "
                f"{len(output_devices)} output(s)")
            return True

        except Exception as e:
            print(f"Error starting multi-input stream: {e}")
            self.stop_routing()
            return False
    
    def stop_routing(self):
        """Stop audio routing — flush silence to prevent residual hum."""
        dbg("Stopping audio routing...")

        self.is_routing = False

        # Zero out all input buffers so the callback outputs silence immediately
        for i in range(len(self.input_raw_data)):
            self.input_raw_data[i] = None
        self.channel_levels = [0.0] * 8
        self.audio_data = np.zeros((self.buffer_size, 8), dtype=np.float32)

        # Brief pause to let the callback emit at least one silent buffer
        time.sleep(0.05)

        # Close all input streams (None entries are duplex placeholders)
        for i, stream in enumerate(self.input_streams):
            if stream is None:
                continue
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                dbg(f"Input stream {chr(ord('A') + i)} closed")
            except Exception as e:
                dbg(f"Error closing input stream {chr(ord('A') + i)}: {e}")
        self.input_streams = []

        # Close all output streams
        for i, stream in enumerate(self.output_streams):
            if stream is None:
                continue
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                dbg(f"Output stream {i} closed")
            except Exception as e:
                dbg(f"Error closing output stream {i}: {e}")
        self.output_streams = []

        # Destroy CoreAudio aggregate devices (macOS only)
        if self._aggregate_device_ids and _HAS_COREAUDIO_AGG:
            for agg_id in self._aggregate_device_ids:
                try:
                    destroy_aggregate_device(agg_id)
                    dbg(f"CoreAudio aggregate {agg_id} destroyed")
                except Exception as e:
                    dbg(f"Error destroying aggregate {agg_id}: {e}")
            self._aggregate_device_ids = []

        # Close diagnostics log
        if self._diag_file:
            import datetime
            self._diag_file.write(f"{'='*60}\nStopped: {datetime.datetime.now().isoformat()}\n")
            self._diag_file.close()
            self._diag_file = None

        dbg("Audio routing stopped")

    def get_channel_levels(self):
        """Get levels for all channels"""
        return self.channel_levels
    
    def get_audio_data(self, channel=0):
        """Get audio data for specific channel"""
        channel = min(channel, 7)  # Limit to available channels
        return self.audio_data[:, channel]
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_routing()
        # Safety net: destroy aggregates if stop_routing didn't
        if self._aggregate_device_ids and _HAS_COREAUDIO_AGG:
            for agg_id in self._aggregate_device_ids:
                try:
                    destroy_aggregate_device(agg_id)
                except Exception:
                    pass
            self._aggregate_device_ids = []
        self.pa.terminate()
        dbg("Audio resources cleaned up")


class MixDialog(QtWidgets.QDialog):
    """Allows selecting up to N inputs with level sliders."""

    def __init__(self, input_count, max_sources, current_mix=None, input_labels=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mix Inputs")
        self.input_count = max(1, min(16, input_count))
        self.max_sources = max_sources
        self.rows = []
        self.input_labels = input_labels or [f"Input {i+1}" for i in range(self.input_count)]
        current_mix = current_mix or []
        mix_map = {entry['index']: entry.get('gain', 1.0) for entry in current_mix if 'index' in entry}

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Select up to three inputs and set their mix levels:"))

        for idx in range(self.input_count):
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            checkbox = QtWidgets.QCheckBox(self.input_labels[idx] if idx < len(self.input_labels) else f"Input {idx+1}")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setSingleStep(5)
            slider.setEnabled(False)
            value_label = QtWidgets.QLabel("0%")
            value_label.setFixedWidth(40)

            gain = mix_map.get(idx, None)
            if gain is not None:
                checkbox.setChecked(True)
                slider.setEnabled(True)
                slider.setValue(int(max(0.0, min(1.0, gain)) * 100))
                value_label.setText(f"{slider.value()}%")
            else:
                slider.setValue(100)

            checkbox.toggled.connect(lambda checked, i=idx: self._handle_checkbox(i, checked))
            slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(f"{value}%"))

            row_layout.addWidget(checkbox)
            row_layout.addWidget(slider, 1)
            row_layout.addWidget(value_label)

            layout.addWidget(row_widget)
            self.rows.append({'checkbox': checkbox, 'slider': slider})

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _handle_checkbox(self, idx, checked):
        row = self.rows[idx]
        slider = row['slider']
        checkbox = row['checkbox']
        if checked:
            if sum(1 for r in self.rows if r['checkbox'].isChecked()) > self.max_sources:
                checkbox.blockSignals(True)
                checkbox.setChecked(False)
                checkbox.blockSignals(False)
                return
            slider.setEnabled(True)
            if slider.value() == 0:
                slider.setValue(100)
        else:
            slider.setEnabled(False)

    def get_mix(self):
        mix = []
        for idx, row in enumerate(self.rows):
            if row['checkbox'].isChecked():
                gain = row['slider'].value() / 100.0
                if gain > 0:
                    mix.append({'index': idx, 'gain': gain})
        return mix[:self.max_sources]


class MultiInputSelector(QtWidgets.QWidget):
    """Button + readout that opens a dialog for mixing up to 3 input sources."""

    mix_changed = Signal()

    def __init__(self, parent=None, max_sources=3):
        super().__init__(parent)
        self.max_sources = max_sources
        self._input_count = 2
        self._input_labels = ["A1", "A2"]
        self._mix = []

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.display = QtWidgets.QLineEdit("Silence")
        self.display.setReadOnly(True)
        self.display.setObjectName("mixDisplay")
        layout.addWidget(self.display, 1)

        self.mix_button = QtWidgets.QPushButton("Mix")
        self.mix_button.setToolTip("Choose up to three inputs to blend for this channel")
        self.mix_button.clicked.connect(self.open_dialog)
        layout.addWidget(self.mix_button)

        self.set_inputs(self._input_count, self._input_labels)

    def set_input_count(self, input_count):
        """Backwards-compatible wrapper."""
        labels = [f"Input {i+1}" for i in range(input_count)]
        self.set_inputs(input_count, labels)

    def set_inputs(self, input_count, input_labels):
        self._input_count = max(1, min(16, input_count))
        self._input_labels = input_labels or [f"Input {i+1}" for i in range(self._input_count)]
        self._mix = [entry for entry in self._mix if entry['index'] < self._input_count]
        self.update_display()

    def open_dialog(self):
        dialog = MixDialog(self._input_count, self.max_sources, self._mix, self._input_labels, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self._mix = dialog.get_mix()
            self.update_display()
            self.mix_changed.emit()

    def get_mix(self):
        return [dict(entry) for entry in self._mix]

    def set_mix(self, mix, emit=True):
        sanitized = []
        for entry in mix:
            if isinstance(entry, dict):
                idx = entry.get('index')
                if isinstance(idx, int) and 0 <= idx < self._input_count:
                    gain = float(entry.get('gain', 1.0))
                    sanitized.append({'index': idx, 'gain': max(0.0, min(1.0, gain))})
            elif isinstance(entry, int) and 0 <= entry < self._input_count:
                sanitized.append({'index': entry, 'gain': 1.0})
            if len(sanitized) >= self.max_sources:
                break
        self._mix = sanitized
        self.update_display()
        if emit:
            self.mix_changed.emit()

    def set_checked_inputs(self, inputs):
        mix = []
        for item in inputs:
            if isinstance(item, dict):
                mix.append(item)
            elif isinstance(item, int):
                mix.append({'index': item, 'gain': 1.0})
        self.set_mix(mix)

    def clear_selection(self):
        self.set_mix([])

    def update_display(self):
        if not self._mix:
            text = "Silence"
        else:
            parts = []
            for entry in self._mix:
                idx = entry['index']
                label = self._input_labels[idx] if idx < len(self._input_labels) else f"In {idx+1}"
                parts.append(f"{label} ({entry['gain']*100:.0f}%)")
            text = " + ".join(parts)
        self.display.setText(text)

class TrackMapperWidget(QtWidgets.QWidget):
    """Widget for mapping input channels to output tracks"""

    mapping_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 300)
        
        # Create layout
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        
        # Title label
        title = QtWidgets.QLabel("Track Mapping")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Inline helper text keeps the UI self-explanatory without popups
        helper = QtWidgets.QLabel(
            "Choose which input feeds each left/right channel below. "
            "Use the Mix button beside each channel to blend up to three inputs, "
            "or use the presets for quick routings."
        )
        helper.setStyleSheet("color: #BBBBBB; font-size: 11px;")
        helper.setWordWrap(True)
        layout.addWidget(helper)
        
        # Create grid for track mapping
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)  # Add more spacing between elements
        grid.setContentsMargins(0, 5, 0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 2)
        
        # Headers
        header_style = "font-weight: bold; color: #CCCCCC; font-size: 12px;"
        track_header = QtWidgets.QLabel("Track")
        track_header.setStyleSheet(header_style)
        grid.addWidget(track_header, 0, 0)
        
        left_header = QtWidgets.QLabel("Left Channel")
        left_header.setStyleSheet(header_style)
        grid.addWidget(left_header, 0, 1)
        
        right_header = QtWidgets.QLabel("Right Channel")
        right_header.setStyleSheet(header_style)
        grid.addWidget(right_header, 0, 2)
        
        assign_header = QtWidgets.QLabel("Quick Set")
        assign_header.setStyleSheet(header_style)
        grid.addWidget(assign_header, 0, 3)
        
        # Track selectors
        self.channel_selectors = []
        self.channel_selector_map = {}
        self.track_button_containers = {}  # Store button container per track for rebuilding
        self.input_channel_count = 2
        self.input_labels = ["A1", "A2"]
        self.current_input_infos = [{'label': 'A', 'channels': 2}]
        self._track_colors = THEMES['Dark']['palette']['track_colors']
        self._track_label_widgets = []
        self._header_labels = [track_header, left_header, right_header, assign_header]
        self._title_widget = title
        self._helper_widget = helper

        # Create 4 rows for 4 tracks
        for i in range(4):  # 4 tracks
            # Track number with custom styling
            track_label = QtWidgets.QLabel(f"Track {i+1}")
            track_label.setStyleSheet(f"font-weight: bold; color: {self.get_track_color(i)};")
            self._track_label_widgets.append(track_label)
            grid.addWidget(track_label, i+1, 0)

            # Left channel (X.1)
            left_combo = MultiInputSelector(max_sources=3)
            if i == 0:
                left_combo.set_checked_inputs([0])
            left_combo.mix_changed.connect(self.mapping_changed)
            grid.addWidget(left_combo, i+1, 1)
            self.channel_selectors.append(left_combo)
            self.channel_selector_map[(i, 'left')] = left_combo

            # Right channel (X.2)
            right_combo = MultiInputSelector(max_sources=3)
            if i == 0:
                right_combo.set_checked_inputs([1])
            right_combo.mix_changed.connect(self.mapping_changed)
            grid.addWidget(right_combo, i+1, 2)
            self.channel_selectors.append(right_combo)
            self.channel_selector_map[(i, 'right')] = right_combo

            # Quick assign button container (rebuilt dynamically when inputs change)
            btn_container = QtWidgets.QWidget()
            btn_container.setLayout(QtWidgets.QHBoxLayout())
            btn_container.layout().setContentsMargins(0, 0, 0, 0)
            btn_container.layout().setSpacing(5)
            grid.addWidget(btn_container, i+1, 3)
            self.track_button_containers[i] = btn_container

        # Build initial quick buttons
        self._rebuild_quick_buttons()
        
        layout.addLayout(grid)
        
        # Add separator line
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        line.setStyleSheet("background-color: #444444; margin: 10px 0;")
        layout.addWidget(line)
        
        # Quick setup buttons
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.setSpacing(10)
        
        preset_label = QtWidgets.QLabel("Presets:")
        preset_label.setStyleSheet("font-weight: bold; color: #CCCCCC;")
        preset_layout.addWidget(preset_label)
        
        # Stereo to track 1 preset
        preset1_btn = QtWidgets.QPushButton("Stereo on Track 1")
        preset1_btn.setProperty("preset", "true")
        preset1_btn.clicked.connect(self.preset_stereo_track1)
        preset_layout.addWidget(preset1_btn)
        
        # All tracks mono preset
        preset2_btn = QtWidgets.QPushButton("Mono Input 1 to All")
        preset2_btn.setProperty("preset", "true")
        preset2_btn.clicked.connect(self.preset_mono_all)
        preset_layout.addWidget(preset2_btn)
        
        # All tracks stereo preset
        preset3_btn = QtWidgets.QPushButton("Stereo to All Tracks")
        preset3_btn.setProperty("preset", "true")
        preset3_btn.clicked.connect(self.preset_stereo_all)
        preset_layout.addWidget(preset3_btn)
        
        # Different mono to each track
        preset4_btn = QtWidgets.QPushButton("Input 1 & 2 to Different Tracks")
        preset4_btn.setProperty("preset", "true")
        preset4_btn.clicked.connect(self.preset_different_tracks)
        preset_layout.addWidget(preset4_btn)
        
        # Use all inputs (NEW)
        preset5_btn = QtWidgets.QPushButton("Use All Inputs")
        preset5_btn.setProperty("preset", "true")
        preset5_btn.clicked.connect(self.preset_use_all_inputs)
        preset_layout.addWidget(preset5_btn)
        
        preset_layout.addStretch()
        
        layout.addLayout(preset_layout)
        
        # Status message
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #AAAAAA; margin-top: 5px;")
        layout.addWidget(self.status_label)

    def get_track_color(self, track_idx):
        """Get color for a specific track from theme track_colors."""
        return self._track_colors[track_idx % len(self._track_colors)]

    def _refresh_track_colors(self):
        """Re-apply track colors after a theme change."""
        for i, lbl in enumerate(self._track_label_widgets):
            lbl.setStyleSheet(f"font-weight: bold; color: {self.get_track_color(i)};")

    def _rebuild_quick_buttons(self):
        """Rebuild quick-set buttons for each track based on current input devices."""
        for track_idx in range(4):
            container = self.track_button_containers.get(track_idx)
            if not container:
                continue
            layout = container.layout()
            # Clear existing buttons
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            # Add per-device buttons
            ch_offset = 0
            for info in self.current_input_infos:
                dev_label = info['label']
                dev_channels = info['channels']
                if dev_channels >= 2:
                    # Stereo pair buttons — one per pair of channels.
                    # 2-channel device: "B Stereo"
                    # 4-channel device: "B 1-2" (e.g. Dry) and "B 3-4" (e.g. Wet/FX)
                    num_pairs = dev_channels // 2
                    for pair_idx in range(num_pairs):
                        left_idx = ch_offset + pair_idx * 2
                        right_idx = left_idx + 1
                        if num_pairs == 1:
                            label_text = f"{dev_label} Stereo"
                            tip = f"Set track to stereo ({dev_label}1->L, {dev_label}2->R)"
                        else:
                            ch_lo = pair_idx * 2 + 1
                            ch_hi = ch_lo + 1
                            label_text = f"{dev_label} {ch_lo}-{ch_hi}"
                            tip = f"Set track to {dev_label}{ch_lo}->L, {dev_label}{ch_hi}->R"
                        btn = QtWidgets.QPushButton(label_text)
                        btn.setToolTip(tip)
                        btn.setProperty("preset", "true")
                        btn.clicked.connect(lambda checked, t=track_idx, l=left_idx, r=right_idx: self._set_stereo_pair(t, l, r))
                        layout.addWidget(btn)
                # Mono button for each channel of this device
                for ch in range(dev_channels):
                    global_idx = ch_offset + ch
                    ch_label = f"{dev_label}{ch+1}"
                    btn = QtWidgets.QPushButton(ch_label)
                    btn.setToolTip(f"Set track to mono with {ch_label}")
                    btn.setProperty("preset", "true")
                    btn.clicked.connect(lambda checked, t=track_idx, idx=global_idx: self.set_mono(t, idx))
                    layout.addWidget(btn)
                ch_offset += dev_channels

    def _set_stereo_pair(self, track_idx, left_idx, right_idx):
        """Set a track to a specific stereo pair by global channel indices."""
        left_combo = self.channel_selector_map.get((track_idx, 'left'))
        right_combo = self.channel_selector_map.get((track_idx, 'right'))
        if not left_combo or not right_combo:
            return
        left_combo.set_checked_inputs([left_idx])
        right_combo.set_checked_inputs([right_idx])
        left_label = self.input_labels[left_idx] if left_idx < len(self.input_labels) else str(left_idx)
        right_label = self.input_labels[right_idx] if right_idx < len(self.input_labels) else str(right_idx)
        self.status_label.setText(f"Track {track_idx+1}: {left_label}->L, {right_label}->R")

    def update_for_input_device(self, input_channels):
        """Update channel options based on input device (single-device compat)"""
        self.update_for_inputs([{'label': 'A', 'channels': input_channels}])

    def update_for_inputs(self, input_infos):
        """Update channel options based on all active input devices.

        input_infos: list of {'label': str, 'channels': int}
        """
        labels = []
        for info in input_infos:
            dev_label = info['label']
            for ch in range(info['channels']):
                labels.append(f"{dev_label}{ch+1}")
        self.input_channel_count = len(labels)
        self.input_labels = labels
        self.current_input_infos = input_infos
        for combo in self.channel_selectors:
            previous = combo.get_mix()
            combo.set_inputs(self.input_channel_count, self.input_labels)
            combo.set_mix(previous)
        self._rebuild_quick_buttons()

    def get_mapping(self):
        """Get the current channel mapping"""
        mapping = []
        for track_idx in range(4):
            left_combo = self.channel_selector_map.get((track_idx, 'left'))
            right_combo = self.channel_selector_map.get((track_idx, 'right'))
            if left_combo:
                mapping.append(left_combo.get_mix())
            else:
                mapping.append([])
            if right_combo:
                mapping.append(right_combo.get_mix())
            else:
                mapping.append([])
        return mapping
    
    def preset_use_all_inputs(self):
        """Preset: Use all available inputs across tracks without duplication"""
        num_tracks = len(self.channel_selector_map) // 2
        input_count = self.input_channel_count
        
        if input_count <= 1:
            self.status_label.setText("Not enough inputs available")
            return
            
        # Loop through tracks and assign unique inputs when possible
        for track_idx in range(num_tracks):
            left_combo = self.channel_selector_map.get((track_idx, 'left'))
            right_combo = self.channel_selector_map.get((track_idx, 'right'))
            if not left_combo or not right_combo:
                continue
            if track_idx < input_count:
                left_combo.set_checked_inputs([track_idx])
                right_combo.set_checked_inputs([track_idx])
            else:
                input_to_use = track_idx % input_count
                left_combo.set_checked_inputs([input_to_use])
                right_combo.set_checked_inputs([input_to_use])
        
        self.status_label.setText("Applied preset: Using all available inputs")
    
    def auto_assign_devices(self, input_infos, num_output_tracks=4):
        """Auto-assign each device to its own track, respecting available output tracks.

        If there are more devices than output tracks, devices share tracks (mix).
        input_infos: list of {'label': str, 'channels': int}
        num_output_tracks: number of stereo tracks available on the output device
        """
        num_tracks = min(4, max(1, num_output_tracks))

        # First clear all tracks
        for combo in self.channel_selectors:
            combo.clear_selection()

        # Build per-track assignment lists (which global channels go to each track)
        track_left = [[] for _ in range(num_tracks)]
        track_right = [[] for _ in range(num_tracks)]
        ch_offset = 0
        for dev_idx, info in enumerate(input_infos):
            track_idx = dev_idx % num_tracks  # Wrap around if more devices than tracks
            # First channel → Left
            track_left[track_idx].append({'index': ch_offset, 'gain': 1.0})
            # Second channel (if stereo) → Right, else same channel
            if info['channels'] >= 2:
                track_right[track_idx].append({'index': ch_offset + 1, 'gain': 1.0})
            else:
                track_right[track_idx].append({'index': ch_offset, 'gain': 1.0})
            ch_offset += info['channels']

        # Apply to selectors
        assignments = []
        for track_idx in range(num_tracks):
            left_combo = self.channel_selector_map.get((track_idx, 'left'))
            right_combo = self.channel_selector_map.get((track_idx, 'right'))
            if not left_combo or not right_combo:
                continue
            if track_left[track_idx]:
                left_combo.set_mix(track_left[track_idx])
            if track_right[track_idx]:
                right_combo.set_mix(track_right[track_idx])

        # Build status text
        parts = []
        ch_offset = 0
        for dev_idx, info in enumerate(input_infos):
            track_idx = dev_idx % num_tracks
            parts.append(f"{info['label']}->Track {track_idx + 1}")
            ch_offset += info['channels']
        self.status_label.setText(f"Auto-assigned: {', '.join(parts)}")

    def set_stereo(self, track_idx):
        """Set a track to stereo using first available stereo pair."""
        self._set_stereo_pair(track_idx, 0, min(1, self.input_channel_count - 1))

    def set_mono(self, track_idx, input_idx):
        """Set a track to mono with specified global input index."""
        left_combo = self.channel_selector_map.get((track_idx, 'left'))
        right_combo = self.channel_selector_map.get((track_idx, 'right'))
        if not left_combo or not right_combo:
            return
        wrapped_input = input_idx % max(1, self.input_channel_count)
        left_combo.set_checked_inputs([wrapped_input])
        right_combo.set_checked_inputs([wrapped_input])
        label = self.input_labels[wrapped_input] if wrapped_input < len(self.input_labels) else str(wrapped_input)
        self.status_label.setText(f"Track {track_idx+1}: mono {label}")
    
    def preset_stereo_track1(self):
        """Preset: Stereo on track 1, silence on others"""
        # First, set all to silence
        for combo in self.channel_selectors:
            combo.clear_selection()
        
        # Then set track 1 to stereo
        self.set_stereo(0)
        
        self.status_label.setText("Applied preset: Stereo on Track 1 only")
    
    def preset_mono_all(self):
        """Preset: Input 1 to all tracks"""
        for combo in self.channel_selectors:
            combo.set_checked_inputs([0])
            
        self.status_label.setText("Applied preset: Input 1 to all channels")
    
    def preset_stereo_all(self):
        """Preset: Stereo to all tracks"""
        for i in range(4):
            self.set_stereo(i)
            
        self.status_label.setText("Applied preset: Stereo on all tracks")
    
    def preset_different_tracks(self):
        """Preset: Input 1 to tracks 1 & 3, Input 2 to tracks 2 & 4"""
        # Track 1: both channels Input 1
        self.set_mono(0, 0)
        
        # Track 2: both channels Input 2
        self.set_mono(1, 1)
        
        # Track 3: both channels Input 1
        self.set_mono(2, 0)
        
        # Track 4: both channels Input 2
        self.set_mono(3, 1)
        
        self.status_label.setText("Applied preset: Input 1 to tracks 1 & 3, Input 2 to tracks 2 & 4")


# ============== ENHANCED VISUALIZATION COMPONENTS ==============

class SpectrogramWaterfall3D(QtWidgets.QWidget):
    """3D spectrogram waterfall: frequency x time x amplitude with rich color mapping."""

    HISTORY_ROWS = 48       # ~2.4 seconds of history at 20fps
    FFT_BINS = 64           # frequency resolution
    COLOR_STEPS = 256       # color lookup table size

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)

        # Spectrogram history buffer (newest row = index 0)
        self.history = np.zeros((self.HISTORY_ROWS, self.FFT_BINS), dtype=np.float32)
        self.smooth_history = np.zeros_like(self.history)

        # Precompute color lookup table: blue → cyan → green → yellow → orange → red → white
        self.color_lut = self._build_color_lut()
        # Numpy RGBA LUT for vectorized image rendering (no per-pixel Python calls)
        self._color_lut_np = np.zeros((self.COLOR_STEPS, 4), dtype=np.uint8)
        for i, c in enumerate(self.color_lut):
            self._color_lut_np[i] = [c.red(), c.green(), c.blue(), 255]
        self._img_buffer = None  # prevent GC of QImage backing memory
        self._bg_color = QtGui.QColor(8, 8, 15)  # spectrogram background

        # Perspective parameters
        self.vanish_x = 0.5    # vanishing point X (fraction of width)
        self.vanish_y = 0.08   # vanishing point Y (fraction of height)
        self.front_y = 0.95    # front edge Y position
        self.depth_shrink = 0.35  # how much the back row shrinks horizontally

        # FFT window
        self.window = np.hanning(512).astype(np.float32)

        # Cached pixmap for rendering (avoids per-frame polygon draws)
        self._pixmap = None

        # Animation timer — 30fps is plenty for visual smoothness
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.timeout.connect(self._animate)
        self.update_timer.start(33)  # 30fps

    def _build_color_lut(self):
        """Build a 256-entry color lookup table: silence→blue→cyan→green→yellow→red→white."""
        lut = []
        for i in range(self.COLOR_STEPS):
            t = i / (self.COLOR_STEPS - 1)  # 0.0 → 1.0
            if t < 0.02:
                # Near-silence: very dark blue
                r, g, b = 5, 5, 20
            elif t < 0.15:
                # Deep blue to cyan
                p = (t - 0.02) / 0.13
                r = int(5 + p * 0)
                g = int(5 + p * 180)
                b = int(20 + p * 215)
            elif t < 0.35:
                # Cyan to green
                p = (t - 0.15) / 0.20
                r = int(5 + p * 10)
                g = int(185 + p * 55)
                b = int(235 - p * 185)
            elif t < 0.55:
                # Green to yellow
                p = (t - 0.35) / 0.20
                r = int(15 + p * 240)
                g = int(240 + p * 15)
                b = int(50 - p * 40)
            elif t < 0.75:
                # Yellow to orange-red
                p = (t - 0.55) / 0.20
                r = 255
                g = int(255 - p * 175)
                b = int(10 + p * 5)
            elif t < 0.92:
                # Red to hot pink
                p = (t - 0.75) / 0.17
                r = 255
                g = int(80 - p * 40)
                b = int(15 + p * 80)
            else:
                # Hot to white bloom
                p = (t - 0.92) / 0.08
                r = 255
                g = int(40 + p * 215)
                b = int(95 + p * 160)
            lut.append(QtGui.QColor(min(255, r), min(255, g), min(255, b)))
        return lut

    def _amp_color(self, amp):
        """Map amplitude (0.0-1.0) to a QColor via the LUT."""
        idx = int(max(0.0, min(1.0, amp)) * (self.COLOR_STEPS - 1))
        return self.color_lut[idx]

    def set_color_lut(self, lut, bg_rgb=None):
        """Replace the color LUT (e.g. when switching themes).

        Args:
            lut: list[QColor] of length COLOR_STEPS.
            bg_rgb: optional (r,g,b) tuple for the spectrogram background.
        """
        self.color_lut = lut
        self._color_lut_np = np.zeros((len(lut), 4), dtype=np.uint8)
        for i, c in enumerate(lut):
            self._color_lut_np[i] = [c.red(), c.green(), c.blue(), 255]
        if bg_rgb is not None:
            self._bg_color = QtGui.QColor(*bg_rgb)

    def update_audio_data(self, left_data, right_data):
        """Perform FFT on incoming audio and push a new row into history."""
        # Mix to mono for the spectrogram
        combined = (left_data + right_data) * 0.5
        n = min(len(combined), len(self.window))
        windowed = combined[:n] * self.window[:n]

        # FFT
        spectrum = np.abs(fft(windowed))[:n // 2]
        if len(spectrum) == 0:
            return

        # Log-scale and normalize
        spectrum = np.maximum(spectrum, 1e-10)
        spectrum = np.log10(spectrum) * 0.25 + 0.6  # shift into visible range
        spectrum = np.clip(spectrum, 0.0, 1.0)

        # Resample to FFT_BINS using log-frequency mapping
        log_bins = np.logspace(np.log10(1), np.log10(len(spectrum)), self.FFT_BINS, dtype=int)
        log_bins = np.clip(log_bins - 1, 0, len(spectrum) - 1)
        row = spectrum[log_bins]

        # Scroll history and insert new row at front
        self.history[1:] = self.history[:-1]
        self.history[0] = row

    def _animate(self):
        """Smooth animation toward target history, render to offscreen pixmap."""
        self.smooth_history += (self.history - self.smooth_history) * 0.35
        self._render_to_pixmap()
        self.update()

    def _render_to_pixmap(self):
        """Render spectrogram with 3D height ridges using numpy image + lightweight QPainter.

        1. Color mapping in numpy (releases GIL) → 48 drawImage strip calls
        2. Height-displaced ridge polygons → ~48 drawPolygon calls (1 per row)
        3. Bright ridge lines for definition → ~48 drawPolyline calls
        Total: ~144 QPainter calls vs. the original ~1500+ polygon draws.
        """
        w = self.width()
        h = self.height()
        if w < 10 or h < 10:
            return

        num_rows = self.HISTORY_ROWS
        num_bins = self.FFT_BINS

        # --- Step 1: Build spectrogram image in numpy (vectorized, GIL-free) ---
        indices = (self.smooth_history * 255).clip(0, 255).astype(np.uint8)
        img_rgba = self._color_lut_np[indices]  # (num_rows, num_bins, 4)

        alpha_vals = np.linspace(255, 100, num_rows).astype(np.uint8)
        img_rgba[:, :, 3] = alpha_vals.reshape(-1, 1)

        # Flip: QImage row 0 = oldest (back/top), row N-1 = newest (front/bottom)
        img_flipped = np.ascontiguousarray(img_rgba[::-1])
        self._img_buffer = img_flipped  # prevent GC of QImage backing memory

        bytes_per_line = num_bins * 4
        qimg = QtGui.QImage(self._img_buffer.data, num_bins, num_rows,
                            bytes_per_line, QtGui.QImage.Format_RGBA8888)

        # --- Step 2: Render perspective strips + 3D height ridges ---
        pixmap = QtGui.QPixmap(w, h)
        painter = QtGui.QPainter(pixmap)
        painter.fillRect(0, 0, w, h, self._bg_color)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        vx = w * self.vanish_x
        vy = h * self.vanish_y
        front_y = h * self.front_y

        # Precompute y positions for strip boundaries
        y_positions = []
        for i in range(num_rows + 1):
            t = 1.0 - i / num_rows
            y_positions.append(front_y + (vy - front_y) * t)

        # Draw each row back-to-front: color strip + height polygon + ridge line
        for strip_idx in range(num_rows):
            t_center = 1.0 - (strip_idx + 0.5) / num_rows
            y_top = y_positions[strip_idx]
            y_bottom = y_positions[strip_idx + 1]

            strip_left = vx * t_center * self.depth_shrink
            strip_right = w + (vx - w) * t_center * self.depth_shrink
            strip_width = strip_right - strip_left

            # (a) Flat color strip — rich per-bin coloring from numpy image
            src_rect = QtCore.QRectF(0, strip_idx, num_bins, 1)
            dst_rect = QtCore.QRectF(strip_left, y_top, strip_width, y_bottom - y_top)
            painter.drawImage(dst_rect, qimg, src_rect)

            # (b) Height-displaced ridge polygon — 3D terrain effect
            hist_idx = num_rows - 1 - strip_idx
            row_data = self.smooth_history[hist_idx]
            max_height = h * 0.22 * (1.0 - t_center * 0.75)
            alpha = int(220 * (1.0 - t_center * 0.7))

            if max_height < 2:
                continue

            # Fewer vertices for distant rows
            n_pts = 16 if t_center > 0.5 else 32
            step = max(1, num_bins // n_pts)

            top_points = []
            for b in range(0, num_bins, step):
                frac = b / num_bins
                bx = strip_left + frac * strip_width
                by = y_top - row_data[b] * max_height
                top_points.append(QtCore.QPointF(bx, by))
            top_points.append(QtCore.QPointF(strip_right,
                                             y_top - row_data[-1] * max_height))

            # Bottom edge closes the polygon at strip baseline
            bottom_points = [QtCore.QPointF(strip_right, y_top),
                             QtCore.QPointF(strip_left, y_top)]

            poly = QtGui.QPolygonF(top_points + bottom_points)

            # Fill with row's average color (semi-transparent)
            avg_amp = float(row_data.mean())
            cidx = min(255, int(min(1.0, avg_amp * 1.5) * 255))
            c = self.color_lut[cidx]
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(c.red(), c.green(), c.blue(), alpha // 2))
            painter.drawPolygon(poly)

            # (c) Bright ridge line on top for definition
            lr = min(255, c.red() + 50)
            lg = min(255, c.green() + 50)
            lb = min(255, c.blue() + 50)
            pen = QtGui.QPen(QtGui.QColor(lr, lg, lb, int(alpha * 0.85)))
            pen.setWidthF(max(0.5, 1.8 * (1.0 - t_center)))
            painter.setPen(pen)
            painter.drawPolyline(QtGui.QPolygonF(top_points))

        # Frequency labels along the front edge
        painter.setPen(QtGui.QColor(180, 180, 180, 200))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        freq_labels = [("20", 0.0), ("100", 0.18), ("500", 0.40),
                       ("1k", 0.55), ("5k", 0.75), ("10k", 0.87), ("20k", 0.98)]
        label_y = int(front_y + 2)
        for text, frac in freq_labels:
            lx = int(frac * w)
            painter.drawText(lx - 10, label_y, 30, 16, QtCore.Qt.AlignCenter, text)
            painter.drawLine(lx, int(front_y) - 3, lx, int(front_y) + 2)

        painter.end()
        self._pixmap = pixmap

    def paintEvent(self, event):
        """Just blit the pre-rendered pixmap — keeps paintEvent fast."""
        if self._pixmap:
            painter = QtGui.QPainter(self)
            painter.drawPixmap(0, 0, self._pixmap)
            painter.end()


class EnhancedMultiLevelMeter(QtWidgets.QWidget):
    """Enhanced widget for visualizing multiple channel levels with animations and clipping detection"""
    def __init__(self, parent=None, channels=8):
        super().__init__(parent)
        self.setMinimumSize(20 * channels, 200)
        self.levels = [0.0] * channels
        self.target_levels = [0.0] * channels  # Target for animation
        self.peak_levels = [0.0] * channels  # Peak hold values
        self.clip_indicators = [0] * channels  # Clip detection (frames remaining)
        self.clip_duration = 60  # Frames to show clip indicator (1 second at 60fps)
        self.num_channels = channels
        self.tick_levels = [0, -3, -6, -12, -24, -40, -60]

        # Theme-driven meter colors (defaults match Dark theme)
        dark = THEMES['Dark']
        self.meter_zones = dark['meter_zones']
        self.meter_peak_zones = dark['meter_peak_zones']
        self._meter_bg = QtGui.QColor(28, 28, 30)
        self._meter_outer_bg = QtGui.QColor(35, 35, 37)
        
        # Animation timer
        self.animation_timer = QtCore.QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(16)  # ~60fps
        
        # Peak decay timer
        self.peak_decay_timer = QtCore.QTimer(self)
        self.peak_decay_timer.timeout.connect(self.decay_peaks)
        self.peak_decay_timer.start(150)  # Slower decay
    
    def set_channels(self, num_channels):
        """Update the number of channels shown"""
        self.num_channels = min(num_channels, 8)  # Limit to 8 channels max
        self.setMinimumSize(20 * self.num_channels, 200)
        self.update()

    def set_meter_zones(self, zones, peak_zones, palette):
        """Update meter colors from a theme.

        Args:
            zones: list of (threshold, (top_r,g,b), (bottom_r,g,b))
            peak_zones: list of (threshold, (r,g,b))
            palette: theme palette dict (for meter_bg)
        """
        self.meter_zones = zones
        self.meter_peak_zones = peak_zones
        bg = palette.get('meter_bg', '#1C1C1E')
        r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
        self._meter_bg = QtGui.QColor(r + 6, g + 6, b + 8)
        self._meter_outer_bg = QtGui.QColor(r + 13, g + 13, b + 15)
        self.update()

    def set_levels(self, levels):
        """Set the target levels for all channels (0.0 to 1.0)"""
        for i in range(min(len(levels), self.num_channels)):
            val = min(max(levels[i], 0.0), 1.0)
            self.target_levels[i] = val
            
            # Update peaks
            if val > self.peak_levels[i]:
                self.peak_levels[i] = val
                
            # Check for clipping
            if val > 0.95:  # Near clipping
                self.clip_indicators[i] = self.clip_duration
    
    def update_animation(self):
        """Update animation frame - smooth transition to target levels"""
        changed = False
        
        for i in range(self.num_channels):
            # Move current level towards target level with easing
            diff = self.target_levels[i] - self.levels[i]
            
            if abs(diff) > 0.001:
                # Faster rise, slower fall for natural meter behavior
                if diff > 0:
                    self.levels[i] += diff * 0.3  # Fast rise
                else:
                    self.levels[i] += diff * 0.1  # Slow fall
                changed = True
                
            # Handle clip indicators
            if self.clip_indicators[i] > 0:
                self.clip_indicators[i] -= 1
                changed = True
                
        if changed:
            self.update()
    
    def decay_peaks(self):
        """Decay peak levels over time"""
        changed = False
        for i in range(self.num_channels):
            if self.peak_levels[i] > 0:
                self.peak_levels[i] *= 0.95  # Gradual decay
                if self.peak_levels[i] < 0.01:
                    self.peak_levels[i] = 0
                changed = True
                
        if changed:
            self.update()
            
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        meter_width = width // self.num_channels if self.num_channels > 0 else width
        
        # Draw background for all meters
        painter.fillRect(0, 0, width, height, self._meter_outer_bg)

        # Draw each channel meter
        for ch in range(self.num_channels):
            level = self.levels[ch]
            peak_level = self.peak_levels[ch]
            level_height = int(height * level)
            peak_height = int(height * peak_level)
            x = ch * meter_width

            # Draw meter gradient background
            painter.fillRect(
                x, 0, meter_width - 1, height,
                self._meter_bg
            )

            # Track/channel identifier at bottom
            track_num = ch // 2 + 1
            ch_num = ch % 2 + 1

            # Draw rounded meter with zone-driven gradient
            if level_height > 0:
                gradient = QtGui.QLinearGradient(
                    x, height - level_height,
                    x, height
                )

                # Pick colors from self.meter_zones
                top_rgb = self.meter_zones[-1][1]
                bot_rgb = self.meter_zones[-1][2]
                for threshold, t_rgb, b_rgb in self.meter_zones:
                    if level < threshold:
                        top_rgb, bot_rgb = t_rgb, b_rgb
                        break
                gradient.setColorAt(0, QtGui.QColor(*top_rgb))
                gradient.setColorAt(1, QtGui.QColor(*bot_rgb))

                # Fill rounded meter
                meter_rect = QtCore.QRectF(
                    x + 1, height - level_height,
                    meter_width - 2, level_height
                )
                painter.setBrush(gradient)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawRoundedRect(meter_rect, 2, 2)

            # Draw peak indicator with zone-driven colors
            if peak_height > 0:
                peak_rgb = self.meter_peak_zones[-1][1]
                for threshold, p_rgb in self.meter_peak_zones:
                    if peak_level < threshold:
                        peak_rgb = p_rgb
                        break

                # Clip blink override
                if peak_level >= 0.95 and self.clip_indicators[ch] > 0:
                    frame = self.clip_indicators[ch]
                    if frame % 10 >= 5:
                        peak_rgb = (min(255, peak_rgb[0]), min(255, peak_rgb[1] + 170), min(255, peak_rgb[2] + 170))

                peak_color = QtGui.QColor(*peak_rgb)
                
                peak_rect = QtCore.QRectF(
                    x + 1, height - peak_height - 2, 
                    meter_width - 2, 3  # Slightly thicker peak indicator
                )
                
                # Draw with slight glow effect
                if peak_level > 0.7:  # Add glow to higher peaks
                    glow_rect = QtCore.QRectF(
                        x + 1, height - peak_height - 3, 
                        meter_width - 2, 5
                    )
                    glow_color = QtGui.QColor(peak_color)
                    glow_color.setAlpha(100)
                    painter.setBrush(glow_color)
                    painter.drawRoundedRect(glow_rect, 2, 2)
                
                painter.fillRect(peak_rect, peak_color)
            
            # Draw simplified tick marks/grid so the meter looks cleaner
            grid_color = QtGui.QColor(75, 75, 80, 150)
            painter.setPen(grid_color)
            for db in self.tick_levels:
                linear = 10 ** (db / 20)
                if linear <= 0:
                    continue
                y = height - (linear * height)
                tick_len = meter_width - 6 if db >= -12 else max(meter_width // 2, 8)
                painter.drawLine(x + 3, int(y), x + 3 + tick_len, int(y))
            
            # Draw track/channel label 
            painter.setPen(QtGui.QColor(180, 180, 180))
            painter.setFont(QtGui.QFont("Arial", 8))
            
            # Format as Track.Channel
            label = f"{track_num}.{ch_num}"
            text_rect = QtCore.QRectF(x, height - 18, meter_width, 16)
            painter.drawText(text_rect, QtCore.Qt.AlignCenter, label)
            
            # Also draw track number at top
            if ch % 2 == 0:  # Only for left channels
                top_rect = QtCore.QRectF(x, 2, meter_width * 2, 16)
                painter.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))
                
                # Color the track label based on channel activity
                track_activity = max(self.levels[ch], self.levels[ch+1]) if ch+1 < self.num_channels else self.levels[ch]
                
                if track_activity > 0.7:
                    # Active track - use color based on level
                    if track_activity > 0.95:
                        painter.setPen(QtGui.QColor(255, 100, 100))  # Red for high levels
                    elif track_activity > 0.85:
                        painter.setPen(QtGui.QColor(255, 180, 0))  # Orange for medium-high
                    else:
                        painter.setPen(QtGui.QColor(220, 220, 0))  # Yellow for medium
                else:
                    # Normal or inactive track
                    painter.setPen(QtGui.QColor(150, 150, 150))
                
                painter.drawText(top_rect, QtCore.Qt.AlignCenter, f"Track {track_num}")
                
            # Draw clip indicator text at top of meter if clipping
            if self.clip_indicators[ch] > 0:
                painter.setFont(QtGui.QFont("Arial", 7, QtGui.QFont.Bold))
                
                # Alternate colors for attention
                if self.clip_indicators[ch] % 10 < 5:
                    painter.setPen(QtGui.QColor(255, 50, 50))  # Red
                else:
                    painter.setPen(QtGui.QColor(255, 255, 255))  # White
                
                clip_rect = QtCore.QRectF(x, 25, meter_width, 16)
                painter.drawText(clip_rect, QtCore.Qt.AlignCenter, "CLIP")


class PatchbayView(QtWidgets.QWidget):
    """Visual patchbay matrix: inputs on rows, output channels on columns.

    Click intersections to patch. Scroll wheel to adjust gain.
    Fully custom-painted for a polished, professional look.
    """

    mapping_changed = Signal()

    # Minimum layout constants (used when space allows; shrink for many inputs)
    HEADER_WIDTH_MIN = 48
    HEADER_HEIGHT_MIN = 36
    ROW_HEIGHT_MIN = 18
    ROW_HEIGHT_MAX = 30
    COL_WIDTH_MIN = 30
    COL_WIDTH_MAX = 42
    NUM_OUTPUTS = 8  # 4 tracks x 2 channels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 150)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)

        self._track_colors = THEMES['Dark']['palette']['track_colors']
        self.input_infos = [{'label': 'A', 'channels': 2}]
        self.input_labels = ['A1', 'A2']
        self.input_channel_count = 2

        # Connections: (input_idx, output_ch) -> gain float 0.0-1.0
        self._connections = {}

        # Hover state
        self._hover_row = -1
        self._hover_col = -1

        # Status label (API compat with other views)
        self.status_label = QtWidgets.QLabel("")

        # Build layout: matrix + preset buttons + status
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(4, 4, 4, 4)

        # The matrix is drawn in paintEvent, we just need the widget area
        self._matrix_widget = _PatchbayMatrix(self)
        layout.addWidget(self._matrix_widget, stretch=1)

        # Preset buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        for label, slot in [
            ("Clear All", self._preset_clear),
            ("Auto Stereo", self._preset_auto_stereo),
            ("Mono All", self._preset_mono_all),
        ]:
            btn = QtWidgets.QPushButton(label)
            btn.setProperty("preset", True)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)
        layout.addWidget(self.status_label)

    def _metrics(self, widget_w, widget_h):
        """Compute dynamic layout metrics that scale to fit all inputs.

        Returns dict with: HW, HH, RH, CW, node_max_r, node_min_r
        """
        n_inputs = max(1, self.input_channel_count)
        n_outputs = self.NUM_OUTPUTS

        # Header sizes — scale down for tight layouts
        HW = max(self.HEADER_WIDTH_MIN, min(56, int(widget_w * 0.1)))
        HH = max(self.HEADER_HEIGHT_MIN, min(44, int(widget_h * 0.08)))

        # Available area for the grid
        avail_h = widget_h - HH - 4
        avail_w = widget_w - HW - 4

        # Row height: fit all inputs, clamped to min/max
        RH = max(self.ROW_HEIGHT_MIN, min(self.ROW_HEIGHT_MAX, avail_h // n_inputs))
        # Column width: fit 8 outputs, clamped to min/max
        CW = max(self.COL_WIDTH_MIN, min(self.COL_WIDTH_MAX, avail_w // n_outputs))

        # Node radius scales with cell size
        cell_min = min(RH, CW)
        node_max_r = max(4, int(cell_min * 0.38))
        node_min_r = max(2, int(node_max_r * 0.4))

        return {
            'HW': HW, 'HH': HH, 'RH': RH, 'CW': CW,
            'node_max_r': node_max_r, 'node_min_r': node_min_r,
        }

    # -- Input management --------------------------------------------------

    def update_for_inputs(self, input_infos):
        self.input_infos = input_infos
        labels = []
        for info in input_infos:
            for ch in range(info['channels']):
                labels.append(f"{info['label']}{ch + 1}")
        self.input_channel_count = len(labels)
        self.input_labels = labels
        # Prune connections that reference removed inputs
        to_remove = [k for k in self._connections if k[0] >= self.input_channel_count]
        for k in to_remove:
            del self._connections[k]
        self._matrix_widget.update()

    # -- Mapping API (same format as MixerStripView / TrackMapperWidget) ---

    def get_mapping(self):
        mapping = []
        for out_ch in range(8):
            sources = []
            for (inp, och), gain in self._connections.items():
                if och == out_ch and gain > 0:
                    sources.append({'index': inp, 'gain': round(gain, 3)})
            mapping.append(sources)
        return mapping

    def set_mapping(self, mapping):
        self._connections.clear()
        for out_ch, sources in enumerate(mapping):
            if not isinstance(sources, list):
                continue
            for src in sources:
                if isinstance(src, dict):
                    idx = src.get('index')
                    gain = src.get('gain', 1.0)
                    if isinstance(idx, int) and idx >= 0 and gain > 0:
                        self._connections[(idx, out_ch)] = gain
        self._matrix_widget.update()

    # -- Presets -----------------------------------------------------------

    def _preset_clear(self):
        self._connections.clear()
        self._matrix_widget.update()
        self.mapping_changed.emit()
        self.status_label.setText("Cleared all connections")

    def _preset_auto_stereo(self):
        self._connections.clear()
        ch_offset = 0
        out_ch = 0
        for info in self.input_infos:
            pairs = info['channels'] // 2
            for p in range(pairs):
                if out_ch + 1 < 8:
                    self._connections[(ch_offset + p * 2, out_ch)] = 1.0
                    self._connections[(ch_offset + p * 2 + 1, out_ch + 1)] = 1.0
                    out_ch += 2
            ch_offset += info['channels']
        self._matrix_widget.update()
        self.mapping_changed.emit()
        self.status_label.setText("Auto stereo routing applied")

    def _preset_mono_all(self):
        self._connections.clear()
        ch_offset = 0
        out_ch = 0
        for info in self.input_infos:
            for ch in range(info['channels']):
                if out_ch < 8:
                    self._connections[(ch_offset + ch, out_ch)] = 1.0
                    out_ch += 1
            ch_offset += info['channels']
        self._matrix_widget.update()
        self.mapping_changed.emit()
        self.status_label.setText("Mono routing applied")

    def auto_assign_devices(self, input_infos, num_output_tracks=4):
        """Auto-assign each device to its own track (same logic as TrackMapperWidget)."""
        self._connections.clear()
        num_tracks = min(4, max(1, num_output_tracks))
        ch_offset = 0
        for dev_idx, info in enumerate(input_infos):
            track_idx = dev_idx % num_tracks
            out_l = track_idx * 2
            out_r = track_idx * 2 + 1
            self._connections[(ch_offset, out_l)] = 1.0
            if info['channels'] >= 2:
                self._connections[(ch_offset + 1, out_r)] = 1.0
            else:
                self._connections[(ch_offset, out_r)] = 1.0
            ch_offset += info['channels']
        self._matrix_widget.update()
        self.mapping_changed.emit()

    # -- Theme support -----------------------------------------------------

    def get_track_color(self, track_idx):
        return self._track_colors[track_idx % len(self._track_colors)]

    def _refresh_track_colors(self):
        self._matrix_widget.update()

    def _device_color_for_input(self, input_idx):
        """Get the device color for a given global input index."""
        offset = 0
        for i, info in enumerate(self.input_infos):
            if input_idx < offset + info['channels']:
                return self._track_colors[i % len(self._track_colors)]
            offset += info['channels']
        return self._track_colors[0]

    def _device_index_for_input(self, input_idx):
        """Get which device (0, 1, 2) owns a global input index."""
        offset = 0
        for i, info in enumerate(self.input_infos):
            if input_idx < offset + info['channels']:
                return i
            offset += info['channels']
        return 0


class _PatchbayMatrix(QtWidgets.QWidget):
    """Inner widget that handles painting and mouse interaction for PatchbayView."""

    def __init__(self, patchbay):
        super().__init__(patchbay)
        self.pb = patchbay
        self.setMouseTracking(True)
        self.setMinimumSize(300, 100)
        self._hover_row = -1
        self._hover_col = -1
        self._gain_popup = None

    def _m(self):
        """Get current dynamic metrics."""
        return self.pb._metrics(self.width(), self.height())

    def _hit_test(self, pos):
        """Return (row, col) for a mouse position, or (-1,-1) if outside grid."""
        m = self._m()
        x = pos.x() - m['HW']
        y = pos.y() - m['HH']
        if x < 0 or y < 0:
            return -1, -1
        col = int(x / m['CW'])
        row = int(y / m['RH'])
        if row >= self.pb.input_channel_count or col >= self.pb.NUM_OUTPUTS:
            return -1, -1
        return row, col

    # -- Mouse events ------------------------------------------------------

    def mouseMoveEvent(self, event):
        row, col = self._hit_test(event.pos())
        if row != self._hover_row or col != self._hover_col:
            self._hover_row = row
            self._hover_col = col
            self.update()

    def leaveEvent(self, event):
        self._hover_row = -1
        self._hover_col = -1
        self.update()

    def mousePressEvent(self, event):
        row, col = self._hit_test(event.pos())
        if row < 0 or col < 0:
            return

        key = (row, col)
        if event.button() == QtCore.Qt.LeftButton:
            if key in self.pb._connections:
                del self.pb._connections[key]
            else:
                self.pb._connections[key] = 1.0
            self.update()
            self.pb.mapping_changed.emit()
        elif event.button() == QtCore.Qt.RightButton:
            if key in self.pb._connections:
                self._show_gain_popup(key, event.globalPos())

    def wheelEvent(self, event):
        pos = event.position().toPoint() if hasattr(event.position(), 'toPoint') else event.pos()
        row, col = self._hit_test(pos)
        if row < 0 or col < 0:
            return
        key = (row, col)
        if key not in self.pb._connections:
            return
        delta = event.angleDelta().y()
        step = 0.05 if delta > 0 else -0.05
        new_gain = max(0.05, min(1.0, self.pb._connections[key] + step))
        self.pb._connections[key] = round(new_gain, 2)
        self.update()
        self.pb.mapping_changed.emit()

    def _show_gain_popup(self, key, global_pos):
        """Show a small popup with a gain slider for the given connection."""
        if self._gain_popup:
            self._gain_popup.close()
        popup = QtWidgets.QWidget(self, QtCore.Qt.Popup)
        popup.setFixedSize(160, 50)
        lay = QtWidgets.QHBoxLayout(popup)
        lay.setContentsMargins(8, 4, 8, 4)
        lbl = QtWidgets.QLabel(f"{int(self.pb._connections[key] * 100)}%")
        lbl.setFixedWidth(36)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(5, 100)
        slider.setValue(int(self.pb._connections[key] * 100))

        def on_change(val):
            self.pb._connections[key] = val / 100.0
            lbl.setText(f"{val}%")
            self.update()
            self.pb.mapping_changed.emit()

        slider.valueChanged.connect(on_change)
        lay.addWidget(slider)
        lay.addWidget(lbl)
        popup.move(global_pos)
        popup.show()
        self._gain_popup = popup

    # -- Painting ----------------------------------------------------------

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        num_inputs = self.pb.input_channel_count
        num_outputs = self.pb.NUM_OUTPUTS

        m = self.pb._metrics(w, h)
        HW, HH, RH, CW = m['HW'], m['HH'], m['RH'], m['CW']
        node_max_r = m['node_max_r']
        node_min_r = m['node_min_r']

        # Adaptive font sizes based on cell height
        font_large = max(7, min(9, RH // 3))
        font_small = max(6, font_large - 1)
        font_tiny = max(5, font_large - 2)

        # Get theme colors
        mw = self.window()
        theme_name = getattr(mw, '_current_theme', 'Dark')
        theme = THEMES.get(theme_name, THEMES['Dark'])
        pal = theme['palette']

        bg = QtGui.QColor(pal['bg_secondary'])
        bg_alt = QtGui.QColor(pal['bg_tertiary'])
        border_color = QtGui.QColor(pal['border'])
        text_dim = QtGui.QColor(pal['text_dim'])

        # 1. Fill background
        painter.fillRect(0, 0, w, h, bg)

        # 2. Alternating track-pair column bands
        for track in range(4):
            x = HW + track * 2 * CW
            band_color = bg_alt if track % 2 == 0 else bg
            painter.fillRect(
                QtCore.QRectF(x, HH, CW * 2, num_inputs * RH),
                band_color
            )

        # 3. Crosshair highlight for hovered cell
        if self._hover_row >= 0 and self._hover_col >= 0:
            cross_color = QtGui.QColor(pal['accent'])
            cross_color.setAlpha(25)
            painter.fillRect(
                QtCore.QRectF(HW, HH + self._hover_row * RH, num_outputs * CW, RH),
                cross_color
            )
            painter.fillRect(
                QtCore.QRectF(HW + self._hover_col * CW, HH, CW, num_inputs * RH),
                cross_color
            )

        # 4. Grid lines
        grid_pen = QtGui.QPen(border_color, 0.5)
        painter.setPen(grid_pen)
        for row in range(num_inputs + 1):
            y = HH + row * RH
            painter.drawLine(QtCore.QPointF(HW, y), QtCore.QPointF(HW + num_outputs * CW, y))
        for col in range(num_outputs + 1):
            x = HW + col * CW
            painter.drawLine(QtCore.QPointF(x, HH), QtCore.QPointF(x, HH + num_inputs * RH))

        # 5. Device separator lines (thicker)
        sep_pen = QtGui.QPen(border_color, 2.0)
        offset = 0
        for info in self.pb.input_infos:
            if offset > 0:
                y = HH + offset * RH
                painter.setPen(sep_pen)
                painter.drawLine(QtCore.QPointF(0, y), QtCore.QPointF(HW + num_outputs * CW, y))
            offset += info['channels']

        # 6. Track pair separator lines (thicker vertical)
        for track in range(1, 4):
            x = HW + track * 2 * CW
            painter.setPen(sep_pen)
            painter.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, HH + num_inputs * RH))

        # 7. Column headers — Track name + L/R
        painter.setFont(QtGui.QFont("Arial", font_large, QtGui.QFont.Bold))
        for track in range(4):
            tc = QtGui.QColor(self.pb.get_track_color(track))
            x = HW + track * 2 * CW
            painter.setPen(tc)
            track_rect = QtCore.QRectF(x, 1, CW * 2, HH * 0.55)
            painter.drawText(track_rect, QtCore.Qt.AlignCenter, f"Track {track + 1}")
            painter.setFont(QtGui.QFont("Arial", font_small))
            painter.setPen(text_dim)
            l_rect = QtCore.QRectF(x, HH * 0.5, CW, HH * 0.5)
            r_rect = QtCore.QRectF(x + CW, HH * 0.5, CW, HH * 0.5)
            painter.drawText(l_rect, QtCore.Qt.AlignCenter, "L")
            painter.drawText(r_rect, QtCore.Qt.AlignCenter, "R")
            painter.setFont(QtGui.QFont("Arial", font_large, QtGui.QFont.Bold))

        # 8. Row headers — input channel labels
        painter.setFont(QtGui.QFont("Arial", font_large, QtGui.QFont.Bold))
        for row in range(num_inputs):
            label = self.pb.input_labels[row] if row < len(self.pb.input_labels) else f"?{row}"
            color = QtGui.QColor(self.pb._device_color_for_input(row))
            painter.setPen(color)
            rect = QtCore.QRectF(2, HH + row * RH, HW - 4, RH)
            painter.drawText(rect, QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, label)

        # 9. Connection nodes
        for row in range(num_inputs):
            for col in range(num_outputs):
                key = (row, col)
                cx = HW + col * CW + CW / 2
                cy = HH + row * RH + RH / 2
                is_hover = (row == self._hover_row and col == self._hover_col)
                gain = self.pb._connections.get(key, 0)

                if gain > 0:
                    node_color = QtGui.QColor(self.pb._device_color_for_input(row))
                    radius = node_min_r + (node_max_r - node_min_r) * gain

                    # Glow
                    glow_color = QtGui.QColor(node_color)
                    glow_color.setAlpha(60)
                    glow_r = radius + max(2, node_max_r * 0.3)
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.setBrush(glow_color)
                    painter.drawEllipse(QtCore.QPointF(cx, cy), glow_r, glow_r)

                    # Filled node
                    painter.setBrush(node_color)
                    painter.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

                    # Gain text
                    if (gain < 0.99 or is_hover) and radius >= 6:
                        painter.setPen(QtGui.QColor(255, 255, 255, 200))
                        painter.setFont(QtGui.QFont("Arial", font_tiny, QtGui.QFont.Bold))
                        txt = f"{int(gain * 100)}"
                        painter.drawText(
                            QtCore.QRectF(cx - 14, cy - 6, 28, 12),
                            QtCore.Qt.AlignCenter, txt
                        )
                        painter.setFont(QtGui.QFont("Arial", font_large, QtGui.QFont.Bold))

                    # Hover ring
                    if is_hover:
                        ring_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 180), 1.5)
                        painter.setPen(ring_pen)
                        painter.setBrush(QtCore.Qt.NoBrush)
                        painter.drawEllipse(QtCore.QPointF(cx, cy), radius + 2, radius + 2)

                elif is_hover:
                    dot_color = QtGui.QColor(pal['text_dim'])
                    dot_color.setAlpha(80)
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.setBrush(dot_color)
                    dot_r = max(2, node_min_r * 0.7)
                    painter.drawEllipse(QtCore.QPointF(cx, cy), dot_r, dot_r)

                    ring_color = QtGui.QColor(pal['accent'])
                    ring_color.setAlpha(120)
                    ring_pen = QtGui.QPen(ring_color, 1.5)
                    painter.setPen(ring_pen)
                    painter.setBrush(QtCore.Qt.NoBrush)
                    painter.drawEllipse(QtCore.QPointF(cx, cy), node_min_r + 2, node_min_r + 2)

        painter.end()

    def sizeHint(self):
        m = self.pb._metrics(500, 400)
        w = m['HW'] + self.pb.NUM_OUTPUTS * m['CW'] + 10
        h = m['HH'] + max(2, self.pb.input_channel_count) * m['RH'] + 10
        return QtCore.QSize(w, h)


class MixerStripView(QtWidgets.QWidget):
    """Alternative mixer-strip UI: per-track on/off, mono/stereo, volume knob, input selector."""

    mapping_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._track_colors = THEMES['Dark']['palette']['track_colors']
        self.input_infos = [{'label': 'A', 'channels': 2}]
        self.input_labels = ['A1', 'A2']
        self.input_channel_count = 2

        self.strips = []  # list of strip state dicts

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(6)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setStyleSheet("color: #AAAAAA; margin-top: 5px;")

        # Strip container
        self._strip_layout = QtWidgets.QHBoxLayout()
        self._strip_layout.setSpacing(16)
        for i in range(4):
            strip = self._create_strip(i)
            self.strips.append(strip)
            self._strip_layout.addWidget(strip['widget'])

        main_layout.addLayout(self._strip_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addStretch()

    def _create_strip(self, track_idx):
        """Create a single track strip with on/off, mono/stereo, dial, and input selector."""
        color = self._track_colors[track_idx % len(self._track_colors)]
        frame = QtWidgets.QFrame()
        frame.setFrameStyle(QtWidgets.QFrame.StyledPanel)
        frame.setMinimumWidth(130)
        layout = QtWidgets.QVBoxLayout(frame)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # Track title
        title = QtWidgets.QLabel(f"Track {track_idx + 1}")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {color};")
        layout.addWidget(title)

        # On/Off toggle
        on_btn = QtWidgets.QPushButton("ON")
        on_btn.setCheckable(True)
        on_btn.setChecked(True)
        on_btn.setStyleSheet(self._toggle_style(True, color))
        on_btn.toggled.connect(lambda checked, b=on_btn, c=color: self._on_toggle(b, checked, c))
        layout.addWidget(on_btn)

        # Mono/Stereo toggle
        stereo_btn = QtWidgets.QPushButton("STEREO")
        stereo_btn.setCheckable(True)
        stereo_btn.setChecked(True)  # default stereo
        stereo_btn.toggled.connect(lambda checked, b=stereo_btn: self._stereo_toggle(b, checked))
        layout.addWidget(stereo_btn)

        # Volume knob
        vol_label = QtWidgets.QLabel("Volume")
        vol_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(vol_label)

        dial = QtWidgets.QDial()
        dial.setRange(0, 100)
        dial.setValue(100)
        dial.setNotchesVisible(True)
        dial.setFixedSize(80, 80)
        dial.valueChanged.connect(lambda v, lbl=vol_label: lbl.setText(f"Vol: {v}%"))
        dial.valueChanged.connect(lambda: self.mapping_changed.emit())
        dial_container = QtWidgets.QHBoxLayout()
        dial_container.addStretch()
        dial_container.addWidget(dial)
        dial_container.addStretch()
        layout.addLayout(dial_container)

        # Input selector
        input_label = QtWidgets.QLabel("Input:")
        input_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(input_label)

        input_combo = QtWidgets.QComboBox()
        self._populate_input_combo(input_combo)
        input_combo.currentIndexChanged.connect(lambda: self.mapping_changed.emit())
        layout.addWidget(input_combo)

        layout.addStretch()

        return {
            'widget': frame,
            'title': title,
            'on_btn': on_btn,
            'stereo_btn': stereo_btn,
            'dial': dial,
            'vol_label': vol_label,
            'input_combo': input_combo,
            'track_idx': track_idx,
        }

    def _toggle_style(self, is_on, color):
        if is_on:
            return f"background-color: {color}; color: white; font-weight: bold; padding: 6px;"
        return "background-color: #555555; color: #999999; font-weight: bold; padding: 6px;"

    def _on_toggle(self, btn, checked, color):
        btn.setText("ON" if checked else "OFF")
        btn.setStyleSheet(self._toggle_style(checked, color))
        self.mapping_changed.emit()

    def _stereo_toggle(self, btn, checked):
        btn.setText("STEREO" if checked else "MONO")
        self.mapping_changed.emit()

    def _populate_input_combo(self, combo):
        """Populate a single input combo with stereo pairs."""
        combo.clear()
        offset = 0
        for info in self.input_infos:
            dev_label = info['label']
            ch = info['channels']
            num_pairs = max(1, ch // 2)
            for p in range(num_pairs):
                ch_lo = p * 2 + 1
                ch_hi = ch_lo + 1
                combo.addItem(f"{dev_label} {ch_lo}-{ch_hi}", offset + p * 2)
            offset += ch

    def update_for_inputs(self, input_infos):
        """Rebuild input pair combos when input devices change."""
        self.input_infos = input_infos
        labels = []
        for info in input_infos:
            for ch in range(info['channels']):
                labels.append(f"{info['label']}{ch+1}")
        self.input_channel_count = len(labels)
        self.input_labels = labels
        for strip in self.strips:
            combo = strip['input_combo']
            prev = combo.currentData()
            # Block signals during repopulation to prevent spam
            combo.blockSignals(True)
            self._populate_input_combo(combo)
            # Try to restore previous selection
            if prev is not None:
                idx = combo.findData(prev)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def get_mapping(self):
        """Return mapping in the same format as TrackMapperWidget.get_mapping().

        Returns a flat list of 8 entries (4 tracks x 2 channels).
        Each entry is a list of {index, gain} dicts.
        """
        mapping = []
        for strip in self.strips:
            enabled = strip['on_btn'].isChecked()
            is_stereo = strip['stereo_btn'].isChecked()
            volume = strip['dial'].value() / 100.0
            pair_start = strip['input_combo'].currentData()
            if pair_start is None:
                pair_start = 0

            if not enabled:
                mapping.extend([[], []])
                continue

            left_idx = pair_start
            right_idx = pair_start + (1 if is_stereo else 0)
            mapping.append([{'index': left_idx, 'gain': volume}])
            mapping.append([{'index': right_idx, 'gain': volume}])
        return mapping

    def set_mapping(self, mapping):
        """Best-effort restore from a TrackMapperWidget-style mapping."""
        for track_idx in range(4):
            if track_idx >= len(self.strips):
                break
            strip = self.strips[track_idx]
            left_entry = mapping[track_idx * 2] if track_idx * 2 < len(mapping) else []
            right_entry = mapping[track_idx * 2 + 1] if track_idx * 2 + 1 < len(mapping) else []

            has_signal = bool(left_entry) or bool(right_entry)
            strip['on_btn'].setChecked(has_signal)

            if left_entry and isinstance(left_entry, list) and len(left_entry) > 0:
                src = left_entry[0]
                if isinstance(src, dict):
                    idx = src.get('index', 0)
                    gain = src.get('gain', 1.0)
                    strip['dial'].setValue(int(gain * 100))
                    # Find the closest pair start
                    pair_start = (idx // 2) * 2
                    combo_idx = strip['input_combo'].findData(pair_start)
                    if combo_idx >= 0:
                        strip['input_combo'].setCurrentIndex(combo_idx)

                    # Detect mono vs stereo
                    if right_entry and isinstance(right_entry, list) and len(right_entry) > 0:
                        r_src = right_entry[0]
                        if isinstance(r_src, dict):
                            r_idx = r_src.get('index', 0)
                            strip['stereo_btn'].setChecked(r_idx != idx)

    def auto_assign_devices(self, input_infos, num_output_tracks=4):
        """Auto-assign each input device to its own track strip."""
        num_tracks = min(4, max(1, num_output_tracks))
        # Block signals to avoid spamming mapping_changed per widget change
        self.blockSignals(True)
        for s in self.strips:
            s['input_combo'].blockSignals(True)
            s['on_btn'].blockSignals(True)
            s['stereo_btn'].blockSignals(True)
            s['dial'].blockSignals(True)
        try:
            ch_offset = 0
            for dev_idx, info in enumerate(input_infos):
                track_idx = dev_idx % num_tracks
                if track_idx >= len(self.strips):
                    break
                strip = self.strips[track_idx]
                combo_idx = strip['input_combo'].findData(ch_offset)
                if combo_idx >= 0:
                    strip['input_combo'].setCurrentIndex(combo_idx)
                strip['on_btn'].setChecked(True)
                strip['stereo_btn'].setChecked(True)
                strip['dial'].setValue(100)
                ch_offset += info['channels']
        finally:
            for s in self.strips:
                s['input_combo'].blockSignals(False)
                s['on_btn'].blockSignals(False)
                s['stereo_btn'].blockSignals(False)
                s['dial'].blockSignals(False)
            self.blockSignals(False)
        # Emit once after all changes
        self.mapping_changed.emit()

    def get_track_color(self, track_idx):
        return self._track_colors[track_idx % len(self._track_colors)]

    def _refresh_track_colors(self):
        for strip in self.strips:
            i = strip['track_idx']
            color = self.get_track_color(i)
            strip['title'].setStyleSheet(f"font-weight: bold; font-size: 13px; color: {color};")
            if strip['on_btn'].isChecked():
                strip['on_btn'].setStyleSheet(self._toggle_style(True, color))


class MainWindow(QtWidgets.QMainWindow):
    """Main application window with enhanced visualizations"""
    def __init__(self):
        super().__init__()

        self.audio_manager = AudioManager()
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(50)  # 50ms = 20fps
        
        # Update title to reflect enhanced version
        self.setWindowTitle("LF Music Mapper - By: @LofiFren")
        self.setMinimumSize(1400, 820)  # Wider default to keep routing presets readable
        
        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setSpacing(12)
        
        # Theme state
        self._current_theme = 'Dark'

        # Menu bar — View > Theme
        menubar = self.menuBar()
        view_menu = menubar.addMenu("View")
        theme_menu = view_menu.addMenu("Theme")
        self._theme_actions = {}
        theme_group = QtWidgets.QActionGroup(self)
        theme_group.setExclusive(True)
        for name in THEMES:
            action = theme_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == 'Dark')
            theme_group.addAction(action)
            action.triggered.connect(lambda checked, n=name: self.apply_theme(n))
            self._theme_actions[name] = action

        # Title and version
        title_layout = QtWidgets.QHBoxLayout()
        self._title_label = QtWidgets.QLabel("LF Music Mapper")
        self._title_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #EEEEEE;")
        title_layout.addWidget(self._title_label)

        self._version_label = QtWidgets.QLabel("By: @LofiFren")
        self._version_label.setStyleSheet("font-size: 14px; color: #AAAAAA;")
        self._version_label.setAlignment(QtCore.Qt.AlignBottom)
        title_layout.addWidget(self._version_label)

        title_layout.addStretch()

        main_layout.addLayout(title_layout)

        # Inline helper text to replace the old popup dialog
        self._intro_label = QtWidgets.QLabel(
            "Route any available input into dedicated multichannel outputs. "
            "Choose your devices, define the mapping on the left, and monitor your signal on the right."
        )
        self._intro_label.setStyleSheet("color: #BBBBBB; font-size: 12px;")
        self._intro_label.setWordWrap(True)
        main_layout.addWidget(self._intro_label)
        
        # Split the primary UI into routing controls (left) and visualization (right)
        content_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        content_splitter.setObjectName("contentSplitter")
        content_splitter.setChildrenCollapsible(False)
        content_splitter.setHandleWidth(2)
        
        left_panel = QtWidgets.QWidget()
        left_panel.setMinimumWidth(720)  # ensure routing controls have enough width for button text
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        
        # Device selection — symmetric inputs (left) and outputs (right)
        device_group = QtWidgets.QGroupBox("Audio Devices")
        device_layout = QtWidgets.QGridLayout()
        device_layout.setSpacing(10)
        device_layout.setColumnStretch(1, 1)  # input combos
        device_layout.setColumnStretch(3, 1)  # output combos

        # -- Inputs (left) --
        # Input A (always active)
        self._input_a_label = QtWidgets.QLabel("Input A:")
        self._input_a_label.setStyleSheet("font-weight: bold; color: #4B9DE0;")
        device_layout.addWidget(self._input_a_label, 0, 0)
        self.input_combo_a = QtWidgets.QComboBox()
        self.input_combo_a.currentIndexChanged.connect(self.input_device_changed)
        device_layout.addWidget(self.input_combo_a, 0, 1)

        # Input B (optional)
        self._input_b_label = QtWidgets.QLabel("Input B:")
        self._input_b_label.setStyleSheet("font-weight: bold; color: #50C878;")
        device_layout.addWidget(self._input_b_label, 1, 0)
        self.input_combo_b = QtWidgets.QComboBox()
        self.input_combo_b.currentIndexChanged.connect(self.input_device_changed)
        device_layout.addWidget(self.input_combo_b, 1, 1)

        # Input C (optional)
        self._input_c_label = QtWidgets.QLabel("Input C:")
        self._input_c_label.setStyleSheet("font-weight: bold; color: #E6A23C;")
        device_layout.addWidget(self._input_c_label, 2, 0)
        self.input_combo_c = QtWidgets.QComboBox()
        self.input_combo_c.currentIndexChanged.connect(self.input_device_changed)
        device_layout.addWidget(self.input_combo_c, 2, 1)

        # -- Outputs (right) — same color scheme as inputs --
        # Output A (always active)
        self._out_a_label = QtWidgets.QLabel("Output A:")
        self._out_a_label.setStyleSheet("font-weight: bold; color: #4B9DE0;")
        device_layout.addWidget(self._out_a_label, 0, 2)
        self.output_combo_a = QtWidgets.QComboBox()
        self.output_combo_a.currentIndexChanged.connect(lambda: self.output_device_changed('A'))
        device_layout.addWidget(self.output_combo_a, 0, 3)
        self.channels_combo_a = QtWidgets.QComboBox()
        device_layout.addWidget(self.channels_combo_a, 0, 4)

        # Output B (optional)
        self._out_b_label = QtWidgets.QLabel("Output B:")
        self._out_b_label.setStyleSheet("font-weight: bold; color: #50C878;")
        device_layout.addWidget(self._out_b_label, 1, 2)
        self.output_combo_b = QtWidgets.QComboBox()
        self.output_combo_b.currentIndexChanged.connect(lambda: self.output_device_changed('B'))
        device_layout.addWidget(self.output_combo_b, 1, 3)
        self.channels_combo_b = QtWidgets.QComboBox()
        device_layout.addWidget(self.channels_combo_b, 1, 4)

        # Output C (optional)
        self._out_c_label = QtWidgets.QLabel("Output C:")
        self._out_c_label.setStyleSheet("font-weight: bold; color: #E6A23C;")
        device_layout.addWidget(self._out_c_label, 2, 2)
        self.output_combo_c = QtWidgets.QComboBox()
        self.output_combo_c.currentIndexChanged.connect(lambda: self.output_device_changed('C'))
        device_layout.addWidget(self.output_combo_c, 2, 3)
        self.channels_combo_c = QtWidgets.QComboBox()
        device_layout.addWidget(self.channels_combo_c, 2, 4)

        # Bottom row: Refresh, Start, Sample Rate
        refresh_btn = QtWidgets.QPushButton("Refresh Devices")
        refresh_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        device_layout.addWidget(refresh_btn, 3, 0)
        refresh_btn.clicked.connect(self.refresh_devices)

        self.route_btn = QtWidgets.QPushButton("Start Routing")
        self.route_btn.setObjectName("applyButton")
        self.route_btn.clicked.connect(self.toggle_routing)
        device_layout.addWidget(self.route_btn, 3, 1)

        device_layout.addWidget(QtWidgets.QLabel("Sample Rate:"), 3, 2)
        self.sample_rate_combo = QtWidgets.QComboBox()
        for rate_val, rate_label in [(44100, "44100 Hz"), (48000, "48000 Hz"), (96000, "96000 Hz")]:
            self.sample_rate_combo.addItem(rate_label, rate_val)
        self.sample_rate_combo.setCurrentIndex(1)  # Default to 48000 Hz
        device_layout.addWidget(self.sample_rate_combo, 3, 3)

        device_group.setLayout(device_layout)
        left_layout.addWidget(device_group)
        
        # Per-output track mapping via tabs
        mapper_group = QtWidgets.QGroupBox("Track Routing")
        mapper_layout = QtWidgets.QVBoxLayout()
        mapper_layout.setContentsMargins(10, 10, 10, 10)
        mapper_layout.setSpacing(8)

        # View toggle: Detail vs Mixer
        view_row = QtWidgets.QHBoxLayout()
        self._mapper_hint = QtWidgets.QLabel(
            "Assign which input feeds every left/right channel per output. Use presets for quick stereo or mono setups."
        )
        self._mapper_hint.setStyleSheet("color: #BBBBBB; font-size: 12px;")
        self._mapper_hint.setWordWrap(True)
        view_row.addWidget(self._mapper_hint, stretch=1)

        self._view_detail_btn = QtWidgets.QPushButton("Patchbay")
        self._view_mixer_btn = QtWidgets.QPushButton("Mixer")
        self._view_detail_btn.setCheckable(True)
        self._view_mixer_btn.setCheckable(True)
        self._view_mixer_btn.setChecked(True)  # Mixer is default
        self._view_detail_btn.setFixedWidth(80)
        self._view_mixer_btn.setFixedWidth(70)
        self._view_detail_btn.clicked.connect(lambda: self._switch_view('detail'))
        self._view_mixer_btn.clicked.connect(lambda: self._switch_view('mixer'))
        view_row.addWidget(self._view_detail_btn)
        view_row.addWidget(self._view_mixer_btn)
        mapper_layout.addLayout(view_row)

        self._current_view = 'mixer'  # Mixer is the default view

        # Patchbay view: per-output PatchbayView tabs
        self.mapper_tabs = QtWidgets.QTabWidget()
        self.track_mappers = {}  # 'A'/'B'/'C' -> PatchbayView

        mapper_a = PatchbayView()
        mapper_a.mapping_changed.connect(lambda: self.apply_mapping(0))
        self.track_mappers['A'] = mapper_a
        self.mapper_tabs.addTab(mapper_a, "Output A")
        self.track_mapper = mapper_a  # backward compat
        self.mapper_tabs.setVisible(False)  # hidden by default (mixer is default)

        # Mixer view: per-output MixerStripView tabs
        self.mixer_tabs = QtWidgets.QTabWidget()
        self.mixer_views = {}  # 'A'/'B'/'C' -> MixerStripView

        mixer_a = MixerStripView()
        mixer_a.mapping_changed.connect(lambda: self.apply_mapping(0))
        self.mixer_views['A'] = mixer_a
        self.mixer_tabs.addTab(mixer_a, "Output A")

        mapper_layout.addWidget(self.mapper_tabs)
        mapper_layout.addWidget(self.mixer_tabs)
        mapper_group.setLayout(mapper_layout)
        left_layout.addWidget(mapper_group)
        left_layout.addStretch()
        
        # Visualization area
        viz_group = QtWidgets.QGroupBox("Audio Visualization")
        viz_layout = QtWidgets.QVBoxLayout()
        
        # Create enhanced visualization widgets
        self.viz_tabs = self.create_enhanced_visualization_widgets()
        viz_layout.addWidget(self.viz_tabs)
        
        viz_group.setLayout(viz_layout)
        right_layout.addWidget(viz_group)
        right_layout.addStretch()
        
        # Add panels to splitter and place splitter in the main layout
        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)
        content_splitter.setSizes([840, 560])
        main_layout.addWidget(content_splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready - Enhanced visualization enabled")
        
        # Store the device map for looking up device info
        self.device_map = {}
        self._cached_input_devices = []
        self._cached_output_devices = []
        self._updating_inputs = False  # Guard against re-entrant input_device_changed

        # Initialize
        self.refresh_devices()
    
    def create_enhanced_visualization_widgets(self):
        """Create the 3D spectrogram waterfall with level meters strip."""
        viz_container = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_container)
        viz_layout.setSpacing(4)
        viz_layout.setContentsMargins(0, 0, 0, 0)

        # 3D Spectrogram waterfall (main visualization)
        self.spectrogram_3d = SpectrogramWaterfall3D()
        viz_layout.addWidget(self.spectrogram_3d, stretch=1)

        # Level meters strip at bottom
        self.level_meters = EnhancedMultiLevelMeter(channels=8)
        self.level_meters.setMaximumHeight(120)
        viz_layout.addWidget(self.level_meters)

        # Track status labels — single row, evenly spaced
        track_row = QtWidgets.QHBoxLayout()
        track_row.setSpacing(20)
        self.track_labels = []

        self._track_label_states = []  # track previous state to avoid redundant updates
        for i in range(4):
            label = QtWidgets.QLabel(f"Track {i+1}: Silent")
            label.setStyleSheet(f"color: {self.get_track_color(i)}; font-weight: bold; padding: 8px 12px; border-radius: 3px; background-color: {THEMES[self._current_theme]['palette']['bg_primary']};")
            label.setMinimumHeight(30)
            label.setMinimumWidth(140)
            label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            track_row.addWidget(label)
            self.track_labels.append(label)
            self._track_label_states.append(None)

        viz_layout.addLayout(track_row)

        return viz_container
    
    def get_track_color(self, track_idx):
        """Get color for a specific track from the current theme."""
        palette = THEMES.get(self._current_theme, THEMES['Dark'])['palette']
        colors = palette['track_colors']
        return colors[track_idx % len(colors)]

    def update_track_labels(self, num_tracks):
        """Update track labels based on number of tracks"""
        # Show only the active track labels
        for i, label in enumerate(self.track_labels):
            if i < num_tracks:
                label.setVisible(True)
            else:
                label.setVisible(False)

    def apply_theme(self, theme_name):
        """Switch the entire UI to a new color theme."""
        if theme_name not in THEMES:
            return
        self._current_theme = theme_name
        theme = THEMES[theme_name]
        palette = theme['palette']

        # 1. App-level QSS
        QtWidgets.QApplication.instance().setStyleSheet(generate_qss(palette))

        # 2. Inline-styled widgets
        self._apply_inline_styles(palette)

        # 3. Spectrogram colors
        lut = build_spectrogram_lut(theme['spectrogram_stops'])
        self.spectrogram_3d.set_color_lut(lut, theme['spectrogram_bg'])

        # 4. Level meter colors
        self.level_meters.set_meter_zones(
            theme['meter_zones'], theme['meter_peak_zones'], palette
        )

        # 5. Update track label colors in visualization
        for i, label in enumerate(self.track_labels):
            color = self.get_track_color(i)
            label.setStyleSheet(
                f"color: {color}; font-weight: bold; padding: 8px 12px; "
                f"border-radius: 3px; background-color: {palette['bg_primary']};"
            )
        self._track_label_states = [None] * len(self._track_label_states)

        # 6. Update track mapper and mixer view track colors
        for mapper in self.track_mappers.values():
            mapper._track_colors = palette['track_colors']
            mapper._refresh_track_colors()
        for mixer in self.mixer_views.values():
            mixer._track_colors = palette['track_colors']
            mixer._refresh_track_colors()

        # 7. Check the right menu action
        if theme_name in self._theme_actions:
            self._theme_actions[theme_name].setChecked(True)

    def _apply_inline_styles(self, palette):
        """Re-apply inline styles to widgets that use setStyleSheet directly."""
        tc = palette['track_colors']

        # Title / version / hint labels
        self._title_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {palette['text_bright']};"
        )
        self._version_label.setStyleSheet(
            f"font-size: 14px; color: {palette['text_secondary']};"
        )
        self._intro_label.setStyleSheet(
            f"color: {palette['text_dim']}; font-size: 12px;"
        )
        self._mapper_hint.setStyleSheet(
            f"color: {palette['text_dim']}; font-size: 12px;"
        )

        # Device labels — A/B/C use first 3 track colors
        self._input_a_label.setStyleSheet(f"font-weight: bold; color: {tc[0]};")
        self._input_b_label.setStyleSheet(f"font-weight: bold; color: {tc[1]};")
        self._input_c_label.setStyleSheet(f"font-weight: bold; color: {tc[2]};")
        self._out_a_label.setStyleSheet(f"font-weight: bold; color: {tc[0]};")
        self._out_b_label.setStyleSheet(f"font-weight: bold; color: {tc[1]};")
        self._out_c_label.setStyleSheet(f"font-weight: bold; color: {tc[2]};")

        # Route button — danger style only when routing is active
        if self.audio_manager.output_streams:
            self.route_btn.setStyleSheet(
                f"background-color: {palette['danger']}; color: white; "
                f"font-weight: bold; min-height: 35px; border-radius: 5px;"
            )

    def _switch_view(self, mode):
        """Switch between 'detail' (patchbay) and 'mixer' view modes."""
        if mode == self._current_view:
            return

        # Transfer mapping from current view → target view
        for label in list(self.track_mappers.keys()):
            if mode == 'mixer' and label in self.mixer_views:
                # Patchbay → Mixer
                mapping = self.track_mappers[label].get_mapping()
                self.mixer_views[label].set_mapping(mapping)
            elif mode == 'detail' and label in self.mixer_views:
                # Mixer → Patchbay
                mapping = self.mixer_views[label].get_mapping()
                self.track_mappers[label].set_mapping(mapping)

        if mode == 'mixer':
            self.mapper_tabs.setVisible(False)
            self.mixer_tabs.setVisible(True)
            self._view_detail_btn.setChecked(False)
            self._view_mixer_btn.setChecked(True)
        else:
            self.mixer_tabs.setVisible(False)
            self.mapper_tabs.setVisible(True)
            self._view_detail_btn.setChecked(True)
            self._view_mixer_btn.setChecked(False)

        self._current_view = mode

    def get_input_infos(self):
        """Get info for all enabled input devices.

        A device counts as enabled when a real device (not '-- Off --') is selected.
        Returns list of {'label': str, 'channels': int, 'index': int}
        """
        infos = []
        combos = [
            ('A', self.input_combo_a),
            ('B', self.input_combo_b),
            ('C', self.input_combo_c),
        ]
        for label, combo in combos:
            dev_idx = combo.currentData()
            if dev_idx is not None and dev_idx in self.device_map:
                channels = int(self.device_map[dev_idx]['inputs'])
                infos.append({'label': label, 'channels': channels, 'index': dev_idx})
        return infos

    def get_enabled_input_devices(self):
        """Get list of {'index': int, 'channels': int} for start_routing."""
        return [{'index': info['index'], 'channels': info['channels']}
                for info in self.get_input_infos()]

    def get_enabled_output_devices(self):
        """Get list of {'index': int, 'channels': int, 'label': str} for start_routing."""
        devices = []
        combos = [
            ('A', self.output_combo_a, self.channels_combo_a),
            ('B', self.output_combo_b, self.channels_combo_b),
            ('C', self.output_combo_c, self.channels_combo_c),
        ]
        for label, dev_combo, ch_combo in combos:
            dev_idx = dev_combo.currentData()
            channels = ch_combo.currentData()
            if dev_idx is not None and channels is not None:
                devices.append({'index': dev_idx, 'channels': channels, 'label': label})
        return devices

    def input_device_changed(self):
        """Handle input device change — recompute total channels across all enabled inputs."""
        if self._updating_inputs:
            return
        self._updating_inputs = True
        try:
            input_infos = self.get_input_infos()
            if not input_infos:
                return

            total_channels = sum(info['channels'] for info in input_infos)

            # Update the default mapping in the audio manager
            self.audio_manager.update_default_mapping(total_channels)

            # Update all active track mapper tabs with device labels
            for slot, mapper in self.track_mappers.items():
                mapper.update_for_inputs(input_infos)
            for slot, mixer in self.mixer_views.items():
                mixer.update_for_inputs(input_infos)

            # Auto-assign all views when NOT routing (initial setup).
            # During routing, only update labels — don't overwrite user routing.
            if not self.audio_manager.is_routing:
                for slot in list(self.track_mappers.keys()):
                    ch_combo = {'A': self.channels_combo_a,
                                'B': self.channels_combo_b,
                                'C': self.channels_combo_c}.get(slot)
                    out_ch = ch_combo.currentData() if ch_combo else None
                    num_tracks = (out_ch // 2) if out_ch else 4
                    self.track_mappers[slot].auto_assign_devices(input_infos, num_tracks)
                    if slot in self.mixer_views:
                        self.mixer_views[slot].auto_assign_devices(input_infos, num_tracks)

            # Update status message
            device_summary = ", ".join(f"{info['label']}({info['channels']}ch)" for info in input_infos)
            self.statusBar().showMessage(f"Inputs: {device_summary} = {total_channels} total channels")
        finally:
            self._updating_inputs = False
    
    def update_visualization(self):
        """Update all visualization widgets with latest audio data"""
        if self.audio_manager.is_routing:
            # Get channel levels
            levels = self.audio_manager.get_channel_levels()

            # Update level meters
            self.level_meters.set_levels(levels)

            # Feed 3D spectrogram with mix of ALL active output channels
            output_channels = self.channels_combo_a.currentData() or 2
            num_ch = min(output_channels, 8)
            left_data = np.zeros_like(self.audio_manager.get_audio_data(0))
            right_data = np.zeros_like(left_data)
            for ch in range(0, num_ch, 2):
                left_data = np.maximum(left_data, np.abs(self.audio_manager.get_audio_data(ch)))
                if ch + 1 < num_ch:
                    right_data = np.maximum(right_data, np.abs(self.audio_manager.get_audio_data(ch + 1)))
            self.spectrogram_3d.update_audio_data(left_data, right_data)

            # Get selected channel count (primary output)
            output_channels = self.channels_combo_a.currentData()
            tracks = output_channels // 2 if output_channels else 1

            # Update track status labels (only on state change to avoid layout churn)
            for i in range(min(4, tracks)):
                left_idx = i * 2
                right_idx = i * 2 + 1

                left_level = levels[left_idx] if left_idx < len(levels) else 0
                right_level = levels[right_idx] if right_idx < len(levels) else 0

                avg_level = (left_level + right_level) / 2

                if avg_level > 0.01:
                    state = "active"
                elif avg_level > 0.001:
                    state = "low"
                else:
                    state = "silent"

                if self._track_label_states[i] != state:
                    self._track_label_states[i] = state
                    color = self.get_track_color(i)
                    pal = THEMES[self._current_theme]['palette']
                    if state == "active":
                        self.track_labels[i].setText(f"Track {i+1}: Active")
                        self.track_labels[i].setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px; border-radius: 3px; background-color: {pal['signal_active_bg']};")
                    elif state == "low":
                        self.track_labels[i].setText(f"Track {i+1}: Low signal")
                        self.track_labels[i].setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px; border-radius: 3px; background-color: {pal['signal_low_bg']};")
                    else:
                        self.track_labels[i].setText(f"Track {i+1}: Silent")
                        self.track_labels[i].setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px; border-radius: 3px; background-color: {pal['bg_primary']};")
    
    def _populate_input_combo(self, combo, add_placeholder=False):
        """Populate an input combo with available input devices."""
        combo.blockSignals(True)
        combo.clear()
        if add_placeholder:
            combo.addItem("-- Off --", None)
        for device in self._cached_input_devices:
            combo.addItem(f"{device['name']} ({device['inputs']}ch, {int(device['rate'])}Hz)", device['index'])
        combo.blockSignals(False)

    def _populate_output_combo(self, combo, add_placeholder=False):
        """Populate an output combo with available output devices."""
        combo.blockSignals(True)
        combo.clear()
        if add_placeholder:
            combo.addItem("-- Off --", None)
        for device in self._cached_output_devices:
            combo.addItem(f"{device['name']} ({device['outputs']}ch, {int(device['rate'])}Hz)", device['index'])
        combo.blockSignals(False)

    def refresh_devices(self):
        """Refresh the device lists (reinitializes PyAudio to detect USB changes)"""
        # Reinitialize PyAudio so newly connected USB devices are detected
        self.audio_manager.reinitialize()

        # Block all input-related updates during bulk population
        self._updating_inputs = True

        devices = self.audio_manager.get_devices()

        # Store device information for later use
        self.device_map = {device['index']: device for device in devices}
        self._cached_input_devices = [d for d in devices if d['inputs'] > 0]
        self._cached_output_devices = [d for d in devices if d['outputs'] > 0]

        # Populate Input A (no placeholder — always active)
        self._populate_input_combo(self.input_combo_a, add_placeholder=False)

        # Input B and C get "-- Off --" placeholder + device list
        self._populate_input_combo(self.input_combo_b, add_placeholder=True)
        self._populate_input_combo(self.input_combo_c, add_placeholder=True)

        # Populate Output A (no placeholder — always active)
        self._populate_output_combo(self.output_combo_a, add_placeholder=False)

        # Output B and C get "-- Off --" placeholder + device list
        self._populate_output_combo(self.output_combo_b, add_placeholder=True)
        self._populate_output_combo(self.output_combo_c, add_placeholder=True)

        # Auto-select S4 on Output A if found
        for i in range(self.output_combo_a.count()):
            if "S4" in self.output_combo_a.itemText(i) or "S-4" in self.output_combo_a.itemText(i):
                self.output_combo_a.setCurrentIndex(i)
                break

        # Auto-select Digitakt on Input A if found
        for i in range(self.input_combo_a.count()):
            if "Digitakt" in self.input_combo_a.itemText(i):
                self.input_combo_a.setCurrentIndex(i)
                break

        # Update channel combos for initially selected outputs
        self.output_device_changed('A')
        self.output_device_changed('B')
        self.output_device_changed('C')

        # Unblock and do a single update for the initial state
        self._updating_inputs = False
        self.input_device_changed()

        self.statusBar().showMessage("Devices refreshed - select your input and output devices")
    
    def _get_output_combos(self, slot):
        """Return (device_combo, channels_combo) for the given output slot."""
        return {
            'A': (self.output_combo_a, self.channels_combo_a),
            'B': (self.output_combo_b, self.channels_combo_b),
            'C': (self.output_combo_c, self.channels_combo_c),
        }[slot]

    def output_device_changed(self, slot='A'):
        """Update the channel options based on the selected output device"""
        dev_combo, ch_combo = self._get_output_combos(slot)
        output_idx = dev_combo.currentData()

        # Clear existing options
        ch_combo.clear()

        is_enabled = output_idx is not None and output_idx in self.device_map

        if is_enabled:
            # Get max channels for the selected output device
            device = self.device_map[output_idx]
            max_channels = int(device['outputs'])

            # Keep track of whether we support at least one stereo track
            has_stereo = False

            # Add options in pairs for stereo tracks up to 8 channels
            for i in range(1, min(5, (max_channels // 2) + 1)):
                channels = i * 2
                ch_combo.addItem(f"{channels} channels ({i} tracks)", channels)
                has_stereo = True

            # If no stereo options were added, add mono option
            if not has_stereo and max_channels > 0:
                ch_combo.addItem(f"{max_channels} channel(s) (mono)", max_channels)

            # Set default selection to maximum available
            if ch_combo.count() > 0:
                ch_combo.setCurrentIndex(ch_combo.count() - 1)

            # Update input selectors for the track mapper (only on primary output)
            if slot == 'A':
                self.input_device_changed()

        # Manage per-output mapper tabs AND mixer tabs for B and C
        if slot in ('B', 'C') and hasattr(self, 'mapper_tabs'):
            out_idx = {'B': 1, 'C': 2}[slot]
            if is_enabled and slot not in self.track_mappers:
                # Add new patchbay tab
                mapper = PatchbayView()
                mapper.mapping_changed.connect(lambda oi=out_idx: self.apply_mapping(oi))
                self.track_mappers[slot] = mapper
                self.mapper_tabs.addTab(mapper, f"Output {slot}")
                # Add new mixer tab
                mixer = MixerStripView()
                mixer.mapping_changed.connect(lambda oi=out_idx: self.apply_mapping(oi))
                self.mixer_views[slot] = mixer
                self.mixer_tabs.addTab(mixer, f"Output {slot}")
                # Initialize with current input info and auto-assign
                input_infos = self.get_input_infos()
                if input_infos:
                    mapper.update_for_inputs(input_infos)
                    mixer.update_for_inputs(input_infos)
                    # Auto-assign input routing so B/C outputs get
                    # their own input channels (not all defaulting to A)
                    out_ch = ch_combo.currentData()
                    num_tracks = (out_ch // 2) if out_ch else 4
                    mapper.auto_assign_devices(input_infos, num_tracks)
                    mixer.auto_assign_devices(input_infos, num_tracks)
            elif not is_enabled and slot in self.track_mappers:
                # Remove detail mapper tab
                mapper = self.track_mappers.pop(slot)
                idx = self.mapper_tabs.indexOf(mapper)
                if idx >= 0:
                    self.mapper_tabs.removeTab(idx)
                # Remove mixer tab
                if slot in self.mixer_views:
                    mixer = self.mixer_views.pop(slot)
                    idx = self.mixer_tabs.indexOf(mixer)
                    if idx >= 0:
                        self.mixer_tabs.removeTab(idx)

    
    def toggle_routing(self):
        """Start or stop audio routing"""
        if self.audio_manager.is_routing:
            self.audio_manager.stop_routing()
            self.route_btn.setText("Start Routing")
            self.route_btn.setStyleSheet("")
            self.statusBar().showMessage("Audio routing stopped")

            # Reset guard so device changes work again
            self._updating_inputs = False

            # Enable device selectors
            for combo in [self.input_combo_a, self.input_combo_b, self.input_combo_c,
                          self.output_combo_a, self.output_combo_b, self.output_combo_c,
                          self.channels_combo_a, self.channels_combo_b, self.channels_combo_c,
                          self.sample_rate_combo]:
                combo.setEnabled(True)
        else:
            # Collect enabled devices
            input_devices = self.get_enabled_input_devices()
            output_devices = self.get_enabled_output_devices()

            if not input_devices or not output_devices:
                QtWidgets.QMessageBox.warning(
                    self, "Cannot Start",
                    "Please select at least one input and one output device.")
                return

            # Duplicate output validation
            seen_indices = {}
            for od in output_devices:
                if od['index'] in seen_indices:
                    name = self.device_map.get(od['index'], {}).get('name', '?')
                    prev = seen_indices[od['index']]
                    QtWidgets.QMessageBox.warning(
                        self, "Duplicate Output Device",
                        f"'{name}' is selected for both Output {prev} and Output {od['label']}.\n\n"
                        f"Each output slot must use a different device.\n"
                        f"Set one of them to '-- Off --' or choose a different device.")
                    return
                seen_indices[od['index']] = od['label']

            sample_rate = self.sample_rate_combo.currentData() or 48000

            # Apply all active output mappings
            for oi, label in enumerate(['A', 'B', 'C']):
                if label in self.track_mappers:
                    self.apply_mapping(oi)

            # Start routing with all enabled inputs and outputs
            if self.audio_manager.start_routing(input_devices, output_devices, sample_rate=sample_rate):
                self.route_btn.setText("Stop Routing")
                danger = THEMES[self._current_theme]['palette']['danger']
                self.route_btn.setStyleSheet(
                    f"background-color: {danger}; color: white; "
                    f"font-weight: bold; min-height: 35px; border-radius: 5px;"
                )

                # Disable all device selectors while routing
                for combo in [self.input_combo_a, self.input_combo_b, self.input_combo_c,
                              self.output_combo_a, self.output_combo_b, self.output_combo_c,
                              self.channels_combo_a, self.channels_combo_b, self.channels_combo_c,
                              self.sample_rate_combo]:
                    combo.setEnabled(False)

                # Update UI elements for the primary output's channel/track count
                output_channels = output_devices[0]['channels']
                tracks = output_channels // 2
                self.level_meters.set_channels(output_channels)
                self.update_track_labels(tracks)

                # Build status message
                input_infos = self.get_input_infos()
                in_names = " + ".join(f"{info['label']}:{self.device_map[info['index']]['name']}" for info in input_infos)
                out_names = " + ".join(f"{od['label']}:{self.device_map[od['index']]['name']}" for od in output_devices if od['index'] in self.device_map)
                self.statusBar().showMessage(
                    f"Routing [{in_names}] -> [{out_names}] ({sample_rate}Hz 32-bit float)")
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Routing Failed",
                    "Failed to start audio routing.\nCheck device settings and ensure devices are not in use.")
                self.statusBar().showMessage("Failed to start audio routing")
    
    def apply_mapping(self, output_idx=0):
        """Apply the current track mapping to the audio manager for a specific output."""
        label = ['A', 'B', 'C'][output_idx] if output_idx < 3 else 'A'
        if self._current_view == 'mixer':
            mapper = self.mixer_views.get(label)
        else:
            mapper = self.track_mappers.get(label)
        if mapper is None:
            return

        mapping = mapper.get_mapping()
        dbg(f">>> apply_mapping(output_idx={output_idx}, label={label}, "
            f"view={self._current_view}, routing={self.audio_manager.is_routing})")
        dbg(f"    raw mapping: {mapping}")

        def describe(entry):
            if isinstance(entry, dict):
                entry = [entry]
            elif isinstance(entry, int):
                entry = [{'index': entry, 'gain': 1.0}]
            parts = []
            if isinstance(entry, (list, tuple)):
                for item in entry:
                    if isinstance(item, dict):
                        idx = item.get('index')
                        gain = item.get('gain', 1.0)
                        if isinstance(idx, int) and idx >= 0:
                            parts.append(f"{idx + 1}@{gain * 100:.0f}%")
                    elif isinstance(item, int) and item >= 0:
                        parts.append(f"{item + 1}@100%")
            if not parts:
                return "X"
            return "+".join(parts)

        mapping_str = ", ".join([
            f"{i//2+1}.{i%2+1}→{describe(m)}" for i, m in enumerate(mapping)
        ])
        self.statusBar().showMessage(f"Output {label} mapping: {mapping_str}")

        # Apply to audio manager
        self.audio_manager.set_track_mapping(mapping, output_idx)

        # Update the track mapper status
        mapper.status_label.setText(f"Mapping applied: {mapping_str}")
    
    def closeEvent(self, event):
        """Handle window close event — stop visualization before audio cleanup."""
        # Stop all timers first so they don't touch audio during teardown
        self.update_timer.stop()
        self.spectrogram_3d.update_timer.stop()
        self.level_meters.animation_timer.stop()
        self.level_meters.peak_decay_timer.stop()
        self.audio_manager.cleanup()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for consistent look
    
    # Apply the default Dark theme via themes.py
    app.setStyleSheet(generate_qss(THEMES['Dark']['palette']))
    
    window = MainWindow()
    window.show()
    
    dbg("Application started with enhanced visualizations")
    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        dbg("LF Music Mapper with Enhanced Visualization")
        dbg(f"Python version: {sys.version}")
        dbg(f"Platform: {sys.platform}")
        dbg("-" * 40)
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        input("Press Enter to exit...")
