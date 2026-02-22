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
        self.output_stream = None
        self.is_routing = False
        self.buffer_size = 1024
        self.audio_data = np.zeros((self.buffer_size, 8), dtype=np.float32)
        self.channel_levels = [0.0] * 8
        self.debug_counter = 0
        self.input_raw_data = []  # Raw numpy arrays from each input device
        self.input_channel_counts = []  # Channel count per input device
        self.total_input_channels = 2  # Total channels across all input devices
        self.num_output_channels = 8  # Default to 4 stereo tracks (8 channels)

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

        # Create a dynamic mapping that uses all available inputs
        self.track_mapping = []
        for i in range(self.num_output_channels):
            self.track_mapping.append([{'index': i % self.total_input_channels, 'gain': 1.0}])
        
    def update_default_mapping(self, total_input_channels):
        """Update the default mapping based on total available input channels across all devices"""
        if total_input_channels <= 0:
            total_input_channels = 1

        self.total_input_channels = total_input_channels
        new_mapping = []
        for i in range(self.num_output_channels):
            new_mapping.append([{'index': i % total_input_channels, 'gain': 1.0}])
        self.track_mapping = new_mapping
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
    
    def set_track_mapping(self, mapping):
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

        self.track_mapping = normalized
        dbg(f"Track mapping set: {self.track_mapping}")
    
    def start_routing(self, input_devices, output_idx, output_channels=8, sample_rate=48000):
        """Start routing audio from multiple input devices to output device.

        input_devices: list of {'index': int, 'channels': int} dicts
        sample_rate: desired sample rate in Hz (default 48000)
        """
        if self.is_routing:
            self.stop_routing()

        try:
            self.num_output_channels = output_channels

            output_info = self.pa.get_device_info_by_index(output_idx)
            max_out_channels = int(output_info['maxOutputChannels'])

            dbg(f"\n----- AUDIO DEVICE DETAILS -----")
            for i, dev in enumerate(input_devices):
                info = self.pa.get_device_info_by_index(dev['index'])
                label = chr(ord('A') + i)
                dbg(f"Input {label}: {info['name']} ({dev['channels']} ch)")
            dbg(f"Output: {output_info['name']} ({max_out_channels} ch)")

            if output_channels > max_out_channels:
                dbg(f"Warning: Limiting output to {max_out_channels} channels")
                self.num_output_channels = max_out_channels

            rate = int(sample_rate)

            self.input_channel_counts = [d['channels'] for d in input_devices]
            self.total_input_channels = sum(self.input_channel_counts)

            tracks = self.num_output_channels // 2
            dbg(f"\n*** {len(input_devices)} input device(s), {self.total_input_channels} total input channels ***")
            dbg(f"Output: {self.num_output_channels} channels ({tracks} stereo tracks)")

            # Dump current mapping
            dbg(f"\nTrack mapping ({len(self.track_mapping)} entries):")
            for out_ch, entry in enumerate(self.track_mapping):
                t = out_ch // 2 + 1
                lr = "L" if out_ch % 2 == 0 else "R"
                if entry:
                    sources = ", ".join(f"ch{e.get('index')}@{e.get('gain',1.0)*100:.0f}%" for e in (entry if isinstance(entry, list) else [entry]))
                    dbg(f"  Track {t}.{lr} (out_ch {out_ch}) <- {sources}")
                else:
                    dbg(f"  Track {t}.{lr} (out_ch {out_ch}) <- [silence]")

            return self._start_multi_input_stream(input_devices, output_idx, rate)

        except Exception as e:
            print(f"Error in audio routing setup: {e}")
            return False

    def _start_multi_input_stream(self, input_devices, output_idx, rate):
        """Open input streams for each device, mix via output callback."""
        try:
            self.debug_counter = 0
            self.input_raw_data = [None] * len(input_devices)
            # Ring buffers decouple input/output hardware clocks — input pushes, output pops
            # maxlen=4 gives ~80ms of jitter absorption at 1024/48kHz
            self._input_ringbufs = [deque(maxlen=4) for _ in range(len(input_devices))]
            self._prev_output = None  # for fade-out on underrun
            self._diag_stale_reads = 0

            # Open diagnostics log file (only when DEBUG enabled)
            if DEBUG:
                import os, datetime
                log_dir = os.path.dirname(os.path.abspath(__file__))
                log_path = os.path.join(log_dir, "audio_diag.log")
                self._diag_file = open(log_path, "w")
                self._diag_file.write(f"=== LF MusicMapper Audio Diagnostics ===\n")
                self._diag_file.write(f"Started: {datetime.datetime.now().isoformat()}\n")
                self._diag_file.write(f"Sample rate: {rate} Hz | Buffer: {self.buffer_size} frames "
                                      f"({self.buffer_size/rate*1000:.1f}ms) | Output channels: {self.num_output_channels}\n")
                for i, dev in enumerate(input_devices):
                    info = self.pa.get_device_info_by_index(dev['index'])
                    self._diag_file.write(f"Input {chr(ord('A')+i)}: {info['name']} ({dev['channels']}ch)\n")
                out_info = self.pa.get_device_info_by_index(output_idx)
                self._diag_file.write(f"Output: {out_info['name']}\n")
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

            # Create a callback closure for each input device
            def make_input_callback(dev_idx, num_channels):
                def callback(in_data, frame_count, time_info, status):
                    if status:
                        if status & pyaudio.paInputUnderflow or status & pyaudio.paInputOverflow:
                            self._diag_overruns += 1
                    data = np.frombuffer(in_data, dtype=np.float32)
                    if len(data) > 0 and num_channels > 0:
                        try:
                            buf = data.reshape(-1, num_channels).copy()
                            self._input_ringbufs[dev_idx].append(buf)
                            self.input_raw_data[dev_idx] = buf  # latest ref for viz
                        except Exception:
                            pass
                    return (None, pyaudio.paContinue)
                return callback

            # Crossfade ramp — precomputed for underrun fade-out
            _xfade_out = np.linspace(1.0, 0.0, self.buffer_size, dtype=np.float32).reshape(-1, 1)

            # Output callback: pop fresh buffers from ring buffers, mix, and output.
            # Ring buffers absorb clock drift between input and output devices.
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

                    if frame_count != self.buffer_size:
                        self._diag_frame_mismatch += 1

                    # Pop fresh data from each input's ring buffer
                    has_data = False
                    input_bufs = [None] * len(self.input_channel_counts)
                    for i in range(len(self.input_channel_counts)):
                        rb = self._input_ringbufs[i]
                        if rb:
                            input_bufs[i] = rb.popleft()
                            has_data = True
                        else:
                            # Underrun on this input — no fresh data available
                            self._diag_stale_reads += 1

                    if not has_data:
                        self._diag_no_data += 1
                        # Fade out previous output to avoid hard cut to silence
                        if self._prev_output is not None:
                            xf_len = min(frame_count, len(_xfade_out))
                            faded = self._prev_output[:xf_len] * _xfade_out[:xf_len]
                            self._prev_output = None
                            return (faded.flatten().tobytes(), pyaudio.paContinue)
                        silence = np.zeros(frame_count * self.num_output_channels, dtype=np.float32)
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

                    # Apply the channel mapping
                    output_frames = np.zeros((frame_count, self.num_output_channels), dtype=np.float32)
                    for out_ch in range(self.num_output_channels):
                        mapping_entry = []
                        if out_ch < len(self.track_mapping):
                            mapping_entry = self.track_mapping[out_ch]
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

                        total_gain = sum(item['gain'] for item in valid_entries)
                        if total_gain > 0:
                            mixed = np.zeros(frame_count, dtype=np.float32)
                            for item in valid_entries:
                                mixed += combined[:, item['index']] * item['gain']
                            mixed /= total_gain
                            output_frames[:, out_ch] = mixed

                    # Soft-clip output to prevent popping from hot signals
                    output_frames = np.tanh(output_frames)

                    # Save for fade-out on future underrun
                    self._prev_output = output_frames.copy()

                    # Track peak level
                    peak = float(np.abs(output_frames).max())
                    if peak > self._diag_peak:
                        self._diag_peak = peak

                    # Update visualization data
                    self.channel_levels = [
                        np.abs(output_frames[:, ch]).mean() if ch < self.num_output_channels else 0.0
                        for ch in range(8)
                    ]
                    sample_count = min(self.buffer_size, len(output_frames))
                    for ch in range(min(8, self.num_output_channels)):
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

                    # Write diagnostic summary every 2 seconds (DEBUG only)
                    now = time.time()
                    if now - self._diag_last_report >= 2.0 and self._diag_file:
                        elapsed = now - self._diag_last_report
                        cbs = self._diag_cb_count
                        cb_rate = cbs / elapsed if elapsed > 0 else 0
                        self._diag_file.write(
                            f"[{now:.1f}] callbacks={cbs} ({cb_rate:.0f}/s) | "
                            f"underruns={self._diag_underruns} overruns={self._diag_overruns} | "
                            f"no_data={self._diag_no_data} stale={self._diag_stale_reads} "
                            f"frame_mismatch={self._diag_frame_mismatch} | "
                            f"slow_cb(>{self._diag_buffer_time_ms*0.5:.1f}ms)={self._diag_slow_cb} "
                            f"max_cb={self._diag_cb_max_ms:.2f}ms | "
                            f"peak={self._diag_peak:.4f} | "
                            f"exceptions={self._diag_exceptions}\n"
                        )
                        self._diag_file.flush()
                        # Reset counters for next period
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

                    return result

                except Exception as e:
                    self._diag_exceptions += 1
                    if self._diag_file:
                        self._diag_file.write(f"!!! EXCEPTION in callback: {e}\n")
                        self._diag_file.flush()
                    silence = np.zeros(frame_count * self.num_output_channels, dtype=np.float32)
                    return (silence.tobytes(), pyaudio.paContinue)

            # Open one input stream per device
            self.input_streams = []
            for dev_idx, dev in enumerate(input_devices):
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

            # Open output stream with callback (driven by audio clock)
            dbg(f"Opening output stream with {self.num_output_channels} channels at {rate}Hz, buffer={self.buffer_size}")
            self.output_stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self.num_output_channels,
                rate=rate,
                output=True,
                output_device_index=output_idx,
                frames_per_buffer=self.buffer_size,
                stream_callback=output_callback
            )

            self.is_routing = True
            dbg(f"\nStarted routing with {len(input_devices)} input device(s)")
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

        # Close all input streams
        for i, stream in enumerate(self.input_streams):
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
                dbg(f"Input stream {chr(ord('A') + i)} closed")
            except Exception as e:
                dbg(f"Error closing input stream {chr(ord('A') + i)}: {e}")
        self.input_streams = []

        # Close output stream
        if self.output_stream:
            try:
                if self.output_stream.is_active():
                    self.output_stream.stop_stream()
                self.output_stream.close()
                dbg("Output stream closed")
            except Exception as e:
                dbg(f"Error closing output stream: {e}")
            self.output_stream = None

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

        # Create 4 rows for 4 tracks
        for i in range(4):  # 4 tracks
            # Track number with custom styling
            track_label = QtWidgets.QLabel(f"Track {i+1}")
            track_label.setStyleSheet(f"font-weight: bold; color: {self.get_track_color(i)};")
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
        """Get color for a specific track"""
        colors = [
            "#4B9DE0",  # Track 1 - Blue
            "#50C878",  # Track 2 - Green
            "#E6A23C",  # Track 3 - Orange
            "#E77F7F"   # Track 4 - Red
        ]
        return colors[track_idx % len(colors)]

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
                    # Stereo button for this device
                    btn = QtWidgets.QPushButton(f"{dev_label} Stereo")
                    btn.setToolTip(f"Set track to stereo ({dev_label}1->L, {dev_label}2->R)")
                    btn.setProperty("preset", "true")
                    left_idx = ch_offset
                    right_idx = ch_offset + 1
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
        painter.fillRect(0, 0, w, h, QtGui.QColor(8, 8, 15))
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
        painter.fillRect(0, 0, width, height, QtGui.QColor(35, 35, 37))
        
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
                QtGui.QColor(28, 28, 30)
            )
            
            # Track/channel identifier at bottom
            track_num = ch // 2 + 1
            ch_num = ch % 2 + 1
            
            # Draw rounded meter with improved gradient
            if level_height > 0:
                # Create gradient for smoother transitions between colors
                gradient = QtGui.QLinearGradient(
                    x, height - level_height, 
                    x, height
                )
                
                # Choose colors based on level with better gradations
                if level < 0.4:  # Safe zone (green)
                    gradient.setColorAt(0, QtGui.QColor(50, 205, 50))  # Green
                    gradient.setColorAt(1, QtGui.QColor(30, 160, 30))
                elif level < 0.7:  # Caution zone (yellow-green to yellow)
                    gradient.setColorAt(0, QtGui.QColor(180, 210, 40))  # Yellow-green
                    gradient.setColorAt(1, QtGui.QColor(170, 170, 20))
                elif level < 0.85:  # Warning zone (yellow to orange)
                    gradient.setColorAt(0, QtGui.QColor(255, 215, 0))  # Yellow
                    gradient.setColorAt(1, QtGui.QColor(240, 170, 0))
                elif level < 0.95:  # Danger zone (orange to red)
                    gradient.setColorAt(0, QtGui.QColor(255, 140, 0))  # Orange
                    gradient.setColorAt(1, QtGui.QColor(220, 80, 0))
                else:  # Clipping zone (bright red)
                    gradient.setColorAt(0, QtGui.QColor(255, 30, 30))  # Bright red
                    gradient.setColorAt(1, QtGui.QColor(180, 0, 0))
                    
                # Fill rounded meter
                meter_rect = QtCore.QRectF(
                    x + 1, height - level_height, 
                    meter_width - 2, level_height
                )
                painter.setBrush(gradient)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawRoundedRect(meter_rect, 2, 2)
            
            # Draw peak indicator with enhanced style
            if peak_height > 0:
                # Create gradient for peak indicator
                peak_color = QtGui.QColor(255, 255, 255)  # White base
                
                if peak_level < 0.4:
                    peak_color = QtGui.QColor(120, 255, 120)  # Light green
                elif peak_level < 0.7:
                    peak_color = QtGui.QColor(230, 230, 100)  # Light yellow
                elif peak_level < 0.85:
                    peak_color = QtGui.QColor(255, 200, 0)  # Gold
                elif peak_level < 0.95:
                    peak_color = QtGui.QColor(255, 140, 0)  # Orange
                else:
                    # Animate clip indicator by alternating colors
                    if self.clip_indicators[ch] > 0:
                        frame = self.clip_indicators[ch]
                        if frame % 10 < 5:  # Blink effect
                            peak_color = QtGui.QColor(255, 50, 50)  # Red
                        else:
                            peak_color = QtGui.QColor(255, 220, 220)  # Light red
                    else:
                        peak_color = QtGui.QColor(255, 50, 50)  # Red
                
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
        
        # Title and version
        title_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("LF Music Mapper")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #EEEEEE;")
        title_layout.addWidget(title)
        
        version = QtWidgets.QLabel("By: @LofiFren")
        version.setStyleSheet("font-size: 14px; color: #AAAAAA;")
        version.setAlignment(QtCore.Qt.AlignBottom)
        title_layout.addWidget(version)
        
        title_layout.addStretch()
        
        main_layout.addLayout(title_layout)
        
        # Inline helper text to replace the old popup dialog
        intro_text = QtWidgets.QLabel(
            "Route any available input into dedicated multichannel outputs. "
            "Choose your devices, define the mapping on the left, and monitor your signal on the right."
        )
        intro_text.setStyleSheet("color: #BBBBBB; font-size: 12px;")
        intro_text.setWordWrap(True)
        main_layout.addWidget(intro_text)
        
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
        
        # Device selection
        device_group = QtWidgets.QGroupBox("Audio Devices")
        device_layout = QtWidgets.QGridLayout()
        device_layout.setSpacing(10)
        device_layout.setColumnStretch(1, 1)
        device_layout.setColumnStretch(3, 1)

        # Input A (always active)
        input_a_label = QtWidgets.QLabel("Input A:")
        input_a_label.setStyleSheet("font-weight: bold; color: #4B9DE0;")
        device_layout.addWidget(input_a_label, 0, 0)
        self.input_combo_a = QtWidgets.QComboBox()
        self.input_combo_a.currentIndexChanged.connect(self.input_device_changed)
        device_layout.addWidget(self.input_combo_a, 0, 1)

        # Input B (optional — select a device to enable, "Off" to disable)
        input_b_label = QtWidgets.QLabel("Input B:")
        input_b_label.setStyleSheet("font-weight: bold; color: #50C878;")
        device_layout.addWidget(input_b_label, 1, 0)
        self.input_combo_b = QtWidgets.QComboBox()
        self.input_combo_b.currentIndexChanged.connect(self.input_device_changed)
        device_layout.addWidget(self.input_combo_b, 1, 1)

        # Input C (optional — select a device to enable, "Off" to disable)
        input_c_label = QtWidgets.QLabel("Input C:")
        input_c_label.setStyleSheet("font-weight: bold; color: #E6A23C;")
        device_layout.addWidget(input_c_label, 2, 0)
        self.input_combo_c = QtWidgets.QComboBox()
        self.input_combo_c.currentIndexChanged.connect(self.input_device_changed)
        device_layout.addWidget(self.input_combo_c, 2, 1)

        # Output device (right column, top)
        device_layout.addWidget(QtWidgets.QLabel("Output Device:"), 0, 2)
        self.output_combo = QtWidgets.QComboBox()
        self.output_combo.currentIndexChanged.connect(self.output_device_changed)
        device_layout.addWidget(self.output_combo, 0, 3)

        # Output channels (right column, middle)
        device_layout.addWidget(QtWidgets.QLabel("Output Channels:"), 1, 2)
        self.channels_combo = QtWidgets.QComboBox()
        device_layout.addWidget(self.channels_combo, 1, 3)

        # Sample rate (right column, bottom)
        device_layout.addWidget(QtWidgets.QLabel("Sample Rate:"), 2, 2)
        self.sample_rate_combo = QtWidgets.QComboBox()
        for rate_val, rate_label in [(44100, "44100 Hz"), (48000, "48000 Hz"), (96000, "96000 Hz")]:
            self.sample_rate_combo.addItem(rate_label, rate_val)
        self.sample_rate_combo.setCurrentIndex(1)  # Default to 48000 Hz
        device_layout.addWidget(self.sample_rate_combo, 2, 3)

        # Add refresh button
        refresh_btn = QtWidgets.QPushButton("Refresh Devices")
        refresh_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        device_layout.addWidget(refresh_btn, 3, 0)
        refresh_btn.clicked.connect(self.refresh_devices)

        # Add start/stop button
        self.route_btn = QtWidgets.QPushButton("Start Routing")
        self.route_btn.setObjectName("applyButton")
        self.route_btn.clicked.connect(self.toggle_routing)
        device_layout.addWidget(self.route_btn, 3, 1)

        device_group.setLayout(device_layout)
        left_layout.addWidget(device_group)
        
        # Track mapper widget with its own group box to keep controls tidy
        mapper_group = QtWidgets.QGroupBox("Track Routing")
        mapper_layout = QtWidgets.QVBoxLayout()
        mapper_layout.setContentsMargins(10, 10, 10, 10)
        mapper_layout.setSpacing(8)
        
        mapper_hint = QtWidgets.QLabel(
            "Assign which input feeds every left/right channel. Use presets for quick stereo or mono setups."
        )
        mapper_hint.setStyleSheet("color: #BBBBBB; font-size: 12px;")
        mapper_hint.setWordWrap(True)
        mapper_layout.addWidget(mapper_hint)
        
        self.track_mapper = TrackMapperWidget()
        self.track_mapper.mapping_changed.connect(self.apply_mapping)
        mapper_layout.addWidget(self.track_mapper)
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
            label.setStyleSheet(f"color: {self.get_track_color(i)}; font-weight: bold; padding: 8px 12px; border-radius: 3px; background-color: #2D2D30;")
            label.setMinimumHeight(30)
            label.setMinimumWidth(140)
            label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            track_row.addWidget(label)
            self.track_labels.append(label)
            self._track_label_states.append(None)

        viz_layout.addLayout(track_row)

        return viz_container
    
    def get_track_color(self, track_idx):
        """Get color for a specific track"""
        colors = [
            "#4B9DE0",  # Track 1 - Blue
            "#50C878",  # Track 2 - Green
            "#E6A23C",  # Track 3 - Orange
            "#E77F7F"   # Track 4 - Red
        ]
        return colors[track_idx % len(colors)]
    
    def update_track_labels(self, num_tracks):
        """Update track labels based on number of tracks"""
        # Show only the active track labels
        for i, label in enumerate(self.track_labels):
            if i < num_tracks:
                label.setVisible(True)
            else:
                label.setVisible(False)
    
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

            # Update track mapper UI with device labels
            self.track_mapper.update_for_inputs(input_infos)

            # Auto-assign devices to tracks, respecting output track count
            output_channels = self.channels_combo.currentData()
            num_output_tracks = (output_channels // 2) if output_channels else 1
            self.track_mapper.auto_assign_devices(input_infos, num_output_tracks)

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

            # Feed 3D spectrogram with first stereo pair (combined overview)
            left_data = self.audio_manager.get_audio_data(0)
            right_data = self.audio_manager.get_audio_data(1)
            self.spectrogram_3d.update_audio_data(left_data, right_data)

            # Get selected channel count
            output_channels = self.channels_combo.currentData()
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
                    if state == "active":
                        self.track_labels[i].setText(f"Track {i+1}: Active")
                        self.track_labels[i].setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px; border-radius: 3px; background-color: #1E3A29;")
                    elif state == "low":
                        self.track_labels[i].setText(f"Track {i+1}: Low signal")
                        self.track_labels[i].setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px; border-radius: 3px; background-color: #3A3A1E;")
                    else:
                        self.track_labels[i].setText(f"Track {i+1}: Silent")
                        self.track_labels[i].setStyleSheet(f"color: {color}; font-weight: bold; padding: 5px; border-radius: 3px; background-color: #2D2D30;")
    
    def _populate_input_combo(self, combo, add_placeholder=False):
        """Populate an input combo with available input devices."""
        combo.blockSignals(True)
        combo.clear()
        if add_placeholder:
            combo.addItem("-- Off --", None)
        for device in self._cached_input_devices:
            combo.addItem(f"{device['name']} ({device['inputs']}ch, {int(device['rate'])}Hz)", device['index'])
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

        # Populate Input A (no placeholder — always active)
        self._populate_input_combo(self.input_combo_a, add_placeholder=False)

        # Input B and C get "-- Off --" placeholder + device list
        self._populate_input_combo(self.input_combo_b, add_placeholder=True)
        self._populate_input_combo(self.input_combo_c, add_placeholder=True)

        # Populate output combo
        self.output_combo.clear()
        for device in devices:
            if device['outputs'] > 0:
                self.output_combo.addItem(f"{device['name']} ({device['outputs']}ch, {int(device['rate'])}Hz)", device['index'])

        # Auto-select S4 output device if found
        for i in range(self.output_combo.count()):
            if "S4" in self.output_combo.itemText(i) or "S-4" in self.output_combo.itemText(i):
                self.output_combo.setCurrentIndex(i)
                break

        # Auto-select Digitakt on Input A if found
        for i in range(self.input_combo_a.count()):
            if "Digitakt" in self.input_combo_a.itemText(i):
                self.input_combo_a.setCurrentIndex(i)
                break

        # Update the channel combo for initially selected output
        self.output_device_changed()

        # Unblock and do a single update for the initial state
        self._updating_inputs = False
        self.input_device_changed()

        self.statusBar().showMessage("Devices refreshed - select your input and output devices")
    
    def output_device_changed(self):
        """Update the channel options based on the selected output device"""
        output_idx = self.output_combo.currentData()
        
        # Clear existing options
        self.channels_combo.clear()
        
        if output_idx is not None and output_idx in self.device_map:
            # Get max channels for the selected output device
            device = self.device_map[output_idx]
            max_channels = int(device['outputs'])
            
            # Keep track of whether we support at least one stereo track
            has_stereo = False
            
            # Add options in pairs for stereo tracks up to 8 channels
            for i in range(1, min(5, (max_channels // 2) + 1)):
                channels = i * 2
                self.channels_combo.addItem(f"{channels} channels ({i} tracks)", channels)
                has_stereo = True
            
            # If no stereo options were added, add mono option
            if not has_stereo and max_channels > 0:
                self.channels_combo.addItem(f"{max_channels} channel(s) (mono)", max_channels)
            
            # Set default selection to maximum available
            if self.channels_combo.count() > 0:
                self.channels_combo.setCurrentIndex(self.channels_combo.count() - 1)
                
            # Update input selectors for the track mapper
            self.input_device_changed()

    
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
            self.input_combo_a.setEnabled(True)
            self.input_combo_b.setEnabled(True)
            self.input_combo_c.setEnabled(True)
            self.output_combo.setEnabled(True)
            self.channels_combo.setEnabled(True)
            self.sample_rate_combo.setEnabled(True)
        else:
            # Collect enabled input devices
            input_devices = self.get_enabled_input_devices()
            output_idx = self.output_combo.currentData()

            if not input_devices or output_idx is None:
                self.statusBar().showMessage("Please select at least one input and an output device")
                return

            # Get selected channel count and sample rate
            output_channels = self.channels_combo.currentData()
            sample_rate = self.sample_rate_combo.currentData() or 48000

            # Apply current mapping
            self.apply_mapping()

            # Start routing with all enabled inputs
            if self.audio_manager.start_routing(input_devices, output_idx, output_channels, sample_rate):
                self.route_btn.setText("Stop Routing")
                self.route_btn.setStyleSheet("""
                    background-color: #B33A3A;
                    color: white;
                    font-weight: bold;
                    min-height: 35px;
                    border-radius: 5px;
                """)

                # Disable all device selectors while routing
                self.input_combo_a.setEnabled(False)
                self.input_combo_b.setEnabled(False)
                self.input_combo_c.setEnabled(False)
                self.output_combo.setEnabled(False)
                self.channels_combo.setEnabled(False)
                self.sample_rate_combo.setEnabled(False)

                # Update UI elements for the current channel/track count
                tracks = output_channels // 2
                self.level_meters.set_channels(output_channels)
                self.update_track_labels(tracks)

                # Build status message with all input device names
                input_infos = self.get_input_infos()
                in_names = " + ".join(f"{info['label']}:{self.device_map[info['index']]['name']}" for info in input_infos)
                out_name = self.output_combo.currentText().split(" (")[0]
                self.statusBar().showMessage(f"Routing [{in_names}] -> {out_name} ({output_channels}ch, {tracks} tracks, {sample_rate}Hz 32-bit float)")
            else:
                self.statusBar().showMessage("Failed to start audio routing - Check device settings")
    
    def apply_mapping(self):
        """Apply the current track mapping to the audio manager"""
        # Get current mapping
        mapping = self.track_mapper.get_mapping()
        
        # Display the mapping in status bar
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
        self.statusBar().showMessage(f"Applied mapping: {mapping_str}")
        
        # Apply to audio manager
        self.audio_manager.set_track_mapping(mapping)
        
        # Update the track mapper status
        self.track_mapper.status_label.setText(f"Mapping applied: {mapping_str}")
    
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
    
    # Set application stylesheet for modern flat design
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #2D2D30;
            color: #F0F0F0;
        }
        QGroupBox {
            border: 1px solid #3F3F46;
            border-radius: 8px;
            margin-top: 1.5ex;
            font-weight: bold;
            padding: 10px;
            background-color: #252526;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 8px;
            background-color: #2D2D30;
        }
        QComboBox {
            background-color: #333337;
            border: 1px solid #3F3F46;
            border-radius: 5px;
            padding: 6px;
            color: #F0F0F0;
            min-height: 25px;
        }
        QComboBox:hover {
            border-color: #007ACC;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QPushButton {
            background-color: #333337;
            border: 1px solid #3F3F46;
            border-radius: 5px;
            padding: 6px;
            color: #F0F0F0;
            min-height: 25px;
        }
        QPushButton:hover {
            background-color: #3E3E40;
            border-color: #007ACC;
        }
        QPushButton:pressed {
            background-color: #007ACC;
        }
        QLabel {
            padding: 2px;
        }
        QStatusBar {
            background-color: #1E1E1E;
            color: #CCCCCC;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 3px;
        }
        QTabWidget::pane {
            border: 1px solid #3F3F46;
            border-radius: 5px;
            background: #252526;
        }
        QTabBar::tab {
            background: #2D2D30;
            border: 1px solid #3F3F46;
            border-bottom-color: #252526;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            padding: 6px 10px;
            margin-right: 2px;
        }
        QTabBar::tab:selected, QTabBar::tab:hover {
            background: #333337;
        }
        QTabBar::tab:selected {
            border-bottom-color: #252526;
        }
        QSlider::groove:horizontal {
            border: 1px solid #3F3F46;
            height: 8px;
            background: #2D2D30;
            margin: 2px 0;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #007ACC;
            border: 1px solid #007ACC;
            width: 18px;
            height: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }
        QSlider::handle:horizontal:hover {
            background: #1C97EA;
        }
        
        /* Apply button style */
        #applyButton {
            background-color: #2C5AA0;
            color: white;
            font-weight: bold;
            min-height: 35px;
            border-radius: 5px;
        }
        #applyButton:hover {
            background-color: #3771C8;
        }
        
        /* Preset button styles */
        QPushButton[preset="true"] {
            background-color: #424245;
            border-width: 1px;
        }
        QPushButton[preset="true"]:hover {
            background-color: #4D4D52;
        }
    """)
    
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
