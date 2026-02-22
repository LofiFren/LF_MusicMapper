# LF Music Mapper

Picture this: your Digitakt is on Input A pumping beats, a turntable on Input B spinning vinyl, and a mic on Input C catching vocals — all three flowing into a single multichannel output, each landing on its own stereo track. Need to blend the Digitakt and turntable together into one track for a live mashup? Dial the gain and it's done. Want to isolate the mic on its own output for separate processing? One click. A 3D spectrogram waterfall lights up every frequency in real time so you can see exactly what's happening across the full spectrum.

LF Music Mapper connects up to 3 input devices to a multichannel output with per-channel mixing, gain control, and live visualization. Whether you're routing hardware into a multichannel interface for a live set, layering a drum machine with a synth and a sampler, or just need a clean way to get three sources onto four stereo tracks — this is the tool.

Built by [@LofiFren](https://github.com/LofiFren)

![LF Music Mapper in action](demo.gif)

## Features

- **Up to 3 simultaneous input devices** - connect a Digitakt, turntable, synth, or any audio interface and mix them together
- **Flexible channel mapping** - route any input channel to any output track with per-channel gain, blend up to 3 sources per channel
- **4 stereo output tracks** - supports multichannel output devices (up to 8 channels)
- **3D spectrogram waterfall** - real-time perspective-projected visualization with frequency, time, and amplitude; rich color mapping from deep blue (silence) through cyan, green, yellow, orange, red to white (loudest)
- **Level meters** - 8-channel VU meters with peak hold and clip detection
- **32-bit float / 48kHz+** - professional audio quality with selectable sample rate (44.1kHz, 48kHz, 96kHz)
- **Callback-based audio engine** - low-latency output driven by the hardware audio clock with ring buffer synchronization between input and output devices
- **Soft limiter** - `tanh` soft clipping prevents digital distortion on hot signals
- **Quick presets** - one-click stereo, mono, and multi-device routing setups per input device
- **Auto-apply mapping** - routing changes take effect immediately, no apply button needed
- **USB hot-plug support** - refresh device list to detect newly connected USB audio interfaces

## Requirements

- Python 3.11+
- macOS, Windows, or Linux
- An audio interface with multiple outputs for full multichannel routing (stereo output works too)

## Installation

```bash
git clone https://github.com/LofiFren/LF_MusicMapper.git
cd LF_MusicMapper
pip install -r requirements.txt
```

### Core dependencies

| Package | Purpose |
|---------|---------|
| PyAudio | Audio stream I/O via PortAudio |
| NumPy | Audio buffer processing and visualization rendering |
| SciPy | FFT for spectrum analysis |
| PyQt5 or PySide2 | GUI framework |

## Usage

```bash
python3.11 mapper.py
```

### Quick start

1. **Input A** selects your primary input device automatically
2. **Input B / C** - pick a second or third input device (select "-- Off --" to disable)
3. **Output Device** - choose your multichannel output (e.g. Roland S-4, Torso S-4)
4. **Output Channels** - select how many channels/tracks to use
5. **Sample Rate** - choose 44100, 48000, or 96000 Hz
6. **Track Routing** - configure which inputs feed each output track, or use the quick-set buttons
7. Click **Start Routing** to begin

### Track mapping

Each output track has Left and Right channels. For each channel you can:

- Click **Mix** to open a dialog where you blend up to 3 input sources with individual gain sliders
- Use per-device **Stereo** / **Mono** quick buttons (e.g. "A Stereo", "B1", "B2")
- Use the **Presets** row for common routing patterns

Input channels are labeled by device: `A1`, `A2` (Input A), `B1`, `B2` (Input B), etc.

### Diagnostics

Set `DEBUG = True` at the top of `mapper.py` to enable:

- Console debug logging for device detection, mapping, and stream events
- `audio_diag.log` file with per-second metrics: callback rate, underruns, peak levels, stale reads, and callback timing

## Architecture

- **Callback-based I/O** - input and output streams use PortAudio callbacks (not blocking reads/writes) for minimal latency
- **Ring buffer sync** - each input device pushes audio into a `deque(maxlen=4)` ring buffer; the output callback pops fresh frames, absorbing clock drift between separate hardware devices
- **Soft clipping** - `np.tanh()` applied to output prevents digital clipping artifacts
- **GIL-friendly visualization** - spectrogram colors computed in numpy (releases GIL), then rendered as image strips with lightweight QPainter calls to avoid starving the audio thread

## License

MIT
