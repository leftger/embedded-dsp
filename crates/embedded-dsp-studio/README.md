# embedded-dsp-studio

<p align="center">
  <b>An interactive, real-time Digital Signal Processing workbench and testbench for embedded systems and microcontrollers.</b>
</p>

<p align="center">
  <a href="https://leftger.github.io/embedded-dsp/"><b>🌐 Launch WebAssembly Studio in Browser</b></a>
</p>

---

## Highlights

- **Dual-Engine Signal Lab**: Interactive PolyBLEP oscillators (Sine, Triangle, Square, Sawtooth), linear & exponential chirp sweeps, Paul Kellett pink noise ($1/f$), white noise, and impulse/step signals.
- **Processor & Filter Pipeline**: Direct Form I Biquad cascades, Andrew Simper 2x oversampled State Variable Filters (simultaneous Lowpass, Highpass, Bandpass, Notch, Peak), Peaking EQ, Q15 fixed-point truncation floor, and soft-knee dynamics compression.
- **Spectral & Phase Forensics**: 4096-point FFT magnitude spectrum, Welch Power Spectral Density (PSD) estimation, real-time oscilloscope, Lissajous goniometer phase correlator, and RMS/Peak metering.
- **Impulse Response Analyzer**: Capture 4096-sample forensic snapshot buffers on Dirac delta excitation, computing peak gain, settling time, and total energy ($\sum h^2$).
- **WAV & CSV Export / Import**: Export forensic snapshots and synthesized waveforms to canonical 16-bit PCM `.wav` and `.csv` files for verification in Python (`scipy.signal`), MATLAB, or Audacity.
- **Host Micro-Benchmarking**: Direct CPU micro-benchmarks measuring algorithm latency ($\mu\text{s/block}$), throughput ($M\text{ samples/sec}$), and speedup ratios of `fast_math` approximations.
- **Zero-Allocation MCU Codegen**: Generates production-ready, zero-allocation C (ARM CMSIS-DSP) and `#![no_std]` Rust code snippets tuned directly in the studio.

---

## Quick Start

### Native Desktop (Linux, macOS, Windows)

```bash
cargo run -p embedded-dsp-studio
```

### In-Browser WebAssembly (WASM)

To run the studio locally in your browser via Trunk:

```bash
cd crates/embedded-dsp-studio
trunk serve
```
Then navigate to `http://127.0.0.1:8080/`.

---

## Testbench Workflow Presets

- **🎯 Quick A/B Null Test**: Feeds identical signals into Processor Slot A and Inverted Processor Slot B to verify bit-exact cancellation down to $-\infty\text{ dB}$.
- **🔬 Q15 Fixed-Point Noise**: Compares Float32 precision against Q15 fixed-point quantization noise floor.
- **📸 Impulse Snapshot**: Fires a deterministic Dirac impulse into Slot A to inspect filter stability and ring-down.
- **⚡ CFFT & Chirp Analysis**: Sweeps an exponential chirp from 20 Hz to 20,000 Hz to analyze filter roll-off in the spectral domain.
- **📉 Notch Rejection Test**: Verifies deep attenuation notch rejection at target frequency.

---

## Headless Test Suite

The studio includes a headless unit test suite verifying presets, math stability, and code generation:

```bash
cargo test -p embedded-dsp-studio
```

---

## License

Dual-licensed under MIT or Apache-2.0.
