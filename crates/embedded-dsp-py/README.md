# embedded-dsp (Python)

Python bindings for [`embedded-dsp`](https://github.com/leftger/embedded-dsp),
built with PyO3's stable ABI (CPython 3.9+). They expose the same verified Rust
kernels as the Rust and C APIs: the audio-EQ designer, FIR, and Direct Form I
biquad cascades.

```python
import embedded_dsp

# Audio-EQ biquad coefficients [b0, b1, b2, a1, a2]
peaking = embedded_dsp.eq_coeffs("peaking", 1_000.0, 48_000.0, 0.707, gain_db=6.0)
iho = embedded_dsp.eq_coeffs("iho", 2_000.0, 48_000.0, 0.707, gain_db=-6.0)

# Streaming kernels over a block
out = embedded_dsp.fir_f32([0.25, 0.5, 0.25], [1.0, 0.0, 0.0, 0.0])
filtered = embedded_dsp.biquad_cascade_f32(peaking, [1.0, 2.0, 3.0, 4.0])
```

## Building

Wheels are built with [maturin](https://github.com/PyO3/maturin):

```
maturin build --release    # from crates/embedded-dsp-py
```

`eq_coeffs` accepts `lowpass`, `highpass`, `bandpass`, `allpass`, `notch`,
`peaking`, `lowshelf`, `highshelf`, and `iho`; `gain_db` applies to `peaking`,
the shelves, and `iho`.
