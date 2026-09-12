# Bare-Metal Microcontroller Benchmarks for embedded-dsp

Measures cycle counts per sample on Cortex-M processors (e.g. STM32H7 / Cortex-M7):
- Run-from-RAM memory layout with deterministic cycle timing
- Direct measurement using ARM Cortex-M `DWT::cycle_count()`
- Measures:
  - Single-sample processing throughput
  - 4-sample chunk processing
  - 1024-sample slice processing
  - In-place buffer overwriting

## Binaries
- `biquad` — `f32` clamped DF1/DF2T biquads and the fixed-point noise-shaper.
- `idsp_parity` — the integer primitives shared with `idsp` (`cossin`, `atan2_i32`,
  `IntPll`, `NormalForm`, half-band cascade), for direct comparison with
  `idsp`'s published `tests/embedded` cycle counts.

## Build
```bash
cargo build --release --target thumbv7em-none-eabihf
```
