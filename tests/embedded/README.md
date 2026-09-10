# Bare-Metal Microcontroller Benchmarks for embedded-dsp

Measures cycle counts per sample on Cortex-M processors (e.g. STM32H7 / Cortex-M7):
- Run-from-RAM memory layout with deterministic cycle timing
- Direct measurement using ARM Cortex-M `DWT::cycle_count()`
- Measures:
  - Single-sample processing throughput
  - 4-sample chunk processing
  - 1024-sample slice processing
  - In-place buffer overwriting

## Build
```bash
cargo build --release --target thumbv7em-none-eabihf
```
