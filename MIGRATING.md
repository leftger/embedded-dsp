# Migrating to 0.6

0.6 is a breaking minor release. It removes two layers of backwards-compatibility
shims that existed only to keep pre-0.6 call sites compiling, so that the public
API tells the truth about what this crate actually supports. There is one
additive change you may want to adopt too.

If you are on 0.5.x, the compiler will point at every call site that needs an
edit; the tables below give the mechanical replacements.

---

## 1. The width-suffixed aliases and wrappers are gone

Since the sample-genericization work, every filter could be written once over
`DspSample`, but the old per-width names were kept as type aliases and thin
wrappers. They are removed; call the generic API directly.

### Type aliases

| Removed | Replacement |
| :-- | :-- |
| `FirInstanceF32<'a>` / `FirInstanceQ31<'a>` / `FirInstanceQ15<'a>` | `FirInstance<'a, f32>` / `FirInstance<'a, q31>` / `FirInstance<'a, q15>` |
| `BiquadCascadeInstanceF32<'a>` / `…Q15` / `…Q31` | `BiquadCascadeInstance<'a, f32>` / `…, q15` / `…, q31` |
| `BiquadCascadeDf2tInstanceF32<'a>` / `…Q15` / `…Q31` | `BiquadCascadeDf2tInstance<'a, f32>` / … |
| `LmsInstanceF32<'a>` / `LmsInstanceQ15<'a>` | `LmsInstance<'a, f32>` / `LmsInstance<'a, q15>` |
| `NlmsInstanceF32<'a>` / `NlmsInstanceQ15<'a>` | `NlmsInstance<'a, f32>` / `NlmsInstance<'a, q15>` |
| `PidInstanceF32` / `PidInstanceQ31` / `PidInstanceQ15` | `PidInstance<f32>` / `PidInstance<q31>` / `PidInstance<q15>` |
| `HilbertTransformF32<'a>` / `HilbertTransformQ15<'a>` | `HilbertTransform<'a, f32>` / `HilbertTransform<'a, q15>` |
| `RecursiveMovingAverageQ15<N>` | `RecursiveMovingAverage<q15, N>` |

### Wrapper functions

| Removed | Replacement |
| :-- | :-- |
| `fir_f32` / `fir_q31` / `fir_q15` | `fir` |
| `biquad_cascade_df1_f32` / `…_q15` / `…_q31` | `biquad_cascade_df1` |
| `biquad_cascade_df2t_f32` / `…_q15` / `…_q31` | `biquad_cascade_df2t` |
| `lms_f32` / `lms_q15` | `lms` |
| `lms_leaky_f32` / `lms_leaky_q15` | `lms_leaky` |
| `nlms_f32` / `nlms_q15` | `nlms` |
| `pid_f32` / `pid_q31` / `pid_q15` | `PidInstance::<T>::process` (or `instance.process(x)`) |

The per-type `*_f32` / `*_q15` / `*_q31` **functions** in `distance`,
`transform`, `interpolation`, and `audio` are *not* affected — those are the
primary API, not shims.

### Before / after

```rust
// 0.5
use embedded_dsp::filtering::{
    fir_q15, FirInstanceQ15, biquad_cascade_df1_f32, BiquadCascadeInstanceF32,
};
let mut fir = FirInstanceQ15::init(32, &coeffs, &mut state);
fir_q15(&mut fir, &input, &mut output);

// 0.6
use embedded_dsp::filtering::{fir, FirInstance};
let mut fir = FirInstance::<q15>::init(32, &coeffs, &mut state);
fir(&mut fir, &input, &mut output);
```

> Watch for locals that shadow a generic function name. A local named `fir`,
> `lms`, `nlms`, or `biquad_cascade_df1` will hide the free function once the
> suffixed wrapper is gone; rename the local (`fir_inst`, …) or call through the
> fully-qualified path.

### Features that changed meaning

`filtering` no longer re-exports `Dsm` or `XorShift32`. Enable the `dsm` /
`dither` features (both are in `full`) and use those modules.

---

## 2. `filtering::Dsm` and `filtering::XorShift32` are gone

Two public types were duplicated: `filtering::Dsm` / `filtering::XorShift32`
shadowed the dedicated `dsm` / `dither` modules, which could not be re-exported
at the crate root while the duplicates existed. The dedicated modules are now the
single implementation.

| Removed | Replacement |
| :-- | :-- |
| `filtering::Dsm<K>` | `dsm::Dsm<K>` (also `embedded_dsp::Dsm`) |
| `filtering::XorShift32` | `dither::XorShift32` (also `embedded_dsp::XorShift32`) |

The surviving `dsm::Dsm` is the `idsp`-verified carry-chained MASH implementation,
not the old `filtering::Dsm` (whose accumulator chain differed). Its API is
`Default` + `process(x) -> i8` + `reset()`. The historical `new()` and
`process_sample()` convenience names are deliberately **not** carried over;
`XorShift32` keeps `next_u32` and gains the `next_f32` / `tpdf_dither_f32`
helpers that only the removed duplicate had.

```rust
// 0.5
use embedded_dsp::filtering::Dsm;
let mut dsm = Dsm::<3>::new();
let out = dsm.process_sample(x);

// 0.6
use embedded_dsp::dsm::Dsm;
let mut dsm = Dsm::<3>::default();
let out = dsm.process(x);
```

---

## 3. Additive: unified audio-EQ builder

Not a migration, but the replacement for hand-rolling RBJ coefficient calls.
`filter_design` now has one validating builder for every Audio EQ Cookbook
response — low/high/band-pass, all-pass, notch, peaking, both shelves, and the
`IHo` integrator-over-harmonic-oscillator section — plus a WebAudio export.

```rust
use embedded_dsp::filter_design::{BiquadType, EqFilter, WebAudioFilter};

// Fluent, validating, generic over every response type.
let peaking = EqFilter::new(1_000.0, 48_000.0).q(0.707).gain_db(6.0).peaking();
let band = EqFilter::new(1_000.0, 48_000.0).bandwidth_octaves(1.0).bandpass();

// `try_build` reports out-of-range parameters instead of emitting a bad biquad.
let iho = EqFilter::new(2_000.0, 48_000.0)
    .q(0.707)
    .gain_db(-6.0)
    .try_build(BiquadType::Iho)?;

// WebAudio `BiquadFilterNode` parameters, detune included.
let wa = WebAudioFilter {
    frequency_hz: 1_000.0,
    detune_cents: 1_200.0, // +1 octave
    ..Default::default()
};
let coeffs = wa.build();
```

Output is `[b0, b1, b2, a1, a2]` in this crate's Direct Form I convention, ready
for `BiquadCascadeInstance` or `Biquad`.
