//! Dithering utilities: PRNG and triangular/uniform noise generators.
//!
//! Provides a tiny `no_std`-compatible PRNG (Marsaglia xorshift32) plus
//! uniform and triangular dither generators for ADC/DAC quantisation noise shaping.
//!
//! Ported from the `idsp` crate by the Sinara/ARTIQ project.

/// Marsaglia 32-bit xorshift PRNG. Period: 2^32 − 1 (never produces 0).
///
/// # Example
/// ```
/// use embedded_dsp::dither::XorShift32;
/// let mut rng = XorShift32::new(1);
/// assert_ne!(rng.next_u32(), 0);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct XorShift32(u32);

impl Default for XorShift32 {
    fn default() -> Self {
        Self(1)
    }
}

impl XorShift32 {
    /// Create a new generator. Seeds of 0 are remapped to 1.
    pub const fn new(seed: u32) -> Self {
        Self(if seed == 0 { 1 } else { seed })
    }

    /// Generate the next pseudo-random `u32` (never 0).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

impl Iterator for XorShift32 {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_u32())
    }
}

/// Uniform byte-granularity noise in `[0, 255]`. Caches 4 bytes per PRNG call.
///
/// # Example
/// ```
/// use embedded_dsp::dither::Uniform;
/// let mut u = Uniform::default();
/// let _: u8 = u.sample();
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct Uniform {
    prng: XorShift32,
    cache: u32,
    idx: u8,
}

impl Uniform {
    /// Create from a seed.
    pub fn new(seed: u32) -> Self {
        Self {
            prng: XorShift32::new(seed),
            cache: 0,
            idx: 0,
        }
    }

    /// Draw the next uniform byte.
    #[inline]
    pub fn sample(&mut self) -> u8 {
        if self.idx == 0 {
            self.cache = self.prng.next_u32();
            self.idx = 3;
        } else {
            self.idx -= 1;
            self.cache >>= 8;
        }
        (self.cache & 0xff) as u8
    }
}

impl Iterator for Uniform {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sample())
    }
}

/// Triangular-distributed noise in `[-(1<<8), (1<<8) - 1]`.
///
/// Produces triangular noise by subtracting two uniform byte samples —
/// the optimal dither distribution for minimising quantisation distortion.
///
/// # Example
/// ```
/// use embedded_dsp::dither::Triangular;
/// let mut t = Triangular::default();
/// let s = t.sample();
/// assert!(s >= -(1 << 8) && s < (1 << 8));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct Triangular {
    uniform: Uniform,
}

impl Triangular {
    /// Create from a seed.
    pub fn new(seed: u32) -> Self {
        Self {
            uniform: Uniform::new(seed),
        }
    }

    /// Draw the next triangular sample.
    #[inline]
    pub fn sample(&mut self) -> i16 {
        self.uniform.sample() as i8 as i16 - self.uniform.sample() as i8 as i16
    }
}

impl Iterator for Triangular {
    type Item = i16;
    fn next(&mut self) -> Option<Self::Item> {
        Some(self.sample())
    }
}
