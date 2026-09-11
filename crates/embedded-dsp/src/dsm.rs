//! Delta-Sigma Modulator (MASH-1^K architecture).
//!
//! Provides noise-shaped output from a `u32` input, useful for DAC
//! oversampling and precision frequency synthesis on embedded targets.
//!
//! Ported from the `idsp` crate by the Sinara/ARTIQ project.

/// MASH-(1)^K Delta-Sigma Modulator.
///
/// - `K` stages (0 ≤ K ≤ 8; `K = 0` always outputs 0).
/// - Output range: `1 - (1 << (K-1))..=(1 << (K-1))`.
/// - For a constant input `x0`, long-run average output = `x0 / 2^32`.
/// - Noise spectral density rises at `K × 20 dB/decade` — ideal for
///   high-order noise shaping in audio DACs and NCOs.
///
/// # Example
///
/// ```
/// use embedded_dsp::dsm::Dsm;
/// let mut d = Dsm::<3>::default();
/// let x = 0x8765_4321u32;
/// let n: u32 = 1 << 20;
/// let sum: f64 = (0..n).map(|_| d.process(x) as f64).sum();
/// let mean = sum / n as f64;
/// let expected = x as f64 / (1u64 << 32) as f64;
/// assert!((mean - expected).abs() < 0.01);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dsm<const K: usize> {
    a: [u32; K],
    c: [i8; K],
}

impl<const K: usize> Default for Dsm<K> {
    fn default() -> Self {
        Self {
            a: [0u32; K],
            c: [0i8; K],
        }
    }
}

impl<const K: usize> Dsm<K> {
    /// Process one input sample, returning the noise-shaped output.
    pub fn process(&mut self, x: u32) -> i8 {
        if K == 0 {
            return 0;
        }
        let mut d = 0i8;
        // MASH accumulator chain
        let mut carry_in = x;
        for a in self.a.iter_mut() {
            let (new_a, carry) = a.overflowing_add(carry_in);
            *a = new_a;
            d = (d << 1) | carry as i8;
            carry_in = new_a;
        }
        if K == 1 {
            return d & 1;
        }
        // Error-feedback noise shaping
        let mut y = d & 1;
        let mut d_shifted = d >> 1;
        for c in self.c.iter_mut().take(K - 1) {
            let new_y = (d_shifted & 1) + y - *c;
            d_shifted >>= 1;
            *c = y;
            y = new_y;
        }
        y
    }

    /// Reset internal accumulators.
    pub fn reset(&mut self) {
        self.a = [0u32; K];
        self.c = [0i8; K];
    }
}
