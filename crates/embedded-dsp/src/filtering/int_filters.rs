//! Integer lowpass filters ported from `idsp`.

// ─────────────────────────────────────────────────────────────────────────────
// Integer Lowpass Filter (ported from idsp)
// ─────────────────────────────────────────────────────────────────────────────

/// Arbitrary-order integer lowpass filter with high dynamic range. DC gain is 1.
///
/// Supports order `N = 1` (first-order) and `N = 2` (second-order Butterworth);
/// any other `N` is rejected at compile time. The filter saturates cleanly
/// towards the `i32` range.
///
/// # Coefficient Calculation
///
/// **First-order** (`N = 1`): `k[0] = π * (1 << 31) * f0 / fn`  
/// where `f0` is the 3 dB corner frequency and `fn` is the Nyquist frequency.
///
/// **Second-order Butterworth** (`N = 2`): `k = [k_sq >> 32, -k / q]`  
/// where `q = 1/sqrt(2)` and `k` is as above.
///
/// Both variants have zeros at Nyquist, optimised for Cortex-M7.
///
/// ```
/// # use embedded_dsp::filtering::IntLowpass;
/// let mut lp = IntLowpass::<1>::new([674_651_885]);
/// assert_eq!(lp.process(1 << 24), 2_635_358);
/// ```
///
/// Unsupported orders are a compile error rather than a runtime panic:
///
/// ```compile_fail
/// # use embedded_dsp::filtering::IntLowpass;
/// let _ = IntLowpass::<3>::new([0, 0, 0]);
/// ```
///
/// Ported from the `idsp` crate by the Sinara/ARTIQ project.
#[derive(Clone, Debug)]
pub struct IntLowpass<const N: usize> {
    /// Lead/lag gain coefficients in Q1.31 fixed-point.
    pub k: [i32; N],
    /// Wide internal state accumulators.
    state: [i64; N],
}

impl<const N: usize> Default for IntLowpass<N>
where
    [i32; N]: Default,
{
    fn default() -> Self {
        const { assert!(N == 1 || N == 2, "IntLowpass supports only N = 1 or N = 2") };
        Self { k: Default::default(), state: [0i64; N] }
    }
}

impl<const N: usize> IntLowpass<N> {
    /// Create a new filter from gain coefficients.
    ///
    /// # Panics
    /// Fails to compile unless `N` is `1` or `2`.
    pub fn new(k: [i32; N]) -> Self {
        const { assert!(N == 1 || N == 2, "IntLowpass supports only N = 1 or N = 2") };
        Self { k, state: [0i64; N] }
    }

    /// Reset internal state to zero.
    pub fn reset(&mut self) {
        self.state = [0i64; N];
    }

    /// Process a single sample and return the filtered output.
    ///
    /// # Panics
    /// Fails to compile unless `N` is `1` or `2`.
    pub fn process(&mut self, x: i32) -> i32 {
        const { assert!(N == 1 || N == 2, "IntLowpass supports only N = 1 or N = 2") };
        if N == 1 {
            let d = x.saturating_sub((self.state[0] >> 32) as i32) as i64
                * self.k[0] as i64;
            self.state[0] += d;
            let y = (self.state[0] >> 32) as i32;
            self.state[0] += d;
            y
        } else {
            let mut d = x.saturating_sub((self.state[0] >> 32) as i32) as i64
                * self.k[0] as i64;
            d += (self.state[1] >> 32) * self.k[1] as i64;
            self.state[1] += d;
            self.state[0] += self.state[1];
            let y = (self.state[0] >> 32) as i32;
            self.state[0] += self.state[1];
            self.state[1] += d;
            y
        }
    }
}

/// First-order integer lowpass (alias for `IntLowpass<1>`).
pub type IntLowpass1 = IntLowpass<1>;
/// Second-order integer lowpass (alias for `IntLowpass<2>`).
pub type IntLowpass2 = IntLowpass<2>;
