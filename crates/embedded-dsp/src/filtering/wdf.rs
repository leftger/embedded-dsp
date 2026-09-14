//! Wave digital filters (allpass chain).

// ─────────────────────────────────────────────────────────────────────────────
// Wave Digital Filters (allpass chain)
// Ported from the `idsp` crate by the Sinara/ARTIQ project, with a corrected
// per-stage state update.
// ─────────────────────────────────────────────────────────────────────────────

/// Two-port adapter architecture selector.
///
/// Each architecture is a nibble in the const generic of [`Wdf`] and encodes
/// the optimal scaled form for a given allpass coefficient range.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Tpa {
    /// Terminate (coefficient 0).
    Z = 0x0,
    /// `1 > g > 1/2`: `a = g - 1`.
    A = 0xA,
    /// `1/2 >= g > 0`: `a = -g`.
    B = 0xB,
    /// Alternative to `B`.
    B1 = 0xE,
    /// `g = 0`.
    X = 0x1,
    /// `-1/2 <= g < 0`: `a = g`.
    C = 0xC,
    /// Alternative to `C`.
    C1 = 0xF,
    /// `-1 < g < -1/2`: `a = -(1 + g)`.
    D = 0xD,
}

impl From<u8> for Tpa {
    #[inline]
    fn from(value: u8) -> Self {
        match value {
            0xa => Tpa::A,
            0xb => Tpa::B,
            0xe => Tpa::B1,
            0x1 => Tpa::X,
            0xc => Tpa::C,
            0xf => Tpa::C1,
            0xd => Tpa::D,
            _ => Tpa::Z,
        }
    }
}

impl Tpa {
    /// Quantize the allpass coefficient `g` for this architecture.
    ///
    /// Returns the Q32.32 fixed-point adapter coefficient, or `None` if `g`
    /// does not fit the architecture's scaled range.
    fn quantize(self, g: f64) -> Option<i32> {
        // Use -0.5 <= a <= 0 instead of the usual positive range so that -0.5
        // exactly fits the Q32.32 fixed-point range.
        let a = match self {
            Self::Z => 0.0,
            Self::A => g - 1.0,
            Self::B | Self::B1 => -g,
            Self::X => 0.0,
            Self::C | Self::C1 => g,
            Self::D => -1.0 - g,
        };
        (-0.5..=0.0).contains(&a).then_some((a * 4294967296.0) as i32)
    }

    /// Fixed-point multiply: `(c * a) >> 32` with wrapping (Q32.32 coefficient).
    #[cfg(feature = "pipeline")]
    #[inline]
    fn mul(self, c: i32, a: i32) -> i32 {
        ((c as i64).wrapping_mul(a as i64) >> 32) as i32
    }

    /// Two-port adapter wave computation.
    ///
    /// Takes `[a1, a2]` (incident wave from the previous stage and the delay
    /// state) and returns `[b1, b2]`: the output wave to the next stage and
    /// the new delay state.
    #[cfg(feature = "pipeline")]
    #[inline]
    fn adapt(&self, x: [i32; 2], a: i32) -> [i32; 2] {
        match self {
            Tpa::A => {
                let c = x[1] - x[0];
                let y = self.mul(c, a).wrapping_add(x[1]);
                [y.wrapping_add(c), y]
            }
            Tpa::B => {
                let c = x[0] - x[1];
                let y = self.mul(c, a).wrapping_add(x[1]);
                [y, y.wrapping_add(c)]
            }
            Tpa::B1 => {
                let c = x[0] - x[1];
                let y = self.mul(c, a);
                [y.wrapping_add(x[1]), y.wrapping_add(x[0])]
            }
            Tpa::X => [x[1], x[0]],
            Tpa::C => {
                let c = x[1] - x[0];
                let y = self.mul(c, a).wrapping_sub(x[1]);
                [y, y.wrapping_add(c)]
            }
            Tpa::C1 => {
                let c = x[1] - x[0];
                let y = self.mul(c, a);
                [y.wrapping_sub(x[1]), y.wrapping_sub(x[0])]
            }
            Tpa::D => {
                let c = x[0] - x[1];
                let y = self.mul(c, a).wrapping_sub(x[1]);
                [y.wrapping_add(c), y]
            }
            Tpa::Z => x,
        }
    }
}

/// Wave digital filter: a cascade of `N` first-order allpass sections.
///
/// The `M` const generic encodes the two-port adapter architecture, one nibble
/// per stage (least significant nibble = first stage). All arithmetic is
/// wrapping 32-bit integer with Q32.32 coefficients — no floating point.
///
/// # Ported from
/// The `idsp` crate by the Sinara/ARTIQ project.
#[derive(Debug, Clone)]
pub struct Wdf<const N: usize, const M: u32> {
    /// Q32.32 adapter coefficients, one per allpass section.
    pub a: [i32; N],
}

impl<const N: usize, const M: u32> Default for Wdf<N, M> {
    fn default() -> Self {
        Self { a: [0; N] }
    }
}

impl<const N: usize, const M: u32> Wdf<N, M> {
    /// Quantize allpass pole coefficients `g` (one per section, `|g| < 1`)
    /// using the architecture encoded in `M`.
    pub fn quantize(g: &[f64; N]) -> Option<Self> {
        let mut a = [0i32; N];
        let mut m = M;
        for (a, g) in a.iter_mut().zip(g) {
            *a = Tpa::from((m & 0xf) as u8).quantize(*g)?;
            m >>= 4;
        }
        debug_assert_eq!(m, 0);
        Some(Self { a })
    }
}

/// State for [`Wdf`]: one delay element per allpass section.
#[derive(Clone, Debug)]
pub struct WdfState<const N: usize> {
    /// Section delay states.
    pub z: [i32; N],
}

impl<const N: usize> Default for WdfState<N> {
    fn default() -> Self {
        Self { z: [0; N] }
    }
}

#[cfg(feature = "pipeline")]
impl<const N: usize, const M: u32> crate::pipeline::SplitProcess<i32, i32, WdfState<N>>
    for Wdf<N, M>
{
    #[inline]
    fn process_with_state(&mut self, state: &mut WdfState<N>, x: i32) -> i32 {
        let mut x = x;
        let mut m = M;
        for (a, z) in self.a.iter().zip(state.z.iter_mut()) {
            let [y, next] = Tpa::from((m & 0xf) as u8).adapt([x, *z], *a);
            *z = next; // update this section's delay state
            x = y;     // output wave feeds the next section
            m >>= 4;
        }
        debug_assert_eq!(m, 0);
        x
    }
}
