//! Maximal-length LFSR sequences (m-sequences).
//!
//! Fibonacci LFSR using liquid-dsp default generator polynomials for degree
//! `m` in `2..=31`. Period is `2^m − 1`. No heap.

/// Default generator polynomials for degree `m = 2..=31` (liquid-dsp
/// `LIQUID_MSEQUENCE_GENPOLY_M*`).
const DEFAULT_GENPOLY: [u32; 30] = [
    0x0000_0003, // m=2
    0x0000_0006, 0x0000_000c, 0x0000_0014, 0x0000_0030, 0x0000_0060, 0x0000_00b8,
    0x0000_0110, 0x0000_0240, 0x0000_0500, 0x0000_0e08, 0x0000_1c80, 0x0000_3802,
    0x0000_6000, 0x0000_d008, 0x0001_2000, 0x0002_0400, 0x0007_2000, 0x0009_0000,
    0x0014_0000, 0x0030_0000, 0x0042_0000, 0x00e1_0000, 0x0100_0004, 0x0200_0023,
    0x0400_0013, 0x0800_0004, 0x1000_0002, 0x2000_0029, 0x4000_0004, // m=31
];

#[inline]
fn bdotprod(x: u32, y: u32) -> u32 {
    (x & y).count_ones() & 1
}

/// Fibonacci LFSR that produces a maximal-length binary sequence.
#[derive(Debug, Clone, Copy)]
pub struct MSequence {
    m: u32,
    g: u32,
    a: u32,
    n: u32,
    state: u32,
}

impl MSequence {
    /// Smallest supported generator degree.
    pub const MIN_DEGREE: u32 = 2;
    /// Largest supported generator degree.
    pub const MAX_DEGREE: u32 = 31;

    /// Builds an LFSR of degree `m` with generator `g` and initial state `a`.
    ///
    /// `m` must be in `2..=31`. `g` should include the `x^m` term. State `0`
    /// locks the generator (all zeros).
    pub fn new(m: u32, g: u32, a: u32) -> Result<Self, crate::types::Status> {
        if !(Self::MIN_DEGREE..=Self::MAX_DEGREE).contains(&m) {
            return Err(crate::types::Status::ArgumentError);
        }
        let n = (1u32 << m) - 1;
        Ok(Self {
            m,
            g,
            a,
            n,
            state: a,
        })
    }

    /// Degree from the MSB of `g`, initial state `1`.
    pub fn from_genpoly(g: u32) -> Result<Self, crate::types::Status> {
        if g < 2 {
            return Err(crate::types::Status::ArgumentError);
        }
        let m = 32 - g.leading_zeros();
        Self::new(m, g, 1)
    }

    /// Default primitive polynomial for degree `m` (`2..=31`), initial state `1`.
    pub fn default_degree(m: u32) -> Result<Self, crate::types::Status> {
        if !(Self::MIN_DEGREE..=Self::MAX_DEGREE).contains(&m) {
            return Err(crate::types::Status::ArgumentError);
        }
        let g = DEFAULT_GENPOLY[(m - 2) as usize];
        Self::from_genpoly(g)
    }

    /// Generator degree `m`.
    #[inline]
    pub fn degree(&self) -> u32 {
        self.m
    }

    /// Sequence period `2^m − 1`.
    #[inline]
    pub fn period(&self) -> u32 {
        self.n
    }

    /// Generator polynomial.
    #[inline]
    pub fn genpoly(&self) -> u32 {
        self.g
    }

    /// Current shift-register state.
    #[inline]
    pub fn state(&self) -> u32 {
        self.state
    }

    /// Overwrite the shift-register state.
    #[inline]
    pub fn set_state(&mut self, state: u32) {
        self.state = state;
    }

    /// Restore the shift register to the initial state given at construction.
    #[inline]
    pub fn reset(&mut self) {
        self.state = self.a;
    }

    /// Advance one step and return the new output bit (`0` or `1`).
    #[inline]
    pub fn advance(&mut self) -> u32 {
        let b = bdotprod(self.state, self.g);
        self.state = (self.state << 1) | b;
        self.state &= self.n;
        b
    }

    /// Pack `bits` successive output bits into an integer (MSB first).
    pub fn generate_symbol(&mut self, bits: u32) -> u32 {
        let mut s = 0u32;
        for _ in 0..bits {
            s = (s << 1) | self.advance();
        }
        s
    }

    /// XOR `buf` with a keystream of 8-bit symbols from this LFSR.
    pub fn scramble_bytes(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() {
            *b ^= self.generate_symbol(8) as u8;
        }
    }
}
