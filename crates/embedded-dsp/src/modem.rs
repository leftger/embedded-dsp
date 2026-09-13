//! Analog-style AM, FM, and SSB at complex baseband.
//!
//! Frequency modulation uses a phase accumulator (no sine LUT). AM-DSB is
//! envelope modulation. SSB reuses [`HilbertTransformF32`] for the analytic
//! signal (USB = `I + jQ`, LSB = `I − jQ`).

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::transform::HilbertTransformF32;
use crate::types::{Complex, Status};

/// Frequency modulator: `s = exp(j · φ)`, `φ += 2π kf m`.
#[derive(Debug, Clone, Copy)]
pub struct FmMod {
    kf: f32,
    phase: f32,
}

impl FmMod {
    /// `kf` is the modulation factor in cycles per sample per unit message (`> 0`).
    pub fn new(kf: f32) -> Result<Self, Status> {
        if !kf.is_finite() || kf <= 0.0 {
            return Err(Status::ArgumentError);
        }
        Ok(Self { kf, phase: 0.0 })
    }

    /// Current modulation factor.
    #[inline]
    pub fn kf(&self) -> f32 {
        self.kf
    }

    /// Clears the phase accumulator.
    #[inline]
    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Modulates one real message sample to complex baseband.
    pub fn modulate(&mut self, m: f32) -> Complex<f32> {
        let two_pi = 2.0 * core::f32::consts::PI;
        self.phase += two_pi * self.kf * m;
        self.phase %= two_pi;
        Complex {
            real: self.phase.cos(),
            imag: self.phase.sin(),
        }
    }
}

/// Polar FM discriminator: `m = arg(conj(z[n−1]) z[n]) / (2π kf)`.
#[derive(Debug, Clone, Copy)]
pub struct FmDemod {
    ref_scale: f32,
    prev: Complex<f32>,
}

impl FmDemod {
    /// Same `kf` as the matching [`FmMod`].
    pub fn new(kf: f32) -> Result<Self, Status> {
        if !kf.is_finite() || kf <= 0.0 {
            return Err(Status::ArgumentError);
        }
        Ok(Self {
            ref_scale: 1.0 / (2.0 * core::f32::consts::PI * kf),
            prev: Complex::new(0.0, 0.0),
        })
    }

    /// Clears the previous-sample memory.
    #[inline]
    pub fn reset(&mut self) {
        self.prev = Complex::new(0.0, 0.0);
    }

    /// Demodulates one complex baseband sample.
    pub fn demodulate(&mut self, z: Complex<f32>) -> f32 {
        let re = self.prev.real * z.real + self.prev.imag * z.imag;
        let im = self.prev.real * z.imag - self.prev.imag * z.real;
        self.prev = z;
        im.atan2(re) * self.ref_scale
    }
}

/// Double-sideband AM at complex baseband: `y = μ m + [1 if carrier]`.
#[derive(Debug, Clone, Copy)]
pub struct AmDsb {
    mod_index: f32,
    suppressed_carrier: bool,
}

impl AmDsb {
    /// `mod_index` is `μ` (`> 0`). `suppressed_carrier` omits the `+ 1` term.
    pub fn new(mod_index: f32, suppressed_carrier: bool) -> Result<Self, Status> {
        if !mod_index.is_finite() || mod_index <= 0.0 {
            return Err(Status::ArgumentError);
        }
        Ok(Self {
            mod_index,
            suppressed_carrier,
        })
    }

    /// Modulation depth `μ`.
    #[inline]
    pub fn mod_index(&self) -> f32 {
        self.mod_index
    }

    /// Whether the carrier term is omitted.
    #[inline]
    pub fn suppressed_carrier(&self) -> bool {
        self.suppressed_carrier
    }

    /// Modulates one real message sample.
    pub fn modulate(&self, m: f32) -> Complex<f32> {
        let a = self.mod_index * m + if self.suppressed_carrier { 0.0 } else { 1.0 };
        Complex { real: a, imag: 0.0 }
    }

    /// Envelope demodulator. With carrier this is `(|y| − 1) / μ`; suppressed-carrier
    /// AM recovers `|m|`, not the signed message (use a Costas loop for that).
    pub fn demodulate_envelope(&self, y: Complex<f32>) -> f32 {
        let mag = (y.real * y.real + y.imag * y.imag).sqrt();
        if self.suppressed_carrier {
            mag / self.mod_index
        } else {
            (mag - 1.0) / self.mod_index
        }
    }
}

/// SSB sideband selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SsbSideband {
    /// Upper sideband: analytic `I + j Q`.
    Usb,
    /// Lower sideband: analytic `I − j Q`.
    Lsb,
}

/// SSB modulator using a caller-owned [`HilbertTransformF32`].
pub struct SsbMod<'a> {
    ht: HilbertTransformF32<'a>,
    mod_index: f32,
    sideband: SsbSideband,
    suppressed_carrier: bool,
}

impl<'a> SsbMod<'a> {
    /// Wraps an existing Hilbert transformer. `mod_index` must be `> 0`.
    pub fn new(
        ht: HilbertTransformF32<'a>,
        sideband: SsbSideband,
        mod_index: f32,
        suppressed_carrier: bool,
    ) -> Result<Self, Status> {
        if !mod_index.is_finite() || mod_index <= 0.0 {
            return Err(Status::ArgumentError);
        }
        Ok(Self {
            ht,
            mod_index,
            sideband,
            suppressed_carrier,
        })
    }

    /// Hilbert group delay in samples (message alignment).
    #[inline]
    pub fn group_delay(&self) -> usize {
        self.ht.group_delay()
    }

    /// Clears Hilbert delay state.
    #[inline]
    pub fn reset(&mut self) {
        self.ht.reset();
    }

    /// Modulates one real message sample to complex baseband.
    pub fn modulate(&mut self, m: f32) -> Complex<f32> {
        let (i, q) = self.ht.process_sample(m);
        let q = match self.sideband {
            SsbSideband::Usb => q,
            SsbSideband::Lsb => -q,
        };
        let mut y = Complex {
            real: self.mod_index * i,
            imag: self.mod_index * q,
        };
        if !self.suppressed_carrier {
            y.real += 1.0;
        }
        y
    }
}

/// SSB demodulator: delayed `I` plus Hilbert(`Q`), then USB `I − H(Q)` / LSB `I + H(Q)`.
///
/// `i_delay` must be the same length as the Hilbert tap/state buffers.
pub struct SsbDemod<'a> {
    ht: HilbertTransformF32<'a>,
    i_delay: &'a mut [f32],
    mod_index: f32,
    sideband: SsbSideband,
}

impl<'a> SsbDemod<'a> {
    /// Wraps a Hilbert transformer and an I-channel delay line of equal length.
    pub fn new(
        ht: HilbertTransformF32<'a>,
        i_delay: &'a mut [f32],
        sideband: SsbSideband,
        mod_index: f32,
    ) -> Result<Self, Status> {
        if !mod_index.is_finite() || mod_index <= 0.0 {
            return Err(Status::ArgumentError);
        }
        if i_delay.len() != ht.num_taps {
            return Err(Status::LengthError);
        }
        i_delay.fill(0.0);
        Ok(Self {
            ht,
            i_delay,
            mod_index,
            sideband,
        })
    }

    /// Hilbert / I-delay group delay in samples.
    #[inline]
    pub fn group_delay(&self) -> usize {
        self.ht.group_delay()
    }

    /// Clears Hilbert and I-delay state.
    pub fn reset(&mut self) {
        self.ht.reset();
        self.i_delay.fill(0.0);
    }

    /// Demodulates one complex baseband sample.
    pub fn demodulate(&mut self, y: Complex<f32>) -> f32 {
        let n = self.i_delay.len();
        for k in (1..n).rev() {
            self.i_delay[k] = self.i_delay[k - 1];
        }
        self.i_delay[0] = y.real;
        let yi = self.i_delay[self.ht.group_delay()];
        let (_, yq) = self.ht.process_sample(y.imag);
        let comb = match self.sideband {
            SsbSideband::Lsb => yi + yq,
            SsbSideband::Usb => yi - yq,
        };
        0.5 * comb / self.mod_index
    }
}
