//! Control-plane configuration types (enabled by the `miniconf` feature).
//!
//! Runtime-tunable settings implement [`miniconf::Tree`], so they can be
//! configured over a network control plane (for example MQTT with
//! `miniconf_mqtt`) without pulling the filter implementation into the
//! configuration schema:
//!
//! ```no_run
//! # #[cfg(feature = "miniconf")] {
//! use embedded_dsp::config::BiquadSettings;
//! use miniconf::json_core;
//!
//! let mut settings = BiquadSettings::default();
//! json_core::set(&mut settings, "/b0", b"0.5").unwrap();
//! let filter = settings.build();
//! # }
//! ```

use miniconf::Tree;

use crate::filtering::{Biquad, BiquadClamp};

/// Runtime-tunable second-order-section (SOS) settings.
///
/// The default output limits are unbounded ([`f32::NEG_INFINITY`] /
/// [`f32::INFINITY`]) and all coefficients are zero.
#[derive(Clone, Copy, Debug, PartialEq, Tree)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct BiquadSettings {
    /// Numerator coefficient `b0`.
    pub b0: f32,
    /// Numerator coefficient `b1`.
    pub b1: f32,
    /// Numerator coefficient `b2`.
    pub b2: f32,
    /// Denominator coefficient `a1` (recurrence sign, added to `y1`).
    pub a1: f32,
    /// Denominator coefficient `a2` (recurrence sign, added to `y2`).
    pub a2: f32,
    /// Summing-junction offset added to the output.
    pub offset: f32,
    /// Lower output clamp.
    pub min: f32,
    /// Upper output clamp.
    pub max: f32,
}

impl Default for BiquadSettings {
    fn default() -> Self {
        Self {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            offset: 0.0,
            min: f32::NEG_INFINITY,
            max: f32::INFINITY,
        }
    }
}

impl BiquadSettings {
    /// Coefficient set `[b0, b1, b2, a1, a2]`.
    pub const fn coefficients(&self) -> Biquad<f32> {
        Biquad::new(self.b0, self.b1, self.b2, self.a1, self.a2)
    }

    /// Build the clamped biquad described by these settings.
    pub const fn build(&self) -> BiquadClamp<f32> {
        BiquadClamp::new(self.coefficients(), self.min, self.max, self.offset)
    }
}
