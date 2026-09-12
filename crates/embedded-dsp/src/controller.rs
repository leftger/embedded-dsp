//! Controller functions (PID motor control, Clarke transform, Park transform, Inverse Clarke, Inverse Park).

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::*;

// --- PID Controller (f32) ---

/// Instance structure for the floating-point PID Control.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PidInstanceF32 {
    pub a0: f32,
    pub a1: f32,
    pub a2: f32,
    pub state: [f32; 3],
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
}

impl PidInstanceF32 {
    pub fn new(kp: f32, ki: f32, kd: f32) -> Self {
        let mut pid = Self {
            a0: 0.0,
            a1: 0.0,
            a2: 0.0,
            state: [0.0; 3],
            kp,
            ki,
            kd,
        };
        pid.init(1);
        pid
    }

    pub fn init(&mut self, reset_state_flag: i32) {
        self.a0 = self.kp + self.ki + self.kd;
        self.a1 = -self.kp - 2.0 * self.kd;
        self.a2 = self.kd;
        if reset_state_flag != 0 {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.state = [0.0; 3];
    }

    pub fn process(&mut self, in_val: f32) -> f32 {
        let out =
            self.state[2] + self.a0 * in_val + self.a1 * self.state[0] + self.a2 * self.state[1];
        self.state[1] = self.state[0];
        self.state[0] = in_val;
        self.state[2] = out;
        out
    }
}

pub fn pid_f32(instance: &mut PidInstanceF32, in_val: f32) -> f32 {
    instance.process(in_val)
}

// --- PID Controller (Q31) ---

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PidInstanceQ31 {
    pub a0: q31,
    pub a1: q31,
    pub a2: q31,
    pub state: [q31; 3],
    pub kp: q31,
    pub ki: q31,
    pub kd: q31,
}

impl PidInstanceQ31 {
    pub fn new(kp: q31, ki: q31, kd: q31) -> Self {
        let mut pid = Self {
            a0: q31::ZERO,
            a1: q31::ZERO,
            a2: q31::ZERO,
            state: [q31::ZERO; 3],
            kp,
            ki,
            kd,
        };
        pid.init(1);
        pid
    }

    pub fn init(&mut self, reset_state_flag: i32) {
        self.a0 = self.kp.saturating_add(self.ki).saturating_add(self.kd);
        self.a1 = (-self.kp).saturating_sub(self.kd.wrapping_mul_int(2));
        self.a2 = self.kd;
        if reset_state_flag != 0 {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.state = [q31::ZERO; 3];
    }

    /// Each MAC term wraps individually (`i32`-wide, not the wider
    /// intermediate a naive i64 accumulator would allow), and only the final
    /// sum saturates. At the extreme edge where a coefficient and its paired
    /// value are both exactly Q31 `MIN` (`-1.0`), the term wraps to `MIN`
    /// instead of the mathematically exact `+2^31`, which can flip that
    /// term's sign in the final sum. This only affects that single boundary
    /// input combination.
    pub fn process(&mut self, in_val: q31) -> q31 {
        let t0 = self.a0.wrapping_mul(in_val).to_bits();
        let t1 = self.a1.wrapping_mul(self.state[0]).to_bits();
        let t2 = self.a2.wrapping_mul(self.state[1]).to_bits();
        let acc = (self.state[2].to_bits() as i64) + (t0 as i64) + (t1 as i64) + (t2 as i64);
        let out = q31::from_bits(acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32);
        self.state[1] = self.state[0];
        self.state[0] = in_val;
        self.state[2] = out;
        out
    }
}

pub fn pid_q31(instance: &mut PidInstanceQ31, in_val: q31) -> q31 {
    instance.process(in_val)
}

// --- PID Controller (Q15) ---

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PidInstanceQ15 {
    pub a0: q15,
    pub a1: q15,
    pub a2: q15,
    pub state: [q15; 3],
    pub kp: q15,
    pub ki: q15,
    pub kd: q15,
}

impl PidInstanceQ15 {
    pub fn new(kp: q15, ki: q15, kd: q15) -> Self {
        let mut pid = Self {
            a0: q15::ZERO,
            a1: q15::ZERO,
            a2: q15::ZERO,
            state: [q15::ZERO; 3],
            kp,
            ki,
            kd,
        };
        pid.init(1);
        pid
    }

    pub fn init(&mut self, reset_state_flag: i32) {
        self.a0 = self.kp.saturating_add(self.ki).saturating_add(self.kd);
        self.a1 = (-self.kp).saturating_sub(self.kd.wrapping_mul_int(2));
        self.a2 = self.kd;
        if reset_state_flag != 0 {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.state = [q15::ZERO; 3];
    }

    /// Same per-term wrapping caveat as [`PidInstanceQ31::process`]: at the
    /// extreme edge where a coefficient and its paired value are both
    /// exactly Q15 `MIN` (`-1.0`), that term wraps to `MIN` instead of the
    /// mathematically exact `+2^15`.
    pub fn process(&mut self, in_val: q15) -> q15 {
        let t0 = self.a0.wrapping_mul(in_val).to_bits();
        let t1 = self.a1.wrapping_mul(self.state[0]).to_bits();
        let t2 = self.a2.wrapping_mul(self.state[1]).to_bits();
        let acc = (self.state[2].to_bits() as i32) + (t0 as i32) + (t1 as i32) + (t2 as i32);
        let out = q15::from_bits(acc.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        self.state[1] = self.state[0];
        self.state[0] = in_val;
        self.state[2] = out;
        out
    }
}

pub fn pid_q15(instance: &mut PidInstanceQ15, in_val: q15) -> q15 {
    instance.process(in_val)
}

// --- Clarke Transform ---

/// Forward Clarke transform for f32: 3-phase (ia, ib) -> 2-phase (alpha, beta).
pub fn clarke_f32(ia: f32, ib: f32, p_alpha: &mut f32, p_beta: &mut f32) {
    *p_alpha = ia;
    let inv_sqrt_3 = 0.57735026919f32; // 1 / sqrt(3)
    *p_beta = (ia + 2.0 * ib) * inv_sqrt_3;
}

/// Inverse Clarke transform for f32: 2-phase (alpha, beta) -> 3-phase (ia, ib).
pub fn inv_clarke_f32(alpha: f32, beta: f32, p_ia: &mut f32, p_ib: &mut f32) {
    *p_ia = alpha;
    let sqrt_3_div_2 = 0.86602540378f32; // sqrt(3) / 2
    *p_ib = -0.5 * alpha + sqrt_3_div_2 * beta;
}

// --- Park Transform ---

/// Forward Park transform for f32: 2-phase stationary (alpha, beta) + angle theta (rad) -> 2-phase rotating (d, q).
pub fn park_f32(alpha: f32, beta: f32, theta: f32, p_d: &mut f32, p_q: &mut f32) {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    *p_d = alpha * cos_t + beta * sin_t;
    *p_q = -alpha * sin_t + beta * cos_t;
}

/// Inverse Park transform for f32: 2-phase rotating (d, q) + angle theta (rad) -> 2-phase stationary (alpha, beta).
pub fn inv_park_f32(d: f32, q: f32, theta: f32, p_alpha: &mut f32, p_beta: &mut f32) {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    *p_alpha = d * cos_t - q * sin_t;
    *p_beta = d * sin_t + q * cos_t;
}

const INV_SQRT3_Q15: i32 = 18919; // 1/√3 in Q15
const SQRT3_2_Q15: i32 = 28378; // √3/2 in Q15

#[inline]
fn sat_q15_i32(v: i32) -> q15 {
    q15::from_bits(v.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
}

/// Forward Clarke transform in Q15.
pub fn clarke_q15(ia: q15, ib: q15, p_alpha: &mut q15, p_beta: &mut q15) {
    *p_alpha = ia;
    let acc = (ia.to_bits() as i32 + 2 * ib.to_bits() as i32) * INV_SQRT3_Q15;
    *p_beta = sat_q15_i32(acc >> 15);
}

/// Inverse Clarke transform in Q15.
pub fn inv_clarke_q15(alpha: q15, beta: q15, p_ia: &mut q15, p_ib: &mut q15) {
    *p_ia = alpha;
    let acc = -((alpha.to_bits() as i32) << 14) + SQRT3_2_Q15 * beta.to_bits() as i32;
    *p_ib = sat_q15_i32(acc >> 15);
}

/// Forward Park transform in Q15. `sin_t` / `cos_t` are Q15 sine/cosine of θ
/// (CMSIS-style; e.g. take `sin_cos_q31` and shift `>> 16`).
pub fn park_q15(alpha: q15, beta: q15, sin_t: q15, cos_t: q15, p_d: &mut q15, p_q: &mut q15) {
    let (alpha, beta, sin_t, cos_t) = (
        alpha.to_bits() as i32,
        beta.to_bits() as i32,
        sin_t.to_bits() as i32,
        cos_t.to_bits() as i32,
    );
    let d = (alpha * cos_t + beta * sin_t) >> 15;
    let q = (-alpha * sin_t + beta * cos_t) >> 15;
    *p_d = sat_q15_i32(d);
    *p_q = sat_q15_i32(q);
}

/// Inverse Park transform in Q15. `sin_t` / `cos_t` are Q15 sine/cosine of θ.
pub fn inv_park_q15(d: q15, q: q15, sin_t: q15, cos_t: q15, p_alpha: &mut q15, p_beta: &mut q15) {
    let (d, q, sin_t, cos_t) = (
        d.to_bits() as i32,
        q.to_bits() as i32,
        sin_t.to_bits() as i32,
        cos_t.to_bits() as i32,
    );
    let alpha = (d * cos_t - q * sin_t) >> 15;
    let beta = (d * sin_t + q * cos_t) >> 15;
    *p_alpha = sat_q15_i32(alpha);
    *p_beta = sat_q15_i32(beta);
}

// ─────────────────────────────────────────────────────────────────────────────
// Precision PI²D² Biquad Controller Synthesis (PidBuilder)
// ─────────────────────────────────────────────────────────────────────────────

use crate::filtering::{Biquad, BiquadClamp};

/// Five possible PID-style actions of a biquad SOS controller.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub enum PidAction {
    /// Double integrator (-40 dB/decade)
    I2 = 0,
    /// Single integrator (-20 dB/decade)
    I = 1,
    /// Proportional (flat gain)
    P = 2,
    /// Derivative (+20 dB/decade)
    D = 3,
    /// Double derivative (+40 dB/decade)
    D2 = 4,
}

/// Feedback-term order: the lowest-order action included in the controller.
///
/// The builder uses three consecutive actions `order ..= order + 2`; gains for
/// actions outside that window are ignored.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub enum PidOrder {
    /// Double integrator (lowest order): uses `I², I, P`.
    I2 = 0,
    /// Single integrator (default): uses `I, P, D`.
    #[default]
    I = 1,
    /// Proportional only: uses `P, D, D²`.
    P = 2,
}

/// Errors returned by [`PidBuilder::validate`] and [`PidBuilder::try_build`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub enum PidError {
    /// A parameter is NaN or infinite.
    NonFinite(&'static str),
    /// A period or gain limit is not strictly positive.
    NonPositive(&'static str),
    /// A gain and its limit have opposite signs.
    SignMismatch(&'static str),
    /// The output minimum exceeds the maximum.
    InvertedRange(&'static str),
}

/// Precision PI²D² biquad controller builder with gain limits and anti-windup clamping.
///
/// Synthesizes a discrete second-order section (SOS) [`BiquadClamp`] from up to
/// five controller actions `[I², I, P, D, D²]`. This is a faithful port of the
/// `idsp` PI²D² builder: gains are accurate in the low-frequency limit and
/// integral/derivative actions are warped towards Nyquist. Per-action gain
/// limits roll off `I`/`D`/`D²` at their respective band edges, which keeps
/// sensor noise from being amplified into actuator chatter.
///
/// ```rust
/// # use embedded_dsp::controller::PidBuilder;
/// let c = PidBuilder::new()
///     .kp(1.0).ki(1e-3).kd(1e2)
///     .limit_i(1e3).limit_d(1e1)
///     .build(1.0);
/// // The denominator carries an integrator pole.
/// assert!(c.coeff.ba[3] != 0.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct PidBuilder {
    order: PidOrder,
    gains: [f32; 5],
    limits: [f32; 5],
    min_clamp: f32,
    max_clamp: f32,
    offset: f32,
}

impl Default for PidBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl PidBuilder {
    /// Create a new builder with zero gains and default unbounded limits.
    pub const fn new() -> Self {
        Self {
            order: PidOrder::I,
            gains: [0.0; 5],
            limits: [f32::INFINITY; 5],
            min_clamp: -f32::INFINITY,
            max_clamp: f32::INFINITY,
            offset: 0.0,
        }
    }

    /// Select the feedback-term order (which three actions are used).
    pub const fn order(mut self, order: PidOrder) -> Self {
        self.order = order;
        self
    }

    /// Set the gain for an arbitrary action.
    pub const fn gain(mut self, action: PidAction, gain: f32) -> Self {
        self.gains[action as usize] = gain;
        self
    }

    /// Set the gain limit for an arbitrary action.
    pub const fn limit(mut self, action: PidAction, limit: f32) -> Self {
        self.limits[action as usize] = limit;
        self
    }

    /// Set proportional gain `Kp`.
    pub const fn kp(mut self, kp: f32) -> Self {
        self.gains[PidAction::P as usize] = kp;
        self
    }

    /// Set integral gain `Ki` (in 1/sec).
    pub const fn ki(mut self, ki: f32) -> Self {
        self.gains[PidAction::I as usize] = ki;
        self
    }

    /// Set double-integral gain `Ki2` (in 1/sec^2).
    pub const fn ki2(mut self, ki2: f32) -> Self {
        self.gains[PidAction::I2 as usize] = ki2;
        self
    }

    /// Set derivative gain `Kd` (in sec).
    pub const fn kd(mut self, kd: f32) -> Self {
        self.gains[PidAction::D as usize] = kd;
        self
    }

    /// Set double-derivative gain `Kd2` (in sec^2).
    pub const fn kd2(mut self, kd2: f32) -> Self {
        self.gains[PidAction::D2 as usize] = kd2;
        self
    }

    /// Set the low-frequency gain limit for the integral `I` action.
    pub const fn limit_i(mut self, max_i_gain: f32) -> Self {
        self.limits[PidAction::I as usize] = max_i_gain;
        self
    }

    /// Set the low-frequency gain limit for the double-integral `I²` action.
    pub const fn limit_i2(mut self, max_i2_gain: f32) -> Self {
        self.limits[PidAction::I2 as usize] = max_i2_gain;
        self
    }

    /// Set the high-frequency gain limit for the derivative `D` action.
    ///
    /// Essential in practical motion/optical control to prevent amplifying
    /// high-frequency sensor quantization noise into motor chatter.
    pub const fn limit_d(mut self, max_d_gain: f32) -> Self {
        self.limits[PidAction::D as usize] = max_d_gain;
        self
    }

    /// Set the high-frequency gain limit for the double-derivative `D²` action.
    pub const fn limit_d2(mut self, max_d2_gain: f32) -> Self {
        self.limits[PidAction::D2 as usize] = max_d2_gain;
        self
    }

    /// Set output anti-windup saturation limits `[min, max]`.
    pub const fn output_limits(mut self, min: f32, max: f32) -> Self {
        self.min_clamp = min;
        self.max_clamp = max;
        self
    }

    /// Set summing-junction setpoint / offset `u`.
    pub const fn offset(mut self, u: f32) -> Self {
        self.offset = u;
        self
    }

    /// Check whether the parametrization is valid for the given sample period.
    pub fn validate(&self, ts: f32) -> Result<(), PidError> {
        if !ts.is_finite() {
            return Err(PidError::NonFinite("period"));
        }
        if ts <= 0.0 {
            return Err(PidError::NonPositive("period"));
        }
        if self.gains.iter().any(|g| !g.is_finite()) {
            return Err(PidError::NonFinite("gain"));
        }
        if self.limits.iter().any(|l| l.is_nan()) {
            return Err(PidError::NonFinite("limit"));
        }
        for action in [PidAction::I2, PidAction::I, PidAction::D, PidAction::D2] {
            let gain = self.gains[action as usize];
            let limit = self.limits[action as usize];
            if limit.is_finite() {
                if limit == 0.0 {
                    return Err(PidError::NonPositive("limit"));
                }
                if gain != 0.0 && gain.signum() != limit.signum() {
                    return Err(PidError::SignMismatch("gain/limit"));
                }
            }
        }
        if self.min_clamp > self.max_clamp {
            return Err(PidError::InvertedRange("output_limits"));
        }
        Ok(())
    }

    /// Validate and then build.
    pub fn try_build(&self, ts: f32) -> Result<BiquadClamp<f32>, PidError> {
        self.validate(ts)?;
        Ok(self.build(ts))
    }

    /// Compute the normalized SOS coefficients `[b0, b1, b2, a1, a2]` (`a0 = 1`).
    ///
    /// Direct port of idsp's PI²D² coefficient synthesis: the three actions
    /// starting at [`PidOrder`] are mapped through difference kernels and
    /// normalized by the summed gain limits.
    pub fn coefficients(&self, ts: f32) -> [f32; 5] {
        let order = self.order as i32;
        // z starts at period^(-order) and is multiplied by the period each step.
        let mut z = match order {
            0 => 1.0,
            1 => 1.0 / ts,
            _ => 1.0 / (ts * ts),
        };

        // Gain/limit triples for actions `order..=order+2`, paired with gl in
        // reverse so that gl[0] receives the lowest-order action.
        let mut gl = [[0.0f32; 2]; 3];
        for (gl, (i, (gain, limit))) in gl
            .iter_mut()
            .zip(
                self.gains
                    .iter()
                    .zip(self.limits.iter())
                    .enumerate()
                    .skip(self.order as usize),
            )
            .rev()
        {
            gl[0] = *gain * z;
            gl[1] = if i == PidAction::P as usize {
                1.0
            } else {
                gl[0] / *limit
            };
            z *= ts;
        }

        // Normalization by the summed gain limits.
        let a0i = 1.0 / (gl[0][1] + gl[1][1] + gl[2][1]);

        // Difference kernels for the three selected actions.
        const KERNELS: [[i32; 3]; 3] = [[1, 0, 0], [1, -1, 0], [1, -2, 1]];

        let mut ba = [[0.0f32; 2]; 3];
        for (gli, ki) in gl.into_iter().zip(KERNELS) {
            let gli = [gli[0] * a0i, gli[1] * a0i];
            for (baj, kij) in ba.iter_mut().zip(ki) {
                if kij > 0 {
                    for _ in 0..kij {
                        baj[0] += gli[0];
                        baj[1] -= gli[1];
                    }
                } else {
                    for _ in 0..-kij {
                        baj[0] -= gli[0];
                        baj[1] += gli[1];
                    }
                }
            }
        }

        [ba[0][0], ba[1][0], ba[2][0], ba[1][1], ba[2][1]]
    }

    /// Synthesize a clamped biquad controller for the given sampling period
    /// `ts` (in seconds).
    ///
    /// # Panics
    /// Panics if `ts` is not finite and positive.
    pub fn build(&self, ts: f32) -> BiquadClamp<f32> {
        assert!(
            ts > 0.0 && ts.is_finite(),
            "Sampling period ts must be finite and positive"
        );
        let ba = self.coefficients(ts);
        BiquadClamp::new(
            Biquad::new(ba[0], ba[1], ba[2], ba[3], ba[4]),
            self.min_clamp,
            self.max_clamp,
            self.offset,
        )
    }
}

