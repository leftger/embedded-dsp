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
