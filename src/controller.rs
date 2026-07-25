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
        let out = self.state[2] + self.a0 * in_val + self.a1 * self.state[0] + self.a2 * self.state[1];
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
            a0: 0,
            a1: 0,
            a2: 0,
            state: [0; 3],
            kp,
            ki,
            kd,
        };
        pid.init(1);
        pid
    }

    pub fn init(&mut self, reset_state_flag: i32) {
        self.a0 = self.kp.saturating_add(self.ki).saturating_add(self.kd);
        self.a1 = (-self.kp).saturating_sub(2 * self.kd);
        self.a2 = self.kd;
        if reset_state_flag != 0 {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.state = [0; 3];
    }

    pub fn process(&mut self, in_val: q31) -> q31 {
        let acc = (self.state[2] as i64)
            + ((self.a0 as i64 * in_val as i64) >> 31)
            + ((self.a1 as i64 * self.state[0] as i64) >> 31)
            + ((self.a2 as i64 * self.state[1] as i64) >> 31);
        let out = acc.clamp(i32::MIN as i64, i32::MAX as i64) as q31;
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
            a0: 0,
            a1: 0,
            a2: 0,
            state: [0; 3],
            kp,
            ki,
            kd,
        };
        pid.init(1);
        pid
    }

    pub fn init(&mut self, reset_state_flag: i32) {
        self.a0 = self.kp.saturating_add(self.ki).saturating_add(self.kd);
        self.a1 = (-self.kp).saturating_sub(2 * self.kd);
        self.a2 = self.kd;
        if reset_state_flag != 0 {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.state = [0; 3];
    }

    pub fn process(&mut self, in_val: q15) -> q15 {
        let acc = (self.state[2] as i32)
            + ((self.a0 as i32 * in_val as i32) >> 15)
            + ((self.a1 as i32 * self.state[0] as i32) >> 15)
            + ((self.a2 as i32 * self.state[1] as i32) >> 15);
        let out = acc.clamp(i16::MIN as i32, i16::MAX as i32) as q15;
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
