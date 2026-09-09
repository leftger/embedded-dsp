//! Phase-Locked Loops (PLL) and carrier recovery for power electronics, motor resolvers, and SDR.
//!
//! Includes:
//! - [`SogiPll`]: Second-Order Generalized Integrator PLL for single-phase grid synchronization (solar inverters, UPS) and resolver angle tracking.
//! - [`CostasLoop`]: Carrier phase and frequency recovery loop for BPSK/QPSK demodulation.

#[allow(unused_imports)]
use crate::math::FloatMath;

/// Second-Order Generalized Integrator Phase-Locked Loop (SOGI-PLL).
///
/// Implements orthogonal signal generation ($v_\alpha, v_\beta$) from a single-phase input $v(t)$
/// and tracks fundamental frequency and phase in real-time.
///
/// Discretized using trapezoidal (Tustin) integration for zero frequency warping at the center frequency.
#[derive(Debug, Clone, Copy)]
pub struct SogiPll {
    // SOGI internal filter states and coefficients
    sample_rate_hz: f32,
    omega_center: f32, // Nominal center frequency in rad/s
    k_sogi: f32,       // SOGI damping factor (typically sqrt(2) ≈ 1.414)
    v_alpha: f32,      // In-phase filtered orthogonal component
    v_beta: f32,       // Quadrature (90 deg lagging) filtered component
    x1: f32,           // Integrator 1 state
    x2: f32,           // Integrator 2 state

    // Loop filter (PI) & NCO states
    kp: f32,           // Proportional gain
    ki: f32,           // Integral gain
    phase: f32,        // Estimated phase θ in [-π, π]
    omega_est: f32,    // Estimated frequency in rad/s
    integrator_pi: f32,// PI controller accumulator
}

impl SogiPll {
    /// Creates a new SOGI-PLL tuned to `center_freq_hz` at `sample_rate_hz`.
    ///
    /// - `k_sogi`: SOGI damping factor (default `1.414`).
    /// - `kp`: Loop filter proportional gain (e.g. `60.0`).
    /// - `ki`: Loop filter integral gain (e.g. `1400.0`).
    pub fn new(center_freq_hz: f32, sample_rate_hz: f32, k_sogi: f32, kp: f32, ki: f32) -> Self {
        let omega_center = 2.0 * core::f32::consts::PI * center_freq_hz;
        Self {
            sample_rate_hz,
            omega_center,
            k_sogi,
            v_alpha: 0.0,
            v_beta: 0.0,
            x1: 0.0,
            x2: 0.0,
            kp,
            ki,
            phase: 0.0,
            omega_est: omega_center,
            integrator_pi: 0.0,
        }
    }

    /// Process a single input sample and return the tracked instantaneous phase $\theta \in [-\pi, \pi]$.
    pub fn process(&mut self, input: f32) -> f32 {
        let ts = 1.0 / self.sample_rate_hz;
        let half_ts = 0.5 * ts;

        // 1. SOGI Orthogonal Signal Generation (Tustin integration)
        let err = input - self.v_alpha;
        let k_err = self.k_sogi * err;
        let w = self.omega_est;

        // State update for SOGI
        let d_x1 = (k_err - self.v_beta) * w;
        let d_x2 = self.v_alpha * w;

        let x1_new = self.x1 + half_ts * d_x1;
        let x2_new = self.x2 + half_ts * d_x2;

        self.v_alpha = x1_new;
        self.v_beta = x2_new;

        self.x1 += ts * (k_err - self.v_beta) * w;
        self.x2 += ts * self.v_alpha * w;

        // 2. Park Transform Phase Detector: q-axis error = -v_alpha * sin(θ) + v_beta * cos(θ)
        let sin_p = self.phase.sin();
        let cos_p = self.phase.cos();
        let v_q = -self.v_alpha * sin_p + self.v_beta * cos_p;

        // 3. Loop Filter (PI controller on v_q)
        self.integrator_pi += self.ki * ts * v_q;
        let delta_omega = self.kp * v_q + self.integrator_pi;
        self.omega_est = self.omega_center + delta_omega;

        // 4. Integrator NCO -> Phase update
        self.phase += self.omega_est * ts;

        // Wrap phase to [-π, π]
        let pi = core::f32::consts::PI;
        let two_pi = 2.0 * pi;
        while self.phase > pi {
            self.phase -= two_pi;
        }
        while self.phase < -pi {
            self.phase += two_pi;
        }

        self.phase
    }

    /// Returns the estimated fundamental frequency in Hz.
    #[inline(always)]
    pub fn frequency_hz(&self) -> f32 {
        self.omega_est / (2.0 * core::f32::consts::PI)
    }

    /// Returns the filtered orthogonal components `(v_alpha, v_beta)`.
    #[inline(always)]
    pub fn orthogonal_components(&self) -> (f32, f32) {
        (self.v_alpha, self.v_beta)
    }

    /// Returns the instantaneous phase $\theta \in [-\pi, \pi]$.
    #[inline(always)]
    pub fn phase(&self) -> f32 {
        self.phase
    }

    /// Reset PLL internal states.
    pub fn reset(&mut self) {
        self.v_alpha = 0.0;
        self.v_beta = 0.0;
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.phase = 0.0;
        self.omega_est = self.omega_center;
        self.integrator_pi = 0.0;
    }
}

/// Costas Loop for BPSK / QPSK carrier phase and frequency tracking.
#[derive(Debug, Clone, Copy)]
pub struct CostasLoop {
    sample_rate_hz: f32,
    phase: f32,
    freq_rad_per_sample: f32,
    center_freq_rad: f32,
    alpha: f32, // Proportional loop filter parameter
    beta: f32,  // Integral loop filter parameter
}

impl CostasLoop {
    /// Create a new Costas Loop.
    pub fn new(center_freq_hz: f32, sample_rate_hz: f32, loop_bandwidth_hz: f32, damping: f32) -> Self {
        let center_freq_rad = 2.0 * core::f32::consts::PI * center_freq_hz / sample_rate_hz;
        let theta = 2.0 * core::f32::consts::PI * loop_bandwidth_hz / sample_rate_hz;
        let d = 1.0 + 2.0 * damping * theta + theta * theta;
        let alpha = (4.0 * damping * theta) / d;
        let beta = (4.0 * theta * theta) / d;

        Self {
            sample_rate_hz,
            phase: 0.0,
            freq_rad_per_sample: center_freq_rad,
            center_freq_rad,
            alpha,
            beta,
        }
    }

    /// Process a modulated carrier sample and return the demodulated baseband in-phase (I) sample.
    pub fn process_sample(&mut self, sample: f32) -> (f32, f32) {
        let cos_val = self.phase.cos();
        let sin_val = (-self.phase).sin();

        let i_arm = sample * cos_val;
        let q_arm = sample * sin_val;

        // BPSK phase error detector: e = I * sign(Q) or e = I * Q
        let error = (i_arm * q_arm).clamp(-1.0, 1.0);

        // Loop filter update
        self.freq_rad_per_sample += self.beta * error;
        self.phase += self.freq_rad_per_sample + self.alpha * error;

        // Wrap phase to [-π, π]
        let pi = core::f32::consts::PI;
        while self.phase > pi {
            self.phase -= 2.0 * pi;
        }
        while self.phase < -pi {
            self.phase += 2.0 * pi;
        }

        (i_arm, q_arm)
    }

    /// Current tracked carrier frequency in Hz.
    #[inline(always)]
    pub fn frequency_hz(&self) -> f32 {
        self.freq_rad_per_sample * self.sample_rate_hz / (2.0 * core::f32::consts::PI)
    }

    /// Nominal center frequency in Hz.
    #[inline(always)]
    pub fn center_frequency_hz(&self) -> f32 {
        self.center_freq_rad * self.sample_rate_hz / (2.0 * core::f32::consts::PI)
    }
}
