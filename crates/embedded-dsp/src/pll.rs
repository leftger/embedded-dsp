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
///
/// # Structure
///
/// A quadrature (I/Q) Costas loop, one sample at a time:
///
/// 1. **Mix.** The input is multiplied by the NCO to form the in-phase and
///    quadrature arms, `i = x·cos θ` and `q = x·sin θ`.
/// 2. **Arm filter.** Both arms are low-pass filtered. This is what makes the
///    loop work: the mixer output contains the wanted baseband term plus a
///    2·f_c image, and the phase detector below only produces a usable DC term
///    once the image has been attenuated. The cutoff is therefore tied to the
///    *centre frequency*, not to the loop bandwidth.
/// 3. **Phase detector.** The amplitude-normalised Costas product
///    `2·i·q/(i² + q²)`, which equals `sin(2Δθ)` — proportional to twice the
///    phase error for small errors, bounded to ±1, and independent of signal
///    amplitude (so the loop gain does not change when the input level does).
/// 4. **Loop filter.** Proportional-integral, driving the NCO frequency and
///    phase.
///
/// # Bandwidth
///
/// The arm filter sits inside the loop and contributes lag, so the achievable
/// bandwidth is limited by the arm cutoff: the effective bandwidth used for the
/// loop coefficients is `min(loop_bandwidth_hz, f_arm / 20)`. Requesting a loop
/// bandwidth anywhere near the centre frequency cannot be realised with a
/// single-pole arm filter, and the loop will fail to lock — `|frequency_hz()|`
/// collapses towards zero instead of tracking. Prefer `loop_bandwidth_hz` of a
/// few percent of `center_freq_hz`.
#[derive(Debug, Clone, Copy)]
pub struct CostasLoop {
    sample_rate_hz: f32,
    phase: f32,
    freq_rad_per_sample: f32,
    center_freq_rad: f32,
    alpha: f32, // Proportional loop filter parameter
    beta: f32,  // Integral loop filter parameter
    arm_k: f32, // One-pole arm filter coefficient
    i_lp: f32,  // Filtered in-phase arm
    q_lp: f32,  // Filtered quadrature arm
}

impl CostasLoop {
    /// Create a new Costas Loop.
    ///
    /// `loop_bandwidth_hz` is capped at `center_freq_hz / 20`; see the type-level
    /// documentation for why.
    pub fn new(center_freq_hz: f32, sample_rate_hz: f32, loop_bandwidth_hz: f32, damping: f32) -> Self {
        let pi = core::f32::consts::PI;
        let center_freq_rad = 2.0 * pi * center_freq_hz / sample_rate_hz;

        // Arm filter cutoff at the centre frequency: it has to reject the
        // 2*f_c mixer image, which a filter scaled to the loop bandwidth cannot
        // do whenever the carrier is only a small multiple of the bandwidth.
        let f_arm = center_freq_hz.abs().clamp(1.0e-6, 0.49 * sample_rate_hz);
        let arm_k = 1.0 - (-2.0 * pi * f_arm / sample_rate_hz).exp();

        // Derive the PI coefficients from the realisable bandwidth. The
        // arm-filter lag inside the loop makes much larger values unstable.
        let bw_eff = loop_bandwidth_hz.abs().min(f_arm / 20.0);
        let theta = 2.0 * pi * bw_eff / sample_rate_hz;
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
            arm_k,
            i_lp: 0.0,
            q_lp: 0.0,
        }
    }

    /// Process a modulated carrier sample and return the filtered baseband I and Q arms.
    pub fn process_sample(&mut self, sample: f32) -> (f32, f32) {
        let i_arm = sample * self.phase.cos();
        let q_arm = sample * self.phase.sin();

        // Arm low-pass.
        self.i_lp += self.arm_k * (i_arm - self.i_lp);
        self.q_lp += self.arm_k * (q_arm - self.q_lp);

        // Normalised Costas phase detector: 2*i*q/(i^2 + q^2) == sin(2*d_theta).
        // Always in [-1, 1], so no clamp is needed; the guard only avoids a
        // division by zero before the arm filters have any energy in them.
        let power = self.i_lp * self.i_lp + self.q_lp * self.q_lp;
        let error = if power > 1.0e-20 {
            2.0 * self.i_lp * self.q_lp / power
        } else {
            0.0
        };

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

        (self.i_lp, self.q_lp)
    }

    /// Reset the phase, frequency estimate, and arm-filter state, keeping the
    /// configured centre frequency and loop coefficients.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.freq_rad_per_sample = self.center_freq_rad;
        self.i_lp = 0.0;
        self.q_lp = 0.0;
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

// ─────────────────────────────────────────────────────────────────────────────
// Integer PLLs (ported from idsp)
// ─────────────────────────────────────────────────────────────────────────────

/// Round an `f64` to the nearest `i32` (saturating), matching idsp's Q32.32
/// coefficient quantization. `f64::round` is unavailable in `core`.
#[inline]
fn round_to_i32(v: f64) -> i32 {
    (v + if v < 0.0 { -0.5 } else { 0.5 }) as i32
}

/// Type-2, order-3 integer sampled-phase PLL.
///
/// Tracks frequency and phase of an input signal with respect to the sampling
/// clock. Open-loop transfer function is type-2 (double DC integrator).
///
/// All arithmetic is wrapping 32-bit integer — stable for any numerically valid
/// gain; no floating-point rounding errors. Phase and frequency are understood
/// modulo the i32 range (first Nyquist zone).
///
/// Single parameter (`bandwidth`) controls loop bandwidth, expressed as a
/// fraction of the sample rate (`0 < bw < 0.5`).
///
/// # Ported from
/// The `idsp` crate by the Sinara/ARTIQ project.
#[derive(Copy, Clone, Debug, Default)]
pub struct IntPll {
    /// Lead-lag coefficients `[b0, b1, a1]` stored as scaled i32.
    /// Internal representation: float * 2^32 (Q32.32), matching the idsp `PLL`.
    pub ba: [i32; 3],
}

impl IntPll {
    /// Construct from zero, pole, and gain (all normalised to sample rate, i.e. in `[0,1)`).
    pub fn from_zpk(zero: f32, pole: f32, gain: f32) -> Self {
        const SCALE: f64 = 4294967296.0; // 2^32 (Q32.32)
        Self {
            ba: [
                round_to_i32(gain as f64 * SCALE),
                round_to_i32((-gain * zero) as f64 * SCALE),
                round_to_i32((pole - 1.0) as f64 * SCALE),
            ],
        }
    }

    /// Construct from normalised loop bandwidth `bw` and lead-lag split factor
    /// `split` (typically `4.0`).
    ///
    /// Yields ~1.5 dB peaking and ~62° phase margin for `split = 4`.
    pub fn from_bandwidth(bw: f32, split: f32) -> Self {
        let a = bw * 2.0 * core::f32::consts::PI;
        let zero = 1.0 - a / split;
        let pole = 1.0 - a * split;
        let gain = -a * a * split;
        Self::from_zpk(zero, pole, gain)
    }

    /// Advance the PLL one sample.
    ///
    /// - `state`: mutable per-call state.
    /// - `input_phase`: sampled input phase (wrapping i32).
    ///
    /// Returns the current output phase estimate.
    pub fn process(&self, state: &mut IntPllState, input_phase: i32) -> i32 {
        // Advance output phase using current frequency estimate
        state.y = state.y.wrapping_add((state.f >> 32) as i32);

        // Phase error (additive: output compensates input)
        let raw_err = input_phase.wrapping_add(state.y);

        // Clamp on wrap to prevent integrator wind-up
        let clamped = state.clamp.process(raw_err);

        // Nyquist zero: halved and averaged with previous
        let z0 = clamped >> 1;
        let y0 = z0.wrapping_add(state.z0);
        state.z0 = z0;

        // Lead-lag biquad with wide i64 state, Q32.32 coefficients:
        // f0 += b0*y0 + b1*y1 + a1*f1 + (a1 * low_word(f0)) >> 32.
        // The products are full-width (the Q32.32 scaling is applied when
        // frequency is extracted as f >> 32), matching idsp's wide `Q` math.
        // The low-word feedback term is evaluated with the *old* f0, exactly
        // like idsp's single `+=` expression.
        let f0 = state.f0;
        let f1 = (f0 >> 32) as i32;
        let y = self.ba[0] as i64 * y0 as i64
            + self.ba[1] as i64 * state.y0 as i64
            + self.ba[2] as i64 * f1 as i64
            + ((self.ba[2] as i64 * f0 as u32 as i64) >> 32);
        state.f0 = f0.wrapping_add(y);
        state.y0 = y0;

        // DC pole (frequency integrator)
        state.f = state.f.wrapping_add(state.f0);

        state.y
    }
}

/// Clamp-on-wrap helper: maps positive wraps to `i32::MAX` and negative wraps to `i32::MIN`,
/// recovering only on the corresponding un-wrap.
#[derive(Copy, Clone, Debug, Default)]
pub struct ClampWrap {
    /// Last accepted input
    pub x0: i32,
    /// Current clamp direction: -1, 0, or +1
    pub dir: i8,
}

impl ClampWrap {
    /// Feed a new input; returns a clamped version.
    ///
    /// Mirrors idsp's `ClampWrap`: a wrap in one direction clamps the output to
    /// the corresponding rail and the clamp is only released by a wrap in the
    /// opposite direction.
    #[inline]
    pub fn process(&mut self, x: i32) -> i32 {
        // idsp's `overflowing_sub`: the wrapped delta's sign is compared with
        // the true comparison sign to classify the wrap direction.
        let x0 = self.x0;
        let delta = x.wrapping_sub(x0);
        self.x0 = x;
        let wrap: i8 = match (delta >= 0, x >= x0) {
            (true, false) => 1,
            (false, true) => -1,
            _ => 0,
        };
        self.dir = (self.dir as i32 + wrap as i32).signum() as i8;
        match self.dir.cmp(&0) {
            core::cmp::Ordering::Less => i32::MIN,
            core::cmp::Ordering::Equal => x,
            core::cmp::Ordering::Greater => i32::MAX,
        }
    }
}

/// Mutable state for [`IntPll`].
#[derive(Copy, Clone, Debug, Default)]
pub struct IntPllState {
    /// Phase-error clamper
    pub clamp: ClampWrap,
    /// Pre-Nyquist-zero phase error
    pub z0: i32,
    /// Post-Nyquist-zero phase error
    pub y0: i32,
    /// Lead-lag accumulator (wide)
    pub f0: i64,
    /// DC integrator (wide) — upper 32 bits are frequency
    pub f: i64,
    /// Current phase estimate
    pub y: i32,
}

impl IntPllState {
    /// Current phase estimate.
    #[inline] pub fn phase(&self) -> i32 { self.y }
    /// Current frequency estimate (wrapping i32 increment per sample).
    #[inline] pub fn frequency(&self) -> i32 { (self.f >> 32) as i32 }
}

/// Reciprocal PLL (RPLL).
///
/// Consumes noisy, quantised timestamps of a reference signal and reconstructs
/// the phase and frequency of the update invocations with respect to (and in
/// units of `1 << 32` of) that reference.
///
/// Call [`Rpll::process`] at a fixed rate. Supply `Some(timestamp)` when a
/// reference edge occurs (at most once per `1 << dt2` update cycles), else `None`.
///
/// # Ported from
/// The `idsp` crate by the Sinara/ARTIQ project.
#[derive(Copy, Clone, Debug, Default)]
pub struct Rpll {
    x: i32,  // previous timestamp
    ff: u32, // frequency estimate from frequency loop
    f: u32,  // combined frequency estimate
    y: i32,  // phase estimate
}

/// Static configuration for [`Rpll`].
#[derive(Copy, Clone, Debug)]
pub struct RpllConfig {
    /// `1 << dt2` is the counter-to-update-rate ratio.
    pub dt2: u8,
    /// Frequency lock settling-time exponent (`>= dt2`).
    pub shift_frequency: u8,
    /// Phase lock settling-time exponent (usually `shift_frequency - 1`).
    pub shift_phase: u8,
}

impl Rpll {
    /// Current phase estimate.
    #[inline] pub fn phase(&self) -> i32 { self.y }
    /// Current frequency estimate (u32 per update cycle).
    #[inline] pub fn frequency(&self) -> u32 { self.f }

    /// Advance one update cycle.
    ///
    /// - `cfg`: static RPLL configuration.
    /// - `timestamp`: `Some(counter)` on a reference edge, else `None`.
    ///
    /// Returns `(phase, frequency)`.
    pub fn process(&mut self, cfg: &RpllConfig, timestamp: Option<i32>) -> (i32, u32) {
        // Advance phase using current frequency
        self.y = self.y.wrapping_add(self.f as i32);

        if let Some(x) = timestamp {
            let dx = x.wrapping_sub(self.x);
            self.x = x;

            // Signal phase: ff * dx >> shift_frequency
            let p_sig = ((self.ff as u64)
                .wrapping_mul(dx as u64)
                .wrapping_add(1u64 << (cfg.shift_frequency - 1))
                >> cfg.shift_frequency) as u32;

            // Reference phase for one reference period at this update rate
            let p_ref = 1u32.wrapping_shl(
                (32u32 + cfg.dt2 as u32).saturating_sub(cfg.shift_frequency as u32),
            );

            // Frequency loop
            self.ff = self.ff.wrapping_add(p_ref.wrapping_sub(p_sig));

            // Time between timestamp and "now" in counter cycles
            let dt = (x.wrapping_neg()) & ((1i32 << cfg.dt2) - 1);

            // Estimated reference phase "now"
            let y_ref = ((self.f >> cfg.dt2 as u32) as i64 * dt as i64) as i32;

            // Phase error with gain
            let dy = (y_ref.wrapping_sub(self.y))
                >> (cfg.shift_phase - cfg.dt2) as u32;

            // Combine
            self.f = self.ff.wrapping_add(dy as u32);
        }

        (self.y, self.f)
    }
}
