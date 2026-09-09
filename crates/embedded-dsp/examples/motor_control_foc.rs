//! Comprehensive Field-Oriented Control (FOC) & Motor Control Example
//!
//! Demonstrates:
//! 1. 3-Phase Stator Current Sensing ($I_a, I_b, I_c$) with High-Frequency PWM Switching Noise
//! 2. Real-Time Feedback Filtering: O(1) Recursive Moving Average (`RecursiveMovingAverage<8>`) and Single-Pole Lowpass (`SinglePoleFilter`)
//! 3. Fast Trigonometry Evaluation: Fast Taylor Sine/Cosine (`fast_sin_f32`, `fast_cos_f32`), LUT (`lut_sin_cos_i16`), and Q31 (`sin_q31`, `cos_q31`)
//! 4. Forward Clarke Transform ($I_a, I_b \to I_\alpha, I_\beta$) in f32 and Q15
//! 5. Forward Park Transform ($I_\alpha, I_\beta, \theta \to I_d, I_q$) into the Rotor Synchronous Frame in f32 and Q15
//! 6. Dual-Axis Vector Current PID Controllers: $I_d$ (Flux / MTPA) and $I_q$ (Electromagnetic Torque) in f32 and Q15
//! 7. Outer Speed Regulation PID Loop (RPM Error $\to$ $I_q^*$ Torque Demand)
//! 8. Inverse Park & Inverse Clarke Transforms generating 3-Phase Reference Voltages ($V_a, V_b, V_c$) for Space Vector PWM (SVPWM)

use embedded_dsp::*;

fn main() {
    println!("===============================================================================");
    println!("       embedded-dsp Field-Oriented Control (FOC) & Motor Control Loop          ");
    println!("===============================================================================");
    println!();

    const CONTROL_RATE_HZ: f32 = 20000.0; // 20 kHz FOC current loop (50 μs per step)
    const NUM_CONTROL_CYCLES: usize = 20;

    // -----------------------------------------------------------------------------------------
    // 1. 3-Phase Current Feedback Simulation with Inverter Switching Ripple
    // -----------------------------------------------------------------------------------------
    println!("--- 1. 3-Phase Current Sensing with Inverter PWM Switching Noise ---");
    let target_torque_current = 5.0f32; // 5 Amps commanded torque current (Iq)
    let rotor_speed_rpm = 3000.0f32;
    let pole_pairs = 4.0f32;
    let electrical_freq_hz = (rotor_speed_rpm * pole_pairs) / 60.0; // 200 Hz electrical freq
    let omega_e = 2.0 * core::f32::consts::PI * electrical_freq_hz;

    let mut ia_raw = [0.0f32; NUM_CONTROL_CYCLES];
    let mut ib_raw = [0.0f32; NUM_CONTROL_CYCLES];
    let mut ic_raw = [0.0f32; NUM_CONTROL_CYCLES];
    let mut theta_e = [0.0f32; NUM_CONTROL_CYCLES];

    for i in 0..NUM_CONTROL_CYCLES {
        let t = i as f32 / CONTROL_RATE_HZ;
        let theta = (omega_e * t) % (2.0 * core::f32::consts::PI);
        theta_e[i] = theta;

        // Ideal stator currents (balanced 3-phase, 120° apart, Iq = 5A, Id = 0A)
        // Ia = -Iq * sin(θ), Ib = -Iq * sin(θ - 2π/3), Ic = -Iq * sin(θ + 2π/3)
        let ia_ideal = -target_torque_current * theta.sin();
        let ib_ideal = -target_torque_current * (theta - 2.0 * core::f32::consts::PI / 3.0).sin();
        let ic_ideal = -target_torque_current * (theta + 2.0 * core::f32::consts::PI / 3.0).sin();

        // 20 kHz PWM inverter switching noise (±0.4A ripple)
        let ripple = if i % 2 == 0 { 0.35f32 } else { -0.35f32 };
        ia_raw[i] = ia_ideal + ripple;
        ib_raw[i] = ib_ideal - ripple * 0.5;
        ic_raw[i] = ic_ideal - ripple * 0.5;
    }

    println!(
        "  Initial Raw Stator Currents at t=0: Ia={:.2}A, Ib={:.2}A, Ic={:.2}A",
        ia_raw[0], ib_raw[0], ic_raw[0]
    );

    // -----------------------------------------------------------------------------------------
    // 2. Real-Time Current Filtering (O(1) Recursive Moving Average)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 2. Stator Current Feedback Filtering (O(1) Recursive Moving Average) ---");
    let mut ma_filter_ia = RecursiveMovingAverage::<8>::new();
    let mut ma_filter_ib = RecursiveMovingAverage::<8>::new();
    let mut ia_filtered = [0.0f32; NUM_CONTROL_CYCLES];
    let mut ib_filtered = [0.0f32; NUM_CONTROL_CYCLES];

    for i in 0..NUM_CONTROL_CYCLES {
        ia_filtered[i] = ma_filter_ia.process(ia_raw[i]);
        ib_filtered[i] = ma_filter_ib.process(ib_raw[i]);
    }
    println!(
        "  Raw Ia[4] = {:.3}A -> Filtered Ia[4] = {:.3}A",
        ia_raw[4], ia_filtered[4]
    );
    println!(
        "  Raw Ib[4] = {:.3}A -> Filtered Ib[4] = {:.3}A",
        ib_raw[4], ib_filtered[4]
    );

    // -----------------------------------------------------------------------------------------
    // 3. Fast Trigonometric Evaluation & Angle Transformations
    // -----------------------------------------------------------------------------------------
    println!("\n--- 3. Fast Trigonometry vs Table Lookup for Rotor Angle θ ---");
    let test_angle = core::f32::consts::FRAC_PI_3; // 60 degrees (π/3 rad)
    let fast_s_i16 = fast_sin_i16(test_angle);
    let fast_c_i16 = fast_cos_i16(test_angle);
    let std_s = test_angle.sin();
    let std_c = test_angle.cos();
    println!(
        "  θ = 60°: LUT fast_sin_i16 = {} ({:.4}, Exact: {:.4})",
        fast_s_i16,
        fast_s_i16 as f32 / 32768.0,
        std_s
    );
    println!(
        "  θ = 60°: LUT fast_cos_i16 = {} ({:.4}, Exact: {:.4})",
        fast_c_i16,
        fast_c_i16 as f32 / 32768.0,
        std_c
    );

    // Q31 Fixed-Point CORDIC Sine/Cosine (angle in [-π, π) mapped to Q31)
    let angle_q31 = q31::from_bits(((test_angle / core::f32::consts::PI) * 2147483648.0) as i32);
    let s_q31 = sin_q31(angle_q31);
    let c_q31 = cos_q31(angle_q31);
    println!(
        "  CORDIC Q31 Sine/Cosine: sin = {} ({:.4}, Exact: {:.4}), cos = {} ({:.4}, Exact: {:.4})",
        s_q31,
        s_q31.to_bits() as f64 / 2147483648.0,
        std_s,
        c_q31,
        c_q31.to_bits() as f64 / 2147483648.0,
        std_c
    );

    // -----------------------------------------------------------------------------------------
    // 4. Forward Clarke & Park Transforms (f32 & Q15)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 4. Forward Clarke & Park Transforms ---");
    let mut i_alpha = 0.0f32;
    let mut i_beta = 0.0f32;
    let mut i_d = 0.0f32;
    let mut i_q = 0.0f32;

    // Evaluate step 10
    let step_idx = 10;
    clarke_f32(
        ia_filtered[step_idx],
        ib_filtered[step_idx],
        &mut i_alpha,
        &mut i_beta,
    );
    park_f32(i_alpha, i_beta, theta_e[step_idx], &mut i_d, &mut i_q);

    println!("  f32 Forward Transforms at Step {}:", step_idx);
    println!(
        "    • Stator Stationary Frame: I_alpha = {:>6.3} A, I_beta = {:>6.3} A",
        i_alpha, i_beta
    );
    println!(
        "    • Rotor Synchronous Frame: I_d (Flux) = {:>6.3} A, I_q (Torque) = {:>6.3} A",
        i_d, i_q
    );

    // Fixed-point Q15 Clarke and Park verification
    let q15_scale = 1000.0f32; // 1 Amp = 1000 Q15 counts
    let ia_q15 = q15::from_bits((ia_filtered[step_idx] * q15_scale) as i16);
    let ib_q15 = q15::from_bits((ib_filtered[step_idx] * q15_scale) as i16);
    let mut q15_alpha = q15::ZERO;
    let mut q15_beta = q15::ZERO;
    let mut q15_d = q15::ZERO;
    let mut q15_q = q15::ZERO;

    clarke_q15(ia_q15, ib_q15, &mut q15_alpha, &mut q15_beta);
    let sin_q15 = q15::from_bits(fast_sin_i16(theta_e[step_idx]));
    let cos_q15 = q15::from_bits(fast_cos_i16(theta_e[step_idx]));
    park_q15(
        q15_alpha, q15_beta, sin_q15, cos_q15, &mut q15_d, &mut q15_q,
    );

    println!("  Q15 Forward Transforms:");
    println!(
        "    • Q15 Clarke: alpha = {} ({:.3} A), beta = {} ({:.3} A)",
        q15_alpha,
        q15_alpha.to_bits() as f32 / q15_scale,
        q15_beta,
        q15_beta.to_bits() as f32 / q15_scale
    );
    println!(
        "    • Q15 Park  : d = {} ({:.3} A), q = {} ({:.3} A)",
        q15_d,
        q15_d.to_bits() as f32 / q15_scale,
        q15_q,
        q15_q.to_bits() as f32 / q15_scale
    );

    // -----------------------------------------------------------------------------------------
    // 5. Dual Current Regulators (Id & Iq PID Loops) + Outer Speed Controller
    // -----------------------------------------------------------------------------------------
    println!("\n--- 5. Vector Current Regulators (Dual PID Loops for Id & Iq) ---");
    // Outer Velocity Loop: Target 3000 RPM, Actual 2950 RPM -> Error = 50 RPM
    let mut speed_pid = PidInstanceF32::new(0.08, 0.005, 0.001);
    let speed_error_rpm = 50.0f32;
    let demanded_iq = speed_pid.process(speed_error_rpm).clamp(-15.0, 15.0);
    println!(
        "  Outer Speed PID: Error = {:.1} RPM -> Commanded I_q* = {:.3} A",
        speed_error_rpm, demanded_iq
    );

    // Inner Current Regulators:
    // Id controller: Setpoint = 0.0 A (Zero d-axis current for Maximum Torque Per Ampere)
    // Iq controller: Setpoint = demanded_iq
    let mut id_pid = PidInstanceF32::new(2.5, 0.15, 0.0);
    let mut iq_pid = PidInstanceF32::new(2.5, 0.15, 0.0);

    let id_setpoint = 0.0f32;
    let iq_setpoint = demanded_iq;

    let id_error = id_setpoint - i_d;
    let iq_error = iq_setpoint - i_q;

    let v_d_command = id_pid.process(id_error).clamp(-24.0, 24.0); // 24V DC bus voltage limit
    let v_q_command = iq_pid.process(iq_error).clamp(-24.0, 24.0);

    println!("  Inner Current PIDs:");
    println!(
        "    • d-Axis (Flux)  : Error = {:>6.3} A -> Commanded V_d = {:>6.3} V",
        id_error, v_d_command
    );
    println!(
        "    • q-Axis (Torque): Error = {:>6.3} A -> Commanded V_q = {:>6.3} V",
        iq_error, v_q_command
    );

    // -----------------------------------------------------------------------------------------
    // 6. Inverse Park & Inverse Clarke Transforms (Modulation Voltages)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 6. Inverse Park & Inverse Clarke Transforms (SVPWM Modulation) ---");
    let mut v_alpha = 0.0f32;
    let mut v_beta = 0.0f32;
    let mut v_a = 0.0f32;
    let mut v_b = 0.0f32;

    inv_park_f32(
        v_d_command,
        v_q_command,
        theta_e[step_idx],
        &mut v_alpha,
        &mut v_beta,
    );
    inv_clarke_f32(v_alpha, v_beta, &mut v_a, &mut v_b);
    let v_c = -v_a - v_b; // 3-phase balanced neutral

    println!(
        "  Inverse Park   : V_alpha = {:>6.3} V, V_beta = {:>6.3} V",
        v_alpha, v_beta
    );
    println!(
        "  Inverse Clarke : V_a = {:>6.3} V, V_b = {:>6.3} V, V_c = {:>6.3} V",
        v_a, v_b, v_c
    );

    // Compute PWM Duty Cycles for Inverter Gate Drivers (normalized to 0.0 .. 1.0)
    let v_dc = 24.0f32;
    let duty_a = (v_a / v_dc + 0.5).clamp(0.0, 1.0);
    let duty_b = (v_b / v_dc + 0.5).clamp(0.0, 1.0);
    let duty_c = (v_c / v_dc + 0.5).clamp(0.0, 1.0);

    println!("  Generated Inverter PWM Duty Cycles:");
    println!("    • Phase A Duty : {:>5.1} %", duty_a * 100.0);
    println!("    • Phase B Duty : {:>5.1} %", duty_b * 100.0);
    println!("    • Phase C Duty : {:>5.1} %", duty_c * 100.0);

    println!();
    println!("===============================================================================");
    println!("             Field-Oriented Control (FOC) Execution Complete!                  ");
    println!("===============================================================================");
}
