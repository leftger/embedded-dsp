//! Comprehensive Sensor Fusion & Inertial Navigation Example
//!
//! Demonstrates:
//! 1. 6-DOF IMU Sensor Simulation with Sensor Bias, Gaussian Noise, and Outliers
//! 2. 1D Conditional Median Filtering for Sensor Glitch / Impulse Rejection
//! 3. Weighted Polynomial Least-Squares Sensor Calibration (Gain & Offset Estimation)
//! 4. 3D Orientation & Attitude Tracking with Quaternions (Normalizing, Conjugate, Inverse, Vector Rotation, Matrix conversion)
//! 5. 2D Constant-Velocity Kinematic Kalman Filter (`KalmanFilter2D` / `KalmanFilter<4, 2>`)
//! 6. Non-linear Radar Tracking with Extended Kalman Filter (`ExtendedKalmanFilter` / `EkfModel`)
//! 7. State Estimation Performance Metrics (RMS error, Variance, Standard Deviation, Max Deviation)

use embedded_dsp::*;

// Non-linear Radar Tracking Model for EKF:
// State x = [pos_x, vel_x, pos_y, vel_y] (4 states)
// Measurement z = [range, bearing_angle] (2 measurements)
// Range = sqrt(pos_x^2 + pos_y^2)
// Bearing = atan2(pos_y, pos_x)
struct RadarTrackingModel;

impl EkfModel<4, 2> for RadarTrackingModel {
    fn f(&self, x: &[f32; 4], dt: f32, out: &mut [f32; 4]) {
        // Constant-velocity motion model
        out[0] = x[0] + dt * x[1]; // x_pos = x_pos + dt * x_vel
        out[1] = x[1]; // x_vel = x_vel
        out[2] = x[2] + dt * x[3]; // y_pos = y_pos + dt * y_vel
        out[3] = x[3]; // y_vel = y_vel
    }

    fn h(&self, x: &[f32; 4], out: &mut [f32; 2]) {
        let px = x[0];
        let py = x[2];
        let range = (px * px + py * py).sqrt();
        let bearing = py.atan2(px);
        out[0] = range;
        out[1] = bearing;
    }

    fn jacobian_f(&self, _x: &[f32; 4], dt: f32, out: &mut [[f32; 4]; 4]) {
        // Linear transition Jacobian F
        *out = [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }

    fn jacobian_h(&self, x: &[f32; 4], out: &mut [[f32; 4]; 2]) {
        let px = x[0];
        let py = x[2];
        let r2 = px * px + py * py;
        let r = r2.max(1e-6).sqrt();

        // Row 0: d(range) / dx = [px/r, 0, py/r, 0]
        out[0] = [px / r, 0.0, py / r, 0.0];

        // Row 1: d(atan2(py, px)) / dx = [-py/r^2, 0, px/r^2, 0]
        let denom = r2.max(1e-6);
        out[1] = [-py / denom, 0.0, px / denom, 0.0];
    }
}

fn main() {
    println!("===============================================================================");
    println!("        embedded-dsp Sensor Fusion, Navigation & Attitude Estimation           ");
    println!("===============================================================================");
    println!();

    // -----------------------------------------------------------------------------------------
    // 1. IMU Sensor Outlier Rejection via 1D Conditional Median Filtering
    // -----------------------------------------------------------------------------------------
    println!("--- 1. Sensor Conditioning & Glitch Removal (Conditional Median Filter) ---");
    // Accelerometer readings experiencing occasional mechanical shock / communication glitches
    let raw_accel_z = [
        9.81f32, 9.80, 9.82, 105.4, 9.81, 9.79, 9.83, 9.81, -45.0, 9.82, 9.80, 9.81,
    ];
    let mut cleaned_accel_z = [0.0f32; 12];
    // Replaces spike only if it deviates by more than threshold (10.0 m/s^2) from local median
    median_filter_1d_f32(&raw_accel_z, &mut cleaned_accel_z, 3, 10.0);

    println!("  Raw Accel Z (with spikes)    : {:?}", raw_accel_z);
    println!("  Cleaned Accel Z (spikes fixed): {:?}", cleaned_accel_z);

    // -----------------------------------------------------------------------------------------
    // 2. Sensor Factory Calibration via Weighted Polynomial Least Squares
    // -----------------------------------------------------------------------------------------
    println!("\n--- 2. Sensor Factory Calibration (Polynomial Least-Squares Fit) ---");
    // Calibration fixture measurements: ADC readings vs Known physical quantities (e.g. pressure/temp)
    let adc_counts = [100.0f32, 200.0, 300.0, 400.0, 500.0];
    // True relationship: Output = 12.5 + 0.45 * ADC
    let ref_values = [57.5f32, 102.5, 147.5, 192.5, 237.5];
    let mut calib_params = [0.0f32; 2]; // [offset c0, gain c1]

    let status = polynomial_least_squares_fit(&adc_counts, &ref_values, None, 1, &mut calib_params);
    if status == Status::Success {
        println!(
            "  Fitted Calibration Model: y = {:.4} + {:.4} * x",
            calib_params[0], calib_params[1]
        );
    } else {
        println!("  Least squares fitting error: {:?}", status);
    }

    // -----------------------------------------------------------------------------------------
    // 3. 3D Orientation & Attitude Tracking with Quaternions
    // -----------------------------------------------------------------------------------------
    println!("\n--- 3. 3D Attitude Estimation with Unit Quaternions ---");
    // Initial attitude: Identity (no rotation)
    let q_current = [1.0f32, 0.0, 0.0, 0.0]; // [w, x, y, z]

    // Incremental rotation: Pitch 90 degrees around Y-axis (cos(45°), 0, sin(45°), 0)
    let angle_y = core::f32::consts::FRAC_PI_2;
    let q_pitch_90 = [(angle_y / 2.0).cos(), 0.0, (angle_y / 2.0).sin(), 0.0];

    let mut q_rotated = [0.0f32; 4];
    quaternion_product_f32(&q_pitch_90, &q_current, &mut q_rotated);
    quaternion_normalize_f32(&mut q_rotated);
    println!(
        "  Quaternion after 90° Pitch: [{:.4}, {:.4}, {:.4}, {:.4}]",
        q_rotated[0], q_rotated[1], q_rotated[2], q_rotated[3]
    );

    // Convert attitude to 3x3 Rotation Matrix
    let mut rot_matrix = [0.0f32; 9];
    quaternion_to_rotmat_f32(&q_rotated, &mut rot_matrix);
    println!("  Converted 3x3 Direction Cosine Matrix (DCM):");
    println!(
        "    [{:>7.4}, {:>7.4}, {:>7.4}]",
        rot_matrix[0], rot_matrix[1], rot_matrix[2]
    );
    println!(
        "    [{:>7.4}, {:>7.4}, {:>7.4}]",
        rot_matrix[3], rot_matrix[4], rot_matrix[5]
    );
    println!(
        "    [{:>7.4}, {:>7.4}, {:>7.4}]",
        rot_matrix[6], rot_matrix[7], rot_matrix[8]
    );

    // Rotate body-frame vector (e.g. forward velocity [1.0, 0.0, 0.0]) to navigation-frame
    // v_rot = q * v * q^*
    let mut q_conj = [0.0f32; 4];
    quaternion_conjugate_f32(&q_rotated, &mut q_conj);
    let v_body = [0.0f32, 1.0, 0.0, 0.0]; // pure imaginary quaternion [0, vx, vy, vz]
    let mut q_temp = [0.0f32; 4];
    let mut v_nav_q = [0.0f32; 4];
    quaternion_product_f32(&q_rotated, &v_body, &mut q_temp);
    quaternion_product_f32(&q_temp, &q_conj, &mut v_nav_q);
    println!(
        "  Body Vector [1, 0, 0] rotated to Navigation Frame: [{:.4}, {:.4}, {:.4}]",
        v_nav_q[1], v_nav_q[2], v_nav_q[3]
    );

    // -----------------------------------------------------------------------------------------
    // 4. Linear 2D Kinematic Kalman Filter (GPS Position + Velocity Fusion)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 4. 2D Kinematic Kalman Filter (Position + Velocity Fusion) ---");
    // True trajectory: moving at constant speed 2.0 m/s from pos = 0
    const DT: f32 = 0.1; // 100 ms time step
    const STEPS: usize = 30;

    let mut kf2d = KalmanFilter2D::new(0.0, 0.0, 0.1, 4.0); // q_var = 0.1, r_var = 4.0 (noisy GPS)
    let mut estimated_pos = [0.0f32; STEPS];
    let mut true_pos = [0.0f32; STEPS];

    for k in 0..STEPS {
        let t = k as f32 * DT;
        let true_p = 2.0 * t;
        true_pos[k] = true_p;

        // Noisy GPS reading: True pos + random noise
        let prng =
            ((k as u64).wrapping_mul(1664525).wrapping_add(1013904223) % 1000) as f32 / 1000.0;
        let noisy_gps = true_p + (prng - 0.5) * 3.0;

        kf2d.predict(DT);
        let est = kf2d.update(noisy_gps);
        estimated_pos[k] = est[0];

        if k % 10 == 0 || k == STEPS - 1 {
            println!(
                "  Step {:>2} (t={:.1}s): True Pos={:>5.2}m, Noisy GPS={:>5.2}m, KF Pos={:>5.2}m, KF Vel={:>5.2}m/s",
                k, t, true_p, noisy_gps, est[0], est[1]
            );
        }
    }

    // -----------------------------------------------------------------------------------------
    // 5. Const-Generic Linear Kalman Filter (4-State, 2-Measurement)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 5. Const-Generic Linear Kalman Filter (4x2 Tracking) ---");
    // State: [px, vx, py, vy]
    let f_matrix = [
        [1.0, DT, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, DT],
        [0.0, 0.0, 0.0, 1.0],
    ];
    // Measurement matrix: GPS measures [px, py]
    let h_matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let mut p_cov = [[0.0f32; 4]; 4];
    let mut q_cov = [[0.0f32; 4]; 4];
    for i in 0..4 {
        p_cov[i][i] = 1.0;
        q_cov[i][i] = 0.05;
    }
    let r_cov = [[1.5, 0.0], [0.0, 1.5]];

    let mut kf_4x2 = KalmanFilter::<4, 2>::new(
        [0.0, 1.5, 0.0, -1.0], // Initial state
        p_cov,
        q_cov,
        r_cov,
    );

    kf_4x2.predict(&f_matrix);
    let meas_z = [0.18, -0.09];
    let kf_status = kf_4x2.update(&h_matrix, &meas_z);
    println!(
        "  Const-generic KalmanFilter<4, 2> update status: {:?}",
        kf_status
    );
    println!(
        "  Updated State Vector: px={:.3}m, vx={:.3}m/s, py={:.3}m, vy={:.3}m/s",
        kf_4x2.x[0], kf_4x2.x[1], kf_4x2.x[2], kf_4x2.x[3]
    );

    // -----------------------------------------------------------------------------------------
    // 6. Non-Linear Extended Kalman Filter (Radar Range + Bearing Tracking)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 6. Non-Linear Extended Kalman Filter (Radar Tracking) ---");
    let ekf_model = RadarTrackingModel;
    let mut ekf = ExtendedKalmanFilter::<4, 2, RadarTrackingModel>::from_variances(
        [100.0, 10.0, 50.0, 5.0], // Target starts at (100m, 50m), speed (10m/s, 5m/s)
        10.0,                     // Initial P variance
        0.2,                      // Process Q variance
        1.0,                      // Measurement R variance
        ekf_model,
    );

    println!("  Target True Trajectory vs EKF Non-Linear Estimate:");
    for step in 1..=5 {
        let dt = 0.5;
        // True target state
        let true_px = 100.0 + step as f32 * dt * 10.0;
        let true_py = 50.0 + step as f32 * dt * 5.0;

        // Polar radar measurements with sensor noise
        let true_range = (true_px * true_px + true_py * true_py).sqrt();
        let true_bearing = true_py.atan2(true_px);

        ekf.predict(dt);
        let z = [true_range + 0.5, true_bearing - 0.005]; // Noisy radar ping
        let status = ekf.update(&z);

        println!(
            "    Step {}: Meas [Range={:>6.1}m, Azimuth={:>6.3} rad] -> EKF Est [X={:>6.1}m, Y={:>6.1}m] (Status: {:?})",
            step, z[0], z[1], ekf.x[0], ekf.x[2], status
        );
    }

    // -----------------------------------------------------------------------------------------
    // 7. Statistical Performance Evaluation
    // -----------------------------------------------------------------------------------------
    println!("\n--- 7. Tracking Error Statistical Metrics ---");
    let mut tracking_errors = [0.0f32; STEPS];
    for i in 0..STEPS {
        tracking_errors[i] = estimated_pos[i] - true_pos[i];
    }

    let mut mean_err = 0.0f32;
    let mut std_err = 0.0f32;
    let mut rms_err = 0.0f32;
    let mut max_err = 0.0f32;
    let mut max_idx = 0usize;

    mean_f32(&tracking_errors, &mut mean_err);
    std_f32(&tracking_errors, &mut std_err);
    rms_f32(&tracking_errors, &mut rms_err);
    max_f32(&tracking_errors, &mut max_err, &mut max_idx);

    println!("  Position Tracking Error Metrics (over {} steps):", STEPS);
    println!("    • Mean Error      : {:>7.4} m", mean_err);
    println!("    • Std Deviation   : {:>7.4} m", std_err);
    println!("    • RMS Error       : {:>7.4} m", rms_err);
    println!(
        "    • Max Error       : {:>7.4} m (at step {})",
        max_err, max_idx
    );

    println!();
    println!("===============================================================================");
    println!("             Sensor Fusion & Navigation Execution Complete!                    ");
    println!("===============================================================================");
}
