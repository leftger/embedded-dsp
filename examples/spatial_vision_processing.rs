//! Comprehensive 2D Spatial & Image Processing Pipeline Example
//!
//! Demonstrates:
//! 1. Synthetic 2D Sensor Matrix / Image Generation (8x8 & 16x16 with edges and salt-and-pepper noise)
//! 2. 2D Spatial Convolution (`convolve2d_f32`) with Gaussian Blur and Sharpening kernels
//! 3. Non-Linear 2D Filtering (`nonlin2d_filter_f32` with Min / Max / Median filters) for noise despeckling
//! 4. 2D Sobel Edge Detection (`sobel_edge_detection_f32`) with gradient thresholding
//! 5. 2D Discrete Cosine Transform (`dct2d_f32` & `idct2d_f32`) for JPEG-style energy compaction
//! 6. Image Quality Metrics: 2D Histogram (`histogram_2d_f32`), MSE (`mse_2d_f32`), and PSNR (`psnr_2d_f32`)
//! 7. Q16.16 Fixed-Point Rasterizer & Scanline Interpolation (`ScanlineInterp`, `mul_q16`, `div_q16`, `lerp_q16`)

use embedded_dsp::*;

fn print_matrix_8x8(label: &str, mat: &[f32; 64]) {
    println!("  {}:", label);
    for r in 0..8 {
        print!("    [");
        for c in 0..8 {
            print!("{:>5.1} ", mat[r * 8 + c]);
        }
        println!("]");
    }
}

fn main() {
    println!("===============================================================================");
    println!("        embedded-dsp 2D Spatial Processing & Embedded Vision                   ");
    println!("===============================================================================");
    println!();

    // -----------------------------------------------------------------------------------------
    // 1. Synthetic 8x8 Sensor Matrix with Block Feature & Salt-and-Pepper Noise
    // -----------------------------------------------------------------------------------------
    println!("--- 1. Synthetic 8x8 Image Matrix with Feature & Noise ---");
    let mut raw_image = [0.0f32; 64];

    // Create a 4x4 high-intensity square in the center
    for r in 2..6 {
        for c in 2..6 {
            raw_image[r * 8 + c] = 20.0;
        }
    }

    // Add salt-and-pepper impulsive noise pixels
    raw_image[1] = 50.0; // Salt noise
    raw_image[15] = 50.0; // Salt noise
    raw_image[3 * 8 + 3] = 0.0; // Pepper noise inside feature
    raw_image[6 * 8 + 2] = 50.0; // Salt noise

    print_matrix_8x8("Raw 8x8 Sensor Image", &raw_image);

    // -----------------------------------------------------------------------------------------
    // 2. 2D Spatial Convolution (Gaussian Blur & Sharpening)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 2. 2D Spatial Convolution (Smoothing & Sharpening) ---");
    // 3x3 Gaussian Blur Kernel
    let gaussian_kernel: [f32; 9] = [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    let mut blurred_image = [0.0f32; 64];
    let status = convolve2d_f32(
        &raw_image,
        &mut blurred_image,
        8,
        8,
        &gaussian_kernel,
        3,
        3,
        true,
    );
    println!("  Gaussian 3x3 Convolution Status: {:?}", status);
    print_matrix_8x8("Gaussian Filtered Image (Smoothed)", &blurred_image);

    // 3x3 Sharpening Kernel
    let sharpen_kernel: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
    let mut sharpened_image = [0.0f32; 64];
    convolve2d_f32(
        &blurred_image,
        &mut sharpened_image,
        8,
        8,
        &sharpen_kernel,
        3,
        3,
        false,
    );
    print_matrix_8x8("Sharpened Image", &sharpened_image);

    // -----------------------------------------------------------------------------------------
    // 3. Non-Linear 2D Filtering (Min, Max, Median Despeckling)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 3. Non-Linear 2D Filtering (Despeckling & Morphological Filters) ---");
    let mut median_cleaned = [0.0f32; 64];
    let mut min_filtered = [0.0f32; 64];
    let mut max_filtered = [0.0f32; 64];

    // 3x3 Median filter removes impulsive salt & pepper noise while preserving sharp boundaries
    nonlin2d_filter_f32(
        &raw_image,
        &mut median_cleaned,
        8,
        8,
        3,
        NonlinFilterType::Median,
    );
    print_matrix_8x8("3x3 Median Filtered Image (Noise Removed)", &median_cleaned);

    // Morphological erosion (Min) and dilation (Max)
    nonlin2d_filter_f32(
        &median_cleaned,
        &mut min_filtered,
        8,
        8,
        3,
        NonlinFilterType::Min,
    );
    nonlin2d_filter_f32(
        &median_cleaned,
        &mut max_filtered,
        8,
        8,
        3,
        NonlinFilterType::Max,
    );
    println!("  Morphological Erosion (Min) & Dilation (Max) computed successfully.");

    // -----------------------------------------------------------------------------------------
    // 4. 2D Sobel Edge Detection
    // -----------------------------------------------------------------------------------------
    println!("\n--- 4. 2D Sobel Edge Detection (Horizontal + Vertical Gradients) ---");
    let mut edges = [0.0f32; 64];
    // Threshold set to 15.0 to detect the boundaries of the central block
    sobel_edge_detection_f32(&median_cleaned, &mut edges, 8, 8, 15.0);
    print_matrix_8x8(
        "Sobel Binary Edge Map (1.0 = Edge, 0.0 = Background)",
        &edges,
    );

    // -----------------------------------------------------------------------------------------
    // 5. 2D DCT-II Transform & Energy Compaction (JPEG Block Transform)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 5. 2D Discrete Cosine Transform (DCT-II) & Inverse DCT-II ---");
    let mut dct_coeffs = [0.0f32; 64];
    let mut reconstructed_image = [0.0f32; 64];

    dct2d_f32(&median_cleaned, &mut dct_coeffs, 8, 8);
    println!(
        "  2D DCT DC Coefficient (Top-Left Energy) = {:.2}",
        dct_coeffs[0]
    );
    println!("  Top 2x2 Low-Frequency DCT Coefficients:");
    println!("    [{:>7.2}, {:>7.2}]", dct_coeffs[0], dct_coeffs[1]);
    println!("    [{:>7.2}, {:>7.2}]", dct_coeffs[8], dct_coeffs[9]);

    // Reconstruct via 2D IDCT
    idct2d_f32(&dct_coeffs, &mut reconstructed_image, 8, 8);
    let mut recon_diff = 0.0f32;
    for i in 0..64 {
        recon_diff += (reconstructed_image[i] - median_cleaned[i]).abs();
    }
    println!(
        "  2D IDCT Exact Reconstruction Absolute Error Sum: {:.2e}",
        recon_diff
    );

    // -----------------------------------------------------------------------------------------
    // 6. Quantitative Image Quality Metrics (Histogram, MSE, PSNR)
    // -----------------------------------------------------------------------------------------
    println!("\n--- 6. Image Metrics: 2D Histogram, MSE, and PSNR ---");
    let mut hist_bins = [0usize; 5]; // 5 bins covering range 0.0 .. 50.0
    histogram_2d_f32(&raw_image, &mut hist_bins, 0.0, 50.0);
    println!(
        "  2D Image Intensity Histogram (5 bins across 0..50): {:?}",
        hist_bins
    );

    let mse = mse_2d_f32(&raw_image, &median_cleaned);
    let psnr = psnr_2d_f32(&raw_image, &median_cleaned, 50.0);
    println!("  Raw Noisy vs Median Cleaned Image:");
    println!("    • Mean Squared Error (MSE)       : {:.2}", mse);
    println!("    • Peak Signal-to-Noise Ratio (PSNR): {:.2} dB", psnr);

    // -----------------------------------------------------------------------------------------
    // 7. Q16.16 Fixed-Point Rasterizer & Scanline Interpolation
    // -----------------------------------------------------------------------------------------
    println!("\n--- 7. Q16.16 Fixed-Point Scanline Interpolation (MCU Graphics / Rasterizer) ---");
    // Span of 10 pixels across scanline
    let span = 10;
    // Left endpoint: Depth z = 1.0 (Q16.16 = 65536), Texture u = 0.0, v = 0.0
    // Right endpoint: Depth z = 5.0 (Q16.16 = 327680), Texture u = 1.0 (65536), v = 1.0 (65536)
    let left_z = to_q16(1.0) as u32;
    let right_z = to_q16(5.0) as u32;
    let left_u = to_q16(0.0) as u32;
    let right_u = to_q16(1.0) as u32;
    let left_v = to_q16(0.0) as u32;
    let right_v = to_q16(1.0) as u32;

    let mut scanline = ScanlineInterp::new(left_z, right_z, left_u, right_u, left_v, right_v, span);

    println!("  Interpolating 10 Pixels Across Fixed-Point Scanline (Q16.16):");
    println!(
        "    {:<5} {:<12} {:<12} {:<12}",
        "Pixel", "Depth (z)", "Texcoord (u)", "Texcoord (v)"
    );
    println!("    --------------------------------------------------");

    for px in 0..=span {
        let z_val = from_q16(scanline.z() as i32);
        let u_val = from_q16(scanline.u() as i32);
        let v_val = from_q16(scanline.v() as i32);

        if px == 0 || px == 5 || px == span {
            println!(
                "    {:<5} {:<12.3} {:<12.3} {:<12.3}",
                px, z_val, u_val, v_val
            );
        }
        scanline.step();
    }

    // Fixed-Point arithmetic helpers
    let a_q16 = to_q16(3.5);
    let b_q16 = to_q16(2.0);
    let prod_q16 = mul_q16(a_q16, b_q16);
    let div_res_q16 = div_q16(a_q16, b_q16);
    let lerp_res_q16 = lerp_q16(a_q16, b_q16, 5, 10);

    println!("\n  Q16.16 Arithmetic Verification:");
    println!(
        "    • mul_q16(3.5, 2.0)   = {:.3} (expected 7.000)",
        from_q16(prod_q16)
    );
    println!(
        "    • div_q16(3.5, 2.0)   = {:.3} (expected 1.750)",
        from_q16(div_res_q16)
    );
    println!(
        "    • lerp_q16(3.5, 2.0)  = {:.3} (expected 2.750)",
        from_q16(lerp_res_q16)
    );

    println!();
    println!("===============================================================================");
    println!("             2D Spatial & Vision Pipeline Execution Complete!                  ");
    println!("===============================================================================");
}
