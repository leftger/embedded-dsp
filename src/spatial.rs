//! 2D Spatial and Image Processing routines (2D DCT/IDCT, 2D Convolution, 2D Nonlinear Filters, Sobel Edge Detection, 2D Histogram, MSE/PSNR).

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::Status;

/// Non-linear 2D spatial filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonlinFilterType {
    /// Minimum value in the sliding window.
    Min,
    /// Maximum value in the sliding window.
    Max,
    /// Median value in the sliding window.
    Median,
}

/// Computes the 2D Discrete Cosine Transform (DCT-II) on a `rows x cols` image.
///
/// `src` and `dst` must have length at least `rows * cols`.
pub fn dct2d_f32(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) -> Status {
    let total = rows * cols;
    if rows == 0 || cols == 0 || src.len() < total || dst.len() < total {
        return Status::LengthError;
    }

    let pi_over_2m = core::f32::consts::PI / (2.0 * rows as f32);
    let pi_over_2n = core::f32::consts::PI / (2.0 * cols as f32);
    let norm_row0 = (1.0 / rows as f32).sqrt();
    let norm_rowk = (2.0 / rows as f32).sqrt();
    let norm_col0 = (1.0 / cols as f32).sqrt();
    let norm_colk = (2.0 / cols as f32).sqrt();

    for u in 0..rows {
        let cu = if u == 0 { norm_row0 } else { norm_rowk };
        let u_f = u as f32;

        for v in 0..cols {
            let cv = if v == 0 { norm_col0 } else { norm_colk };
            let v_f = v as f32;

            let mut sum = 0.0f32;
            for x in 0..rows {
                let cos_u = ((2 * x + 1) as f32 * u_f * pi_over_2m).cos();
                for y in 0..cols {
                    let cos_v = ((2 * y + 1) as f32 * v_f * pi_over_2n).cos();
                    sum += src[x * cols + y] * cos_u * cos_v;
                }
            }
            dst[u * cols + v] = cu * cv * sum;
        }
    }

    Status::Success
}

/// Computes the 2D Inverse Discrete Cosine Transform (IDCT-II) on a `rows x cols` coefficient matrix.
pub fn idct2d_f32(src: &[f32], dst: &mut [f32], rows: usize, cols: usize) -> Status {
    let total = rows * cols;
    if rows == 0 || cols == 0 || src.len() < total || dst.len() < total {
        return Status::LengthError;
    }

    let pi_over_2m = core::f32::consts::PI / (2.0 * rows as f32);
    let pi_over_2n = core::f32::consts::PI / (2.0 * cols as f32);
    let norm_row0 = (1.0 / rows as f32).sqrt();
    let norm_rowk = (2.0 / rows as f32).sqrt();
    let norm_col0 = (1.0 / cols as f32).sqrt();
    let norm_colk = (2.0 / cols as f32).sqrt();

    for x in 0..rows {
        for y in 0..cols {
            let mut sum = 0.0f32;
            for u in 0..rows {
                let cu = if u == 0 { norm_row0 } else { norm_rowk };
                let cos_u = ((2 * x + 1) as f32 * u as f32 * pi_over_2m).cos();

                for v in 0..cols {
                    let cv = if v == 0 { norm_col0 } else { norm_colk };
                    let cos_v = ((2 * y + 1) as f32 * v as f32 * pi_over_2n).cos();
                    sum += cu * cv * src[u * cols + v] * cos_u * cos_v;
                }
            }
            dst[x * cols + y] = sum;
        }
    }

    Status::Success
}

/// Performs 2D spatial convolution of `src` image (`rows x cols`) with a `k_rows x k_cols` kernel.
///
/// If `normalize` is `true`, the convolution output is divided by the sum of the absolute kernel weights.
pub fn convolve2d_f32(
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    cols: usize,
    kernel: &[f32],
    k_rows: usize,
    k_cols: usize,
    normalize: bool,
) -> Status {
    let total = rows * cols;
    let k_total = k_rows * k_cols;
    if rows == 0 || cols == 0 || k_rows == 0 || k_cols == 0 {
        return Status::ArgumentError;
    }
    if src.len() < total || dst.len() < total || kernel.len() < k_total {
        return Status::LengthError;
    }

    let dead_r = k_rows / 2;
    let dead_c = k_cols / 2;

    let norm_factor: f32 = if normalize {
        let mut sum = 0.0f32;
        for &v in kernel {
            sum += v.abs();
        }
        if sum != 0.0 {
            sum
        } else {
            1.0
        }
    } else {
        1.0
    };

    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0f32;
            for kr in 0..k_rows {
                let ir = (r as isize + kr as isize - dead_r as isize).clamp(0, (rows - 1) as isize)
                    as usize;
                for kc in 0..k_cols {
                    let ic = (c as isize + kc as isize - dead_c as isize)
                        .clamp(0, (cols - 1) as isize) as usize;
                    acc += src[ir * cols + ic] * kernel[kr * k_cols + kc];
                }
            }
            dst[r * cols + c] = acc / norm_factor;
        }
    }

    Status::Success
}

/// Performs 2D non-linear filtering (`Min`, `Max`, or `Median`) on a `rows x cols` image using a `k_size x k_size` square window.
///
/// `k_size` must be odd and $\le 7$ for embedded zero-allocation stack sorting.
pub fn nonlin2d_filter_f32(
    src: &[f32],
    dst: &mut [f32],
    rows: usize,
    cols: usize,
    k_size: usize,
    filtype: NonlinFilterType,
) -> Status {
    let total = rows * cols;
    if rows == 0 || cols == 0 || k_size == 0 || k_size % 2 == 0 || k_size > 7 {
        return Status::ArgumentError;
    }
    if src.len() < total || dst.len() < total {
        return Status::LengthError;
    }

    let half = k_size / 2;
    let k_len = k_size * k_size;
    let mut sort_buf = [0.0f32; 64];

    for r in 0..rows {
        for c in 0..cols {
            let mut count = 0;
            for kr in 0..k_size {
                let ir = (r as isize + kr as isize - half as isize).clamp(0, (rows - 1) as isize)
                    as usize;
                for kc in 0..k_size {
                    let ic = (c as isize + kc as isize - half as isize)
                        .clamp(0, (cols - 1) as isize) as usize;
                    sort_buf[count] = src[ir * cols + ic];
                    count += 1;
                }
            }

            match filtype {
                NonlinFilterType::Min => {
                    let mut min_v = sort_buf[0];
                    for i in 1..k_len {
                        if sort_buf[i] < min_v {
                            min_v = sort_buf[i];
                        }
                    }
                    dst[r * cols + c] = min_v;
                }
                NonlinFilterType::Max => {
                    let mut max_v = sort_buf[0];
                    for i in 1..k_len {
                        if sort_buf[i] > max_v {
                            max_v = sort_buf[i];
                        }
                    }
                    dst[r * cols + c] = max_v;
                }
                NonlinFilterType::Median => {
                    for a in 1..k_len {
                        let mut b = a;
                        while b > 0 && sort_buf[b - 1] > sort_buf[b] {
                            sort_buf.swap(b - 1, b);
                            b -= 1;
                        }
                    }
                    dst[r * cols + c] = sort_buf[k_len / 2];
                }
            }
        }
    }

    Status::Success
}

/// Applies Sobel edge detection to a 2D image, outputting gradient magnitude and binary edge detection.
///
/// Steps:
/// 1. Convolve with horizontal Sobel operator $G_x$.
/// 2. Convolve with vertical Sobel operator $G_y$.
/// 3. Compute gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$.
/// 4. Threshold magnitude: values $\ge \text{threshold}$ become `1.0`, others `0.0`.
pub fn sobel_edge_detection_f32(
    src: &[f32],
    dst_edges: &mut [f32],
    rows: usize,
    cols: usize,
    threshold: f32,
) -> Status {
    let total = rows * cols;
    if rows < 3 || cols < 3 || src.len() < total || dst_edges.len() < total {
        return Status::LengthError;
    }

    let h_sobel: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let v_sobel: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    for r in 0..rows {
        for c in 0..cols {
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;

            for kr in 0..3 {
                let ir = (r as isize + kr as isize - 1).clamp(0, (rows - 1) as isize) as usize;
                for kc in 0..3 {
                    let ic = (c as isize + kc as isize - 1).clamp(0, (cols - 1) as isize) as usize;
                    let val = src[ir * cols + ic];
                    gx += val * h_sobel[kr * 3 + kc];
                    gy += val * v_sobel[kr * 3 + kc];
                }
            }

            let mag = (gx * gx + gy * gy).sqrt();
            dst_edges[r * cols + c] = if mag >= threshold { 1.0 } else { 0.0 };
        }
    }

    Status::Success
}

/// Computes the histogram of a 2D image / matrix into `bins` spanning `[min_val, max_val]`.
pub fn histogram_2d_f32(src: &[f32], bins: &mut [usize], min_val: f32, max_val: f32) -> Status {
    let num_bins = bins.len();
    if src.is_empty() || num_bins == 0 || max_val <= min_val {
        return Status::ArgumentError;
    }

    bins.fill(0);
    let span = max_val - min_val;
    let scale = (num_bins as f32) / span;

    for &val in src {
        let clamped = val.clamp(min_val, max_val);
        let mut idx = ((clamped - min_val) * scale) as usize;
        if idx >= num_bins {
            idx = num_bins - 1;
        }
        bins[idx] += 1;
    }

    Status::Success
}

/// Computes the Mean Squared Error (MSE) between two 2D images.
pub fn mse_2d_f32(img_a: &[f32], img_b: &[f32]) -> f32 {
    let len = img_a.len().min(img_b.len());
    if len == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0f32;
    for i in 0..len {
        let diff = img_a[i] - img_b[i];
        sum_sq += diff * diff;
    }
    sum_sq / (len as f32)
}

/// Computes the Peak Signal-to-Noise Ratio (PSNR) in dB between two 2D images.
pub fn psnr_2d_f32(img_a: &[f32], img_b: &[f32], max_val: f32) -> f32 {
    let mse = mse_2d_f32(img_a, img_b);
    if mse <= 1e-12 {
        return 99.0; // Near identical
    }
    10.0 * ((max_val * max_val) / mse).log10()
}
