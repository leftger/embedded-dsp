#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::*;

// --- Complex Addition ---

pub fn cmplx_add_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] + src_b[i];
    }
}

pub fn cmplx_add_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].saturating_add(src_b[i]);
    }
}

pub fn cmplx_add_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].saturating_add(src_b[i]);
    }
}

// --- Complex Subtraction ---

pub fn cmplx_sub_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i] - src_b[i];
    }
}

pub fn cmplx_sub_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].saturating_sub(src_b[i]);
    }
}

pub fn cmplx_sub_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].saturating_sub(src_b[i]);
    }
}

// --- Complex Multiplication (Complex * Complex) ---

pub fn cmplx_mult_cmplx_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    let num_samples = src_a.len() / 2;
    let len = num_samples.min(src_b.len() / 2).min(dst.len() / 2);
    for i in 0..len {
        let ar = src_a[2 * i];
        let ai = src_a[2 * i + 1];
        let br = src_b[2 * i];
        let bi = src_b[2 * i + 1];

        dst[2 * i] = ar * br - ai * bi;
        dst[2 * i + 1] = ar * bi + ai * br;
    }
}

pub fn cmplx_mult_cmplx_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    let num_samples = src_a.len() / 2;
    let len = num_samples.min(src_b.len() / 2).min(dst.len() / 2);
    for i in 0..len {
        let ar = src_a[2 * i];
        let ai = src_a[2 * i + 1];
        let br = src_b[2 * i];
        let bi = src_b[2 * i + 1];

        let real = ((ar as i64 * br as i64) >> 31) - ((ai as i64 * bi as i64) >> 31);
        let imag = ((ar as i64 * bi as i64) >> 31) + ((ai as i64 * br as i64) >> 31);

        dst[2 * i] = real.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        dst[2 * i + 1] = imag.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    }
}

pub fn cmplx_mult_cmplx_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    let num_samples = src_a.len() / 2;
    let len = num_samples.min(src_b.len() / 2).min(dst.len() / 2);
    for i in 0..len {
        let ar = src_a[2 * i];
        let ai = src_a[2 * i + 1];
        let br = src_b[2 * i];
        let bi = src_b[2 * i + 1];

        let real = ((ar as i32 * br as i32) >> 15) - ((ai as i32 * bi as i32) >> 15);
        let imag = ((ar as i32 * bi as i32) >> 15) + ((ai as i32 * br as i32) >> 15);

        dst[2 * i] = real.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        dst[2 * i + 1] = imag.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    }
}

// --- Complex Multiplication (Complex * Real) ---

pub fn cmplx_mult_real_f32(src_cmplx: &[f32], src_real: &[f32], dst: &mut [f32]) {
    let num_samples = (src_cmplx.len() / 2).min(src_real.len()).min(dst.len() / 2);
    for i in 0..num_samples {
        let r = src_real[i];
        dst[2 * i] = src_cmplx[2 * i] * r;
        dst[2 * i + 1] = src_cmplx[2 * i + 1] * r;
    }
}

pub fn cmplx_mult_real_q31(src_cmplx: &[q31], src_real: &[q31], dst: &mut [q31]) {
    let num_samples = (src_cmplx.len() / 2).min(src_real.len()).min(dst.len() / 2);
    for i in 0..num_samples {
        let r = src_real[i];
        dst[2 * i] = q31_mult(src_cmplx[2 * i], r);
        dst[2 * i + 1] = q31_mult(src_cmplx[2 * i + 1], r);
    }
}

pub fn cmplx_mult_real_q15(src_cmplx: &[q15], src_real: &[q15], dst: &mut [q15]) {
    let num_samples = (src_cmplx.len() / 2).min(src_real.len()).min(dst.len() / 2);
    for i in 0..num_samples {
        let r = src_real[i];
        dst[2 * i] = q15_mult(src_cmplx[2 * i], r);
        dst[2 * i + 1] = q15_mult(src_cmplx[2 * i + 1], r);
    }
}

// --- Complex Magnitude ---

pub fn cmplx_mag_f32(src: &[f32], dst: &mut [f32]) {
    let num_samples = (src.len() / 2).min(dst.len());
    for i in 0..num_samples {
        let r = src[2 * i];
        let im = src[2 * i + 1];
        dst[i] = (r * r + im * im).sqrt();
    }
}

pub fn cmplx_mag_q31(src: &[q31], dst: &mut [q31]) {
    let num_samples = (src.len() / 2).min(dst.len());
    for i in 0..num_samples {
        let r = src[2 * i] as f64 / 2147483648.0;
        let im = src[2 * i + 1] as f64 / 2147483648.0;
        let mag = (r * r + im * im).sqrt();
        dst[i] = (mag * 2147483647.0).clamp(0.0, 2147483647.0) as q31;
    }
}

pub fn cmplx_mag_q15(src: &[q15], dst: &mut [q15]) {
    let num_samples = (src.len() / 2).min(dst.len());
    for i in 0..num_samples {
        let r = src[2 * i] as f32 / 32768.0;
        let im = src[2 * i + 1] as f32 / 32768.0;
        let mag = (r * r + im * im).sqrt();
        dst[i] = (mag * 32767.0).clamp(0.0, 32767.0) as q15;
    }
}

// --- Complex Magnitude Squared ---

pub fn cmplx_mag_squared_f32(src: &[f32], dst: &mut [f32]) {
    let num_samples = (src.len() / 2).min(dst.len());
    for i in 0..num_samples {
        let r = src[2 * i];
        let im = src[2 * i + 1];
        dst[i] = r * r + im * im;
    }
}

pub fn cmplx_mag_squared_q31(src: &[q31], dst: &mut [q31]) {
    let num_samples = (src.len() / 2).min(dst.len());
    for i in 0..num_samples {
        let r = src[2 * i] as i64;
        let im = src[2 * i + 1] as i64;
        let acc = ((r * r) >> 33) + ((im * im) >> 33);
        dst[i] = acc.clamp(0, i32::MAX as i64) as q31;
    }
}

pub fn cmplx_mag_squared_q15(src: &[q15], dst: &mut [q15]) {
    let num_samples = (src.len() / 2).min(dst.len());
    for i in 0..num_samples {
        let r = src[2 * i] as i32;
        let im = src[2 * i + 1] as i32;
        let acc = ((r * r) >> 17) + ((im * im) >> 17);
        dst[i] = acc.clamp(0, i16::MAX as i32) as q15;
    }
}

// --- Complex Conjugate ---

pub fn cmplx_conj_f32(src: &[f32], dst: &mut [f32]) {
    let num_samples = (src.len() / 2).min(dst.len() / 2);
    for i in 0..num_samples {
        dst[2 * i] = src[2 * i];
        dst[2 * i + 1] = -src[2 * i + 1];
    }
}

pub fn cmplx_conj_q31(src: &[q31], dst: &mut [q31]) {
    let num_samples = (src.len() / 2).min(dst.len() / 2);
    for i in 0..num_samples {
        dst[2 * i] = src[2 * i];
        dst[2 * i + 1] = src[2 * i + 1].saturating_neg();
    }
}

pub fn cmplx_conj_q15(src: &[q15], dst: &mut [q15]) {
    let num_samples = (src.len() / 2).min(dst.len() / 2);
    for i in 0..num_samples {
        dst[2 * i] = src[2 * i];
        dst[2 * i + 1] = src[2 * i + 1].saturating_neg();
    }
}

// --- Complex Dot Product ---

pub fn cmplx_dot_prod_f32(src_a: &[f32], src_b: &[f32]) -> Complex<f32> {
    let num_samples = (src_a.len() / 2).min(src_b.len() / 2);
    let mut real_sum = 0.0f32;
    let mut imag_sum = 0.0f32;
    for i in 0..num_samples {
        let ar = src_a[2 * i];
        let ai = src_a[2 * i + 1];
        let br = src_b[2 * i];
        let bi = src_b[2 * i + 1];

        real_sum += ar * br - ai * bi;
        imag_sum += ar * bi + ai * br;
    }
    Complex::new(real_sum, imag_sum)
}

pub fn cmplx_dot_prod_q31(src_a: &[q31], src_b: &[q31]) -> Complex<q63> {
    let num_samples = (src_a.len() / 2).min(src_b.len() / 2);
    let mut real_sum: q63 = 0;
    let mut imag_sum: q63 = 0;
    for i in 0..num_samples {
        let ar = src_a[2 * i] as i64;
        let ai = src_a[2 * i + 1] as i64;
        let br = src_b[2 * i] as i64;
        let bi = src_b[2 * i + 1] as i64;

        real_sum += (ar * br - ai * bi) >> 14;
        imag_sum += (ar * bi + ai * br) >> 14;
    }
    Complex::new(real_sum, imag_sum)
}

pub fn cmplx_dot_prod_q15(src_a: &[q15], src_b: &[q15]) -> Complex<q63> {
    let num_samples = (src_a.len() / 2).min(src_b.len() / 2);
    let mut real_sum: q63 = 0;
    let mut imag_sum: q63 = 0;
    for i in 0..num_samples {
        let ar = src_a[2 * i] as i32;
        let ai = src_a[2 * i + 1] as i32;
        let br = src_b[2 * i] as i32;
        let bi = src_b[2 * i + 1] as i32;

        real_sum += (ar * br - ai * bi) as q63;
        imag_sum += (ar * bi + ai * br) as q63;
    }
    Complex::new(real_sum, imag_sum)
}
