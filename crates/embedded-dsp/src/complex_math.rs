#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::*;

// --- Complex Addition ---

/// Complex vector addition into `dst`, generic over the sample width. Operates on flat
/// interleaved `[re, im, re, im, ...]` slices, not [`Complex<T>`] values.
pub fn cmplx_add<T: DspSample>(src_a: &[T], src_b: &[T], dst: &mut [T]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].sat_add(src_b[i]);
    }
}

/// Complex vector addition (`f32`) into `dst`.
pub fn cmplx_add_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    cmplx_add(src_a, src_b, dst)
}

/// Complex vector addition (`q31`) into `dst`.
pub fn cmplx_add_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    cmplx_add(src_a, src_b, dst)
}

/// Complex vector addition (`q15`) into `dst`.
pub fn cmplx_add_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    cmplx_add(src_a, src_b, dst)
}

// --- Complex Subtraction ---

/// Complex vector subtraction into `dst`, generic over the sample width.
pub fn cmplx_sub<T: DspSample>(src_a: &[T], src_b: &[T], dst: &mut [T]) {
    let len = src_a.len().min(src_b.len()).min(dst.len());
    for i in 0..len {
        dst[i] = src_a[i].sat_sub(src_b[i]);
    }
}

/// Complex vector subtraction (`f32`) into `dst`.
pub fn cmplx_sub_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    cmplx_sub(src_a, src_b, dst)
}

/// Complex vector subtraction (`q31`) into `dst`.
pub fn cmplx_sub_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    cmplx_sub(src_a, src_b, dst)
}

/// Complex vector subtraction (`q15`) into `dst`.
pub fn cmplx_sub_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    cmplx_sub(src_a, src_b, dst)
}

// --- Complex Multiplication (Complex * Complex) ---

/// Complex-by-complex multiplication into `dst`, generic over the sample width.
///
/// Requires `T::Coeff = T` (true for `f32`/`f64`/`q15`/`q31` today): each real/imaginary part
/// multiplies another sample of its own type, so the coefficient and sample types must coincide.
/// The per-term product runs through [`DspSample::mul_high`] (the same per-term-shift primitive
/// the FIR/biquad kernels use), matching the hand-written kernels' `(a * b) >> FRAC` per term
/// exactly, rather than shifting the combined sum.
pub fn cmplx_mult_cmplx<T: DspSample<Coeff = T>>(src_a: &[T], src_b: &[T], dst: &mut [T]) {
    let num_samples = src_a.len() / 2;
    let len = num_samples.min(src_b.len() / 2).min(dst.len() / 2);
    for i in 0..len {
        let ar = src_a[2 * i];
        let ai = src_a[2 * i + 1];
        let br = src_b[2 * i];
        let bi = src_b[2 * i + 1];

        let real = T::mul_high(ar, br) - T::mul_high(ai, bi);
        let imag = T::mul_high(ar, bi) + T::mul_high(ai, br);

        dst[2 * i] = T::from_accum_shifted(real, 0);
        dst[2 * i + 1] = T::from_accum_shifted(imag, 0);
    }
}

/// Complex-by-complex multiplication (`f32`) into `dst`.
pub fn cmplx_mult_cmplx_f32(src_a: &[f32], src_b: &[f32], dst: &mut [f32]) {
    cmplx_mult_cmplx(src_a, src_b, dst)
}

/// Complex-by-complex multiplication (`q31`) into `dst`.
pub fn cmplx_mult_cmplx_q31(src_a: &[q31], src_b: &[q31], dst: &mut [q31]) {
    cmplx_mult_cmplx(src_a, src_b, dst)
}

/// Complex-by-complex multiplication (`q15`) into `dst`.
pub fn cmplx_mult_cmplx_q15(src_a: &[q15], src_b: &[q15], dst: &mut [q15]) {
    cmplx_mult_cmplx(src_a, src_b, dst)
}

// --- Complex Multiplication (Complex * Real) ---

/// Complex-by-real multiplication into `dst`, generic over the sample width.
pub fn cmplx_mult_real<T: DspSample>(src_cmplx: &[T], src_real: &[T], dst: &mut [T]) {
    let num_samples = (src_cmplx.len() / 2).min(src_real.len()).min(dst.len() / 2);
    for i in 0..num_samples {
        let r = src_real[i];
        dst[2 * i] = src_cmplx[2 * i].sat_mul(r);
        dst[2 * i + 1] = src_cmplx[2 * i + 1].sat_mul(r);
    }
}

/// Complex-by-real multiplication (`f32`) into `dst`.
pub fn cmplx_mult_real_f32(src_cmplx: &[f32], src_real: &[f32], dst: &mut [f32]) {
    cmplx_mult_real(src_cmplx, src_real, dst)
}

/// Complex-by-real multiplication (`q31`) into `dst`.
pub fn cmplx_mult_real_q31(src_cmplx: &[q31], src_real: &[q31], dst: &mut [q31]) {
    cmplx_mult_real(src_cmplx, src_real, dst)
}

/// Complex-by-real multiplication (`q15`) into `dst`.
pub fn cmplx_mult_real_q15(src_cmplx: &[q15], src_real: &[q15], dst: &mut [q15]) {
    cmplx_mult_real(src_cmplx, src_real, dst)
}

// --- Complex Magnitude Squared ---

/// Squared magnitude of a complex vector into `dst`, generic over the sample width.
///
/// Requires `T::Coeff = T` (see [`cmplx_mult_cmplx`]). Each term (`r*r`, `im*im`) is shifted by
/// `FRAC + 2` individually via [`DspSample::mul_shifted`] *before* summing — one more bit of
/// headroom than [`DspSample::mul_high`]'s `FRAC` alone would leave, matching the hand-written
/// kernels' `(r*r) >> (FRAC+2) + (im*im) >> (FRAC+2)` term-by-term rather than shifting the
/// combined sum once (the two are not bit-exact equivalent in general integer arithmetic).
pub fn cmplx_mag_squared<T: DspSample<Coeff = T>>(src: &[T], dst: &mut [T]) {
    let num_samples = (src.len() / 2).min(dst.len());
    for i in 0..num_samples {
        let r = src[2 * i];
        let im = src[2 * i + 1];
        let acc = T::mul_shifted(r, r, T::FRAC + 2) + T::mul_shifted(im, im, T::FRAC + 2);
        dst[i] = T::from_accum_shifted(acc, 0);
    }
}

/// Squared magnitude of a complex vector (`f32`) into `dst`.
pub fn cmplx_mag_squared_f32(src: &[f32], dst: &mut [f32]) {
    cmplx_mag_squared(src, dst)
}

/// Squared magnitude of a complex vector (`q31`) into `dst`.
pub fn cmplx_mag_squared_q31(src: &[q31], dst: &mut [q31]) {
    cmplx_mag_squared(src, dst)
}

/// Squared magnitude of a complex vector (`q15`) into `dst`.
pub fn cmplx_mag_squared_q15(src: &[q15], dst: &mut [q15]) {
    cmplx_mag_squared(src, dst)
}

// --- Complex Magnitude ---

/// Magnitude of a complex vector (`f32`) into `dst`.
pub fn cmplx_mag_f32(src: &[f32], dst: &mut [f32]) {
    let num_samples = (src.len() / 2).min(dst.len());
    cmplx_mag_squared_f32(src, dst);
    for out in &mut dst[..num_samples] {
        *out = out.sqrt();
    }
}

/// Magnitude of a complex vector (`q31`) into `dst`.
pub fn cmplx_mag_q31(src: &[q31], dst: &mut [q31]) {
    let num_samples = (src.len() / 2).min(dst.len());
    cmplx_mag_squared_q31(src, dst);
    for out in &mut dst[..num_samples] {
        let mut mag = q31::ZERO;
        let _ = crate::fast_math::sqrt_q31(*out, &mut mag);
        *out = mag;
    }
}

/// Magnitude of a complex vector (`q15`) into `dst`.
pub fn cmplx_mag_q15(src: &[q15], dst: &mut [q15]) {
    let num_samples = (src.len() / 2).min(dst.len());
    cmplx_mag_squared_q15(src, dst);
    for out in &mut dst[..num_samples] {
        let mut mag = q15::ZERO;
        let _ = crate::fast_math::sqrt_q15(*out, &mut mag);
        *out = mag;
    }
}

// --- Complex Conjugate ---

/// Complex conjugate into `dst`, generic over the sample width.
pub fn cmplx_conj<T: DspSample>(src: &[T], dst: &mut [T]) {
    let num_samples = (src.len() / 2).min(dst.len() / 2);
    for i in 0..num_samples {
        dst[2 * i] = src[2 * i];
        dst[2 * i + 1] = src[2 * i + 1].sat_neg();
    }
}

/// Complex conjugate (`f32`) into `dst`.
pub fn cmplx_conj_f32(src: &[f32], dst: &mut [f32]) {
    cmplx_conj(src, dst)
}

/// Complex conjugate (`q31`) into `dst`.
pub fn cmplx_conj_q31(src: &[q31], dst: &mut [q31]) {
    cmplx_conj(src, dst)
}

/// Complex conjugate (`q15`) into `dst`.
pub fn cmplx_conj_q15(src: &[q15], dst: &mut [q15]) {
    cmplx_conj(src, dst)
}

// --- Complex Dot Product ---

/// Complex dot product, generic over the sample width.
///
/// Requires `T::Coeff = T` (see [`cmplx_mult_cmplx`]). Each term's real/imaginary product is
/// computed *raw* (full-width, unshifted, via [`DspSample::madd`] seeded from `Accum::default()`)
/// and combined before [`DspSample::accum_shift`] scales it by `shift` — a per-call scale factor
/// applied to each term prior to accumulating across the vector, not to the final sum (the two
/// are not bit-exact equivalent in general integer arithmetic; see [`cmplx_mag_squared`] for the
/// same distinction). The result stays in the accumulator domain (`Complex<T::Accum>`): `f32` for
/// `f32`, `q63` (`i64`) for `q15`/`q31`, matching what the hand-written kernels returned.
pub fn cmplx_dot_prod<T: DspSample<Coeff = T>>(
    src_a: &[T],
    src_b: &[T],
    shift: u32,
) -> Complex<T::Accum> {
    let num_samples = (src_a.len() / 2).min(src_b.len() / 2);
    let mut real_sum = T::Accum::default();
    let mut imag_sum = T::Accum::default();
    for i in 0..num_samples {
        let ar = src_a[2 * i];
        let ai = src_a[2 * i + 1];
        let br = src_b[2 * i];
        let bi = src_b[2 * i + 1];

        let raw_real = T::madd(T::Accum::default(), ar, br) - T::madd(T::Accum::default(), ai, bi);
        let raw_imag = T::madd(T::Accum::default(), ar, bi) + T::madd(T::Accum::default(), ai, br);

        real_sum = real_sum + T::accum_shift(raw_real, shift);
        imag_sum = imag_sum + T::accum_shift(raw_imag, shift);
    }
    Complex::new(real_sum, imag_sum)
}

/// Complex dot product (`f32`).
pub fn cmplx_dot_prod_f32(src_a: &[f32], src_b: &[f32]) -> Complex<f32> {
    cmplx_dot_prod(src_a, src_b, 0)
}

/// Complex dot product (`q31`).
pub fn cmplx_dot_prod_q31(src_a: &[q31], src_b: &[q31]) -> Complex<q63> {
    cmplx_dot_prod(src_a, src_b, 14)
}

/// Complex dot product (`q15`).
pub fn cmplx_dot_prod_q15(src_a: &[q15], src_b: &[q15]) -> Complex<q63> {
    cmplx_dot_prod(src_a, src_b, 0)
}
