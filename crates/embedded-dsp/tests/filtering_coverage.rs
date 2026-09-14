//! Coverage for filtering primitives: FIR, biquad cascades, LMS, convolution,
//! correlation, one-pole filters, and circular buffers.

use embedded_dsp::filtering::{
    BiquadCascadeDf2tInstanceF32, BiquadCascadeDf2tInstanceQ15, BiquadCascadeDf2tInstanceQ31,
    BiquadCascadeInstanceF32, BiquadCascadeInstanceQ15, BiquadCascadeInstanceQ31, CircularBuffer,
    DcBlockerQ15, FirInstanceF32, FirInstanceQ15, FirInstanceQ31, LmsInstanceF32, NlmsInstanceF32,
    RecursiveMovingAverage, RecursiveMovingAverageQ15, SinglePoleFilter, biquad_cascade_df1_f32,
    biquad_cascade_df1_q15, biquad_cascade_df1_q31, biquad_cascade_df2t_f32,
    biquad_cascade_df2t_q15, biquad_cascade_df2t_q31, conv_f32, conv_q7, conv_q15, conv_q31,
    correlate_f32, correlate_q15, correlate_q31, fir_f32, fir_q15, fir_q31, lms_f32, lms_leaky_f32,
    nlms_f32,
};
use embedded_dsp::types::{q7, q15, q31};

#[test]
fn fir_f32_q31_q15_run() {
    let coeffs = [0.5f32, 0.25, 0.25];
    let src = [1.0f32, 2.0, 3.0, 4.0];
    let mut state = [0.0f32; 3];
    let mut dst = [0.0f32; 4];
    let mut fir = FirInstanceF32::init(3, &coeffs, &mut state);
    fir_f32(&mut fir, &src, &mut dst);

    let qcoeffs = [
        q31::from_bits(1 << 30),
        q31::from_bits(1 << 29),
        q31::from_bits(1 << 29),
    ];
    let qsrc = [
        q31::from_bits(1 << 20),
        q31::from_bits(2 << 20),
        q31::from_bits(3 << 20),
        q31::from_bits(4 << 20),
    ];
    let mut qstate = [q31::ZERO; 3];
    let mut qdst = [q31::ZERO; 4];
    let mut qfir = FirInstanceQ31::init(3, &qcoeffs, &mut qstate);
    fir_q31(&mut qfir, &qsrc, &mut qdst);

    let q15coeffs = [
        q15::from_bits(1 << 14),
        q15::from_bits(1 << 13),
        q15::from_bits(1 << 13),
    ];
    let q15src = [
        q15::from_bits(100),
        q15::from_bits(200),
        q15::from_bits(300),
        q15::from_bits(400),
    ];
    let mut q15state = [q15::ZERO; 3];
    let mut q15dst = [q15::ZERO; 4];
    let mut q15fir = FirInstanceQ15::init(3, &q15coeffs, &mut q15state);
    fir_q15(&mut q15fir, &q15src, &mut q15dst);
}

#[test]
fn biquad_cascade_variants_run() {
    let coeffs = [1.0f32, 0.0, 0.0, 0.0, 0.0];
    let src = [1.0f32, 2.0, 3.0, 4.0];
    let mut state = [0.0f32; 4];
    let mut dst = [0.0f32; 4];
    let mut inst = BiquadCascadeInstanceF32::init(1, &coeffs, &mut state);
    biquad_cascade_df1_f32(&mut inst, &src, &mut dst);

    let mut state2 = [0.0f32; 2];
    let mut dst2 = [0.0f32; 4];
    let mut inst2 = BiquadCascadeDf2tInstanceF32::init(1, &coeffs, &mut state2);
    biquad_cascade_df2t_f32(&mut inst2, &src, &mut dst2);

    let qcoeffs = [
        q31::from_bits(1 << 31),
        q31::ZERO,
        q31::ZERO,
        q31::ZERO,
        q31::ZERO,
    ];
    let qsrc = [
        q31::from_bits(1000),
        q31::from_bits(2000),
        q31::from_bits(3000),
        q31::from_bits(4000),
    ];
    let mut qstate = [q31::ZERO; 4];
    let mut qdst = [q31::ZERO; 4];
    let mut qinst = BiquadCascadeInstanceQ31::with_post_shift(1, &qcoeffs, &mut qstate, 0);
    biquad_cascade_df1_q31(&mut qinst, &qsrc, &mut qdst);

    let mut qstate2 = [q31::ZERO; 2];
    let mut qdst2 = [q31::ZERO; 4];
    let mut qinst2 = BiquadCascadeDf2tInstanceQ31::with_post_shift(1, &qcoeffs, &mut qstate2, 0);
    biquad_cascade_df2t_q31(&mut qinst2, &qsrc, &mut qdst2);

    let q15coeffs = [
        q15::from_bits(1 << 15),
        q15::ZERO,
        q15::ZERO,
        q15::ZERO,
        q15::ZERO,
    ];
    let q15src = [
        q15::from_bits(100),
        q15::from_bits(200),
        q15::from_bits(300),
        q15::from_bits(400),
    ];
    let mut q15state = [q15::ZERO; 4];
    let mut q15dst = [q15::ZERO; 4];
    let mut q15inst = BiquadCascadeInstanceQ15::with_post_shift(1, &q15coeffs, &mut q15state, 0);
    biquad_cascade_df1_q15(&mut q15inst, &q15src, &mut q15dst);

    let mut q15state2 = [q15::ZERO; 2];
    let mut q15dst2 = [q15::ZERO; 4];
    let mut q15inst2 =
        BiquadCascadeDf2tInstanceQ15::with_post_shift(1, &q15coeffs, &mut q15state2, 0);
    biquad_cascade_df2t_q15(&mut q15inst2, &q15src, &mut q15dst2);
}

/// The generic biquad cascades must reproduce the hand-written per-width kernels bit for bit,
/// including the `post_shift` narrowing — not just track them within a tolerance.
#[test]
fn generic_biquad_cascades_match_reference_bit_for_bit() {
    let coeffs = [
        q15::from_bits(1000),
        q15::from_bits(-200),
        q15::from_bits(300),
        q15::from_bits(4000),
        q15::from_bits(-500),
    ];
    let src: [q15; 8] =
        core::array::from_fn(|i| q15::from_bits(((i as i32 * 1237 + 11) % 20_000 - 10_000) as i16));

    for &post_shift in &[0u8, 1u8] {
        let shift = 15u32.saturating_sub(post_shift as u32).min(31);
        let b: [i64; 5] = core::array::from_fn(|k| coeffs[k].to_bits() as i64);

        // Reference Direct Form I (the pre-genericization kernel).
        let (mut x1, mut x2, mut y1, mut y2) = (0i64, 0i64, 0i64, 0i64);
        let mut want_df1 = [q15::ZERO; 8];
        for (i, &s) in src.iter().enumerate() {
            let in_val = s.to_bits() as i64;
            let acc = b[0] * in_val + b[1] * x1 + b[2] * x2 + b[3] * y1 + b[4] * y2;
            let out = (acc >> shift).clamp(i16::MIN as i64, i16::MAX as i64);
            x2 = x1;
            x1 = in_val;
            y2 = y1;
            y1 = out;
            want_df1[i] = q15::from_bits(out as i16);
        }
        let mut st = [q15::ZERO; 4];
        let mut inst = BiquadCascadeInstanceQ15::with_post_shift(1, &coeffs, &mut st, post_shift);
        let mut got_df1 = [q15::ZERO; 8];
        biquad_cascade_df1_q15(&mut inst, &src, &mut got_df1);
        assert_eq!(got_df1, want_df1, "DF1 diverged at post_shift {post_shift}");

        // Reference transposed Direct Form II.
        let (mut s1, mut s2) = (0i64, 0i64);
        let mut want_df2t = [q15::ZERO; 8];
        for (i, &s) in src.iter().enumerate() {
            let in_val = s.to_bits() as i64;
            let y = (b[0] * in_val + (s1 << shift)).clamp(i64::MIN >> 1, i64::MAX >> 1) >> shift;
            let out = y.clamp(i16::MIN as i64, i16::MAX as i64);
            let s1_new = (b[1] * in_val + b[3] * out + (s2 << shift)) >> shift;
            let s2_new = (b[2] * in_val + b[4] * out) >> shift;
            s1 = s1_new.clamp(i16::MIN as i64, i16::MAX as i64);
            s2 = s2_new.clamp(i16::MIN as i64, i16::MAX as i64);
            want_df2t[i] = q15::from_bits(out as i16);
        }
        let mut st2 = [q15::ZERO; 2];
        let mut inst2 =
            BiquadCascadeDf2tInstanceQ15::with_post_shift(1, &coeffs, &mut st2, post_shift);
        let mut got_df2t = [q15::ZERO; 8];
        biquad_cascade_df2t_q15(&mut inst2, &src, &mut got_df2t);
        assert_eq!(
            got_df2t, want_df2t,
            "DF2T diverged at post_shift {post_shift}"
        );
    }
}

/// The generic FIR must reproduce the per-term high-multiply recurrence bit for bit.
#[test]
fn generic_fir_matches_per_term_reference_bit_for_bit() {
    // q15: `acc += (state * coeff) >> 15`, then saturate.
    let coeffs = [
        q15::from_bits(1000),
        q15::from_bits(-200),
        q15::from_bits(50),
    ];
    let src: [q15; 8] =
        core::array::from_fn(|i| q15::from_bits(((i as i32 * 1237 + 11) % 20_000 - 10_000) as i16));
    let mut want = [q15::ZERO; 8];
    let mut state = [0i32; 3];
    for (i, &s) in src.iter().enumerate() {
        state[2] = state[1];
        state[1] = state[0];
        state[0] = s.to_bits() as i32;
        let mut acc = 0i32;
        for k in 0..3 {
            acc += (state[k] * coeffs[k].to_bits() as i32) >> 15;
        }
        want[i] = q15::from_bits(acc.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
    }
    let mut st = [q15::ZERO; 3];
    let mut got = [q15::ZERO; 8];
    let mut inst = FirInstanceQ15::init(3, &coeffs, &mut st);
    fir_q15(&mut inst, &src, &mut got);
    assert_eq!(got, want, "q15 FIR diverged from the per-term reference");

    // f32: plain dot product.
    let fcoeffs = [0.25f32, -0.5, 0.125];
    let fsrc: [f32; 8] = core::array::from_fn(|i| ((i as f32) * 0.017 - 1.3).sin());
    let mut fwant = [0.0f32; 8];
    let mut fstate = [0.0f32; 3];
    for (i, &s) in fsrc.iter().enumerate() {
        fstate[2] = fstate[1];
        fstate[1] = fstate[0];
        fstate[0] = s;
        let mut acc = 0.0f32;
        for k in 0..3 {
            acc += fstate[k] * fcoeffs[k];
        }
        fwant[i] = acc;
    }
    let mut fst = [0.0f32; 3];
    let mut fgot = [0.0f32; 8];
    let mut finst = FirInstanceF32::init(3, &fcoeffs, &mut fst);
    fir_f32(&mut finst, &fsrc, &mut fgot);
    assert_eq!(fgot, fwant, "f32 FIR diverged from the reference");
}

#[test]
fn lms_and_nlms_adapt() {
    let src = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let reference = [0.1f32, 0.2, 0.3, 0.4, 0.5];
    let mut coeffs = [0.0f32; 2];
    let mut state = [0.0f32; 2];
    let mut out = [0.0f32; 5];
    let mut err = [0.0f32; 5];
    let mut lms = LmsInstanceF32::init(2, &mut coeffs, &mut state, 0.01);
    lms_f32(&mut lms, &src, &reference, &mut out, &mut err);

    let mut coeffs2 = [0.0f32; 2];
    let mut state2 = [0.0f32; 2];
    let mut lms2 = LmsInstanceF32::init(2, &mut coeffs2, &mut state2, 0.01);
    lms_leaky_f32(&mut lms2, &src, &reference, &mut out, &mut err, 0.001);

    let mut coeffs3 = [0.0f32; 2];
    let mut state3 = [0.0f32; 2];
    let mut nlms = NlmsInstanceF32::init(2, &mut coeffs3, &mut state3, 0.01, 1e-5);
    nlms_f32(&mut nlms, &src, &reference, &mut out, &mut err);
}

#[test]
fn convolution_correlation_and_misc_filters() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0];
    let mut conv_out = [0.0f32; 4];
    conv_f32(&a, &b, &mut conv_out);

    let qa = [
        q31::from_bits(100),
        q31::from_bits(200),
        q31::from_bits(300),
    ];
    let qb = [q31::from_bits(4), q31::from_bits(5)];
    let mut qconv = [q31::ZERO; 4];
    conv_q31(&qa, &qb, &mut qconv);

    let q15a = [
        q15::from_bits(100),
        q15::from_bits(200),
        q15::from_bits(300),
    ];
    let q15b = [q15::from_bits(4), q15::from_bits(5)];
    let mut q15conv = [q15::ZERO; 4];
    conv_q15(&q15a, &q15b, &mut q15conv);

    let q7a = [q7::from_bits(10), q7::from_bits(20), q7::from_bits(30)];
    let q7b = [q7::from_bits(4), q7::from_bits(5)];
    let mut q7conv = [q7::ZERO; 4];
    conv_q7(&q7a, &q7b, &mut q7conv);

    let mut corr = [0.0f32; 4];
    correlate_f32(&a, &b, &mut corr);
    let mut qcorr = [q31::ZERO; 4];
    correlate_q31(&qa, &qb, &mut qcorr);
    let mut q15corr = [q15::ZERO; 4];
    correlate_q15(&q15a, &q15b, &mut q15corr);

    let mut sp = SinglePoleFilter::<f32>::lowpass(0.5);
    sp.process(1.0);
    sp.reset();
    let mut sp2 = SinglePoleFilter::<f32>::highpass(0.5);
    sp2.process(1.0);

    let mut spq = SinglePoleFilter::<q15>::lowpass(q15::from_bits(1 << 14));
    spq.process(q15::from_bits(100));
    spq.reset();
    let mut spq2 = SinglePoleFilter::<q15>::highpass_from_f32(0.5);
    spq2.process(q15::from_bits(100));
    let mut dc = DcBlockerQ15::new(q15::from_bits(1 << 14));
    dc.process(q15::from_bits(100));
    dc.reset();

    let mut avg = RecursiveMovingAverage::<f32, 4>::default();
    avg.process(1.0);
    avg.reset();
    let mut avgq = RecursiveMovingAverageQ15::<4>::default();
    avgq.process(q15::from_bits(100));
    avgq.reset();

    let mut cb = CircularBuffer::<f32, 4>::new(0.0);
    cb.push(1.0);
    let _ = cb.latest();
    let _ = cb.oldest();
    let _ = cb.get(0);
    cb.clear(0.0);
}
