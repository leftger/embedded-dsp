//! Coverage for filtering primitives: FIR, biquad cascades, LMS, convolution,
//! correlation, one-pole filters, and circular buffers.

use embedded_dsp::filtering::{
    BiquadCascadeDf2tInstanceF32, BiquadCascadeDf2tInstanceQ15, BiquadCascadeDf2tInstanceQ31,
    BiquadCascadeInstanceF32, BiquadCascadeInstanceQ15, BiquadCascadeInstanceQ31, CircularBuffer,
    DcBlockerQ15, FirInstanceF32, FirInstanceQ15, FirInstanceQ31, LmsInstanceF32, NlmsInstanceF32,
    RecursiveMovingAverage, RecursiveMovingAverageQ15, SinglePoleFilter, SinglePoleFilterQ15,
    biquad_cascade_df1_f32, biquad_cascade_df1_q15, biquad_cascade_df1_q31,
    biquad_cascade_df2t_f32, biquad_cascade_df2t_q15, biquad_cascade_df2t_q31, conv_f32, conv_q7,
    conv_q15, conv_q31, correlate_f32, correlate_q15, correlate_q31, fir_f32, fir_q15, fir_q31,
    lms_f32, lms_leaky_f32, nlms_f32,
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
    let mut qinst = BiquadCascadeInstanceQ31::init(1, &qcoeffs, &mut qstate, 0);
    biquad_cascade_df1_q31(&mut qinst, &qsrc, &mut qdst);

    let mut qstate2 = [q31::ZERO; 2];
    let mut qdst2 = [q31::ZERO; 4];
    let mut qinst2 = BiquadCascadeDf2tInstanceQ31::init(1, &qcoeffs, &mut qstate2, 0);
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
    let mut q15inst = BiquadCascadeInstanceQ15::init(1, &q15coeffs, &mut q15state, 0);
    biquad_cascade_df1_q15(&mut q15inst, &q15src, &mut q15dst);

    let mut q15state2 = [q15::ZERO; 2];
    let mut q15dst2 = [q15::ZERO; 4];
    let mut q15inst2 = BiquadCascadeDf2tInstanceQ15::init(1, &q15coeffs, &mut q15state2, 0);
    biquad_cascade_df2t_q15(&mut q15inst2, &q15src, &mut q15dst2);
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

    let mut sp = SinglePoleFilter::lowpass(0.5);
    sp.process(1.0);
    sp.reset();
    let mut sp2 = SinglePoleFilter::highpass(0.5);
    sp2.process(1.0);

    let mut spq = SinglePoleFilterQ15::lowpass(q15::from_bits(1 << 14));
    spq.process(q15::from_bits(100));
    spq.reset();
    let mut spq2 = SinglePoleFilterQ15::highpass_from_f32(0.5);
    spq2.process(q15::from_bits(100));
    let mut dc = DcBlockerQ15::new(q15::from_bits(1 << 14));
    dc.process(q15::from_bits(100));
    dc.reset();

    let mut avg = RecursiveMovingAverage::<4>::default();
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
