//! Coverage-gap tests for `basic_math`: exercises every public elementwise
//! family so the scalar and fixed-point hot paths are actually measured.

use embedded_dsp::basic_math::{
    abs_f32, abs_f64, add_f32, add_f64, and_u8, and_u16, and_u32, clip_f32, dot_prod_f32,
    dot_prod_f64, mult_f32, mult_f64, negate_f32, negate_f64, not_u8, not_u16, not_u32, offset_f32,
    offset_f64, or_u8, or_u16, or_u32, scale_f32, scale_f64, shift_q7, shift_q15, shift_q31,
    sub_f32, sub_f64, xor_u8, xor_u16, xor_u32,
};

#[test]
fn float_elementwise_family() {
    let mut abs = [0.0; 4];
    abs_f32(&[-1.0, 2.0, -3.0, 4.0], &mut abs);
    assert_eq!(abs, [1.0, 2.0, 3.0, 4.0]);

    let mut abs64 = [0.0; 2];
    abs_f64(&[-1.5, 2.5], &mut abs64);
    assert_eq!(abs64, [1.5, 2.5]);

    let a = [1.0, 2.0, 3.0];
    let b = [10.0, 20.0, 30.0];
    let mut out = [0.0; 3];
    add_f32(&a, &b, &mut out);
    assert_eq!(out, [11.0, 22.0, 33.0]);
    sub_f32(&b, &a, &mut out);
    assert_eq!(out, [9.0, 18.0, 27.0]);
    mult_f32(&a, &b, &mut out);
    assert_eq!(out, [10.0, 40.0, 90.0]);
    negate_f32(&a, &mut out);
    assert_eq!(out, [-1.0, -2.0, -3.0]);
    offset_f32(&a, 100.0, &mut out);
    assert_eq!(out, [101.0, 102.0, 103.0]);
    scale_f32(&a, 2.0, &mut out);
    assert_eq!(out, [2.0, 4.0, 6.0]);
    clip_f32(&[-5.0, 0.5, 5.0], -1.0, 1.0, &mut out);
    assert_eq!(out, [-1.0, 0.5, 1.0]);

    let a64 = [1.0, 2.0];
    let b64 = [3.0, 4.0];
    let mut out64 = [0.0; 2];
    add_f64(&a64, &b64, &mut out64);
    assert_eq!(out64, [4.0, 6.0]);
    sub_f64(&b64, &a64, &mut out64);
    assert_eq!(out64, [2.0, 2.0]);
    mult_f64(&a64, &b64, &mut out64);
    assert_eq!(out64, [3.0, 8.0]);
    negate_f64(&a64, &mut out64);
    assert_eq!(out64, [-1.0, -2.0]);
    offset_f64(&a64, 1.5, &mut out64);
    assert_eq!(out64, [2.5, 3.5]);
    scale_f64(&a64, 0.5, &mut out64);
    assert_eq!(out64, [0.5, 1.0]);

    assert_eq!(dot_prod_f32(&a, &b), 10.0 + 40.0 + 90.0);
    assert_eq!(dot_prod_f64(&a64, &b64), 3.0 + 8.0);
}

#[test]
fn logic_elementwise_family() {
    let a32 = [0b1010u32];
    let b32 = [0b1100u32];
    let mut out32 = [0u32; 1];
    and_u32(&a32, &b32, &mut out32);
    assert_eq!(out32[0], 0b1000);
    or_u32(&a32, &b32, &mut out32);
    assert_eq!(out32[0], 0b1110);
    xor_u32(&a32, &b32, &mut out32);
    assert_eq!(out32[0], 0b0110);
    not_u32(&a32, &mut out32);
    assert_eq!(out32[0], !0b1010);

    let a16 = [0b1010u16];
    let b16 = [0b1100u16];
    let mut out16 = [0u16; 1];
    and_u16(&a16, &b16, &mut out16);
    assert_eq!(out16[0], 0b1000);
    or_u16(&a16, &b16, &mut out16);
    assert_eq!(out16[0], 0b1110);
    xor_u16(&a16, &b16, &mut out16);
    assert_eq!(out16[0], 0b0110);
    not_u16(&a16, &mut out16);
    assert_eq!(out16[0], !0b1010);

    let a8 = [0b1010u8];
    let b8 = [0b1100u8];
    let mut out8 = [0u8; 1];
    and_u8(&a8, &b8, &mut out8);
    assert_eq!(out8[0], 0b1000);
    or_u8(&a8, &b8, &mut out8);
    assert_eq!(out8[0], 0b1110);
    xor_u8(&a8, &b8, &mut out8);
    assert_eq!(out8[0], 0b0110);
    not_u8(&a8, &mut out8);
    assert_eq!(out8[0], !0b1010);
}

#[test]
fn fixed_shift_family_smoke() {
    use embedded_dsp::types::{q7, q15, q31};

    let mut out = [q31::from_bits(0); 2];
    shift_q31(
        &[q31::from_bits(1 << 20), q31::from_bits(1 << 20)],
        2,
        &mut out,
    );
    assert_eq!(out, [q31::from_bits(1 << 22), q31::from_bits(1 << 22)]);

    let mut out16 = [q15::from_bits(0); 2];
    shift_q15(&[q15::from_bits(100), q15::from_bits(-100)], -1, &mut out16);
    assert_eq!(out16, [q15::from_bits(50), q15::from_bits(-50)]);

    let mut out8 = [q7::from_bits(0); 2];
    shift_q7(&[q7::from_bits(40), q7::from_bits(-40)], 1, &mut out8);
    assert_eq!(out8, [q7::from_bits(80), q7::from_bits(-80)]);
}
