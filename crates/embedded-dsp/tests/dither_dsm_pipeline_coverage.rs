//! Coverage for the dithering, delta-sigma, and pipeline/streaming APIs.
//!
//! `dither` and `dsm` were previously reachable only from their doc examples
//! (which are not instrumented), and the `pipeline` trait surface was largely
//! unexercised. These tests drive the public API directly.

use embedded_dsp::audio::{
    PeakEnvelopeFollower, PeakEnvelopeFollowerQ15, RmsEnvelopeFollower, RmsEnvelopeFollowerQ15,
};
use embedded_dsp::beamforming::DelayAndSumBeamformer;
use embedded_dsp::dither::{Triangular, Uniform, XorShift32};
use embedded_dsp::dsm::Dsm;
use embedded_dsp::pipeline::{
    Chain, DspNode, Gain, Identity, Inplace, Lanes, Limiter, Offset, Pair, Process, Split,
    SplitInplace, SplitProcess,
};
use embedded_dsp::resampling::{CicDecimator, CicFilter, GardnerSymbolSync, HbfDec, HbfInt};
use embedded_dsp::types::q15;

// ─────────────────────────────────────────────────────────────────────────────
// dither
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn xorshift32_seed_zero_is_remapped_to_one() {
    // `new` maps a zero seed to 1 so the generator never latches at zero.
    assert_eq!(XorShift32::new(0), XorShift32::new(1));
    assert_eq!(XorShift32::default(), XorShift32::new(1));
    assert_ne!(XorShift32::new(2), XorShift32::new(3));
}

#[test]
fn xorshift32_is_deterministic_and_never_zero() {
    let mut a = XorShift32::new(0xDEAD_BEEF);
    let mut b = XorShift32::new(0xDEAD_BEEF);
    for i in 0..10_000 {
        let x = a.next_u32();
        assert_eq!(x, b.next_u32(), "diverged at step {i}");
        assert_ne!(x, 0, "xorshift32 must never produce 0");
    }
}

#[test]
fn xorshift32_iterator_matches_next_u32() {
    let mut iter = XorShift32::new(7);
    let mut direct = XorShift32::new(7);
    for _ in 0..16 {
        assert_eq!(iter.next(), Some(direct.next_u32()));
    }
}

#[test]
fn uniform_byte_stream_is_deterministic_and_recycles_cache() {
    let mut a = Uniform::new(42);
    let mut b = Uniform::new(42);
    // 32 draws exercise both the `idx == 0` refill and the byte-shift path
    // (each PRNG word yields four bytes).
    for i in 0..32 {
        assert_eq!(a.sample(), b.sample(), "diverged at byte {i}");
    }
}

#[test]
fn uniform_default_and_iterator_agree() {
    let mut default = Uniform::default();
    let mut seeded = Uniform::new(1);
    assert_eq!(default.sample(), seeded.sample());

    let mut iter = Uniform::new(9);
    let mut direct = Uniform::new(9);
    for _ in 0..8 {
        assert_eq!(iter.next(), Some(direct.sample()));
    }
}

#[test]
fn triangular_samples_stay_within_documented_range() {
    // Documented as [-(1 << 8), (1 << 8) - 1]: the difference of two i8-cast
    // uniform draws is bounded by -255..=255.
    let mut t = Triangular::new(0x1234_5678);
    for _ in 0..10_000 {
        let s = t.sample() as i32;
        assert!((-256..256).contains(&s), "out of range: {s}");
    }
}

#[test]
fn triangular_is_deterministic_and_iterable() {
    let mut a = Triangular::new(1234);
    let mut b = Triangular::new(1234);
    for i in 0..32 {
        assert_eq!(a.sample(), b.sample(), "diverged at sample {i}");
    }

    let mut iter = Triangular::default();
    let mut _seen: i16 = 0;
    for _ in 0..8 {
        _seen = iter.next().expect("triangular iterator is infinite");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// dsm
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dsm_order_zero_always_outputs_zero() {
    let mut d = Dsm::<0>::default();
    for x in [0u32, 1, 0x8000_0000, u32::MAX] {
        assert_eq!(d.process(x), 0, "K = 0 must always output 0");
    }
}

#[test]
fn dsm_first_order_outputs_a_single_bit() {
    let mut d = Dsm::<1>::default();
    for x in [0u32, 1, 7, 0x4000_0000, u32::MAX] {
        let y = d.process(x);
        assert!(y == 0 || y == 1, "K = 1 must emit one bit, got {y}");
    }
}

#[test]
fn dsm_mean_tracks_normalised_input() {
    let mut d = Dsm::<3>::default();
    let x = 0x8765_4321u32;
    let n: u32 = 1 << 20;
    let sum: f64 = (0..n).map(|_| d.process(x) as f64).sum();
    let mean = sum / f64::from(n);
    let expected = x as f64 / (1u64 << 32) as f64;
    assert!(
        (mean - expected).abs() < 0.01,
        "mean {mean} drifted from expected {expected}"
    );
}

#[test]
fn dsm_higher_orders_are_bounded_and_toggle() {
    let mut d = Dsm::<5>::default();
    let mut out = [0i8; 512];
    for slot in out.iter_mut() {
        *slot = d.process(0x2000_0000);
    }
    // Documented range for order K: 1 - (1 << (K - 1)) ..= (1 << (K - 1)).
    let lo = 1 - (1i8 << 4);
    let hi = 1i8 << 4;
    for &y in &out {
        assert!((lo..=hi).contains(&y), "order-5 output {y} out of range");
    }
    assert!(
        out.iter().any(|&v| v != out[0]),
        "modulator should toggle rather than sit constant"
    );
}

#[test]
fn dsm_reset_restores_a_fresh_modulator() {
    let mut used = Dsm::<4>::default();
    for _ in 0..1_000 {
        used.process(0x1234_5678);
    }
    let mut fresh = Dsm::<4>::default();

    used.reset();
    assert_eq!(used, fresh, "reset must clear accumulators");

    for i in 0..1_000 {
        assert_eq!(
            used.process(0x1234_5678),
            fresh.process(0x1234_5678),
            "diverged at step {i} after reset"
        );
    }
}

#[test]
fn dsm_composes_through_split_process() {
    // `Dsm` changes type (`u32 -> i8`), so it reaches `Process`/`SplitProcess` but not `DspNode`
    // (which requires `X = Y`). Verify the trait path agrees bit-for-bit with the inherent method.
    let mut via_trait = Dsm::<3>::default();
    let mut via_inherent = Dsm::<3>::default();
    for x in [0u32, 1, 0x8765_4321, 0x2000_0000, u32::MAX] {
        assert_eq!(
            via_trait.process_with_state(&mut (), x),
            via_inherent.process(x),
            "SplitProcess path diverged from Dsm::process for x={x:#x}"
        );
    }

    let src = [0x1234_5678u32; 64];
    let mut dst_trait = [0i8; 64];
    let mut dst_inherent = [0i8; 64];
    via_trait.block_with_state(&mut (), &src, &mut dst_trait);
    for (slot, &x) in dst_inherent.iter_mut().zip(src.iter()) {
        *slot = via_inherent.process(x);
    }
    assert_eq!(dst_trait, dst_inherent);
}

// ─────────────────────────────────────────────────────────────────────────────
// beamforming
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn beamformer_composes_through_split_process() {
    // `[f32; MICS] -> f32` collapses channels, a genuine type change, so this reaches
    // `Process`/`SplitProcess` but not `DspNode` (which requires matching input/output types).
    let mut via_trait: DelayAndSumBeamformer<2, 32> = DelayAndSumBeamformer::new();
    let mut via_inherent: DelayAndSumBeamformer<2, 32> = DelayAndSumBeamformer::new();
    via_trait.set_delays(&[0.0, 2.0]);
    via_inherent.set_delays(&[0.0, 2.0]);

    for t in 0..8 {
        let ch0 = if t == 2 { 1.0 } else { 0.0 };
        let ch1 = if t == 0 { 1.0 } else { 0.0 };
        let input = [ch0, ch1];
        assert_eq!(
            via_trait.process_with_state(&mut (), input),
            via_inherent.process_sample(&input)
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// audio: envelope followers
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn envelope_followers_reach_dsp_node_through_the_pipeline_blankets() {
    // f32/q15 -> same type, no external state: each stateless `SplitProcess` bridge should reach
    // `Process` and `DspNode` through the blankets without any bespoke wiring.
    let mut peak_trait = PeakEnvelopeFollower::new(10.0, 100.0);
    let mut peak_inherent = PeakEnvelopeFollower::new(10.0, 100.0);
    let mut rms_trait = RmsEnvelopeFollower::new(50.0);
    let mut rms_inherent = RmsEnvelopeFollower::new(50.0);
    let mut peak_q15_trait = PeakEnvelopeFollowerQ15::new(10.0, 100.0);
    let mut peak_q15_inherent = PeakEnvelopeFollowerQ15::new(10.0, 100.0);
    let mut rms_q15_trait = RmsEnvelopeFollowerQ15::new(50.0);
    let mut rms_q15_inherent = RmsEnvelopeFollowerQ15::new(50.0);

    for n in 0..256 {
        let x = (n as f32 * 0.05).sin();
        let xq = q15::saturating_from_num(x);
        assert_eq!(
            DspNode::process_sample(&mut peak_trait, x),
            peak_inherent.process(x)
        );
        assert_eq!(
            DspNode::process_sample(&mut rms_trait, x),
            rms_inherent.process(x)
        );
        assert_eq!(
            DspNode::process_sample(&mut peak_q15_trait, xq),
            peak_q15_inherent.process(xq)
        );
        assert_eq!(
            DspNode::process_sample(&mut rms_q15_trait, xq),
            rms_q15_inherent.process(xq)
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// resampling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn cic_decimator_composes_through_split_process() {
    // `i32 -> Option<i32>` (an output only every R samples) reaches `Process`/`SplitProcess` but
    // not `DspNode`, which requires matching input/output types.
    let mut via_trait: CicDecimator<3> = CicDecimator::new(4);
    let mut via_inherent: CicDecimator<3> = CicDecimator::new(4);
    for i in 0..32 {
        assert_eq!(
            via_trait.process_with_state(&mut (), i),
            via_inherent.process_sample(i)
        );
    }
}

#[test]
fn cic_filter_interpolate_and_decimate_compose_through_split_process() {
    // `process_interpolate`: `Option<i32> -> i32`. `process_decimate`: `i32 -> Option<i32>`.
    // Different trait instantiations (different `X`), so both bridges coexist on `CicFilter`.
    let mut interp_trait: CicFilter<i32, 3, 1> = CicFilter::new(3);
    let mut interp_inherent: CicFilter<i32, 3, 1> = CicFilter::new(3);
    for i in 0..16 {
        let x = if i % 4 == 0 { Some(i * 100) } else { None };
        assert_eq!(
            interp_trait.process_with_state(&mut (), x),
            interp_inherent.process_interpolate(x)
        );
    }

    let mut dec_trait: CicFilter<i32, 3, 1> = CicFilter::new(3);
    let mut dec_inherent: CicFilter<i32, 3, 1> = CicFilter::new(3);
    for i in 0..16 {
        assert_eq!(
            dec_trait.process_with_state(&mut (), i * 10),
            dec_inherent.process_decimate(i * 10)
        );
    }
}

#[test]
fn gardner_symbol_sync_composes_through_split_process() {
    // `f32 -> Option<f32>` (a decision only on a late strobe) reaches `Process`/`SplitProcess`
    // but not `DspNode`, which requires matching input/output types.
    let mut via_trait = GardnerSymbolSync::new(4.0, 1e-2).unwrap();
    let mut via_inherent = GardnerSymbolSync::new(4.0, 1e-2).unwrap();
    for n in 0..128 {
        let x = (n as f32 * 0.1).sin();
        assert_eq!(
            via_trait.process_with_state(&mut (), x),
            via_inherent.push(x)
        );
    }
}

#[test]
fn hbf_dec_process_pair_matches_the_block_method_and_composes() {
    // `process()` is now built on top of `process_pair`; verify the block path still matches a
    // manual per-pair reconstruction, and that the `SplitProcess` bridge agrees with both.
    const M: usize = 3;
    let coeffs = [0.1f32, -0.05, 0.02];
    let src: [f32; 16] = core::array::from_fn(|i| (i as f32 * 0.3).sin());

    let mut via_block = HbfDec::<M>::new(coeffs);
    let mut dst = [0.0f32; 8];
    via_block.process(&src, &mut dst);

    let mut via_trait = HbfDec::<M>::new(coeffs);
    for (i, out) in dst.iter().enumerate() {
        let pair = [src[2 * i], src[2 * i + 1]];
        assert_eq!(via_trait.process_with_state(&mut (), pair), *out);
    }
}

#[test]
fn hbf_int_process_sample_matches_the_block_method_and_composes() {
    const M: usize = 3;
    let coeffs = [0.1f32, -0.05, 0.02];
    let src: [f32; 8] = core::array::from_fn(|i| (i as f32 * 0.3).sin());

    let mut via_block = HbfInt::<M>::new(coeffs);
    let mut dst = [0.0f32; 16];
    via_block.process(&src, &mut dst);

    let mut via_trait = HbfInt::<M>::new(coeffs);
    for (i, &x) in src.iter().enumerate() {
        let pair = via_trait.process_with_state(&mut (), x);
        assert_eq!(pair, [dst[2 * i], dst[2 * i + 1]]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// pipeline: helper nodes
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal `DspNode` used to exercise the trait's provided `process_block`,
/// `process_in_place`, and `then` methods.
struct Doubler;

impl SplitProcess<i32, i32, ()> for Doubler {
    fn process_with_state(&mut self, _state: &mut (), input: i32) -> i32 {
        input * 2
    }
}

/// Minimal `Process` + `Inplace` node.
struct Inc;

impl Process<i32, i32> for Inc {
    fn process(&mut self, x: i32) -> i32 {
        x + 1
    }
}

impl Inplace<i32> for Inc {}

/// Configuration whose state lives in an explicit `&mut i32`, used for the
/// `SplitProcess` / `SplitInplace` traits and the `Split` adapter.
struct RunningScale;

impl SplitProcess<i32, i32, i32> for RunningScale {
    fn process_with_state(&mut self, state: &mut i32, x: i32) -> i32 {
        *state += 1;
        x * *state
    }
}

impl SplitInplace<i32, i32> for RunningScale {}

/// Step one sample through a split configuration.
fn split_step<C, X: Copy, Y, S>(config: &mut C, state: &mut S, x: X) -> Y
where
    C: SplitProcess<X, Y, S>,
{
    config.process_with_state(state, x)
}

/// In-place step through a split configuration.
fn split_inplace<C, X: Copy, S>(config: &mut C, state: &mut S, xy: &mut [X])
where
    C: SplitInplace<X, S>,
{
    config.inplace_with_state(state, xy)
}

// ─────────────────────────────────────────────────────────────────────────────
// pipeline: DspNode provided methods and combinators
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dsp_node_provided_block_and_inplace_methods() {
    let mut node = Doubler;
    let input = [1i32, 2, 3, 4];

    let mut out = [0i32; 4];
    node.process_block(&input, &mut out);
    assert_eq!(out, [2, 4, 6, 8]);

    // A shorter output buffer exercises the `.min()` clamp.
    let mut short = [0i32; 2];
    node.process_block(&input, &mut short);
    assert_eq!(short, [2, 4]);

    let mut buf = [1i32, 2, 3];
    node.process_in_place(&mut buf);
    assert_eq!(buf, [2, 4, 6]);
}

#[test]
fn dsp_node_then_builds_a_chain() {
    let mut chain = Doubler.then(Doubler);
    assert_eq!(chain.process_sample(3), 12);
}

#[test]
fn process_block_and_inplace_apply_sample_wise() {
    let mut node = Inc;

    let mut out = [0i32; 3];
    node.block(&[1, 2, 3], &mut out);
    assert_eq!(out, [2, 3, 4]);

    let mut buf = [1i32, 2];
    node.inplace(&mut buf);
    assert_eq!(buf, [2, 3]);
}

#[test]
fn chain_composes_process_and_inplace() {
    let mut via_process = Chain {
        first: Inc,
        second: Inc,
    };
    assert_eq!(via_process.process(1), 3);

    let mut via_inplace = Chain {
        first: Inc,
        second: Inc,
    };
    let mut buf = [1i32, 2, 3];
    via_inplace.inplace(&mut buf);
    assert_eq!(buf, [3, 4, 5]);
}

// ─────────────────────────────────────────────────────────────────────────────
// pipeline: split config/state
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn split_process_block_and_inplace() {
    let mut config = RunningScale;
    let mut state = 0i32;
    let mut out = [0i32; 3];
    config.block_with_state(&mut state, &[1, 2, 3], &mut out);
    assert_eq!(out, [1, 4, 9]);

    let mut state = 0i32;
    let mut xy = [1i32, 1];
    config.inplace_with_state(&mut state, &mut xy);
    assert_eq!(xy, [1, 2]);
}

#[test]
fn split_adapter_wraps_config_and_state() {
    let mut split = Split::new(RunningScale, 0i32);
    assert_eq!(split.process(5), 5);

    let mut out = [0i32; 2];
    split.block(&[1, 1], &mut out);
    assert_eq!(out, [2, 3]);

    let mut xy = [1i32, 1];
    split.inplace(&mut xy);
    assert_eq!(xy, [4, 5]);

    // `Split` also acts as a `DspNode`.
    let mut node = Split::new(RunningScale, 0i32);
    assert_eq!(node.process_sample(2), 2);
    let mut block = [0i32; 2];
    node.process_block(&[1, 1], &mut block);
    assert_eq!(block, [2, 3]);
}

#[test]
fn lanes_share_one_config_across_states() {
    let mut lanes = Lanes::new(RunningScale);
    let mut state = [0i32; 3];
    let out = lanes.process_with_state(&mut state, [1, 2, 3]);
    assert_eq!(out, [1, 2, 3]);
    assert_eq!(state, [1, 1, 1]);

    // `into_inner` hands the shared configuration back.
    let mut inner = Lanes::new(RunningScale).into_inner();
    let mut state = 0i32;
    assert_eq!(inner.process_with_state(&mut state, 4), 4);
}

#[test]
fn pair_runs_two_independent_branches() {
    let mut pair = Pair::new(RunningScale, RunningScale);
    let mut state = (0i32, 0i32);
    let out = pair.process_with_state(&mut state, [2, 3]);
    assert_eq!(out, [2, 3]);
    assert_eq!(state, (1, 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// pipeline: leaf nodes
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn offset_adds_and_supports_split_traits() {
    // `Offset` is bounded on `DspSample` (f32/f64/q15/q31), not on integers.
    let mut offset = Offset(0.5f32);
    assert_eq!(Process::<f32>::process(&mut offset, 1.0), 1.5);

    let mut buf = [1.0f32, 2.0];
    Inplace::<f32>::inplace(&mut offset, &mut buf);
    assert_eq!(buf, [1.5, 2.5]);

    // Split variants ignore the (unit) state.
    let mut config = Offset(0.25f32);
    let mut state = ();
    assert_eq!(
        split_step::<_, f32, f32, ()>(&mut config, &mut state, 0.5),
        0.75
    );

    let mut xy = [1.0f32, 2.0];
    split_inplace::<_, f32, ()>(&mut config, &mut state, &mut xy);
    assert_eq!(xy, [1.25, 2.25]);
}

#[test]
fn identity_passes_samples_through_unchanged() {
    let mut id = Identity;
    assert_eq!(Process::<i32>::process(&mut id, 5), 5);

    let mut buf = [1i32, 2, 3];
    Inplace::<i32>::inplace(&mut id, &mut buf);
    assert_eq!(buf, [1, 2, 3]);

    let mut state = ();
    assert_eq!(split_step::<_, i32, i32, ()>(&mut id, &mut state, 9), 9);

    let mut xy = [1i32, 2];
    split_inplace::<_, i32, ()>(&mut id, &mut state, &mut xy);
    assert_eq!(xy, [1, 2]);
}

#[test]
fn gain_scales_dsp_sample_types() {
    let mut gain = Gain::new(2.0f32);
    assert_eq!(gain.process_sample(3.0), 6.0);
    assert_eq!(Process::<f32>::process(&mut gain, 3.0), 6.0);

    let mut buf = [1.0f32, 2.0];
    Inplace::<f32>::inplace(&mut gain, &mut buf);
    assert_eq!(buf, [2.0, 4.0]);
}

#[test]
fn gain_q15_and_q31_specialisations() {
    // i16 uses the Q15 path: 16384/32768 == 0.5, so 1000 * 0.5 == 500.
    let mut q15_gain = Gain::new(16_384i16);
    assert_eq!(q15_gain.process_sample(1_000i16), 500);

    // i32 uses the Q31 path: 1 << 30 is 0.5 in Q31.
    let mut q31_gain = Gain::new(1i32 << 30);
    assert_eq!(q31_gain.process_sample(1_000i32), 500);

    // i32::MIN * i32::MIN overflows i32 after the Q31 shift and must clamp.
    let mut saturating = Gain::new(i32::MIN);
    assert_eq!(saturating.process_sample(i32::MIN), i32::MAX);
}

#[test]
fn limiter_clamps_both_rails() {
    let mut limiter = Limiter::new(-1.0f32, 1.0f32);
    assert_eq!(limiter.process_sample(-2.0), -1.0);
    assert_eq!(limiter.process_sample(2.0), 1.0);
    assert_eq!(limiter.process_sample(0.5), 0.5);

    assert_eq!(Process::<f32>::process(&mut limiter, 3.0), 1.0);

    let mut buf = [-5.0f32, 0.0, 5.0];
    limiter.inplace(&mut buf);
    assert_eq!(buf, [-1.0, 0.0, 1.0]);
}

#[test]
fn built_in_nodes_delegate_to_their_inherent_process() {
    use embedded_dsp::controller::{PidInstanceF32, PidInstanceQ15, PidInstanceQ31};
    use embedded_dsp::filtering::{DcBlockerQ15, SinglePoleFilter};
    use embedded_dsp::types::{q15, q31};

    let mut via_node = PidInstanceF32::new(1.0, 0.1, 0.01);
    let mut direct = PidInstanceF32::new(1.0, 0.1, 0.01);
    assert_eq!(
        DspNode::process_sample(&mut via_node, 0.5),
        direct.process(0.5)
    );

    let mut via_node =
        PidInstanceQ15::new(q15::from_bits(100), q15::from_bits(10), q15::from_bits(1));
    let mut direct =
        PidInstanceQ15::new(q15::from_bits(100), q15::from_bits(10), q15::from_bits(1));
    assert_eq!(
        DspNode::process_sample(&mut via_node, q15::from_bits(1_000)),
        direct.process(q15::from_bits(1_000))
    );

    let mut via_node =
        PidInstanceQ31::new(q31::from_bits(100), q31::from_bits(10), q31::from_bits(1));
    let mut direct =
        PidInstanceQ31::new(q31::from_bits(100), q31::from_bits(10), q31::from_bits(1));
    assert_eq!(
        DspNode::process_sample(&mut via_node, q31::from_bits(1_000)),
        direct.process(q31::from_bits(1_000))
    );

    let mut via_node = SinglePoleFilter::<f32>::lowpass(0.9);
    let mut direct = SinglePoleFilter::<f32>::lowpass(0.9);
    assert_eq!(
        DspNode::process_sample(&mut via_node, 1.0),
        direct.process(1.0)
    );

    let mut via_node = SinglePoleFilter::<q15>::lowpass(q15::from_bits(16_384));
    let mut direct = SinglePoleFilter::<q15>::lowpass(q15::from_bits(16_384));
    assert_eq!(
        DspNode::process_sample(&mut via_node, q15::from_bits(1_000)),
        direct.process(q15::from_bits(1_000))
    );

    let mut via_node = DcBlockerQ15::new(q15::from_bits(32_000));
    let mut direct = DcBlockerQ15::new(q15::from_bits(32_000));
    assert_eq!(
        DspNode::process_sample(&mut via_node, q15::from_bits(1_000)),
        direct.process(q15::from_bits(1_000))
    );
}
