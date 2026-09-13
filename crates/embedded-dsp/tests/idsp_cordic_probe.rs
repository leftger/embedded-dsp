//! Pins the claims the README makes about `idsp`'s CORDIC modes.
//!
//! The README compares this crate against `idsp` and has to say something about the modes upstream
//! offers and this crate does not. That comparison was originally written from a hand transcription
//! of upstream's algorithm, and three of its claims turned out to be wrong — `div` was said to
//! compute `z − y/x` when it computes `z + y/x` perfectly well, and `mul`'s band was described
//! backwards. `idsp` is a dev-dependency, so the comparison can be *measured* instead, and these
//! tests are what keep it honest.
//!
//! The assertions split in two on purpose:
//!
//! * Claims that stay true if upstream fixes its bug — `div` is correct, `mul` is correct on
//!   `|z| ≤ 0.5` — are asserted directly.
//! * Claims that only hold *because* of the bug (the fold outside the band, the panic on
//!   `i32::MIN`) are asserted too, but with messages saying to update the README if they fail. A
//!   dev-dependency bump that fixes upstream should break these loudly rather than leave the
//!   README quietly wrong, which is exactly what happened the first time.

const Q31: f64 = 2147483648.0;
/// CORDIC carries a few Q1.31 LSBs of error (~2e-9 observed, including the `f64` conversion at each
/// end), while the failure modes these tests detect are off by ~1e-1 — seven orders of margin, so
/// the tolerance is not what is doing the work.
const TOL: f64 = 1e-8;
/// Reciprocal of the hyperbolic gain, matching upstream's own test module.
const G: f64 = 1.207_497_067_763_072;

fn f2i(x: f64) -> i32 {
    (x * Q31).round() as i64 as i32
}

fn i2f(x: i32) -> f64 {
    x as f64 / Q31
}

#[test]
fn idsp_div_is_correct_over_its_documented_domain() {
    // `z + y/x`, over the quotient range, both signs and a non-zero accumulator.
    for &(x, y, z) in &[
        (0.5, 0.05, 0.0),
        (0.5, 0.1, 0.0),
        (0.5, 0.2, 0.0),
        (0.5, 0.25, 0.0),
        (0.5, 0.3, 0.0),
        (0.5, 0.4, 0.0),
        (0.5, 0.45, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, -0.25, 0.0),
        (0.25, 0.125, 0.0),
        (0.1, 0.05, 0.0),
        (0.8, 0.2, 0.0),
        (0.9, 0.45, 0.0),
        (0.5, 0.25, 0.1),
        (0.5, 0.25, -0.2),
        (0.5, 0.25, 0.4),
    ] {
        let got = i2f(idsp::div(f2i(x), f2i(y), f2i(z)));
        let want = z + y / x;
        assert!(
            (got - want).abs() <= TOL,
            "idsp::div({x}, {y}, {z}) = {got}, documented z + y/x = {want}"
        );
    }
}

#[test]
fn idsp_mul_is_correct_within_half_its_documented_range() {
    // Inside the band it is exact, and this stays true if upstream widens it.
    for &z in &[-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
        let got = i2f(idsp::mul(f2i(0.5), 0, f2i(z)));
        let want = 0.5 * z;
        assert!(
            (got - want).abs() <= TOL,
            "idsp::mul(0.5, 0, {z}) = {got}, documented y + x*z = {want}"
        );
    }

    // Outside it, it folds instead of extrapolating — upstream's signing bug.
    for &(z, folded) in &[(0.6, 0.2), (0.9, 0.05), (-0.6, -0.2), (-0.9, -0.05)] {
        let got = i2f(idsp::mul(f2i(0.5), 0, f2i(z)));
        assert!(
            (got - folded).abs() <= TOL,
            "idsp::mul(0.5, 0, {z}) no longer folds to {folded} (got {got}) — if upstream fixed its \
             signing bug, update the README note about the ±0.5 band"
        );
    }
}

#[test]
fn idsp_cosh_sinh_holds_only_for_small_angles() {
    // Correct on the small-angle band.
    for &z in &[0.0, 0.1, 0.2, 0.3] {
        let (a, _) = idsp::cosh_sinh(f2i(0.5 * G), 0, f2i(z));
        let got = i2f(a);
        let want = 0.5 * z.cosh();
        assert!(
            (got - want).abs() <= TOL,
            "idsp::cosh_sinh(x*G, 0, {z}) = {got}, wanted x*cosh(z) = {want}"
        );
    }

    // Past it the sign flips rather than the value drifting.
    let (a, _) = idsp::cosh_sinh(f2i(0.5 * G), 0, f2i(0.5));
    let got = i2f(a);
    assert!(
        (got + 0.5 * 0.5f64.cosh()).abs() <= TOL,
        "idsp::cosh_sinh no longer sign-flips at z = 0.5 (got {got}) — if upstream fixed this, \
         update the README note"
    );
}

#[test]
fn idsp_vectoring_modes_panic_on_minus_one() {
    // `i32::MIN` is exactly −1.0 in Q1.31: an ordinary value, not an edge case a caller would
    // avoid. Upstream negates `x` in vectoring mode without accounting for it.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let div = std::panic::catch_unwind(|| idsp::div(i32::MIN, 0, 0));
    std::panic::set_hook(previous);

    assert!(
        div.is_err(),
        "idsp::div(i32::MIN, 0, 0) no longer panics — if upstream fixed the negation overflow, \
         update the README note about vectoring-mode inputs"
    );
}
