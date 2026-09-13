//! Diagnostics: does square-root factoring actually beat the plain covariance form in f32?
//!
//! Compares three filters on identical models and measurements:
//!
//! * `KalmanFilter<N, M>` — plain form, `P⁺ = (I − KH)P`.
//! * `SquareRootKalmanFilter<N, M>` — stores a Cholesky factor and re-factorizes each step, so
//!   `P = S Sᵀ` is positive semi-definite by construction, but the arithmetic still runs on `P`
//!   (see the type's documentation).
//! * an `f64` reference, so "error" means deviation from the arithmetic an f32 filter is trying
//!   to emulate.
//!
//! Reported per run: the smallest diagonal of the covariance's Cholesky factor (a scale indicator
//! for the least-determined direction), whether that factor exists at all — i.e. whether `P` is
//! still positive definite — the asymmetry of `P`, the relative Frobenius distance from the `f64`
//! covariance, and the worst state error against the known trajectory.
//!
//! The second half sweeps the state dimension with a weakly observable model: an integrator chain
//! of order `N` observed only at position, which is where the plain form's `I − KH` cancellation
//! has the most room to compound.
//!
//! ```sh
//! cargo run -p embedded-dsp --example sr_spike --release
//! ```

use embedded_dsp::kalman::{KalmanFilter, SquareRootKalmanFilter};

// ─────────────────────────────────────────────────────────────────────────────
// f64 helpers (independent of the crate, so they can serve as a reference)
// ─────────────────────────────────────────────────────────────────────────────

type Mat<const N: usize> = [[f64; N]; N];

fn zeros<const N: usize>() -> Mat<N> {
    [[0.0; N]; N]
}

/// `v` on the diagonal.
fn diag<const N: usize>(v: f64) -> Mat<N> {
    let mut m = zeros::<N>();
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = v;
    }
    m
}

/// Integrator chain of order `N` for `dt = 1`: `F[i][j] = 1 / (j − i)!` for `j ≥ i`.
fn chain_f<const N: usize>() -> Mat<N> {
    let mut f = zeros::<N>();
    for (i, row) in f.iter_mut().enumerate() {
        let mut factorial = 1.0f64;
        for (j, cell) in row.iter_mut().enumerate().skip(i) {
            let d = j - i;
            if d > 0 {
                factorial *= d as f64;
            }
            *cell = 1.0 / factorial;
        }
    }
    f
}

fn mat_mul<const N: usize>(a: &Mat<N>, b: &Mat<N>) -> Mat<N> {
    let mut out = zeros::<N>();
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// `a · bᵀ`.
fn mul_bt<const N: usize>(a: &Mat<N>, b: &Mat<N>) -> Mat<N> {
    let mut out = zeros::<N>();
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += a[i][k] * b[j][k];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// Lower Cholesky factor with **no** floor: the flag is `false` when `a` is not positive definite,
/// which is the property we actually want to measure.
fn chol_lower<const N: usize>(a: &Mat<N>) -> (Mat<N>, bool) {
    let mut l = zeros::<N>();
    for i in 0..N {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                // A NaN pivot counts as "not positive definite" too.
                if sum <= 0.0 || sum.is_nan() {
                    return (l, false);
                }
                l[i][j] = sum.sqrt();
            } else {
                if l[j][j] == 0.0 {
                    return (l, false);
                }
                l[i][j] = sum / l[j][j];
            }
        }
    }
    (l, true)
}

fn min_diag<const N: usize>(m: &Mat<N>) -> f64 {
    (0..N).map(|i| m[i][i]).fold(f64::INFINITY, f64::min)
}

fn frob<const N: usize>(m: &Mat<N>) -> f64 {
    let mut s = 0.0;
    for row in m {
        for v in row {
            s += v * v;
        }
    }
    s.sqrt()
}

fn rel_frob<const N: usize>(a: &Mat<N>, b: &Mat<N>) -> f64 {
    let mut s = 0.0;
    for i in 0..N {
        for j in 0..N {
            let d = a[i][j] - b[i][j];
            s += d * d;
        }
    }
    s.sqrt() / frob(b).max(1e-300)
}

fn asym<const N: usize>(m: &Mat<N>) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..N {
        for j in (i + 1)..N {
            worst = worst.max((m[i][j] - m[j][i]).abs());
        }
    }
    worst
}

fn to_f32<const N: usize>(m: &Mat<N>) -> [[f32; N]; N] {
    let mut out = [[0.0f32; N]; N];
    for i in 0..N {
        for j in 0..N {
            out[i][j] = m[i][j] as f32;
        }
    }
    out
}

fn to_f64<const N: usize>(m: &[[f32; N]; N]) -> Mat<N> {
    let mut out = zeros::<N>();
    for i in 0..N {
        for j in 0..N {
            out[i][j] = m[i][j] as f64;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// The three filters
// ─────────────────────────────────────────────────────────────────────────────

struct Scenario<const N: usize> {
    name: &'static str,
    steps: usize,
    f: Mat<N>,
    q: Mat<N>,
    r: f64,
    p0: Mat<N>,
    /// Measure every `n`-th step; 1 means every step.
    measure_every: usize,
}

/// Position of the ground-truth trajectory: constant velocity from the origin.
fn truth_position(step: usize) -> f64 {
    step as f64
}

fn measurement_row<const N: usize>() -> [[f32; N]; 1] {
    let mut h = [[0.0f32; N]; 1];
    h[0][0] = 1.0;
    h
}

fn run_plain<const N: usize>(s: &Scenario<N>) -> (Mat<N>, [f64; N], bool) {
    let x0 = [0.0f32; N];
    let mut kf = KalmanFilter::<N, 1>::new(x0, to_f32(&s.p0), to_f32(&s.q), [[s.r as f32]]);
    let f = to_f32(&s.f);
    let h = measurement_row::<N>();

    for step in 1..=s.steps {
        kf.predict(&f);
        if step % s.measure_every == 0 {
            let z = [truth_position(step) as f32];
            if kf.update(&h, &z) != embedded_dsp::Status::Success {
                return (to_f64(&kf.p), kf.x.map(f64::from), false);
            }
        }
        if !kf.p[0][0].is_finite() || !kf.x[0].is_finite() {
            return (to_f64(&kf.p), kf.x.map(f64::from), false);
        }
    }
    (to_f64(&kf.p), kf.x.map(f64::from), true)
}

fn run_sr<const N: usize>(s: &Scenario<N>) -> (Mat<N>, [f64; N], bool) {
    let (s0, ok0) = chol_lower(&s.p0);
    let (s_q, ok_q) = chol_lower(&s.q);
    if !ok0 || !ok_q {
        return (s.p0, [0.0; N], false);
    }

    let mut sr = SquareRootKalmanFilter::<N, 1>::new(
        [0.0f32; N],
        to_f32(&s0),
        to_f32(&s.f),
        to_f32(&s_q),
        measurement_row::<N>(),
        [[s.r.max(0.0).sqrt() as f32]],
    );

    for step in 1..=s.steps {
        sr.predict();
        if step % s.measure_every == 0 {
            let z = [truth_position(step) as f32];
            if sr.update(&z) != embedded_dsp::Status::Success {
                return (to_f64(&sr.covariance()), sr.x.map(f64::from), false);
            }
        }
        if !sr.s[0][0].is_finite() {
            return (to_f64(&sr.covariance()), sr.x.map(f64::from), false);
        }
    }
    (to_f64(&sr.covariance()), sr.x.map(f64::from), true)
}

/// f64 reference using the same simple-form update as the plain filter.
fn run_reference<const N: usize>(s: &Scenario<N>) -> (Mat<N>, [f64; N]) {
    let mut x = [0.0f64; N];
    let mut p = s.p0;
    let mut h = [0.0f64; N];
    h[0] = 1.0;

    for step in 1..=s.steps {
        // x ← F x
        let mut next = [0.0f64; N];
        for (i, n) in next.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in 0..N {
                acc += s.f[i][k] * x[k];
            }
            *n = acc;
        }
        x = next;

        // P ← F P Fᵀ + Q
        let fp = mat_mul(&s.f, &p);
        p = mul_bt(&fp, &s.f);
        for i in 0..N {
            for j in 0..N {
                p[i][j] += s.q[i][j];
            }
        }

        if step % s.measure_every == 0 {
            let z = truth_position(step);

            // P Hᵀ (column vector), S = H P Hᵀ + R
            let mut ph = [0.0f64; N];
            for (i, v) in ph.iter_mut().enumerate() {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += p[i][k] * h[k];
                }
                *v = acc;
            }
            let mut var = s.r;
            for k in 0..N {
                var += h[k] * ph[k];
            }

            let innovation = z - x[0];
            let gain: [f64; N] = core::array::from_fn(|i| ph[i] / var);
            for i in 0..N {
                x[i] += gain[i] * innovation;
            }

            // P ← (I − K H) P
            let mut next_p = zeros::<N>();
            for i in 0..N {
                for j in 0..N {
                    let mut acc = 0.0;
                    for t in 0..N {
                        let mut ikh = -gain[i] * h[t];
                        if i == t {
                            ikh += 1.0;
                        }
                        acc += ikh * p[t][j];
                    }
                    next_p[i][j] = acc;
                }
            }
            p = next_p;
        }
    }
    (p, x)
}

fn report<const N: usize>(s: &Scenario<N>) {
    let (pref, xref) = run_reference(s);
    let (pref_chol, pref_pd) = chol_lower(&pref);
    let (pp, xp, okp) = run_plain(s);
    let (ps, xs, oks) = run_sr(s);
    let (pp_chol, pp_pd) = chol_lower(&pp);
    let (ps_chol, ps_pd) = chol_lower(&ps);

    let state_err = |x: &[f64; N]| {
        (0..N)
            .map(|i| (x[i] - xref[i]).abs())
            .fold(0.0f64, f64::max)
    };

    println!(
        "{:<26} N={:<3} every={:<3} steps={}",
        s.name, N, s.measure_every, s.steps
    );
    println!(
        "  {:<6} {:>11} {:>4} {:>10} {:>10} {:>11} {:>5}",
        "filter", "minDiag(L)", "PD", "asym", "relErrP", "stateErr", "ok"
    );
    println!(
        "  {:<6} {:>11.3e} {:>4} {:>10.3e} {:>10.3e} {:>11.3e} {:>5}",
        "plain",
        min_diag(&pp_chol),
        pp_pd,
        asym(&pp),
        rel_frob(&pp, &pref),
        state_err(&xp),
        okp
    );
    println!(
        "  {:<6} {:>11.3e} {:>4} {:>10.3e} {:>10.3e} {:>11.3e} {:>5}",
        "sqrt",
        min_diag(&ps_chol),
        ps_pd,
        asym(&ps),
        rel_frob(&ps, &pref),
        state_err(&xs),
        oks
    );
    println!(
        "  {:<6} {:>11.3e} {:>4} {:>10.3e} {:>10.3e} {:>11.3e} {:>5}",
        "f64 ref",
        min_diag(&pref_chol),
        pref_pd,
        0.0,
        0.0,
        0.0,
        true
    );
    println!();
}

/// An `N`-state integrator chain observed only at position, which leaves the higher states weakly
/// observable — the regime where `I − KH` cancellation has the most room to compound.
fn chain<const N: usize>(
    name: &'static str,
    steps: usize,
    q: f64,
    r: f64,
    every: usize,
) -> Scenario<N> {
    Scenario {
        name,
        steps,
        f: chain_f::<N>(),
        q: diag::<N>(q),
        r,
        p0: diag::<N>(1.0),
        measure_every: every,
    }
}

/// Two-state constant-velocity model, as used for the original sweep.
fn two_state(name: &'static str, steps: usize, q: f64, r: f64, p0: [[f64; 2]; 2]) -> Scenario<2> {
    Scenario {
        name,
        steps,
        f: [[1.0, 1.0], [0.0, 1.0]],
        q: diag::<2>(q),
        r,
        p0,
        measure_every: 1,
    }
}

fn main() {
    println!("=== original 2-state regimes ===\n");

    let eye: [[f64; 2]; 2] = [[1.0, 0.0], [0.0, 1.0]];
    report(&two_state("benign: R ~ P0", 20_000, 1e-6, 1e-2, eye));
    report(&two_state("tight R << P0", 20_000, 1e-12, 1e-8, eye));
    report(&two_state(
        "anisotropic P0 (1e12)",
        20_000,
        1e-10,
        1e-6,
        [[1e6, 0.0], [0.0, 1e-6]],
    ));
    report(&two_state("R = 0 (deterministic)", 2_000, 1e-9, 0.0, eye));
    report(&two_state(
        "long run, tiny Q and R",
        200_000,
        1e-14,
        1e-10,
        eye,
    ));
    report(&two_state("huge Q, tiny R", 20_000, 1.0, 1e-12, eye));
    report(&two_state(
        "long run, P above floor",
        200_000,
        1e-10,
        1e-8,
        eye,
    ));
    report(&two_state(
        "long run, P just above floor",
        200_000,
        1e-12,
        1e-10,
        eye,
    ));

    // Dense and mildly sparse, same step count throughout so the dimensions are comparable. The
    // N = 16 point at the end is the degenerate extreme: even the f64 reference loses positive
    // definiteness there, so it says more about the model than about f32.
    println!("=== state-dimension sweep: weakly observable integrator chain ===\n");

    const SWEEP_STEPS: usize = 30_000;

    println!("-- measuring every step --\n");
    report(&chain::<4>("chain N=4", SWEEP_STEPS, 1e-10, 1e-8, 1));
    report(&chain::<6>("chain N=6", SWEEP_STEPS, 1e-10, 1e-8, 1));
    report(&chain::<8>("chain N=8", SWEEP_STEPS, 1e-10, 1e-8, 1));
    report(&chain::<10>("chain N=10", SWEEP_STEPS, 1e-10, 1e-8, 1));
    report(&chain::<12>("chain N=12", SWEEP_STEPS, 1e-10, 1e-8, 1));

    println!("-- every second step --\n");
    report(&chain::<4>(
        "chain N=4, every 2",
        SWEEP_STEPS,
        1e-10,
        1e-8,
        2,
    ));
    report(&chain::<6>(
        "chain N=6, every 2",
        SWEEP_STEPS,
        1e-10,
        1e-8,
        2,
    ));
    report(&chain::<8>(
        "chain N=8, every 2",
        SWEEP_STEPS,
        1e-10,
        1e-8,
        2,
    ));
    report(&chain::<10>(
        "chain N=10, every 2",
        SWEEP_STEPS,
        1e-10,
        1e-8,
        2,
    ));
    report(&chain::<12>(
        "chain N=12, every 2",
        SWEEP_STEPS,
        1e-10,
        1e-8,
        2,
    ));

    println!("-- degenerate extreme, for contrast --\n");
    report(&chain::<16>("chain N=16", 10_000, 1e-10, 1e-8, 1));
    report(&chain::<16>("chain N=16, sparse", 10_000, 1e-12, 1e-10, 4));
}
