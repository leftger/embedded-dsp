//! Throwaway spike: does square-root factoring actually beat the plain covariance form in f32?
//!
//! Settles one question before committing to a composed factored filter:
//!
//! * `KalmanFilter<N, M>` — plain form, simple `P⁺ = (I − KH)P`.
//! * `SquareRootKalmanFilter<N, M>` — stores a Cholesky factor, re-factorizes each step
//!   (`P = S Sᵀ` is PSD by construction, but the arithmetic is still done on P).
//! * an `f64` simple-form reference, so "error" means deviation from the exact arithmetic that
//!   an f32 filter is trying to emulate.
//!
//! Metrics per run: asymmetry of `P`, the smaller eigenvalue of `P` (negative ⇒ positive
//! definiteness lost), relative Frobenius distance from the f64 `P`, and the state error against
//! the known ground-truth trajectory.

use embedded_dsp::kalman::{KalmanFilter, SquareRootKalmanFilter};

type M2 = [[f64; 2]; 2];

fn chol(p: M2) -> M2 {
    let l00 = p[0][0].max(0.0).sqrt();
    let l10 = if l00 > 0.0 { p[1][0] / l00 } else { 0.0 };
    let l11 = (p[1][1] - l10 * l10).max(0.0).sqrt();
    [[l00, 0.0], [l10, l11]]
}

fn to_f32(m: M2) -> [[f32; 2]; 2] {
    [
        [m[0][0] as f32, m[0][1] as f32],
        [m[1][0] as f32, m[1][1] as f32],
    ]
}

/// Smaller eigenvalue of a symmetric 2×2 matrix.
fn min_eig_f32(p: [[f32; 2]; 2]) -> f64 {
    let (a, b, d) = (p[0][0] as f64, p[0][1] as f64, p[1][1] as f64);
    let tr = a + d;
    let disc = (tr * tr - 4.0 * (a * d - b * b)).max(0.0).sqrt();
    (tr - disc) / 2.0
}

fn asym_f32(p: [[f32; 2]; 2]) -> f64 {
    (p[0][1] as f64 - p[1][0] as f64).abs()
}

fn frob_diff(a: [[f32; 2]; 2], b: M2) -> f64 {
    let mut s = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            let d = a[i][j] as f64 - b[i][j];
            s += d * d;
        }
    }
    s.sqrt()
}

fn frob(b: M2) -> f64 {
    let mut s = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            s += b[i][j] * b[i][j];
        }
    }
    s.sqrt()
}

struct Scenario {
    name: &'static str,
    steps: usize,
    f: M2,
    q: M2,
    r: f64,
    p0: M2,
}

/// Plain f32 filter: `x ← F x`, `P ← F P Fᵀ + Q`, then the simple-form measurement update.
fn run_plain(s: &Scenario) -> ([[f32; 2]; 2], [f32; 2], bool) {
    let mut kf = KalmanFilter::<2, 1>::new([0.0, 1.0], to_f32(s.p0), to_f32(s.q), [[s.r as f32]]);
    let f = to_f32(s.f);
    let h = [[1.0f32, 0.0]];

    for step in 1..=s.steps {
        kf.predict(&f);
        let z = [step as f32]; // constant velocity from x0 = [0, 1]
        if kf.update(&h, &z) != embedded_dsp::Status::Success {
            return (kf.p, kf.x, false);
        }
        if !kf.p[0][0].is_finite() || !kf.x[0].is_finite() {
            return (kf.p, kf.x, false);
        }
    }
    (kf.p, kf.x, true)
}

fn run_sr(s: &Scenario) -> ([[f32; 2]; 2], [f32; 2], bool) {
    let mut sr = SquareRootKalmanFilter::<2, 1>::new(
        [0.0, 1.0],
        to_f32(chol(s.p0)),
        to_f32(s.f),
        to_f32(chol(s.q)),
        [[1.0f32, 0.0]],
        [[(s.r).max(0.0).sqrt() as f32]],
    );

    for step in 1..=s.steps {
        sr.predict();
        let z = [step as f32];
        if sr.update(&z) != embedded_dsp::Status::Success {
            return (sr.covariance(), sr.x, false);
        }
        if !sr.s[0][0].is_finite() {
            return (sr.covariance(), sr.x, false);
        }
    }
    (sr.covariance(), sr.x, true)
}

/// f64 reference, same update form as the plain filter.
fn run_reference(s: &Scenario) -> (M2, [f64; 2]) {
    let (mut x, mut p) = ([0.0f64, 1.0], s.p0);
    let f = s.f;

    for step in 1..=s.steps {
        // x ← F x, P ← F P Fᵀ + Q
        let x1 = f[0][0] * x[0] + f[0][1] * x[1];
        let x2 = f[1][0] * x[0] + f[1][1] * x[1];
        x = [x1, x2];
        let fp = [
            [
                f[0][0] * p[0][0] + f[0][1] * p[1][0],
                f[0][0] * p[0][1] + f[0][1] * p[1][1],
            ],
            [
                f[1][0] * p[0][0] + f[1][1] * p[1][0],
                f[1][0] * p[0][1] + f[1][1] * p[1][1],
            ],
        ];
        p = [
            [
                fp[0][0] * f[0][0] + fp[0][1] * f[0][1] + s.q[0][0],
                fp[0][0] * f[1][0] + fp[0][1] * f[1][1] + s.q[0][1],
            ],
            [
                fp[1][0] * f[0][0] + fp[1][1] * f[0][1] + s.q[1][0],
                fp[1][0] * f[1][0] + fp[1][1] * f[1][1] + s.q[1][1],
            ],
        ];

        // Scalar update with H = [1, 0].
        let h = [1.0f64, 0.0];
        let ph = [
            p[0][0] * h[0] + p[0][1] * h[1],
            p[1][0] * h[0] + p[1][1] * h[1],
        ];
        let var = h[0] * ph[0] + h[1] * ph[1] + s.r;
        let k = [ph[0] / var, ph[1] / var];
        let y = step as f64 - x[0];
        x = [x[0] + k[0] * y, x[1] + k[1] * y];

        let kh = [[k[0] * h[0], k[0] * h[1]], [k[1] * h[0], k[1] * h[1]]];
        let ikh = [[1.0 - kh[0][0], -kh[0][1]], [-kh[1][0], 1.0 - kh[1][1]]];
        p = [
            [
                ikh[0][0] * p[0][0] + ikh[0][1] * p[1][0],
                ikh[0][0] * p[0][1] + ikh[0][1] * p[1][1],
            ],
            [
                ikh[1][0] * p[0][0] + ikh[1][1] * p[1][0],
                ikh[1][0] * p[0][1] + ikh[1][1] * p[1][1],
            ],
        ];
    }
    (p, x)
}

fn main() {
    let scenarios = [
        Scenario {
            name: "benign: R ~ P0",
            steps: 20_000,
            f: [[1.0, 1.0], [0.0, 1.0]],
            q: [[1e-6, 0.0], [0.0, 1e-6]],
            r: 1e-2,
            p0: [[1.0, 0.0], [0.0, 1.0]],
        },
        Scenario {
            name: "tight R << P0",
            steps: 20_000,
            f: [[1.0, 1.0], [0.0, 1.0]],
            q: [[1e-12, 0.0], [0.0, 1e-12]],
            r: 1e-8,
            p0: [[1.0, 0.0], [0.0, 1.0]],
        },
        Scenario {
            name: "anisotropic P0 (1e12)",
            steps: 20_000,
            f: [[1.0, 1.0], [0.0, 1.0]],
            q: [[1e-10, 0.0], [0.0, 1e-10]],
            r: 1e-6,
            p0: [[1e6, 0.0], [0.0, 1e-6]],
        },
        Scenario {
            name: "R = 0 (deterministic)",
            steps: 2_000,
            f: [[1.0, 1.0], [0.0, 1.0]],
            q: [[1e-9, 0.0], [0.0, 1e-9]],
            r: 0.0,
            p0: [[1.0, 0.0], [0.0, 1.0]],
        },
        Scenario {
            name: "long run, tiny Q and R",
            steps: 200_000,
            f: [[1.0, 1.0], [0.0, 1.0]],
            q: [[1e-14, 0.0], [0.0, 1e-14]],
            r: 1e-10,
            p0: [[1.0, 0.0], [0.0, 1.0]],
        },
        Scenario {
            name: "huge Q, tiny R",
            steps: 20_000,
            f: [[1.0, 1.0], [0.0, 1.0]],
            q: [[1.0, 0.0], [0.0, 1.0]],
            r: 1e-12,
            p0: [[1.0, 0.0], [0.0, 1.0]],
        },
        // Diagnostics for the Cholesky diagonal floor (`max(1e-12)` in the crate): the same
        // long-run shape, with the steady-state covariance sitting either side of 1e-12.
        Scenario {
            name: "long run, P well above floor",
            steps: 200_000,
            f: [[1.0, 1.0], [0.0, 1.0]],
            q: [[1e-10, 0.0], [0.0, 1e-10]],
            r: 1e-8,
            p0: [[1.0, 0.0], [0.0, 1.0]],
        },
        Scenario {
            name: "long run, P just above floor",
            steps: 200_000,
            f: [[1.0, 1.0], [0.0, 1.0]],
            q: [[1e-12, 0.0], [0.0, 1e-12]],
            r: 1e-10,
            p0: [[1.0, 0.0], [0.0, 1.0]],
        },
    ];

    println!(
        "{:<30} {:>7} {:>11} {:>9} {:>10} {:>10} {:>11} {:>6}",
        "scenario", "filter", "minEig", "minEig/ref", "asym", "relErrP", "stateErr", "ok"
    );

    for s in &scenarios {
        let (pref, xref) = run_reference(s);
        let scale = frob(pref).max(1e-300);
        let mref = min_eig_f32(to_f32(pref));

        let (pp, xp, okp) = run_plain(s);
        let (ps, xs, oks) = run_sr(s);

        let ratio = |m: f64| if mref.abs() > 0.0 { m / mref } else { f64::NAN };

        println!(
            "{:<30} {:>7} {:>11.3e} {:>9.3} {:>10.3e} {:>10.3e} {:>11.3e} {:>6}",
            s.name,
            "plain",
            min_eig_f32(pp),
            ratio(min_eig_f32(pp)),
            asym_f32(pp),
            frob_diff(pp, pref) / scale,
            (xp[0] as f64 - xref[0]).abs(),
            okp
        );
        println!(
            "{:<30} {:>7} {:>11.3e} {:>9.3} {:>10.3e} {:>10.3e} {:>11.3e} {:>6}",
            "",
            "sqrt",
            min_eig_f32(ps),
            ratio(min_eig_f32(ps)),
            asym_f32(ps),
            frob_diff(ps, pref) / scale,
            (xs[0] as f64 - xref[0]).abs(),
            oks
        );
        println!(
            "{:<30} {:>7} {:>11.3e} {:>9.3} {:>10.3e} {:>10.3e} {:>11.3e} {:>6}",
            "", "f64 ref", mref, 1.0, 0.0, 0.0, 0.0, true
        );
        println!();
    }
}
