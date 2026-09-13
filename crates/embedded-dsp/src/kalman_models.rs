//! Composable Kalman models, re-exported from the upstream
//! [`idsp`](https://crates.io/crates/idsp) crate.
//!
//! [`crate::kalman`] is the other half of this story: self-contained filters with a fixed
//! structure — pick the one that matches your model, or write your own. Here the filter is
//! *composed* instead. A prediction phase and a measurement phase share an `Estimate`, and each
//! is a pluggable type:
//!
//! | Piece | Role |
//! | :--- | :--- |
//! | `Estimate` | state vector plus its error covariance |
//! | `Transition` | dense state-transition matrix `F` and process noise `Q` |
//! | `ConstantVelocity` | the common `[[1, dt], [0, 1]]` motion model |
//! | `RandomWalk` | the smallest complete filter, `F = H = 1` |
//! | `Observation` | dense scalar measurement row `H` and noise `R` |
//! | `Direct` | observe a single state component, with no dense matrix |
//! | `Kalman` | compose a prediction phase with a measurement phase |
//!
//! Composing rather than picking matters once the model is unusual: `Kalman::optional()` skips
//! the update when a measurement is missing, and the same code path serves fixed-point state,
//! which follows the coefficient/raw convention documented in `idsp::iir`. Working through
//! `dsp_process` also means the phases are ordinary streaming stages, so they compose with
//! [`crate::pipeline`] style pipelines rather than needing their own glue.
//!
//! The `dsp_process` crate is re-exported here because the filter is driven through its traits
//! (`Process`, `Split`); you do not need to depend on it yourself.
//!
//! # Feature
//!
//! Behind the `kalman-models` feature, which is deliberately **not** part of `full`: it is the
//! one feature that pulls external crates (`idsp`, and through it `dsp-process` and
//! `dsp-fixedpoint`). Default builds stay dependency-free, exactly as with `nalgebra`.
//!
//! # Examples
//!
//! A scalar random walk — one state, one measurement:
//!
//! ```
//! use embedded_dsp::kalman_models::{dsp_process::Split, Estimate, Process, RandomWalk};
//!
//! let config = RandomWalk::<f32>::new(0.01, 1.0);
//! let mut filter = Split::new(config, Estimate::new([0.0], [[10.0]]));
//!
//! let filtered = filter.process(2.0);
//! assert!(filtered > 0.0 && filtered < 2.0);
//! ```
//!
//! Composing the pieces for a position/velocity tracker, observing position directly:
//!
//! ```
//! use embedded_dsp::kalman_models::{
//!     ConstantVelocity, Direct, Estimate, Kalman, Process, dsp_process::Split,
//! };
//!
//! let predict = ConstantVelocity::<f32>::new(1.0, [[1e-4, 0.0], [0.0, 1e-4]]);
//! let update = Direct::<f32, 0>::new(0.5);
//! let mut filter = Split::new(
//!     Kalman::new(predict, update),
//!     Estimate::new([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]]),
//! );
//!
//! let mut estimated = 0.0;
//! for position in [1.0f32, 2.0, 3.0] {
//!     estimated = filter.process(position);
//! }
//! // The tracker converges on the position it is fed.
//! assert!((estimated - 3.0).abs() < 1.0);
//! ```

pub use dsp_process::{self, Process, Split};
pub use idsp::kalman::{
    ConstantVelocity, DenseKalman, Direct, Estimate, Kalman, Observation, RandomWalk, Transition,
};
