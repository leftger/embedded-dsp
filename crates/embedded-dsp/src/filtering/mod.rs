//! Digital filtering: FIR, IIR/biquad, adaptive, convolution, and recursive
//! filters.
//!
//! Split into one submodule per filter family; every public item is
//! re-exported here, so `embedded_dsp::filtering::Foo` and the crate-root
//! glob are unchanged.

mod adaptive;
mod biquad;
mod convolution;
mod fir;
mod int_filters;
mod lockin;
mod normal_form;
mod recursive;
mod wdf;

pub use adaptive::*;
pub use biquad::*;
pub use convolution::*;
pub use fir::*;
pub use int_filters::*;
pub use lockin::*;
pub use normal_form::*;
pub use recursive::*;
pub use wdf::*;
