//! Const generic safe wrappers for compile-time sized FIR filters, Biquads, and Matrices.

use crate::filtering::{biquad_cascade_df1_f32, fir_f32, BiquadCascadeInstanceF32, FirInstanceF32};
use crate::matrix::{
    mat_add_f32, mat_mult_f32, mat_scale_f32, mat_sub_f32, mat_trans_f32, MatrixInstance,
    MatrixInstanceMut,
};

/// Compile-time fixed-size FIR filter holding its own state buffer.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct FirFilter<const TAPS: usize> {
    pub coeffs: [f32; TAPS],
    state: [f32; TAPS],
}

impl<const TAPS: usize> FirFilter<TAPS> {
    /// Create a new FIR filter with given coefficients.
    pub fn new(coeffs: [f32; TAPS]) -> Self {
        Self {
            coeffs,
            state: [0.0; TAPS],
        }
    }

    /// Process input slice `src` into output slice `dst`.
    pub fn process(&mut self, src: &[f32], dst: &mut [f32]) {
        let mut instance = FirInstanceF32 {
            num_taps: TAPS as u16,
            coeffs: &self.coeffs,
            state: &mut self.state,
        };
        fir_f32(&mut instance, src, dst);
    }

    /// Reset filter state buffer.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// Compile-time fixed-size Biquad Cascade Direct Form I filter holding its state buffer.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct BiquadCascade<const COEFFS_LEN: usize, const STATE_LEN: usize> {
    pub coeffs: [f32; COEFFS_LEN],
    pub state: [f32; STATE_LEN],
    num_stages: u8,
}

impl<const COEFFS_LEN: usize, const STATE_LEN: usize> BiquadCascade<COEFFS_LEN, STATE_LEN> {
    /// Create a new Biquad cascade filter given coefficients and number of stages.
    pub fn new(coeffs: [f32; COEFFS_LEN]) -> Self {
        let num_stages = (COEFFS_LEN / 5) as u8;
        Self {
            coeffs,
            state: [0.0; STATE_LEN],
            num_stages,
        }
    }

    /// Process input slice `src` into output slice `dst`.
    pub fn process(&mut self, src: &[f32], dst: &mut [f32]) {
        let mut instance = BiquadCascadeInstanceF32 {
            num_stages: self.num_stages,
            coeffs: &self.coeffs,
            state: &mut self.state,
        };
        biquad_cascade_df1_f32(&mut instance, src, dst);
    }

    /// Reset internal filter delay state.
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }
}

/// Compile-time fixed-size 2D matrix structure.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "defmt", derive(defmt::Format))]
pub struct Matrix<const R: usize, const C: usize, const N: usize> {
    pub data: [f32; N],
}

impl<const R: usize, const C: usize, const N: usize> Matrix<R, C, N> {
    /// Create matrix from array.
    pub const fn new(data: [f32; N]) -> Self {
        Self { data }
    }

    /// Matrix zero constructor.
    pub fn zeros() -> Self {
        Self { data: [0.0; N] }
    }

    /// Matrix addition: `self + rhs`.
    pub fn add(&self, rhs: &Self) -> Self {
        let mut out = Self::zeros();
        let a_inst = MatrixInstance::new(R as u16, C as u16, &self.data);
        let b_inst = MatrixInstance::new(R as u16, C as u16, &rhs.data);
        let mut out_inst = MatrixInstanceMut::new(R as u16, C as u16, &mut out.data);
        mat_add_f32(&a_inst, &b_inst, &mut out_inst);
        out
    }

    /// Matrix subtraction: `self - rhs`.
    pub fn sub(&self, rhs: &Self) -> Self {
        let mut out = Self::zeros();
        let a_inst = MatrixInstance::new(R as u16, C as u16, &self.data);
        let b_inst = MatrixInstance::new(R as u16, C as u16, &rhs.data);
        let mut out_inst = MatrixInstanceMut::new(R as u16, C as u16, &mut out.data);
        mat_sub_f32(&a_inst, &b_inst, &mut out_inst);
        out
    }

    /// Matrix scaling: `self * scale`.
    pub fn scale(&self, scale: f32) -> Self {
        let mut out = Self::zeros();
        let a_inst = MatrixInstance::new(R as u16, C as u16, &self.data);
        let mut out_inst = MatrixInstanceMut::new(R as u16, C as u16, &mut out.data);
        mat_scale_f32(&a_inst, scale, &mut out_inst);
        out
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> Matrix<C, R, N> {
        let mut out = Matrix::<C, R, N>::zeros();
        let a_inst = MatrixInstance::new(R as u16, C as u16, &self.data);
        let mut out_inst = MatrixInstanceMut::new(C as u16, R as u16, &mut out.data);
        mat_trans_f32(&a_inst, &mut out_inst);
        out
    }

    /// Matrix multiplication: `self * rhs`.
    pub fn mul_mat<const C2: usize, const N2: usize, const N_OUT: usize>(
        &self,
        rhs: &Matrix<C, C2, N2>,
    ) -> Matrix<R, C2, N_OUT> {
        let mut out = Matrix::<R, C2, N_OUT>::zeros();
        let a_inst = MatrixInstance::new(R as u16, C as u16, &self.data);
        let b_inst = MatrixInstance::new(C as u16, C2 as u16, &rhs.data);
        let mut out_inst = MatrixInstanceMut::new(R as u16, C2 as u16, &mut out.data);
        mat_mult_f32(&a_inst, &b_inst, &mut out_inst);
        out
    }
}
