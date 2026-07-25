//! Support Vector Machine (SVM) Classifier (Linear, Polynomial, RBF, Sigmoid kernels).

#[allow(unused_imports)]
use crate::math::FloatMath;
use crate::types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvmKernelType {
    Linear,
    Polynomial,
    Rbf,
    Sigmoid,
}

#[inline(always)]
fn pow_core(x: f32, p: i32) -> f32 {
    let mut res = 1.0f32;
    for _ in 0..p {
        res *= x;
    }
    res
}

/// SVM Classifier Instance structure for f32.
pub struct SvmInstanceF32<'a> {
    pub num_vector_dim: usize,
    pub num_support_vectors: usize,
    pub intercept: f32,
    pub dual_coefs: &'a [f32],      // Size num_support_vectors
    pub support_vectors: &'a [f32], // Size num_support_vectors * num_vector_dim
    pub kernel_type: SvmKernelType,
    pub gamma: f32,
    pub coef0: f32,
    pub degree: i32,
}

impl<'a> SvmInstanceF32<'a> {
    pub fn predict(&self, input: &[f32], result: &mut i32) -> Status {
        if input.len() < self.num_vector_dim {
            return Status::LengthError;
        }

        let mut sum = self.intercept;
        for i in 0..self.num_support_vectors {
            let sv = &self.support_vectors[i * self.num_vector_dim..(i + 1) * self.num_vector_dim];
            let alpha = self.dual_coefs[i];

            let kernel_val = match self.kernel_type {
                SvmKernelType::Linear => {
                    let mut dot = 0.0f32;
                    for d in 0..self.num_vector_dim {
                        dot += sv[d] * input[d];
                    }
                    dot
                }
                SvmKernelType::Polynomial => {
                    let mut dot = 0.0f32;
                    for d in 0..self.num_vector_dim {
                        dot += sv[d] * input[d];
                    }
                    pow_core(self.gamma * dot + self.coef0, self.degree)
                }
                SvmKernelType::Rbf => {
                    let mut dist_sq = 0.0f32;
                    for d in 0..self.num_vector_dim {
                        let diff = input[d] - sv[d];
                        dist_sq += diff * diff;
                    }
                    (-self.gamma * dist_sq).exp()
                }
                SvmKernelType::Sigmoid => {
                    let mut dot = 0.0f32;
                    for d in 0..self.num_vector_dim {
                        dot += sv[d] * input[d];
                    }
                    (self.gamma * dot + self.coef0).tanh()
                }
            };
            sum += alpha * kernel_val;
        }

        *result = if sum >= 0.0 { 1 } else { -1 };
        Status::Success
    }
}

pub fn svm_predict_f32(instance: &SvmInstanceF32, input: &[f32], result: &mut i32) -> Status {
    instance.predict(input, result)
}
