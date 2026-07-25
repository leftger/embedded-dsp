//! Gaussian Naive Bayes Classifier.

#[allow(unused_imports)]
use crate::math::FloatMath;

/// Gaussian Naive Bayes Instance structure for f32.
pub struct GaussianNaiveBayesInstanceF32<'a> {
    pub num_classes: usize,
    pub num_features: usize,
    pub theta: &'a [f32],       // Means array of size num_classes * num_features
    pub sigma: &'a [f32],       // Variances array of size num_classes * num_features
    pub class_prior: &'a [f32], // Class priors array of size num_classes
    pub epsilon: f32,           // Additive variance stabilization
}

impl<'a> GaussianNaiveBayesInstanceF32<'a> {
    /// Predict class index and return log probability buffer.
    pub fn predict(&self, input: &[f32], probs: &mut [f32]) -> usize {
        let mut max_log_prob = f32::NEG_INFINITY;
        let mut best_class = 0;

        let log_2pi = 1.8378770664f32; // ln(2 * pi)

        for c in 0..self.num_classes {
            let prior = self.class_prior[c];
            let mut log_prob = prior.max(1e-12).ln();

            for f in 0..self.num_features {
                let mean = self.theta[c * self.num_features + f];
                let var = self.sigma[c * self.num_features + f] + self.epsilon;
                let diff = input[f] - mean;

                log_prob -= 0.5 * (log_2pi + var.ln() + (diff * diff) / var);
            }

            if c < probs.len() {
                probs[c] = log_prob;
            }

            if log_prob > max_log_prob {
                max_log_prob = log_prob;
                best_class = c;
            }
        }
        best_class
    }
}

pub fn bayes_predict_f32(
    instance: &GaussianNaiveBayesInstanceF32,
    input: &[f32],
    probs: &mut [f32],
) -> usize {
    instance.predict(input, probs)
}
