//! Deterministic snapshot capture buffers and impulse response analyzers.

/// Fixed-capacity zero-allocation snapshot buffer.
#[derive(Debug, Clone)]
pub struct SnapshotBuffer<const N: usize> {
    buffer: [f32; N],
    count: usize,
    frozen: bool,
}

impl<const N: usize> SnapshotBuffer<N> {
    /// Creates a new empty snapshot buffer.
    pub fn new() -> Self {
        Self {
            buffer: [0.0; N],
            count: 0,
            frozen: false,
        }
    }

    /// Appends a sample if the buffer is not full or frozen. Returns true if stored.
    pub fn push(&mut self, sample: f32) -> bool {
        if self.frozen || self.count >= N {
            false
        } else {
            self.buffer[self.count] = sample;
            self.count += 1;
            if self.count >= N {
                self.frozen = true;
            }
            true
        }
    }

    /// Returns the captured sample slice.
    pub fn samples(&self) -> &[f32] {
        &self.buffer[..self.count]
    }

    /// Returns true if the buffer is full and frozen.
    pub fn is_full(&self) -> bool {
        self.count >= N
    }

    /// Unfreezes and clears the snapshot buffer.
    pub fn reset(&mut self) {
        self.count = 0;
        self.frozen = false;
    }
}

impl<const N: usize> Default for SnapshotBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Analytical results of an impulse response test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImpulseResponseInfo {
    /// Maximum absolute peak gain achieved.
    pub peak_gain: f32,
    /// Number of samples taken to settle permanently within the target threshold band.
    pub settling_time_samples: usize,
    /// Total signal energy $\sum h[n]^2$.
    pub total_energy: f32,
    /// True if peak gain and energy remain finite (no numerical explosion).
    pub is_stable: bool,
}

/// Analyzes an impulse response sample vector.
pub fn analyze_impulse_response(response: &[f32], settling_threshold: f32) -> ImpulseResponseInfo {
    if response.is_empty() {
        return ImpulseResponseInfo {
            peak_gain: 0.0,
            settling_time_samples: 0,
            total_energy: 0.0,
            is_stable: true,
        };
    }

    let mut peak = 0.0f32;
    let mut energy = 0.0f32;
    let mut settling_idx = response.len();
    let threshold = settling_threshold.abs();

    for (_idx, &sample) in response.iter().enumerate() {
        let abs_val = sample.abs();
        if abs_val > peak {
            peak = abs_val;
        }
        energy += sample * sample;
    }

    // Work backwards to find where output settled within threshold
    for idx in (0..response.len()).rev() {
        if response[idx].abs() > threshold {
            settling_idx = idx + 1;
            break;
        }
    }

    let is_stable = peak.is_finite() && energy.is_finite() && peak < 1000.0;

    ImpulseResponseInfo {
        peak_gain: peak,
        settling_time_samples: settling_idx,
        total_energy: energy,
        is_stable,
    }
}
