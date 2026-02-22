//! Weight initialization utilities.
//!
//! Provides shared random number generation using the Box-Muller transform
//! for Gaussian (normal) distributions. Used by all crates in the workspace.

use crate::Float;
use rand::Rng;

/// Generates a single sample from the standard normal distribution N(0, 1)
/// using the Box-Muller transform.
#[inline]
pub fn randn_scalar(rng: &mut impl Rng) -> Float {
    let u1: Float = rng.random();
    let u2: Float = rng.random();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Fills a pre-allocated slice with scaled normal samples: N(0, std_dev²).
///
/// This avoids allocating a new Vec when the caller already has a buffer.
pub fn fill_randn(buf: &mut [Float], std_dev: Float, rng: &mut impl Rng) {
    for x in buf.iter_mut() {
        *x = randn_scalar(rng) * std_dev;
    }
}

/// Generates a Vec of `size` random values from N(0, std_dev²)
/// using the Box-Muller transform.
pub fn randn_vec(size: usize, std_dev: Float, rng: &mut impl Rng) -> Vec<Float> {
    (0..size)
        .map(|_| randn_scalar(rng) * std_dev)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randn_scalar_distribution() {
        let mut rng = rand::rng();
        let n = 10_000;
        let samples: Vec<Float> = (0..n).map(|_| randn_scalar(&mut rng)).collect();

        let mean: Float = samples.iter().sum::<Float>() / n as Float;
        let variance: Float =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<Float>() / n as Float;

        // Mean should be close to 0, variance close to 1
        assert!(mean.abs() < 0.1, "mean = {mean}");
        assert!((variance - 1.0).abs() < 0.1, "variance = {variance}");
    }

    #[test]
    fn test_randn_vec_length() {
        let mut rng = rand::rng();
        let v = randn_vec(42, 1.0, &mut rng);
        assert_eq!(v.len(), 42);
    }

    #[test]
    fn test_fill_randn() {
        let mut rng = rand::rng();
        let mut buf = vec![0.0; 100];
        fill_randn(&mut buf, 2.0, &mut rng);

        // Should not be all zeros after fill
        assert!(buf.iter().any(|&x| x != 0.0));
    }
}
