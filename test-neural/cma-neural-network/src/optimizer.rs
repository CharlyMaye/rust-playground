//! Optimizers for neural network training.
//!
//! This module provides different optimization algorithms to update
//! network weights during training.

use crate::Float;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Available optimizer types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent (SGD) - Simple and fast
    SGD { learning_rate: Float },
    
    /// SGD with momentum - Accelerates in the right directions
    Momentum { 
        learning_rate: Float, 
        beta: Float  // Typically 0.9
    },
    
    /// RMSprop - Adapts learning rate per parameter
    RMSprop { 
        learning_rate: Float, 
        beta: Float,      // Typically 0.9
        epsilon: Float    // Typically 1e-8
    },
    
    /// Adam - Adaptive Moment Estimation (modern standard)
    Adam { 
        learning_rate: Float,
        beta1: Float,     // Typically 0.9 (momentum)
        beta2: Float,     // Typically 0.999 (variance)
        epsilon: Float    // Typically 1e-8
    },
    
    /// AdamW - Adam with decoupled Weight Decay
    AdamW { 
        learning_rate: Float,
        beta1: Float,
        beta2: Float,
        epsilon: Float,
        weight_decay: Float  // Typically 0.01
    },
}

impl OptimizerType {
    /// Creates an SGD optimizer with the specified learning rate.
    pub fn sgd(learning_rate: Float) -> Self {
        OptimizerType::SGD { learning_rate }
    }
    
    /// Creates a Momentum optimizer with default parameters.
    pub fn momentum(learning_rate: Float) -> Self {
        OptimizerType::Momentum { 
            learning_rate, 
            beta: 0.9 
        }
    }
    
    /// Creates an RMSprop optimizer with default parameters.
    pub fn rmsprop(learning_rate: Float) -> Self {
        OptimizerType::RMSprop { 
            learning_rate, 
            beta: 0.9, 
            epsilon: 1e-8 
        }
    }
    
    /// Creates an Adam optimizer with default parameters (recommended).
    pub fn adam(learning_rate: Float) -> Self {
        OptimizerType::Adam { 
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8
        }
    }
    
    /// Creates an AdamW optimizer with default parameters.
    pub fn adamw(learning_rate: Float, weight_decay: Float) -> Self {
        OptimizerType::AdamW { 
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay
        }
    }
}

/// Optimizer state for a weight matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState2D {
    /// First moment (momentum) - used by Momentum, Adam, AdamW
    pub m: Option<Array2<Float>>,
    
    /// Second moment (variance) - used by RMSprop, Adam, AdamW
    pub v: Option<Array2<Float>>,
    
    /// Number of iterations (for bias correction in Adam)
    pub t: usize,
}

/// Optimizer state for a bias vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState1D {
    /// First moment (momentum)
    pub m: Option<Array1<Float>>,
    
    /// Second moment (variance)
    pub v: Option<Array1<Float>>,
    
    /// Number of iterations
    pub t: usize,
}

/// Element-wise optimizer step on raw slices.
///
/// All optimizer logic is implemented once here, operating on flat slices.
/// `OptimizerState1D` and `OptimizerState2D` delegate to this function.
///
/// `params`, `m`, and `v` are always contiguous (owned arrays), while
/// `gradient` is passed as a separate slice (caller ensures contiguity).
fn step_slices(
    params: &mut [Float],
    gradient: &[Float],
    m: Option<&mut [Float]>,
    v: Option<&mut [Float]>,
    t: usize,
    optimizer: &OptimizerType,
) {
    match optimizer {
        OptimizerType::SGD { learning_rate } => {
            let lr = *learning_rate;
            for (p, &g) in params.iter_mut().zip(gradient.iter()) {
                *p -= lr * g;
            }
        }

        OptimizerType::Momentum { learning_rate, beta } => {
            let m = m.expect("Momentum state not initialized");
            let b = *beta;
            let lr = *learning_rate;
            for ((p, m_i), &g) in params.iter_mut().zip(m.iter_mut()).zip(gradient.iter()) {
                *m_i = *m_i * b + g;
                *p -= lr * *m_i;
            }
        }

        OptimizerType::RMSprop { learning_rate, beta, epsilon } => {
            let v = v.expect("RMSprop state not initialized");
            let b = *beta;
            let one_minus_b = 1.0 - b;
            let lr = *learning_rate;
            let eps = *epsilon;
            for ((p, v_i), &g) in params.iter_mut().zip(v.iter_mut()).zip(gradient.iter()) {
                *v_i = *v_i * b + g * g * one_minus_b;
                *p -= lr * g / (v_i.sqrt() + eps);
            }
        }

        OptimizerType::Adam { learning_rate, beta1, beta2, epsilon } => {
            let m = m.expect("Adam m state not initialized");
            let v = v.expect("Adam v state not initialized");
            let one_minus_b1 = 1.0 - beta1;
            let one_minus_b2 = 1.0 - beta2;
            let bc1 = 1.0 - beta1.powi(t as i32);
            let bc2 = 1.0 - beta2.powi(t as i32);
            let lr = *learning_rate;
            let eps = *epsilon;
            for (((p, m_i), v_i), &g) in params
                .iter_mut()
                .zip(m.iter_mut())
                .zip(v.iter_mut())
                .zip(gradient.iter())
            {
                *m_i = *m_i * *beta1 + g * one_minus_b1;
                *v_i = *v_i * *beta2 + g * g * one_minus_b2;
                let m_hat = *m_i / bc1;
                let v_hat = *v_i / bc2;
                *p -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }

        OptimizerType::AdamW { learning_rate, beta1, beta2, epsilon, weight_decay } => {
            let m = m.expect("AdamW m state not initialized");
            let v = v.expect("AdamW v state not initialized");
            let one_minus_b1 = 1.0 - beta1;
            let one_minus_b2 = 1.0 - beta2;
            let decay = 1.0 - learning_rate * weight_decay;
            let bc1 = 1.0 - beta1.powi(t as i32);
            let bc2 = 1.0 - beta2.powi(t as i32);
            let lr = *learning_rate;
            let eps = *epsilon;
            for (((p, m_i), v_i), &g) in params
                .iter_mut()
                .zip(m.iter_mut())
                .zip(v.iter_mut())
                .zip(gradient.iter())
            {
                *m_i = *m_i * *beta1 + g * one_minus_b1;
                *v_i = *v_i * *beta2 + g * g * one_minus_b2;
                let m_hat = *m_i / bc1;
                let v_hat = *v_i / bc2;
                *p = *p * decay - lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }
}

/// Determines whether an optimizer needs first-moment (m) state.
fn needs_first_moment(optimizer: &OptimizerType) -> bool {
    matches!(
        optimizer,
        OptimizerType::Momentum { .. } | OptimizerType::Adam { .. } | OptimizerType::AdamW { .. }
    )
}

/// Determines whether an optimizer needs second-moment (v) state.
fn needs_second_moment(optimizer: &OptimizerType) -> bool {
    matches!(
        optimizer,
        OptimizerType::RMSprop { .. } | OptimizerType::Adam { .. } | OptimizerType::AdamW { .. }
    )
}

impl OptimizerState2D {
    /// Creates a new state for a matrix of the given shape.
    pub fn new(shape: (usize, usize), optimizer: &OptimizerType) -> Self {
        OptimizerState2D {
            m: if needs_first_moment(optimizer) { Some(Array2::zeros(shape)) } else { None },
            v: if needs_second_moment(optimizer) { Some(Array2::zeros(shape)) } else { None },
            t: 0,
        }
    }

    /// Updates weights with the computed gradient.
    pub fn step(
        &mut self,
        weights: &mut Array2<Float>,
        gradient: &Array2<Float>,
        optimizer: &OptimizerType,
    ) {
        self.t += 1;
        // Gradient may come from a view (non-contiguous); ensure contiguity.
        let grad_cow = gradient.as_standard_layout();
        step_slices(
            weights.as_slice_mut().expect("weights not contiguous"),
            grad_cow.as_slice().expect("gradient not contiguous after layout fix"),
            self.m.as_mut().map(|a| a.as_slice_mut().expect("m not contiguous")),
            self.v.as_mut().map(|a| a.as_slice_mut().expect("v not contiguous")),
            self.t,
            optimizer,
        );
    }
}

impl OptimizerState1D {
    /// Creates a new state for a vector of the given size.
    pub fn new(size: usize, optimizer: &OptimizerType) -> Self {
        OptimizerState1D {
            m: if needs_first_moment(optimizer) { Some(Array1::zeros(size)) } else { None },
            v: if needs_second_moment(optimizer) { Some(Array1::zeros(size)) } else { None },
            t: 0,
        }
    }

    /// Updates biases with the computed gradient.
    pub fn step(
        &mut self,
        biases: &mut Array1<Float>,
        gradient: &Array1<Float>,
        optimizer: &OptimizerType,
    ) {
        self.t += 1;
        let grad_cow = gradient.as_standard_layout();
        step_slices(
            biases.as_slice_mut().expect("biases not contiguous"),
            grad_cow.as_slice().expect("gradient not contiguous after layout fix"),
            self.m.as_mut().map(|a| a.as_slice_mut().expect("m not contiguous")),
            self.v.as_mut().map(|a| a.as_slice_mut().expect("v not contiguous")),
            self.t,
            optimizer,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_update() {
        let mut weights = Array2::from_elem((2, 2), 1.0);
        let gradient = Array2::from_elem((2, 2), 0.1);
        let mut state = OptimizerState2D::new((2, 2), &OptimizerType::sgd(0.1));
        
        state.step(&mut weights, &gradient, &OptimizerType::sgd(0.1));
        
        // w = 1.0 - 0.1 * 0.1 = 0.99
        assert!((weights[[0, 0]] - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_adam_initialization() {
        let optimizer = OptimizerType::adam(0.001);
        let state = OptimizerState2D::new((3, 3), &optimizer);
        
        assert!(state.m.is_some());
        assert!(state.v.is_some());
        assert_eq!(state.t, 0);
    }

    #[test]
    fn test_optimizer_constructors() {
        let sgd = OptimizerType::sgd(0.1);
        assert!(matches!(sgd, OptimizerType::SGD { .. }));
        
        let adam = OptimizerType::adam(0.001);
        assert!(matches!(adam, OptimizerType::Adam { .. }));
        
        let rmsprop = OptimizerType::rmsprop(0.01);
        assert!(matches!(rmsprop, OptimizerType::RMSprop { .. }));
    }
}
