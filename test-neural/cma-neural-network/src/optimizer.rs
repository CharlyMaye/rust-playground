//! Optimizers for neural network training.
//!
//! This module provides different optimization algorithms to update
//! network weights during training.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Available optimizer types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent (SGD) - Simple and fast
    SGD { learning_rate: f64 },
    
    /// SGD with momentum - Accelerates in the right directions
    Momentum { 
        learning_rate: f64, 
        beta: f64  // Typically 0.9
    },
    
    /// RMSprop - Adapts learning rate per parameter
    RMSprop { 
        learning_rate: f64, 
        beta: f64,      // Typically 0.9
        epsilon: f64    // Typically 1e-8
    },
    
    /// Adam - Adaptive Moment Estimation (modern standard)
    Adam { 
        learning_rate: f64,
        beta1: f64,     // Typically 0.9 (momentum)
        beta2: f64,     // Typically 0.999 (variance)
        epsilon: f64    // Typically 1e-8
    },
    
    /// AdamW - Adam with decoupled Weight Decay
    AdamW { 
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64  // Typically 0.01
    },
}

impl OptimizerType {
    /// Creates an SGD optimizer with the specified learning rate.
    pub fn sgd(learning_rate: f64) -> Self {
        OptimizerType::SGD { learning_rate }
    }
    
    /// Creates a Momentum optimizer with default parameters.
    pub fn momentum(learning_rate: f64) -> Self {
        OptimizerType::Momentum { 
            learning_rate, 
            beta: 0.9 
        }
    }
    
    /// Creates an RMSprop optimizer with default parameters.
    pub fn rmsprop(learning_rate: f64) -> Self {
        OptimizerType::RMSprop { 
            learning_rate, 
            beta: 0.9, 
            epsilon: 1e-8 
        }
    }
    
    /// Creates an Adam optimizer with default parameters (recommended).
    pub fn adam(learning_rate: f64) -> Self {
        OptimizerType::Adam { 
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8
        }
    }
    
    /// Creates an AdamW optimizer with default parameters.
    pub fn adamw(learning_rate: f64, weight_decay: f64) -> Self {
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
    pub m: Option<Array2<f64>>,
    
    /// Second moment (variance) - used by RMSprop, Adam, AdamW
    pub v: Option<Array2<f64>>,
    
    /// Number of iterations (for bias correction in Adam)
    pub t: usize,
}

impl OptimizerState2D {
    /// Creates a new state for a matrix of the given shape.
    pub fn new(shape: (usize, usize), optimizer: &OptimizerType) -> Self {
        let needs_m = matches!(optimizer, 
            OptimizerType::Momentum { .. } | 
            OptimizerType::Adam { .. } | 
            OptimizerType::AdamW { .. }
        );
        
        let needs_v = matches!(optimizer,
            OptimizerType::RMSprop { .. } |
            OptimizerType::Adam { .. } | 
            OptimizerType::AdamW { .. }
        );
        
        OptimizerState2D {
            m: if needs_m { Some(Array2::zeros(shape)) } else { None },
            v: if needs_v { Some(Array2::zeros(shape)) } else { None },
            t: 0,
        }
    }
    
    /// Updates weights with the computed gradient.
    pub fn step(
        &mut self,
        weights: &mut Array2<f64>,
        gradient: &Array2<f64>,
        optimizer: &OptimizerType,
    ) {
        self.t += 1;
        
        match optimizer {
            OptimizerType::SGD { learning_rate } => {
                // Simple gradient descent: w = w - lr * grad
                *weights -= &(gradient * *learning_rate);
            }
            
            OptimizerType::Momentum { learning_rate, beta } => {
                // Momentum: m = beta * m + grad
                //           w = w - lr * m
                let m = self.m.as_mut().expect("Momentum state not initialized");
                // In-place: m *= beta, puis m += gradient
                m.mapv_inplace(|x| x * *beta);
                *m += gradient;
                // w -= lr * m
                weights.scaled_add(-*learning_rate, m);
            }
            
            OptimizerType::RMSprop { learning_rate, beta, epsilon } => {
                // RMSprop: v = beta * v + (1 - beta) * grad^2
                //          w = w - lr * grad / (sqrt(v) + epsilon)
                let v = self.v.as_mut().expect("RMSprop state not initialized");
                let one_minus_beta = 1.0 - beta;
                // In-place update of v
                ndarray::Zip::from(v.view_mut())
                    .and(gradient.view())
                    .for_each(|v_i, &g| {
                        *v_i = *v_i * *beta + g * g * one_minus_beta;
                    });
                // Update weights in-place
                ndarray::Zip::from(weights.view_mut())
                    .and(gradient.view())
                    .and(v.view())
                    .for_each(|w, &g, &v_i| {
                        *w -= *learning_rate * g / (v_i.sqrt() + *epsilon);
                    });
            }
            
            OptimizerType::Adam { learning_rate, beta1, beta2, epsilon } => {
                // Adam: m = beta1 * m + (1 - beta1) * grad
                //       v = beta2 * v + (1 - beta2) * grad^2
                //       m_hat = m / (1 - beta1^t)
                //       v_hat = v / (1 - beta2^t)
                //       w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
                
                let m = self.m.as_mut().expect("Adam m state not initialized");
                let v = self.v.as_mut().expect("Adam v state not initialized");
                
                let one_minus_beta1 = 1.0 - beta1;
                let one_minus_beta2 = 1.0 - beta2;
                
                // Bias correction factors
                let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
                let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
                
                // All updates in a single pass
                ndarray::Zip::from(weights.view_mut())
                    .and(m.view_mut())
                    .and(v.view_mut())
                    .and(gradient.view())
                    .for_each(|w, m_i, v_i, &g| {
                        // Update moments
                        *m_i = *m_i * *beta1 + g * one_minus_beta1;
                        *v_i = *v_i * *beta2 + g * g * one_minus_beta2;
                        // Bias-corrected moments
                        let m_hat = *m_i / bias_correction1;
                        let v_hat = *v_i / bias_correction2;
                        // Update weight
                        *w -= *learning_rate * m_hat / (v_hat.sqrt() + *epsilon);
                    });
            }
            
            OptimizerType::AdamW { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                // AdamW: Same as Adam but with decoupled weight decay
                //        w = w * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
                
                let m = self.m.as_mut().expect("AdamW m state not initialized");
                let v = self.v.as_mut().expect("AdamW v state not initialized");
                
                let one_minus_beta1 = 1.0 - beta1;
                let one_minus_beta2 = 1.0 - beta2;
                let weight_decay_factor = 1.0 - learning_rate * weight_decay;
                
                // Bias correction factors
                let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
                let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
                
                // All updates in a single pass
                ndarray::Zip::from(weights.view_mut())
                    .and(m.view_mut())
                    .and(v.view_mut())
                    .and(gradient.view())
                    .for_each(|w, m_i, v_i, &g| {
                        // Update moments
                        *m_i = *m_i * *beta1 + g * one_minus_beta1;
                        *v_i = *v_i * *beta2 + g * g * one_minus_beta2;
                        // Bias-corrected moments
                        let m_hat = *m_i / bias_correction1;
                        let v_hat = *v_i / bias_correction2;
                        // Weight decay + Adam update
                        *w = *w * weight_decay_factor - *learning_rate * m_hat / (v_hat.sqrt() + *epsilon);
                    });
            }
        }
    }
}

/// Optimizer state for a bias vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState1D {
    /// First moment (momentum)
    pub m: Option<Array1<f64>>,
    
    /// Second moment (variance)
    pub v: Option<Array1<f64>>,
    
    /// Number of iterations
    pub t: usize,
}

impl OptimizerState1D {
    /// Creates a new state for a vector of the given size.
    pub fn new(size: usize, optimizer: &OptimizerType) -> Self {
        let needs_m = matches!(optimizer, 
            OptimizerType::Momentum { .. } | 
            OptimizerType::Adam { .. } | 
            OptimizerType::AdamW { .. }
        );
        
        let needs_v = matches!(optimizer,
            OptimizerType::RMSprop { .. } |
            OptimizerType::Adam { .. } | 
            OptimizerType::AdamW { .. }
        );
        
        OptimizerState1D {
            m: if needs_m { Some(Array1::zeros(size)) } else { None },
            v: if needs_v { Some(Array1::zeros(size)) } else { None },
            t: 0,
        }
    }
    
    /// Updates biases with the computed gradient.
    pub fn step(
        &mut self,
        biases: &mut Array1<f64>,
        gradient: &Array1<f64>,
        optimizer: &OptimizerType,
    ) {
        self.t += 1;
        
        match optimizer {
            OptimizerType::SGD { learning_rate } => {
                biases.scaled_add(-*learning_rate, gradient);
            }
            
            OptimizerType::Momentum { learning_rate, beta } => {
                let m = self.m.as_mut().expect("Momentum state not initialized");
                m.mapv_inplace(|x| x * *beta);
                *m += gradient;
                biases.scaled_add(-*learning_rate, m);
            }
            
            OptimizerType::RMSprop { learning_rate, beta, epsilon } => {
                let v = self.v.as_mut().expect("RMSprop state not initialized");
                let one_minus_beta = 1.0 - beta;
                ndarray::Zip::from(biases.view_mut())
                    .and(v.view_mut())
                    .and(gradient.view())
                    .for_each(|b, v_i, &g| {
                        *v_i = *v_i * *beta + g * g * one_minus_beta;
                        *b -= *learning_rate * g / (v_i.sqrt() + *epsilon);
                    });
            }
            
            OptimizerType::Adam { learning_rate, beta1, beta2, epsilon } => {
                let m = self.m.as_mut().expect("Adam m state not initialized");
                let v = self.v.as_mut().expect("Adam v state not initialized");
                let one_minus_beta1 = 1.0 - beta1;
                let one_minus_beta2 = 1.0 - beta2;
                let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
                let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
                
                ndarray::Zip::from(biases.view_mut())
                    .and(m.view_mut())
                    .and(v.view_mut())
                    .and(gradient.view())
                    .for_each(|b, m_i, v_i, &g| {
                        *m_i = *m_i * *beta1 + g * one_minus_beta1;
                        *v_i = *v_i * *beta2 + g * g * one_minus_beta2;
                        let m_hat = *m_i / bias_correction1;
                        let v_hat = *v_i / bias_correction2;
                        *b -= *learning_rate * m_hat / (v_hat.sqrt() + *epsilon);
                    });
            }
            
            OptimizerType::AdamW { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                let m = self.m.as_mut().expect("AdamW m state not initialized");
                let v = self.v.as_mut().expect("AdamW v state not initialized");
                let one_minus_beta1 = 1.0 - beta1;
                let one_minus_beta2 = 1.0 - beta2;
                let weight_decay_factor = 1.0 - learning_rate * weight_decay;
                let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
                let bias_correction2 = 1.0 - beta2.powi(self.t as i32);
                
                ndarray::Zip::from(biases.view_mut())
                    .and(m.view_mut())
                    .and(v.view_mut())
                    .and(gradient.view())
                    .for_each(|b, m_i, v_i, &g| {
                        *m_i = *m_i * *beta1 + g * one_minus_beta1;
                        *v_i = *v_i * *beta2 + g * g * one_minus_beta2;
                        let m_hat = *m_i / bias_correction1;
                        let v_hat = *v_i / bias_correction2;
                        *b = *b * weight_decay_factor - *learning_rate * m_hat / (v_hat.sqrt() + *epsilon);
                    });
            }
        }
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
