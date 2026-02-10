//! # Optimizers
//!
//! SGD (with momentum) and Adam for parameter updates.

use crate::Float;
use crate::module::Parameter;
use ndarray::ArrayD;

// ═══════════════════════════════════════════════════════════════════════════
// Optimizer trait
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for all optimizers.
pub trait Optimizer {
    /// Perform one optimization step (update parameters using their gradients).
    fn step(&mut self);

    /// Reset all gradients to zero.
    fn zero_grad(&mut self);
}

// ═══════════════════════════════════════════════════════════════════════════
// SGD
// ═══════════════════════════════════════════════════════════════════════════

/// Stochastic Gradient Descent with optional momentum.
pub struct SGD {
    params: Vec<Parameter>,
    lr: Float,
    momentum: Float,
    velocities: Vec<Option<ArrayD<Float>>>,
}

impl SGD {
    /// Create a new SGD optimizer.
    pub fn new(params: Vec<Parameter>, lr: Float) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            velocities: vec![None; n],
        }
    }

    /// Create SGD with momentum.
    pub fn with_momentum(params: Vec<Parameter>, lr: Float, momentum: Float) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum,
            velocities: vec![None; n],
        }
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: Float) {
        self.lr = lr;
    }

    /// Get current learning rate.
    pub fn lr(&self) -> Float {
        self.lr
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        let lr = self.lr;
        let momentum = self.momentum;

        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                if momentum > 0.0 {
                    let v = self.velocities[i]
                        .get_or_insert_with(|| ArrayD::zeros(grad.raw_dim()));
                    // v = momentum * v + grad
                    *v = &*v * momentum + &grad;
                    // param -= lr * v
                    let update = &*v * lr;
                    param.update_data(|data| {
                        *data -= &update;
                    });
                } else {
                    let update = &grad * lr;
                    param.update_data(|data| {
                        *data -= &update;
                    });
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Adam
// ═══════════════════════════════════════════════════════════════════════════

/// Adam optimizer (Adaptive Moment Estimation).
///
/// ```text
/// m = β₁ * m + (1 - β₁) * g
/// v = β₂ * v + (1 - β₂) * g²
/// m̂ = m / (1 - β₁ᵗ)
/// v̂ = v / (1 - β₂ᵗ)
/// θ = θ - lr * m̂ / (√v̂ + ε)
/// ```
pub struct Adam {
    params: Vec<Parameter>,
    lr: Float,
    beta1: Float,
    beta2: Float,
    epsilon: Float,
    weight_decay: Float,
    m: Vec<Option<ArrayD<Float>>>,
    v: Vec<Option<ArrayD<Float>>>,
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer with default hyperparameters.
    pub fn new(params: Vec<Parameter>, lr: Float) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: vec![None; n],
            v: vec![None; n],
            t: 0,
        }
    }

    /// Builder: set beta1.
    pub fn beta1(mut self, beta1: Float) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Builder: set beta2.
    pub fn beta2(mut self, beta2: Float) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Builder: set epsilon.
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Builder: set weight decay (AdamW).
    pub fn weight_decay(mut self, wd: Float) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: Float) {
        self.lr = lr;
    }

    /// Get current learning rate.
    pub fn lr(&self) -> Float {
        self.lr
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        let t = self.t;
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;
        let weight_decay = self.weight_decay;

        let bias_correction1 = 1.0 - beta1.powi(t as i32);
        let bias_correction2 = 1.0 - beta2.powi(t as i32);

        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                // AdamW: weight decay applied to params directly
                if weight_decay > 0.0 {
                    param.update_data(|data| {
                        *data *= 1.0 - lr * weight_decay;
                    });
                }

                // First moment: m = β₁ * m + (1 - β₁) * g
                let m = self.m[i]
                    .get_or_insert_with(|| ArrayD::zeros(grad.raw_dim()));
                *m = &*m * beta1 + &grad * (1.0 - beta1);

                // Second moment: v = β₂ * v + (1 - β₂) * g²
                let v = self.v[i]
                    .get_or_insert_with(|| ArrayD::zeros(grad.raw_dim()));
                let grad_sq = &grad * &grad;
                *v = &*v * beta2 + &grad_sq * (1.0 - beta2);

                // Bias-corrected estimates
                let m_hat = &*m / bias_correction1;
                let v_hat = &*v / bias_correction2;

                // θ = θ - lr * m̂ / (√v̂ + ε)
                let update = &m_hat / &v_hat.mapv(|x| x.sqrt() + epsilon);
                let scaled_update = &update * lr;
                param.update_data(|data| {
                    *data -= &scaled_update;
                });
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }
}
