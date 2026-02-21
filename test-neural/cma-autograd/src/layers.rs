//! # Stateless Layers
//!
//! Layers that have no trainable parameters.
//! They only implement forward via autograd operations.
//! Backward is handled automatically by the GradFn attached to the ops.

use crate::Float;
use crate::ops;
use crate::tensor::Tensor;
use ndarray::{s, ArrayD, IxDyn};

// ═══════════════════════════════════════════════════════════════════════════
// ReLU
// ═══════════════════════════════════════════════════════════════════════════

/// ReLU activation layer (stateless).
///
/// f(x) = max(0, x)
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        ops::relu(input)
    }

    pub fn name(&self) -> &'static str {
        "ReLU"
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sigmoid
// ═══════════════════════════════════════════════════════════════════════════

/// Sigmoid activation layer (stateless).
///
/// f(x) = 1 / (1 + exp(-x))
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        ops::sigmoid(input)
    }

    pub fn name(&self) -> &'static str {
        "Sigmoid"
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tanh
// ═══════════════════════════════════════════════════════════════════════════

/// Tanh activation layer (stateless).
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        ops::tanh_act(input)
    }

    pub fn name(&self) -> &'static str {
        "Tanh"
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten
// ═══════════════════════════════════════════════════════════════════════════

/// Flatten layer: reshapes [batch, ...] → [batch, flat_size].
///
/// Preserves the batch dimension and flattens the rest.
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        if shape.len() <= 2 {
            return input.clone(); // Already flat
        }
        let batch = shape[0];
        let flat_size: usize = shape[1..].iter().product();
        ops::reshape(input, &[batch, flat_size])
    }

    pub fn name(&self) -> &'static str {
        "Flatten"
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MaxPool2D
// ═══════════════════════════════════════════════════════════════════════════

/// Max Pooling 2D layer with autograd backward support.
///
/// Reduces spatial dimensions by taking the max value in each window.
/// Stores argmax indices for proper gradient flow during backward.
pub struct MaxPool2D {
    kernel_size: usize,
    stride: usize,
}

impl MaxPool2D {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self {
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Delegate forward computation to cma-cnn's optimized MaxPool2D
        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        assert_eq!(shape.len(), 4, "MaxPool2D expects 4D input [N, C, H, W]");

        let result = crate::cnn_ops::maxpool2d_optimized(
            &input_data,
            self.kernel_size,
            self.stride,
        );

        use crate::grad_fn::MaxPool2DBackward;
        use crate::tensor::is_grad_enabled;
        use std::sync::Arc;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(MaxPool2DBackward {
                input: input.clone(),
                input_shape: shape,
                max_indices_h: result.max_indices_h,
                max_indices_w: result.max_indices_w,
                kernel_size: self.kernel_size,
                stride: self.stride,
            });
            Tensor::from_op(result.output, grad_fn)
        } else {
            Tensor::new(result.output, false)
        }
    }

    pub fn name(&self) -> &'static str {
        "MaxPool2D"
    }

    /// Kernel size.
    pub fn kernel_size(&self) -> usize { self.kernel_size }
    /// Stride.
    pub fn stride(&self) -> usize { self.stride }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dropout
// ═══════════════════════════════════════════════════════════════════════════

/// Dropout layer: randomly zeroes elements during training.
///
/// During training, each element is zeroed with probability `p`
/// and remaining elements are scaled by `1 / (1 - p)`.
///
/// During evaluation, this is a no-op (identity).
pub struct Dropout {
    p: Float,
    training: bool,
}

impl Dropout {
    pub fn new(p: Float) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1)"
        );
        Self { p, training: true }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return input.clone();
        }

        use rand::Rng;
        let mut rng = rand::rng();
        let data = input.data();
        let scale = 1.0 / (1.0 - self.p);
        let mask = data.mapv(|_| {
            if rng.random::<Float>() >= self.p {
                scale
            } else {
                0.0
            }
        });

        let mask_tensor = Tensor::new(mask, false);
        ops::mul(input, &mask_tensor)
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn name(&self) -> &'static str {
        "Dropout"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Softmax (along last axis)
// ═══════════════════════════════════════════════════════════════════════════

/// Softmax layer (stateless).
///
/// Applies softmax along the last axis: softmax(x_i) = exp(x_i) / Σ exp(x_j)
pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Numerically stable softmax: subtract max before exp
        let data = input.data();
        let shape = data.shape().to_vec();
        let ndim = shape.len();

        if ndim == 1 {
            let max_val: Float = data.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
            let shifted = data.mapv(|x| (x - max_val).exp());
            let sum: Float = shifted.iter().sum();
            let result = shifted.mapv(|x| x / sum);
            // For autograd tracking of softmax, we'd need a dedicated SoftmaxBackward.
            // For now, return as a non-tracked tensor.
            Tensor::new(result, false)
        } else if ndim == 2 {
            // Softmax along axis 1 (last axis for 2D)
            let rows = shape[0];
            let cols = shape[1];
            let mut result = data.clone();
            for i in 0..rows {
                let row_slice = data.slice(ndarray::s![i, ..]);
                let max_val: Float = row_slice
                    .iter()
                    .cloned()
                    .fold(Float::NEG_INFINITY, Float::max);
                let mut sum: Float = 0.0;
                for j in 0..cols {
                    let val = (data[[i, j]] - max_val).exp();
                    result[[i, j]] = val;
                    sum += val;
                }
                for j in 0..cols {
                    result[[i, j]] /= sum;
                }
            }
            Tensor::new(result, false)
        } else {
            panic!("Softmax only supports 1D and 2D tensors for now");
        }
    }

    pub fn name(&self) -> &'static str {
        "Softmax"
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BatchNorm2D
// ═══════════════════════════════════════════════════════════════════════════

/// Batch Normalization 2D layer with learnable parameters.
///
/// Normalizes each channel across the batch and spatial dimensions:
/// `y = gamma * (x - mean) / sqrt(var + eps) + beta`
///
/// During training, uses batch statistics and updates running stats.
/// During eval, uses running statistics.
pub struct BatchNorm2D {
    /// Learnable scale parameter [C]
    gamma: crate::module::Parameter,
    /// Learnable shift parameter [C]
    beta: crate::module::Parameter,
    /// Running mean for eval mode [C]
    running_mean: std::cell::RefCell<Vec<Float>>,
    /// Running variance for eval mode [C]
    running_var: std::cell::RefCell<Vec<Float>>,
    /// Epsilon for numerical stability
    epsilon: Float,
    /// Momentum for running stats EMA
    momentum: Float,
    /// Number of channels
    num_channels: usize,
    /// Whether in training mode
    training: bool,
}

impl BatchNorm2D {
    pub fn new(num_channels: usize) -> Self {
        Self {
            gamma: crate::module::Parameter::new(
                Tensor::from_vec(vec![1.0; num_channels], &[num_channels], true),
            ),
            beta: crate::module::Parameter::new(
                Tensor::zeros(&[num_channels], true),
            ),
            running_mean: std::cell::RefCell::new(vec![0.0; num_channels]),
            running_var: std::cell::RefCell::new(vec![1.0; num_channels]),
            epsilon: 1e-5,
            momentum: 0.1,
            num_channels,
            training: true,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let data = input.data();
        let shape = data.shape().to_vec();
        assert_eq!(shape.len(), 4, "BatchNorm2D expects 4D input [N, C, H, W]");
        let (batch, channels, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(channels, self.num_channels);

        let mut output = ArrayD::zeros(IxDyn(&shape));

        if self.training {
            let mut x_hat = ArrayD::zeros(IxDyn(&shape));
            let mut std_inv_vec = vec![0.0; channels];

            for c in 0..channels {
                let ch = data.slice(s![.., c, .., ..]);
                let mean = ch.mean().expect("empty channel in BatchNorm2D");
                let var = ch.mapv(|x| (x - mean).powi(2)).mean().expect("empty channel");
                let std_inv = 1.0 / (var + self.epsilon).sqrt();
                std_inv_vec[c] = std_inv;

                let gamma_c = self.gamma.tensor().data_ref(|d| d[[c]]);
                let beta_c = self.beta.tensor().data_ref(|d| d[[c]]);

                let x_hat_c = ch.mapv(|x| (x - mean) * std_inv);
                x_hat.slice_mut(s![.., c, .., ..]).assign(&x_hat_c);
                output
                    .slice_mut(s![.., c, .., ..])
                    .assign(&x_hat_c.mapv(|v| v * gamma_c + beta_c));

                let mut rm = self.running_mean.borrow_mut();
                let mut rv = self.running_var.borrow_mut();
                rm[c] = (1.0 - self.momentum) * rm[c] + self.momentum * mean;
                rv[c] = (1.0 - self.momentum) * rv[c] + self.momentum * var;
                drop(rm);
                drop(rv);
            }

            // Attach backward
            use crate::grad_fn::BatchNorm2DBackward;
            use crate::tensor::is_grad_enabled;
            use std::sync::Arc;

            if is_grad_enabled() && input.requires_grad() {
                let grad_fn = Arc::new(BatchNorm2DBackward {
                    input: input.clone(),
                    gamma: self.gamma.tensor().clone(),
                    beta: self.beta.tensor().clone(),
                    x_hat,
                    std_inv: std_inv_vec,
                    batch_size: batch,
                    spatial_size: h * w,
                });
                Tensor::from_op(output, grad_fn)
            } else {
                Tensor::new(output, false)
            }
        } else {
            // Eval mode: use running statistics
            let rm = self.running_mean.borrow();
            let rv = self.running_var.borrow();

            for c in 0..channels {
                let mean = rm[c];
                let var = rv[c];
                let std_inv = 1.0 / (var + self.epsilon).sqrt();
                let gamma_c = self.gamma.tensor().data_ref(|d| d[[c]]);
                let beta_c = self.beta.tensor().data_ref(|d| d[[c]]);

                let out_c = data
                    .slice(s![.., c, .., ..])
                    .mapv(|x| gamma_c * (x - mean) * std_inv + beta_c);
                output.slice_mut(s![.., c, .., ..]).assign(&out_c);
            }

            Tensor::new(output, false)
        }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn parameters(&self) -> Vec<&crate::module::Parameter> {
        vec![&self.gamma, &self.beta]
    }

    pub fn name(&self) -> &'static str {
        "BatchNorm2D"
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize { self.num_channels }
    /// Access gamma parameter.
    pub fn gamma(&self) -> &crate::module::Parameter { &self.gamma }
    /// Access beta parameter.
    pub fn beta(&self) -> &crate::module::Parameter { &self.beta }
    /// Get running mean (clone).
    pub fn running_mean(&self) -> Vec<Float> { self.running_mean.borrow().clone() }
    /// Get running variance (clone).
    pub fn running_var(&self) -> Vec<Float> { self.running_var.borrow().clone() }
    /// Epsilon value.
    pub fn epsilon(&self) -> Float { self.epsilon }
    /// Momentum value.
    pub fn momentum(&self) -> Float { self.momentum }
}

// ═══════════════════════════════════════════════════════════════════════════
// AvgPool2D
// ═══════════════════════════════════════════════════════════════════════════

/// Average Pooling 2D layer.
///
/// ∂L/∂X = (1/k²) * ∂L/∂Y (uniform distribution).
pub struct AvgPool2D {
    kernel_size: usize,
    stride: usize,
}

impl AvgPool2D {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size, stride }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Delegate forward computation to cma-cnn's optimized AvgPool2D
        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        assert_eq!(shape.len(), 4, "AvgPool2D expects 4D input [N, C, H, W]");

        let output = crate::cnn_ops::avgpool2d_optimized(
            &input_data,
            self.kernel_size,
            self.stride,
        );

        use crate::tensor::is_grad_enabled;
        use std::sync::Arc;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(AvgPool2DBackward {
                input: input.clone(),
                input_shape: shape,
                kernel_size: self.kernel_size,
                stride: self.stride,
            });
            Tensor::from_op(output, grad_fn)
        } else {
            Tensor::new(output, false)
        }
    }

    pub fn name(&self) -> &'static str {
        "AvgPool2D"
    }

    /// Kernel size.
    pub fn kernel_size(&self) -> usize { self.kernel_size }
    /// Stride.
    pub fn stride(&self) -> usize { self.stride }
}

/// AvgPool2D backward: distributes gradient uniformly.
#[derive(Debug)]
struct AvgPool2DBackward {
    input: Tensor,
    input_shape: Vec<usize>,
    kernel_size: usize,
    stride: usize,
}

impl crate::grad_fn::GradFn for AvgPool2DBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let (batch, channels, _h, _w) = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
        );
        let go_shape = grad_output.shape();
        let out_h = go_shape[2];
        let out_w = go_shape[3];
        let k2 = (self.kernel_size * self.kernel_size) as Float;

        let mut grad_input = ArrayD::zeros(IxDyn(&self.input_shape));

        for b in 0..batch {
            for c in 0..channels {
                for i in 0..out_h {
                    for j in 0..out_w {
                        let g = grad_output[[b, c, i, j]] / k2;
                        for ki in 0..self.kernel_size {
                            for kj in 0..self.kernel_size {
                                grad_input[[b, c, i * self.stride + ki, j * self.stride + kj]] += g;
                            }
                        }
                    }
                }
            }
        }

        vec![grad_input]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "AvgPool2DBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GlobalAvgPool2D
// ═══════════════════════════════════════════════════════════════════════════

/// Global Average Pooling 2D: [N, C, H, W] → [N, C, 1, 1]
pub struct GlobalAvgPool2D;

impl GlobalAvgPool2D {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Delegate forward computation to cma-cnn's optimized GlobalAvgPool2D
        let input_data = input.data();
        let shape = input_data.shape().to_vec();
        assert_eq!(shape.len(), 4, "GlobalAvgPool2D expects 4D input");

        let output = crate::cnn_ops::global_avgpool2d_optimized(&input_data);

        use crate::tensor::is_grad_enabled;
        use std::sync::Arc;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(GlobalAvgPool2DBackward {
                input: input.clone(),
                input_shape: shape,
            });
            Tensor::from_op(output, grad_fn)
        } else {
            Tensor::new(output, false)
        }
    }

    pub fn name(&self) -> &'static str {
        "GlobalAvgPool2D"
    }
}

impl Default for GlobalAvgPool2D {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct GlobalAvgPool2DBackward {
    input: Tensor,
    input_shape: Vec<usize>,
}

impl crate::grad_fn::GradFn for GlobalAvgPool2DBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let (batch, channels, h, w) = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
        );
        let hw = (h * w) as Float;

        let mut grad_input = ArrayD::zeros(IxDyn(&self.input_shape));

        for b in 0..batch {
            for c in 0..channels {
                let g = grad_output[[b, c, 0, 0]] / hw;
                for i in 0..h {
                    for j in 0..w {
                        grad_input[[b, c, i, j]] = g;
                    }
                }
            }
        }

        vec![grad_input]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "GlobalAvgPool2DBackward"
    }
}
