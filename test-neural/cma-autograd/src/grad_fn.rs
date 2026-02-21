//! # Gradient Functions
//!
//! `GradFn` trait and implementations for each operation.
//! Each `GradFn` knows how to:
//! - Compute the local gradient (backward)
//! - Reference its input tensors (for graph traversal)

use crate::Float;
use crate::tensor::Tensor;
use ndarray::{ArrayD, Axis, IxDyn, Zip};
use std::fmt;

/// Trait for backward functions attached to tensor operations.
///
/// Each operation (add, mul, matmul, relu, ...) implements this trait
/// to define how gradients flow backward through it.
pub trait GradFn: Send + Sync + fmt::Debug {
    /// Compute gradients of inputs given the gradient of the output.
    ///
    /// `grad_output` is ∂L/∂output. Returns gradients for each input tensor.
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>>;

    /// Returns the input tensors that feed into this operation.
    fn inputs(&self) -> Vec<Tensor>;

    /// Name of this gradient function (for debugging).
    fn name(&self) -> &'static str;
}

// ═══════════════════════════════════════════════════════════════════════════
// AddBackward: c = a + b
// ∂L/∂a = ∂L/∂c (broadcast sum if needed)
// ∂L/∂b = ∂L/∂c (broadcast sum if needed)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct AddBackward {
    pub a: Tensor,
    pub b: Tensor,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let grad_a = unbroadcast(grad_output, &self.a_shape);
        let grad_b = unbroadcast(grad_output, &self.b_shape);
        vec![grad_a, grad_b]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SubBackward: c = a - b
// ∂L/∂a = ∂L/∂c
// ∂L/∂b = -∂L/∂c
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct SubBackward {
    pub a: Tensor,
    pub b: Tensor,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for SubBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let grad_a = unbroadcast(grad_output, &self.a_shape);
        let grad_b = unbroadcast(&(-grad_output), &self.b_shape);
        vec![grad_a, grad_b]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MulBackward: c = a * b (element-wise)
// ∂L/∂a = ∂L/∂c * b
// ∂L/∂b = ∂L/∂c * a
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct MulBackward {
    pub a: Tensor,
    pub b: Tensor,
    pub a_data: ArrayD<Float>,
    pub b_data: ArrayD<Float>,
    pub a_shape: Vec<usize>,
    pub b_shape: Vec<usize>,
}

impl GradFn for MulBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let grad_a = unbroadcast(&(grad_output * &self.b_data), &self.a_shape);
        let grad_b = unbroadcast(&(grad_output * &self.a_data), &self.b_shape);
        vec![grad_a, grad_b]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MulScalarBackward: c = a * scalar
// ∂L/∂a = ∂L/∂c * scalar
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct MulScalarBackward {
    pub a: Tensor,
    pub scalar: Float,
}

impl GradFn for MulScalarBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        vec![grad_output * self.scalar]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "MulScalarBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// NegBackward: c = -a
// ∂L/∂a = -∂L/∂c
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct NegBackward {
    pub a: Tensor,
}

impl GradFn for NegBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        vec![-grad_output]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MatMulBackward: c = a @ b  (2D matrix multiply)
// ∂L/∂a = ∂L/∂c @ bᵀ
// ∂L/∂b = aᵀ @ ∂L/∂c
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct MatMulBackward {
    pub a: Tensor,
    pub b: Tensor,
    pub a_data: ArrayD<Float>,
    pub b_data: ArrayD<Float>,
}

impl GradFn for MatMulBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        // grad_output: [M, P], a: [M, N], b: [N, P]
        let grad_2d = grad_output
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("MatMulBackward: grad_output must be 2D");
        let a_2d = self
            .a_data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("MatMulBackward: a must be 2D");
        let b_2d = self
            .b_data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("MatMulBackward: b must be 2D");

        // ∂L/∂a = grad_output @ bᵀ  → [M, N]
        let grad_a = grad_2d.dot(&b_2d.t()).into_dyn();
        // ∂L/∂b = aᵀ @ grad_output   → [N, P]
        let grad_b = a_2d.t().dot(&grad_2d).into_dyn();

        vec![grad_a, grad_b]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn name(&self) -> &'static str {
        "MatMulBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SumBackward: c = sum(a)
// ∂L/∂a = ∂L/∂c * ones_like(a)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct SumBackward {
    pub a: Tensor,
    pub a_shape: Vec<usize>,
}

impl GradFn for SumBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        // grad_output is a scalar, broadcast to a_shape
        let scalar = grad_output.iter().next().copied().unwrap_or(1.0);
        vec![ArrayD::from_elem(IxDyn(&self.a_shape), scalar)]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SumAxisBackward: c = sum(a, axis)
// ∂L/∂a = broadcast grad_output back to a_shape
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct SumAxisBackward {
    pub a: Tensor,
    pub a_shape: Vec<usize>,
    pub axis: usize,
}

impl GradFn for SumAxisBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        // Insert axis back and broadcast
        let mut expanded_shape = grad_output.shape().to_vec();
        expanded_shape.insert(self.axis, 1);
        let expanded = grad_output.clone().into_shape_with_order(IxDyn(&expanded_shape)).unwrap();

        // Broadcast to original shape
        let result = expanded.broadcast(IxDyn(&self.a_shape)).unwrap().to_owned();
        vec![result]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "SumAxisBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MeanBackward: c = mean(a) = sum(a) / N
// ∂L/∂a = ∂L/∂c / N
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct MeanBackward {
    pub a: Tensor,
    pub a_shape: Vec<usize>,
    pub numel: usize,
}

impl GradFn for MeanBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let scalar = grad_output.iter().next().copied().unwrap_or(1.0);
        let n = self.numel as Float;
        vec![ArrayD::from_elem(IxDyn(&self.a_shape), scalar / n)]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PowfBackward: c = a^p
// ∂L/∂a = ∂L/∂c * p * a^(p-1)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct PowfBackward {
    pub a: Tensor,
    pub a_data: ArrayD<Float>,
    pub exponent: Float,
}

impl GradFn for PowfBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let exp = self.exponent;
        // Fused Zip: avoids allocating a_data.mapv(|x| x.powf(exp-1)) + separate multiply
        let grad = Zip::from(grad_output).and(&self.a_data).map_collect(|&g, &a| g * exp * a.powf(exp - 1.0));
        vec![grad]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "PowfBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// LogBackward: c = ln(a)
// ∂L/∂a = ∂L/∂c / a
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct LogBackward {
    pub a: Tensor,
    pub a_data: ArrayD<Float>,
}

impl GradFn for LogBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        vec![grad_output / &self.a_data]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ExpBackward: c = exp(a)
// ∂L/∂a = ∂L/∂c * exp(a) = ∂L/∂c * c
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct ExpBackward {
    pub a: Tensor,
    pub output_data: ArrayD<Float>,
}

impl GradFn for ExpBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        vec![grad_output * &self.output_data]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ReLUBackward: c = max(0, a)
// ∂L/∂a = ∂L/∂c * (a > 0)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct ReLUBackward {
    pub a: Tensor,
    pub a_data: ArrayD<Float>,
}

impl GradFn for ReLUBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        // Fused Zip: avoids float mask array + separate multiply (2 allocs → 1)
        let grad = Zip::from(grad_output).and(&self.a_data).map_collect(|&g, &x| if x > 0.0 { g } else { 0.0 });
        vec![grad]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "ReLUBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SigmoidBackward: c = σ(a)
// ∂L/∂a = ∂L/∂c * σ(a) * (1 - σ(a)) = ∂L/∂c * c * (1 - c)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct SigmoidBackward {
    pub a: Tensor,
    pub output_data: ArrayD<Float>,
}

impl GradFn for SigmoidBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let s = &self.output_data;
        let grad = grad_output * s * &(1.0 - s);
        vec![grad]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TanhBackward: c = tanh(a)
// ∂L/∂a = ∂L/∂c * (1 - tanh²(a)) = ∂L/∂c * (1 - c²)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct TanhBackward {
    pub a: Tensor,
    pub output_data: ArrayD<Float>,
}

impl GradFn for TanhBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let t = &self.output_data;
        let grad = grad_output * &(1.0 - t * t);
        vec![grad]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SoftmaxBackward: c = softmax(a)
// ∂L/∂x_i = y_i * (∂L/∂y_i - Σ_k(∂L/∂y_k * y_k))
//         = y ⊙ (grad_out - dot(grad_out, y))  [1D case]
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct SoftmaxBackward {
    pub a: Tensor,
    /// The softmax output y = softmax(x) — needed for the Jacobian.
    pub output_data: ArrayD<Float>,
}

impl GradFn for SoftmaxBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let y = &self.output_data;
        let grad = if y.ndim() == 1 {
            // grad = y ⊙ (grad_out - dot(grad_out, y))
            let dot: Float = (grad_output * y).sum();
            y * &(grad_output.mapv(|x| x - dot))
        } else {
            // 2D: compute row-wise dot products then subtract and scale
            let rows = y.shape()[0];
            let mut grad = ArrayD::zeros(y.raw_dim());
            for b in 0..rows {
                let yo = y.slice(ndarray::s![b, ..]);
                let go = grad_output.slice(ndarray::s![b, ..]);
                let dot: Float = (&yo * &go).sum();
                // Fused Zip: avoids go.mapv(|x| x - dot) + assign per row (2 allocs → 1 direct write)
                Zip::from(grad.slice_mut(ndarray::s![b, ..]))
                    .and(&yo)
                    .and(&go)
                    .for_each(|g, &y_val, &go_val| *g = y_val * (go_val - dot));
            }
            grad
        };
        vec![grad]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ClampBackward: c = clamp(a, min, max)
// ∂L/∂a = ∂L/∂c * (min < a < max) — gradient passes only where not clamped
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct ClampBackward {
    pub a: Tensor,
    pub a_data: ArrayD<Float>,
    pub min_val: Float,
    pub max_val: Float,
}

impl GradFn for ClampBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let (min, max) = (self.min_val, self.max_val);
        // Fused Zip: avoids float mask array + separate multiply (2 allocs → 1)
        let grad = Zip::from(grad_output).and(&self.a_data).map_collect(|&g, &x| {
            if x > min && x < max { g } else { 0.0 }
        });
        vec![grad]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "ClampBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ReshapeBackward: restore original shape
// ∂L/∂a = reshape(∂L/∂c, original_shape)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct ReshapeBackward {
    pub a: Tensor,
    pub original_shape: Vec<usize>,
}

impl GradFn for ReshapeBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let grad = grad_output
            .clone()
            .into_shape_with_order(IxDyn(&self.original_shape))
            .expect("ReshapeBackward: shape mismatch");
        vec![grad]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TransposeBackward: c = aᵀ
// ∂L/∂a = (∂L/∂c)ᵀ
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct TransposeBackward {
    pub a: Tensor,
}

impl GradFn for TransposeBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        let grad_2d = grad_output
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("TransposeBackward: must be 2D");
        vec![grad_2d.t().to_owned().into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.a.clone()]
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility: reduce grad to match original shape after broadcasting
// ═══════════════════════════════════════════════════════════════════════════

/// Reduce gradient dimensions that were broadcast.
///
/// When ndarray broadcasts [3] + [2, 3] → [2, 3], the gradient for the [3] tensor
/// needs to be summed along axis 0 to get back to shape [3].
fn unbroadcast(grad: &ArrayD<Float>, target_shape: &[usize]) -> ArrayD<Float> {
    let grad_shape = grad.shape();
    if grad_shape == target_shape {
        return grad.clone();
    }

    let mut result = grad.clone();

    // If target has fewer dims, sum leading axes
    while result.ndim() > target_shape.len() {
        result = result.sum_axis(Axis(0));
    }

    // Sum axes where target has size 1 but grad doesn't
    for (i, (&gs, &ts)) in result.shape().to_vec().iter().zip(target_shape.iter()).enumerate() {
        if ts == 1 && gs != 1 {
            result = result.sum_axis(Axis(i));
            // Re-insert the axis with size 1
            let mut new_shape = result.shape().to_vec();
            new_shape.insert(i, 1);
            result = result.into_shape_with_order(IxDyn(&new_shape)).unwrap();
        }
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════
// Conv2DBackward: Y = conv2d(X, W) + b
//
// Uses im2col representation:
//   col = im2col(X)           → [N*OH*OW, C_in*kH*kW]
//   W_2d = reshape(W)         → [C_out, C_in*kH*kW]
//   Y_2d = col @ W_2d^T       → [N*OH*OW, C_out]
//
// ∂L/∂W_2d = col^T @ ∂L/∂Y_2d                     → [C_in*kH*kW, C_out] → transpose → [C_out, C_in*kH*kW]
// ∂L/∂col = ∂L/∂Y_2d @ W_2d                        → [N*OH*OW, C_in*kH*kW]
// ∂L/∂X = col2im(∂L/∂col)                           → [N, C_in, H, W]
// ∂L/∂b = sum(∂L/∂Y_2d, axis=0)                     → [C_out]
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct Conv2DBackward {
    pub input: Tensor,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    /// im2col of input: [N*OH*OW, C_in*kH*kW]
    pub col_data: ArrayD<Float>,
    /// Weight reshaped to 2D: [C_out, C_in*kH*kW]
    pub weight_2d_data: ArrayD<Float>,
    pub input_shape: Vec<usize>,   // [N, C_in, H, W]
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl GradFn for Conv2DBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        // grad_output: [N, C_out, OH, OW] in BCHW format
        let go_shape = grad_output.shape();
        let batch = go_shape[0];
        let out_channels = go_shape[1];
        let out_h = go_shape[2];
        let out_w = go_shape[3];

        // Reshape grad_output: BCHW → [N*OH*OW, C_out]
        // BCHW→BHWC via permuted_axes: replaces 4-level scalar scatter loop
        let rows = batch * out_h * out_w;
        let go_bhwc = grad_output
            .view()
            .into_dimensionality::<ndarray::Ix4>()
            .unwrap()
            .permuted_axes([0usize, 2, 3, 1])
            .as_standard_layout()
            .into_owned()
            .into_dyn();
        let go_2d = go_bhwc
            .into_shape_with_order(IxDyn(&[rows, out_channels]))
            .unwrap();

        let go_2d_2 = go_2d.view().into_dimensionality::<ndarray::Ix2>().unwrap();

        // ∂L/∂W_2d = col^T @ go_2d → [C_in*kH*kW, C_out]
        let col_2d = self.col_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let grad_w_2d = col_2d.t().dot(&go_2d_2); // [C_in*kH*kW, C_out]
        // Transpose to [C_out, C_in*kH*kW] then reshape to 4D weight shape
        // as_standard_layout().into_owned() avoids the Vec<Float> roundtrip (collect + from_shape_vec)
        let c_in = self.input_shape[1];
        let grad_w_4d = grad_w_2d
            .t()
            .as_standard_layout()
            .into_owned()
            .into_dyn()
            .into_shape_with_order(IxDyn(&[out_channels, c_in, self.kernel_size, self.kernel_size]))
            .unwrap();

        // ∂L/∂col = go_2d @ W_2d → [N*OH*OW, C_in*kH*kW]
        let w_2d = self.weight_2d_data.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let grad_col = go_2d_2.dot(&w_2d).into_dyn(); // [N*OH*OW, C_in*kH*kW]

        // ∂L/∂X = col2im(grad_col)
        let (n, c_in, h, w) = (
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
        );
        let grad_input = col2im(
            &grad_col, n, c_in, h, w,
            self.kernel_size, self.stride, self.padding,
        );

        // ∂L/∂b = sum(go_2d, axis=0) → [C_out]
        if self.bias.is_some() {
            let grad_bias = go_2d.sum_axis(Axis(0));
            // Return: grad_input, grad_weight (4D), grad_bias
            vec![grad_input, grad_w_4d, grad_bias]
        } else {
            vec![grad_input, grad_w_4d]
        }
    }

    fn inputs(&self) -> Vec<Tensor> {
        let mut inputs = vec![self.input.clone(), self.weight.clone()];
        if let Some(ref bias) = self.bias {
            inputs.push(bias.clone());
        }
        inputs
    }

    fn name(&self) -> &'static str {
        "Conv2DBackward"
    }
}

/// col2im: inverse of im2col — scatters columns back to image format.
///
/// Converts grad_col [N*OH*OW, C_in*kH*kW] back to [N, C_in, H, W].
#[allow(clippy::too_many_arguments)]
fn col2im(
    grad_col: &ArrayD<Float>,
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> ArrayD<Float> {
    let out_h = (h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (w + 2 * padding - kernel_size) / stride + 1;

    let padded_h = h + 2 * padding;
    let padded_w = w + 2 * padding;

    let mut padded = ArrayD::zeros(IxDyn(&[batch, channels, padded_h, padded_w]));

    for b in 0..batch {
        for i in 0..out_h {
            for j in 0..out_w {
                let row = b * out_h * out_w + i * out_w + j;
                let mut col_idx = 0;
                for c in 0..channels {
                    for ki in 0..kernel_size {
                        for kj in 0..kernel_size {
                            let hi = i * stride + ki;
                            let wi = j * stride + kj;
                            padded[[b, c, hi, wi]] += grad_col[[row, col_idx]];
                            col_idx += 1;
                        }
                    }
                }
            }
        }
    }

    // Remove padding via slice: avoids 4-level scalar copy loop
    if padding > 0 {
        padded
            .slice(ndarray::s![.., .., padding..h + padding, padding..w + padding])
            .to_owned()
            .into_dyn()
    } else {
        padded
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MaxPool2DBackward: Y = maxpool2d(X)
// ∂L/∂X[b,c,h',w'] = ∂L/∂Y[b,c,oh,ow] if (h',w') was the argmax, else 0
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct MaxPool2DBackward {
    pub input: Tensor,
    pub input_shape: Vec<usize>,  // [N, C, H, W]
    /// Flattened argmax indices: [N, C, OH, OW] storing the linear index within each pool window
    pub max_indices_h: ArrayD<usize>,
    pub max_indices_w: ArrayD<usize>,
    pub kernel_size: usize,
    pub stride: usize,
}

impl GradFn for MaxPool2DBackward {
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

        let mut grad_input = ArrayD::zeros(IxDyn(&self.input_shape));

        for b in 0..batch {
            for c in 0..channels {
                for i in 0..out_h {
                    for j in 0..out_w {
                        let max_h = self.max_indices_h[[b, c, i, j]];
                        let max_w = self.max_indices_w[[b, c, i, j]];
                        grad_input[[b, c, max_h, max_w]] += grad_output[[b, c, i, j]];
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
        "MaxPool2DBackward"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BatchNorm2DBackward: Y = gamma * (X - mean) / sqrt(var + eps) + beta
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct BatchNorm2DBackward {
    pub input: Tensor,
    pub gamma: Tensor,
    pub beta: Tensor,
    /// Normalized input: (X - mean) / std
    pub x_hat: ArrayD<Float>,
    /// Inverse standard deviation: 1/sqrt(var + eps) per channel [C]
    pub std_inv: Vec<Float>,
    pub batch_size: usize,
    pub spatial_size: usize,  // H * W
}

impl GradFn for BatchNorm2DBackward {
    fn backward(&self, grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        // grad_output: [N, C, H, W]
        let shape = grad_output.shape();
        let (batch, channels, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let m = (batch * h * w) as Float; // number of elements per channel

        let gamma_data = self.gamma.data_ref(|d| d.clone());

        // ∂L/∂gamma = sum over (N,H,W) of ∂L/∂Y * x_hat
        let mut grad_gamma = ArrayD::zeros(IxDyn(&[channels]));
        // ∂L/∂beta = sum over (N,H,W) of ∂L/∂Y
        let mut grad_beta = ArrayD::zeros(IxDyn(&[channels]));

        // Single channel loop: compute sums + grad_gamma/beta in pass 1, grad_input in pass 2.
        // Previously the sums were computed twice in two separate outer channel loops.
        let mut grad_input = ArrayD::zeros(IxDyn(&[batch, channels, h, w]));

        for c in 0..channels {
            let g = gamma_data[[c]];
            let si = self.std_inv[c];

            // Pass 1: accumulate sum_dy and sum_dy_xhat; set grad_gamma and grad_beta
            let mut sum_dy: Float = 0.0;
            let mut sum_dy_xhat: Float = 0.0;
            for b in 0..batch {
                for i in 0..h {
                    for j in 0..w {
                        let dy = grad_output[[b, c, i, j]];
                        sum_dy += dy;
                        sum_dy_xhat += dy * self.x_hat[[b, c, i, j]];
                    }
                }
            }
            grad_gamma[[c]] = sum_dy_xhat;
            grad_beta[[c]] = sum_dy;

            // Pass 2: compute grad_input using the sums from pass 1
            for b in 0..batch {
                for i in 0..h {
                    for j in 0..w {
                        let dy = grad_output[[b, c, i, j]];
                        let xh = self.x_hat[[b, c, i, j]];
                        grad_input[[b, c, i, j]] =
                            g * si / m * (m * dy - sum_dy - xh * sum_dy_xhat);
                    }
                }
            }
        }

        vec![grad_input, grad_gamma, grad_beta]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.input.clone(), self.gamma.clone(), self.beta.clone()]
    }

    fn name(&self) -> &'static str {
        "BatchNorm2DBackward"
    }
}
