//! # Operations
//!
//! All operations on Tensors. Each operation:
//! 1. Computes the result (forward)
//! 2. If grad is enabled, attaches a `GradFn` to the result

use crate::Float;
use crate::grad_fn::*;
use crate::tensor::{Tensor, is_grad_enabled};
use ndarray::{ArrayD, Axis, IxDyn};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Arithmetic operations
// ═══════════════════════════════════════════════════════════════════════════

/// Element-wise addition: c = a + b (supports broadcasting).
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let result = &a_data + &b_data;

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(AddBackward {
            a: a.clone(),
            b: b.clone(),
            a_shape: a_data.shape().to_vec(),
            b_shape: b_data.shape().to_vec(),
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Element-wise subtraction: c = a - b.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let result = &a_data - &b_data;

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(SubBackward {
            a: a.clone(),
            b: b.clone(),
            a_shape: a_data.shape().to_vec(),
            b_shape: b_data.shape().to_vec(),
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Element-wise multiplication: c = a * b.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();
    let result = &a_data * &b_data;

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(MulBackward {
            a: a.clone(),
            b: b.clone(),
            a_data: a_data.clone(),
            b_data: b_data.clone(),
            a_shape: a_data.shape().to_vec(),
            b_shape: b_data.shape().to_vec(),
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Multiply tensor by a scalar: c = a * scalar.
pub fn mul_scalar(a: &Tensor, scalar: Float) -> Tensor {
    let result = a.data() * scalar;

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(MulScalarBackward {
            a: a.clone(),
            scalar,
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Negate: c = -a.
pub fn neg(a: &Tensor) -> Tensor {
    let result = -a.data();

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(NegBackward { a: a.clone() });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Matrix operations
// ═══════════════════════════════════════════════════════════════════════════

/// Matrix multiplication: c = a @ b (2D tensors only).
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let a_data = a.data();
    let b_data = b.data();

    let a_2d = a_data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("matmul: first argument must be 2D");
    let b_2d = b_data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("matmul: second argument must be 2D");

    let result = a_2d.dot(&b_2d).into_dyn();

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(MatMulBackward {
            a: a.clone(),
            b: b.clone(),
            a_data,
            b_data,
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Transpose a 2D tensor.
pub fn transpose(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let a_2d = a_data
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("transpose: must be 2D");
    let result = a_2d.t().to_owned().into_dyn();

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(TransposeBackward { a: a.clone() });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reduction operations
// ═══════════════════════════════════════════════════════════════════════════

/// Sum all elements → scalar.
pub fn sum(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let total: Float = a_data.iter().sum();
    let result = ArrayD::from_elem(IxDyn(&[]), total);

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(SumBackward {
            a: a.clone(),
            a_shape: a_data.shape().to_vec(),
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Sum along a given axis.
pub fn sum_axis(a: &Tensor, axis: usize) -> Tensor {
    let a_data = a.data();
    let result = a_data.sum_axis(Axis(axis));

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(SumAxisBackward {
            a: a.clone(),
            a_shape: a_data.shape().to_vec(),
            axis,
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Mean of all elements → scalar.
pub fn mean(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let n = a_data.len();
    let total: Float = a_data.iter().sum();
    let result = ArrayD::from_elem(IxDyn(&[]), total / n as Float);

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(MeanBackward {
            a: a.clone(),
            a_shape: a_data.shape().to_vec(),
            numel: n,
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Element-wise math operations
// ═══════════════════════════════════════════════════════════════════════════

/// Element-wise power: c = a^p.
pub fn powf(a: &Tensor, exponent: Float) -> Tensor {
    let a_data = a.data();
    let result = a_data.mapv(|x| x.powf(exponent));

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(PowfBackward {
            a: a.clone(),
            a_data,
            exponent,
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Element-wise natural log: c = ln(a).
pub fn log(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let result = a_data.mapv(|x| x.ln());

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(LogBackward {
            a: a.clone(),
            a_data,
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Element-wise exponential: c = exp(a).
pub fn exp(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let result = a_data.mapv(|x| x.exp());

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(ExpBackward {
            a: a.clone(),
            output_data: result.clone(),
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Activation operations
// ═══════════════════════════════════════════════════════════════════════════

/// ReLU: c = max(0, a).
pub fn relu(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let result = a_data.mapv(|x| x.max(0.0));

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(ReLUBackward {
            a: a.clone(),
            a_data,
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Sigmoid: c = 1 / (1 + exp(-a)).
pub fn sigmoid(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let result = a_data.mapv(|x| 1.0 / (1.0 + (-x).exp()));

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(SigmoidBackward {
            a: a.clone(),
            output_data: result.clone(),
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Tanh activation.
pub fn tanh_act(a: &Tensor) -> Tensor {
    let a_data = a.data();
    let result = a_data.mapv(|x| x.tanh());

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(TanhBackward {
            a: a.clone(),
            output_data: result.clone(),
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

/// Element-wise clamp: c = clamp(a, min, max).
///
/// Gradient passes through where `min < a < max`, zero elsewhere.
pub fn clamp(a: &Tensor, min_val: Float, max_val: Float) -> Tensor {
    let a_data = a.data();
    let result = a_data.mapv(|x| x.clamp(min_val, max_val));

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(ClampBackward {
            a: a.clone(),
            a_data,
            min_val,
            max_val,
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shape operations
// ═══════════════════════════════════════════════════════════════════════════

/// Reshape a tensor (total elements must match).
pub fn reshape(a: &Tensor, shape: &[usize]) -> Tensor {
    let a_data = a.data();
    let original_shape = a_data.shape().to_vec();
    let result = a_data
        .into_shape_with_order(IxDyn(shape))
        .expect("reshape: total elements must match");

    if is_grad_enabled() && a.requires_grad() {
        let grad_fn = Arc::new(ReshapeBackward {
            a: a.clone(),
            original_shape,
        });
        Tensor::from_op(result, grad_fn)
    } else {
        Tensor::new(result, false)
    }
}
