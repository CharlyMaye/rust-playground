//! # Comprehensive Gradient Tests
//!
//! Validates autograd correctness by comparing analytical gradients (from backward)
//! against numerical gradients (finite differences).
//!
//! For each operation, we verify:
//! 1. Forward produces correct values
//! 2. Backward computes correct gradients (vs numerical)
//!
//! Numerical gradient: ∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)

use cma_autograd::prelude::*;
use cma_autograd::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

const EPS: f32 = 1e-4;
const TOL: f32 = 2e-2; // tolerance for numerical vs analytical comparison (f32 precision)

// ═══════════════════════════════════════════════════════════════════════════
// Helper: numerical gradient via finite differences
// ═══════════════════════════════════════════════════════════════════════════

/// Compute numerical gradient of scalar function `f` with respect to `x`
/// using central differences: (f(x+ε) - f(x-ε)) / 2ε
fn numerical_gradient<F>(x_data: &[Float], shape: &[usize], f: F) -> Vec<Float>
where
    F: Fn(&Tensor) -> Tensor,
{
    let n = x_data.len();
    let mut grad = vec![0.0; n];

    for i in 0..n {
        // x + ε
        let mut x_plus = x_data.to_vec();
        x_plus[i] += EPS;
        let t_plus = Tensor::from_vec(x_plus, shape, false);
        let f_plus = f(&t_plus).item();

        // x - ε
        let mut x_minus = x_data.to_vec();
        x_minus[i] -= EPS;
        let t_minus = Tensor::from_vec(x_minus, shape, false);
        let f_minus = f(&t_minus).item();

        grad[i] = (f_plus - f_minus) / (2.0 * EPS);
    }

    grad
}

/// Assert that two gradient vectors are approximately equal.
fn assert_grads_close(analytical: &[Float], numerical: &[Float], op_name: &str) {
    assert_eq!(
        analytical.len(),
        numerical.len(),
        "{}: gradient length mismatch",
        op_name
    );

    for (i, (a, n)) in analytical.iter().zip(numerical.iter()).enumerate() {
        let diff = (a - n).abs();
        let scale = a.abs().max(n.abs()).max(1e-6);
        let relative = diff / scale;
        assert!(
            relative < TOL,
            "{}: gradient mismatch at index {} — analytical={}, numerical={}, relative_err={}",
            op_name,
            i,
            a,
            n,
            relative
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Addition
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_add_backward() {
    let a_data = vec![1.0, 2.0, 3.0];
    let b_data = vec![4.0, 5.0, 6.0];

    let a = Tensor::from_vec(a_data.clone(), &[3], true);
    let b = Tensor::from_vec(b_data.clone(), &[3], true);

    let c = &a + &b;
    let loss = c.sum();
    loss.backward();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    // ∂sum(a+b)/∂a = [1, 1, 1]
    let expected = ArrayD::from_elem(IxDyn(&[3]), 1.0);
    assert_eq!(grad_a, expected);
    assert_eq!(grad_b, expected);

    // Numerical check for a
    let num_grad_a = numerical_gradient(&a_data, &[3], |x| {
        let b_t = Tensor::from_vec(b_data.clone(), &[3], false);
        (x + &b_t).sum()
    });
    assert_grads_close(grad_a.as_slice().unwrap(), &num_grad_a, "add_a");
}

#[test]
fn test_add_broadcast_backward() {
    // [2, 3] + [3] → [2, 3], grad of [3] should sum over axis 0
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.1, 0.2, 0.3];

    let a = Tensor::from_vec(a_data.clone(), &[2, 3], true);
    let b = Tensor::from_vec(b_data.clone(), &[3], true);

    let c = &a + &b;
    let loss = c.sum();
    loss.backward();

    let grad_a = a.grad().unwrap();
    let grad_b = b.grad().unwrap();

    // ∂sum/∂a = all ones (2x3)
    assert_eq!(grad_a, ArrayD::from_elem(IxDyn(&[2, 3]), 1.0));
    // ∂sum/∂b = [2, 2, 2] (summed over the broadcast dim)
    assert_eq!(grad_b, ArrayD::from_elem(IxDyn(&[3]), 2.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Subtraction
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sub_backward() {
    let a_data = vec![5.0, 3.0, 1.0];
    let b_data = vec![1.0, 2.0, 3.0];

    let a = Tensor::from_vec(a_data.clone(), &[3], true);
    let b = Tensor::from_vec(b_data.clone(), &[3], true);

    let c = &a - &b;
    let loss = c.sum();
    loss.backward();

    // ∂sum(a-b)/∂a = [1, 1, 1]
    assert_eq!(a.grad().unwrap(), ArrayD::from_elem(IxDyn(&[3]), 1.0));
    // ∂sum(a-b)/∂b = [-1, -1, -1]
    assert_eq!(b.grad().unwrap(), ArrayD::from_elem(IxDyn(&[3]), -1.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Multiplication (element-wise)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_mul_backward() {
    let a_data = vec![2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0];

    let a = Tensor::from_vec(a_data.clone(), &[3], true);
    let b = Tensor::from_vec(b_data.clone(), &[3], true);

    let c = &a * &b;
    let loss = c.sum();
    loss.backward();

    // ∂sum(a*b)/∂a = b
    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    let grad_b: Vec<Float> = b.grad().unwrap().iter().copied().collect();
    assert_eq!(grad_a, b_data);
    assert_eq!(grad_b, a_data);

    // Numerical check
    let num_grad_a = numerical_gradient(&a_data, &[3], |x| {
        let b_t = Tensor::from_vec(b_data.clone(), &[3], false);
        (x * &b_t).sum()
    });
    assert_grads_close(&grad_a, &num_grad_a, "mul_a");
}

#[test]
fn test_mul_scalar_backward() {
    let a_data = vec![1.0, 2.0, 3.0];

    let a = Tensor::from_vec(a_data.clone(), &[3], true);
    let c = &a * 3.0;
    let loss = c.sum();
    loss.backward();

    // ∂sum(3*a)/∂a = [3, 3, 3]
    assert_eq!(a.grad().unwrap(), ArrayD::from_elem(IxDyn(&[3]), 3.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Negation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_neg_backward() {
    let a_data = vec![1.0, -2.0, 3.0];

    let a = Tensor::from_vec(a_data, &[3], true);
    let c = -&a;
    let loss = c.sum();
    loss.backward();

    // ∂sum(-a)/∂a = [-1, -1, -1]
    assert_eq!(a.grad().unwrap(), ArrayD::from_elem(IxDyn(&[3]), -1.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// MatMul
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_matmul_backward() {
    // A: [2, 3], B: [3, 2]
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];

    let a = Tensor::from_vec(a_data.clone(), &[2, 3], true);
    let b = Tensor::from_vec(b_data.clone(), &[3, 2], true);

    let c = a.matmul(&b);
    let loss = c.sum();
    loss.backward();

    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    let grad_b: Vec<Float> = b.grad().unwrap().iter().copied().collect();

    // Numerical check for A
    let num_grad_a = numerical_gradient(&a_data, &[2, 3], |x| {
        let b_t = Tensor::from_vec(b_data.clone(), &[3, 2], false);
        x.matmul(&b_t).sum()
    });
    assert_grads_close(&grad_a, &num_grad_a, "matmul_a");

    // Numerical check for B
    let num_grad_b = numerical_gradient(&b_data, &[3, 2], |x| {
        let a_t = Tensor::from_vec(a_data.clone(), &[2, 3], false);
        a_t.matmul(x).sum()
    });
    assert_grads_close(&grad_b, &num_grad_b, "matmul_b");
}

// ═══════════════════════════════════════════════════════════════════════════
// Transpose
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_transpose_backward() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a = Tensor::from_vec(a_data.clone(), &[2, 3], true);
    let b = a.t();
    let loss = b.sum();
    loss.backward();

    // ∂sum(aᵀ)/∂a = ones(2, 3) (transpose of gradient just transposes back)
    assert_eq!(a.grad().unwrap(), ArrayD::from_elem(IxDyn(&[2, 3]), 1.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Sum
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sum_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], true);
    let loss = a.sum();
    loss.backward();

    // ∂sum(a)/∂a = ones
    assert_eq!(a.grad().unwrap(), ArrayD::from_elem(IxDyn(&[2, 2]), 1.0));
}

#[test]
fn test_sum_axis_backward() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = Tensor::from_vec(a_data, &[2, 3], true);

    // Sum along axis 0: [2,3] → [3]
    let s = a.sum_axis(0);
    let loss = s.sum();
    loss.backward();

    // Every element gets gradient 1
    assert_eq!(a.grad().unwrap(), ArrayD::from_elem(IxDyn(&[2, 3]), 1.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Mean
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_mean_backward() {
    let a = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0], &[4], true);
    let loss = a.mean();
    loss.backward();

    // ∂mean(a)/∂a = 1/N for each element
    assert_eq!(a.grad().unwrap(), ArrayD::from_elem(IxDyn(&[4]), 0.25));
}

// ═══════════════════════════════════════════════════════════════════════════
// Powf
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_powf_backward() {
    let a_data = vec![1.0, 2.0, 3.0];

    let a = Tensor::from_vec(a_data.clone(), &[3], true);
    let c = a.powf(2.0);
    let loss = c.sum();
    loss.backward();

    // ∂sum(a²)/∂a = 2a
    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    assert_grads_close(&grad_a, &[2.0, 4.0, 6.0], "powf");

    // Numerical check
    let num_grad = numerical_gradient(&a_data, &[3], |x| x.powf(2.0).sum());
    assert_grads_close(&grad_a, &num_grad, "powf_numerical");
}

#[test]
fn test_powf_cubic_backward() {
    let a_data = vec![1.0, 2.0, 3.0];

    let a = Tensor::from_vec(a_data.clone(), &[3], true);
    let c = a.powf(3.0);
    let loss = c.sum();
    loss.backward();

    // ∂sum(a³)/∂a = 3a²
    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    assert_grads_close(&grad_a, &[3.0, 12.0, 27.0], "powf_cubic");
}

// ═══════════════════════════════════════════════════════════════════════════
// Log
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_log_backward() {
    let a_data = vec![1.0, 2.0, 4.0];

    let a = Tensor::from_vec(a_data.clone(), &[3], true);
    let c = a.log();
    let loss = c.sum();
    loss.backward();

    // ∂sum(ln(a))/∂a = 1/a
    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    assert_grads_close(&grad_a, &[1.0, 0.5, 0.25], "log");

    // Numerical check
    let num_grad = numerical_gradient(&a_data, &[3], |x| x.log().sum());
    assert_grads_close(&grad_a, &num_grad, "log_numerical");
}

// ═══════════════════════════════════════════════════════════════════════════
// Exp
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_exp_backward() {
    let a_data = vec![0.0, 1.0, -1.0];

    let a = Tensor::from_vec(a_data.clone(), &[3], true);
    let c = a.exp();
    let loss = c.sum();
    loss.backward();

    // ∂sum(exp(a))/∂a = exp(a)
    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    let expected: Vec<Float> = a_data.iter().map(|x| x.exp()).collect();
    assert_grads_close(&grad_a, &expected, "exp");

    let num_grad = numerical_gradient(&a_data, &[3], |x| x.exp().sum());
    assert_grads_close(&grad_a, &num_grad, "exp_numerical");
}

// ═══════════════════════════════════════════════════════════════════════════
// ReLU
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_relu_backward() {
    let a_data = vec![-2.0, -1.0, 0.5, 1.0, 3.0];

    let a = Tensor::from_vec(a_data.clone(), &[5], true);
    let c = a.relu();
    let loss = c.sum();
    loss.backward();

    // ∂sum(relu(a))/∂a = 1 if a>0, 0 otherwise
    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    assert_eq!(grad_a, vec![0.0, 0.0, 1.0, 1.0, 1.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Sigmoid
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sigmoid_backward() {
    let a_data = vec![-1.0, 0.0, 1.0, 2.0];

    let a = Tensor::from_vec(a_data.clone(), &[4], true);
    let c = a.sigmoid();
    let loss = c.sum();
    loss.backward();

    // ∂sum(σ(a))/∂a = σ(a)(1-σ(a))
    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();

    let num_grad = numerical_gradient(&a_data, &[4], |x| x.sigmoid().sum());
    assert_grads_close(&grad_a, &num_grad, "sigmoid");
}

// ═══════════════════════════════════════════════════════════════════════════
// Tanh
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_backward() {
    let a_data = vec![-1.0, 0.0, 0.5, 2.0];

    let a = Tensor::from_vec(a_data.clone(), &[4], true);
    let c = a.tanh_act();
    let loss = c.sum();
    loss.backward();

    // ∂sum(tanh(a))/∂a = 1 - tanh²(a)
    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();

    let num_grad = numerical_gradient(&a_data, &[4], |x| x.tanh_act().sum());
    assert_grads_close(&grad_a, &num_grad, "tanh");
}

// ═══════════════════════════════════════════════════════════════════════════
// Softmax
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_softmax_backward() {
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let layer = Softmax::new();

    let a = Tensor::from_vec(a_data.clone(), &[4], true);
    let c = layer.forward(&a);
    // Scalar loss: weighted sum so gradient is non-trivial
    let weights = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4], false);
    let loss = (&c * &weights).sum();
    loss.backward();

    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();

    let num_grad = numerical_gradient(&a_data, &[4], |x| {
        let l = Softmax::new();
        let out = l.forward(x);
        let w = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4], false);
        (&out * &w).sum()
    });
    assert_grads_close(&grad_a, &num_grad, "softmax");
}

// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_reshape_backward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
    let b = a.reshape(&[3, 2]);
    let loss = b.sum();
    loss.backward();

    // Gradient should flow back and reshape to original shape
    assert_eq!(a.grad().unwrap().shape(), &[2, 3]);
    assert_eq!(a.grad().unwrap(), ArrayD::from_elem(IxDyn(&[2, 3]), 1.0));
}

// ═══════════════════════════════════════════════════════════════════════════
// Compound operations (chain rule)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_chain_rule_mul_sum() {
    // f(a) = sum(a * a) = sum(a²)
    // ∂f/∂a = 2a
    let a_data = vec![1.0, 2.0, 3.0];
    let a = Tensor::from_vec(a_data.clone(), &[3], true);

    let c = &a * &a;
    let loss = c.sum();
    loss.backward();

    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    assert_grads_close(&grad_a, &[2.0, 4.0, 6.0], "chain_mul_sum");
}

#[test]
fn test_chain_rule_linear() {
    // f(x, w, b) = sum((x @ wᵀ) + b)
    // x: [1, 3], w: [2, 3], b: [2]
    let x_data = vec![1.0, 2.0, 3.0];
    let w_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let b_data = vec![0.01, 0.02];

    let x = Tensor::from_vec(x_data.clone(), &[1, 3], true);
    let w = Tensor::from_vec(w_data.clone(), &[2, 3], true);
    let b = Tensor::from_vec(b_data.clone(), &[2], true);

    let wt = w.t(); // [3, 2]
    let z = x.matmul(&wt); // [1, 2]
    let out = &z + &b; // [1, 2]
    let loss = out.sum();
    loss.backward();

    // Numerical check for w
    let num_grad_w = numerical_gradient(&w_data, &[2, 3], |w_test| {
        let x_t = Tensor::from_vec(x_data.clone(), &[1, 3], false);
        let b_t = Tensor::from_vec(b_data.clone(), &[2], false);
        let wt_test = w_test.t();
        let z_test = x_t.matmul(&wt_test);
        let out_test = &z_test + &b_t;
        out_test.sum()
    });
    let grad_w: Vec<Float> = w.grad().unwrap().iter().copied().collect();
    assert_grads_close(&grad_w, &num_grad_w, "chain_linear_w");

    // Numerical check for x
    let num_grad_x = numerical_gradient(&x_data, &[1, 3], |x_test| {
        let w_t = Tensor::from_vec(w_data.clone(), &[2, 3], false);
        let b_t = Tensor::from_vec(b_data.clone(), &[2], false);
        let wt_test = w_t.t();
        let z_test = x_test.matmul(&wt_test);
        let out_test = &z_test + &b_t;
        out_test.sum()
    });
    let grad_x: Vec<Float> = x.grad().unwrap().iter().copied().collect();
    assert_grads_close(&grad_x, &num_grad_x, "chain_linear_x");
}

#[test]
fn test_chain_relu_mul() {
    // f(a) = sum(relu(a) * 2)
    let a_data = vec![-1.0, 0.0, 1.0, 2.0];
    let a = Tensor::from_vec(a_data.clone(), &[4], true);

    let r = a.relu();
    let c = &r * 2.0;
    let loss = c.sum();
    loss.backward();

    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    // ∂sum(2*relu(a))/∂a = 2 * (a > 0)
    assert_eq!(grad_a, vec![0.0, 0.0, 2.0, 2.0]);
}

#[test]
fn test_compound_sigmoid_mse() {
    // f(x) = mean((sigmoid(x) - target)²)
    let x_data = vec![0.0, 1.0, -1.0];
    let t_data = vec![0.5, 0.8, 0.2];

    let x = Tensor::from_vec(x_data.clone(), &[3], true);
    let target = Tensor::from_vec(t_data.clone(), &[3], false);

    let pred = x.sigmoid();
    let loss = mse_loss(&pred, &target);
    loss.backward();

    let grad_x: Vec<Float> = x.grad().unwrap().iter().copied().collect();

    let num_grad = numerical_gradient(&x_data, &[3], |x_test| {
        let t = Tensor::from_vec(t_data.clone(), &[3], false);
        let p = x_test.sigmoid();
        mse_loss(&p, &t)
    });
    assert_grads_close(&grad_x, &num_grad, "compound_sigmoid_mse");
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear module
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_linear_forward_shape() {
    let linear = Linear::new(4, 3);

    let x = Tensor::from_vec(vec![1.0; 8], &[2, 4], true);
    let y = linear.forward(&x);

    assert_eq!(y.shape(), vec![2, 3]);
    assert!(y.grad_fn().is_some()); // should be tracked
}

#[test]
fn test_linear_backward_updates() {
    let linear = Linear::new(3, 2);
    assert_eq!(linear.parameters().len(), 2); // weight + bias

    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], true);
    let y = linear.forward(&x);
    let loss = y.sum();
    loss.backward();

    // All parameters should have gradients
    for param in linear.parameters() {
        assert!(
            param.grad().is_some(),
            "Parameter should have gradient after backward"
        );
    }
}

#[test]
fn test_linear_num_parameters() {
    let linear = Linear::new(10, 5);
    // weight: 5*10 = 50, bias: 5 → total = 55
    assert_eq!(linear.num_parameters(), 55);
}

// ═══════════════════════════════════════════════════════════════════════════
// Loss functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_mse_loss_value() {
    let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], false);
    let target = Tensor::from_vec(vec![1.5, 2.5, 3.5], &[3], false);
    let loss = mse_loss(&pred, &target);

    // MSE = mean((0.5)²) = mean(0.25) = 0.25
    let val = loss.item();
    assert!((val - 0.25).abs() < 1e-6, "MSE should be 0.25, got {}", val);
}

#[test]
fn test_mse_loss_backward() {
    let x_data = vec![1.0, 2.0, 3.0];
    let t_data = vec![1.5, 2.5, 3.5];

    let x = Tensor::from_vec(x_data.clone(), &[3], true);
    let target = Tensor::from_vec(t_data.clone(), &[3], false);
    let loss = mse_loss(&x, &target);
    loss.backward();

    // ∂MSE/∂x = 2(x-t)/N
    let grad_x: Vec<Float> = x.grad().unwrap().iter().copied().collect();
    let n = 3.0;
    let expected: Vec<Float> = x_data
        .iter()
        .zip(t_data.iter())
        .map(|(x, t)| 2.0 * (x - t) / n)
        .collect();
    assert_grads_close(&grad_x, &expected, "mse_backward");
}

#[test]
fn test_bce_loss_value() {
    // BCE with pred=0.5, target=1.0 → -log(0.5) ≈ 0.693
    let pred = Tensor::from_vec(vec![0.5], &[1], false);
    let target = Tensor::from_vec(vec![1.0], &[1], false);
    let loss = binary_cross_entropy_loss(&pred, &target);

    let val = loss.item();
    let expected = -(1.0_f32 * 0.5_f32.ln() + 0.0 * 0.5_f32.ln());
    assert!(
        (val - expected).abs() < 1e-4,
        "BCE should be ~{}, got {}",
        expected,
        val
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimizer: SGD
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sgd_step() {
    let linear = Linear::new(2, 1);
    let params = linear.parameters().iter().map(|p| (*p).clone()).collect();
    let mut optimizer = SGD::new(params, 0.1);

    let x = Tensor::from_vec(vec![1.0, 2.0], &[1, 2], false);
    let y = linear.forward(&x);
    let target = Tensor::from_vec(vec![1.0], &[1, 1], false);
    let loss = mse_loss(&y, &target);

    let loss_val_before = loss.item();
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();

    // After one step, loss should change
    let y2 = linear.forward(&x);
    let loss2 = mse_loss(&y2, &target);
    let loss_val_after = loss2.item();

    // We can't guarantee loss decreased with one step always,
    // but parameters should have changed
    assert!(
        (loss_val_before - loss_val_after).abs() > 1e-10,
        "Loss should change after optimizer step"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimizer: Adam
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_adam_step() {
    let linear = Linear::new(2, 1);
    let params = linear.parameters().iter().map(|p| (*p).clone()).collect();
    let mut optimizer = Adam::new(params, 0.01);

    let x = Tensor::from_vec(vec![1.0, 2.0], &[1, 2], false);
    let target = Tensor::from_vec(vec![1.0], &[1, 1], false);

    let y = linear.forward(&x);
    let loss = mse_loss(&y, &target);
    loss.backward();
    optimizer.step();
    optimizer.zero_grad();

    // Second step
    let y2 = linear.forward(&x);
    let loss2 = mse_loss(&y2, &target);
    loss2.backward();
    optimizer.step();

    // Parameters should have been updated
    // (Just testing it doesn't crash and parameters change)
    assert!(loss2.item().is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════
// no_grad
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_no_grad_context() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2], true);
    let b = Tensor::from_vec(vec![3.0, 4.0], &[2], true);

    let c = no_grad(|| &a + &b);
    assert!(!c.requires_grad());
    assert!(c.grad_fn().is_none());

    // With grad
    let d = &a + &b;
    assert!(d.requires_grad());
    assert!(d.grad_fn().is_some());
}

#[test]
fn test_no_grad_guard() {
    let a = Tensor::from_vec(vec![1.0], &[1], true);
    let b = Tensor::from_vec(vec![2.0], &[1], true);

    {
        let _g = NoGradGuard::new();
        let c = &a * &b;
        assert!(!c.requires_grad());
    }

    // Grad re-enabled after guard drop
    let d = &a * &b;
    assert!(d.requires_grad());
}

// ═══════════════════════════════════════════════════════════════════════════
// End-to-end: XOR training
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_xor_training() {
    // XOR: 2 inputs → hidden(8) → relu → output(1)
    // Using a wider hidden layer and Adam for reliable convergence
    let hidden = Linear::new(2, 8);
    let output = Linear::new(8, 1);

    let mut all_params: Vec<Parameter> = Vec::new();
    for p in hidden.parameters() {
        all_params.push(p.clone());
    }
    for p in output.parameters() {
        all_params.push(p.clone());
    }

    let mut optimizer = Adam::new(all_params, 0.05);

    // XOR dataset — batch of all 4 samples at once
    let inputs = Tensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &[4, 2],
        false,
    );
    let targets = Tensor::from_vec(vec![0.0, 1.0, 1.0, 0.0], &[4, 1], false);

    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;
    let epochs = 500;

    for epoch in 0..epochs {
        optimizer.zero_grad();

        // Forward (full batch)
        let h = hidden.forward(&inputs);
        let h_relu = h.relu();
        let y = output.forward(&h_relu);
        let loss = mse_loss(&y, &targets);

        let l = loss.item();
        if epoch == 0 {
            initial_loss = l;
        }
        if epoch == epochs - 1 {
            final_loss = l;
        }

        loss.backward();
        optimizer.step();
    }

    assert!(
        final_loss < initial_loss,
        "Loss should decrease during training: initial={}, final={}",
        initial_loss,
        final_loss
    );
    assert!(
        final_loss < 0.05,
        "XOR should be learnable: final_loss={} (should be < 0.05)",
        final_loss
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Detach
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_detach() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2], true);
    let b = Tensor::from_vec(vec![3.0, 4.0], &[2], true);

    let c = &a + &b;
    assert!(c.grad_fn().is_some());

    let d = c.detach();
    assert!(d.grad_fn().is_none());
    assert!(!d.requires_grad());
}

// ═══════════════════════════════════════════════════════════════════════════
// Multiple backward paths (diamond graph)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_diamond_graph() {
    // a → b = a*2
    //   → c = a*3
    // loss = sum(b + c)
    // ∂loss/∂a = 2 + 3 = 5

    let a = Tensor::from_vec(vec![1.0, 1.0], &[2], true);
    let b = &a * 2.0;
    let c = &a * 3.0;
    let d = &b + &c;
    let loss = d.sum();
    loss.backward();

    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    assert_grads_close(&grad_a, &[5.0, 5.0], "diamond_graph");
}

#[test]
fn test_reuse_tensor() {
    // a * a should accumulate gradient from both uses
    // f = sum(a * a) = sum(a²)
    // ∂f/∂a = 2a
    let a = Tensor::from_vec(vec![2.0, 3.0], &[2], true);
    let c = &a * &a;
    let loss = c.sum();
    loss.backward();

    let grad_a: Vec<Float> = a.grad().unwrap().iter().copied().collect();
    assert_grads_close(&grad_a, &[4.0, 6.0], "reuse_tensor");
}

// ═══════════════════════════════════════════════════════════════════════════
// Stateless layers
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_relu_layer() {
    let relu_layer = cma_autograd::layers::ReLU::new();
    let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4], true);
    let y = relu_layer.forward(&x);

    let y_data: Vec<Float> = y.data().iter().copied().collect();
    assert_eq!(y_data, vec![0.0, 0.0, 1.0, 2.0]);

    let loss = y.sum();
    loss.backward();
    let grad_x: Vec<Float> = x.grad().unwrap().iter().copied().collect();
    assert_eq!(grad_x, vec![0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_flatten_layer() {
    let flatten = cma_autograd::layers::Flatten::new();
    let x = Tensor::from_vec(vec![1.0; 24], &[2, 3, 4], true);
    let y = flatten.forward(&x);

    assert_eq!(y.shape(), vec![2, 12]);

    let loss = y.sum();
    loss.backward();
    assert_eq!(x.grad().unwrap().shape(), &[2, 3, 4]);
}
