//! # Tensor: Core data structure with automatic gradient tracking
//!
//! The Tensor is the fundamental building block of autograd. It encapsulates:
//! - The data (`ArrayD<Float>`)
//! - The accumulated gradient (via `RwLock` for thread-safety)
//! - The backward function (`grad_fn`) that created it
//! - A `requires_grad` flag to enable tracking

use crate::Float;
use crate::grad_fn::GradFn;
use ndarray::{ArrayD, IxDyn};
use std::fmt;
use std::sync::{Arc, RwLock};

// Thread-local flag to disable gradient tracking (for no_grad context)
thread_local! {
    static GRAD_ENABLED: std::cell::RefCell<bool> = const { std::cell::RefCell::new(true) };
}

/// Checks whether gradient computation is currently enabled.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| *g.borrow())
}

/// Sets gradient computation enabled/disabled. Returns previous value.
pub fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|g| {
        let prev = *g.borrow();
        *g.borrow_mut() = enabled;
        prev
    })
}

/// Internal shared state for a Tensor.
///
/// Uses `RwLock` for the gradient field to allow concurrent reads
/// and exclusive writes while remaining `Sync`.
struct TensorInner {
    /// N-dimensional data
    data: ArrayD<Float>,
    /// Accumulated gradient (lazily allocated, thread-safe)
    grad: RwLock<Option<ArrayD<Float>>>,
    /// Whether this tensor participates in gradient computation
    requires_grad: bool,
    /// The backward function that produced this tensor (None for leaf tensors)
    grad_fn: Option<Arc<dyn GradFn>>,
    /// Whether this is a leaf tensor (created by user, not by an operation)
    is_leaf: bool,
}

impl fmt::Debug for TensorInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorInner")
            .field("shape", &self.data.shape())
            .field("requires_grad", &self.requires_grad)
            .field("is_leaf", &self.is_leaf)
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish()
    }
}

/// A multi-dimensional array with automatic differentiation support.
///
/// Tensors track operations performed on them and can compute gradients
/// via the `backward()` method.
///
/// # Thread Safety
///
/// `Tensor` is `Send + Sync` thanks to `Arc<TensorInner>` where the mutable
/// gradient field is protected by `RwLock`.
#[derive(Clone)]
pub struct Tensor {
    inner: Arc<TensorInner>,
}

impl Tensor {
    // ═══════════════════════════════════════════════════════════════════
    // Constructors
    // ═══════════════════════════════════════════════════════════════════

    /// Creates a new Tensor from an ndarray with optional gradient tracking.
    pub fn new(data: ArrayD<Float>, requires_grad: bool) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                data,
                grad: RwLock::new(None),
                requires_grad,
                grad_fn: None,
                is_leaf: true,
            }),
        }
    }

    /// Creates a Tensor from a flat Vec and a shape.
    pub fn from_vec(data: Vec<Float>, shape: &[usize], requires_grad: bool) -> Self {
        let array =
            ArrayD::from_shape_vec(IxDyn(shape), data).expect("Shape mismatch with data length");
        Self::new(array, requires_grad)
    }

    /// Creates a Tensor filled with zeros.
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        Self::new(ArrayD::zeros(IxDyn(shape)), requires_grad)
    }

    /// Creates a Tensor filled with ones.
    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        Self::new(ArrayD::from_elem(IxDyn(shape), 1.0), requires_grad)
    }

    /// Creates a Tensor with random values from uniform distribution [0, 1).
    pub fn rand(shape: &[usize], requires_grad: bool) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let size: usize = shape.iter().product();
        let data: Vec<Float> = (0..size).map(|_| rng.random::<Float>()).collect();
        Self::from_vec(data, shape, requires_grad)
    }

    /// Creates a Tensor with random values from normal distribution N(0, 1).
    pub fn randn(shape: &[usize], requires_grad: bool) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let size: usize = shape.iter().product();
        let data: Vec<Float> = (0..size)
            .map(|_| {
                let u1: Float = rng.random();
                let u2: Float = rng.random();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        Self::from_vec(data, shape, requires_grad)
    }

    /// Creates a scalar Tensor (0-dimensional).
    pub fn scalar(value: Float, requires_grad: bool) -> Self {
        Self::new(ArrayD::from_elem(IxDyn(&[]), value), requires_grad)
    }

    /// Internal: creates a Tensor that is the result of an operation.
    pub(crate) fn from_op(data: ArrayD<Float>, grad_fn: Arc<dyn GradFn>) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                data,
                grad: RwLock::new(None),
                requires_grad: true,
                grad_fn: Some(grad_fn),
                is_leaf: false,
            }),
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Accessors
    // ═══════════════════════════════════════════════════════════════════

    /// Returns a clone of the underlying data.
    pub fn data(&self) -> ArrayD<Float> {
        self.inner.data.clone()
    }

    /// Applies a function to a reference of the data (zero-copy read).
    pub fn data_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&ArrayD<Float>) -> R,
    {
        f(&self.inner.data)
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> Vec<usize> {
        self.inner.data.shape().to_vec()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.inner.data.ndim()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.inner.data.len()
    }

    /// Whether this tensor requires gradient computation.
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    /// Whether this is a leaf tensor (not created by an operation).
    pub fn is_leaf(&self) -> bool {
        self.inner.is_leaf
    }

    /// Returns the gradient if it has been computed.
    pub fn grad(&self) -> Option<ArrayD<Float>> {
        self.inner.grad.read().unwrap().clone()
    }

    /// Returns the grad_fn (backward function) if any.
    pub fn grad_fn(&self) -> Option<Arc<dyn GradFn>> {
        self.inner.grad_fn.clone()
    }

    /// Returns a stable pointer-based identity for this tensor.
    pub(crate) fn id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    // ═══════════════════════════════════════════════════════════════════
    // Gradient management
    // ═══════════════════════════════════════════════════════════════════

    /// Accumulates gradient (adds to existing gradient or sets it).
    pub(crate) fn accumulate_grad(&self, grad: &ArrayD<Float>) {
        let mut grad_ref = self.inner.grad.write().unwrap();
        match grad_ref.as_mut() {
            Some(existing) => {
                *existing += grad;
            }
            None => {
                *grad_ref = Some(grad.clone());
            }
        }
    }

    /// Sets the gradient directly.
    pub fn set_grad(&self, grad: ArrayD<Float>) {
        *self.inner.grad.write().unwrap() = Some(grad);
    }

    /// Clears the gradient.
    pub fn zero_grad(&self) {
        *self.inner.grad.write().unwrap() = None;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Backward pass
    // ═══════════════════════════════════════════════════════════════════

    /// Computes gradients via backpropagation from this tensor.
    ///
    /// This tensor must be a scalar (single element). The gradient
    /// is set to 1.0 and propagated back through the computation graph.
    pub fn backward(&self) {
        crate::engine::backward(self);
    }

    // ═══════════════════════════════════════════════════════════════════
    // Tensor operations (convenience methods)
    // ═══════════════════════════════════════════════════════════════════

    /// Sum all elements, returning a scalar tensor.
    pub fn sum(&self) -> Tensor {
        crate::ops::sum(self)
    }

    /// Sum along a given axis.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        crate::ops::sum_axis(self, axis)
    }

    /// Mean of all elements.
    pub fn mean(&self) -> Tensor {
        crate::ops::mean(self)
    }

    /// Element-wise power.
    pub fn powf(&self, exponent: Float) -> Tensor {
        crate::ops::powf(self, exponent)
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Tensor {
        crate::ops::log(self)
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Tensor {
        crate::ops::exp(self)
    }

    /// ReLU activation.
    pub fn relu(&self) -> Tensor {
        crate::ops::relu(self)
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self) -> Tensor {
        crate::ops::sigmoid(self)
    }

    /// Tanh activation.
    pub fn tanh_act(&self) -> Tensor {
        crate::ops::tanh_act(self)
    }

    /// Reshape to a new shape (total elements must match).
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        crate::ops::reshape(self, shape)
    }

    /// Transpose (2D only: swap axes 0 and 1).
    pub fn t(&self) -> Tensor {
        crate::ops::transpose(self)
    }

    /// Matrix multiplication.
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        crate::ops::matmul(self, other)
    }

    /// Detach from computation graph (returns a new tensor with no grad_fn).
    pub fn detach(&self) -> Tensor {
        Tensor::new(self.data(), false)
    }

    /// Returns a scalar value (panics if not a scalar).
    pub fn item(&self) -> Float {
        let data = &self.inner.data;
        assert!(
            data.len() == 1,
            "item() requires a scalar tensor, got shape {:?}",
            data.shape()
        );
        data.iter().next().copied().unwrap()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator overloading
// ═══════════════════════════════════════════════════════════════════════════

impl std::ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        crate::ops::add(self, rhs)
    }
}

impl std::ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        crate::ops::sub(self, rhs)
    }
}

impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        crate::ops::mul(self, rhs)
    }
}

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        crate::ops::neg(self)
    }
}

impl std::ops::Mul<Float> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Float) -> Tensor {
        crate::ops::mul_scalar(self, rhs)
    }
}

impl std::ops::Div<Float> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Float) -> Tensor {
        crate::ops::mul_scalar(self, 1.0 / rhs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Display
// ═══════════════════════════════════════════════════════════════════════════

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, requires_grad={}, is_leaf={}, grad_fn={})",
            self.inner.data.shape(),
            self.inner.requires_grad,
            self.inner.is_leaf,
            match &self.inner.grad_fn {
                Some(gf) => gf.name(),
                None => "None",
            }
        )
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], true);
        assert_eq!(t.shape(), vec![3]);
        assert!(t.requires_grad());
        assert!(t.is_leaf());
        assert!(t.grad_fn().is_none());
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(&[2, 3], false);
        assert_eq!(t.shape(), vec![2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(!t.requires_grad());
    }

    #[test]
    fn test_tensor_scalar() {
        let t = Tensor::scalar(3.14, false);
        assert_eq!(t.numel(), 1);
        assert!((t.item() - 3.14).abs() < 1e-5);
    }

    #[test]
    fn test_grad_accumulation() {
        let t = Tensor::from_vec(vec![1.0, 2.0], &[2], true);
        assert!(t.grad().is_none());

        let g1 = ArrayD::from_elem(IxDyn(&[2]), 0.5);
        t.accumulate_grad(&g1);
        assert_eq!(t.grad().unwrap(), ArrayD::from_elem(IxDyn(&[2]), 0.5));

        t.accumulate_grad(&g1);
        assert_eq!(t.grad().unwrap(), ArrayD::from_elem(IxDyn(&[2]), 1.0));

        t.zero_grad();
        assert!(t.grad().is_none());
    }
}
