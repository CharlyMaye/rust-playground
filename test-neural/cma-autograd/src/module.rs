//! # Modules: Parameter, Layer traits, Linear, Conv2D
//!
//! Trait hierarchy:
//! - `Module`: base trait for composite models
//! - `TrainableLayer`: layer with trainable parameters
//! - Stateless layers (ReLU, MaxPool) are in the `layers` module
//!
//! ## Important: parameter mutability
//!
//! `Parameter` wraps an immutable Tensor (inside an Arc).
//! For optimizer updates, the inner Tensor is recreated
//! via `Parameter::set_data()` which replaces the Arc.

use crate::Float;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};
use std::cell::UnsafeCell;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Parameter
// ═══════════════════════════════════════════════════════════════════════════

/// A learnable parameter wrapping a Tensor.
///
/// Parameters always have `requires_grad = true`.
/// They support mutation via `set_data` for optimizer updates.
///
/// **Shared ownership**: Cloning a `Parameter` creates a shared reference
/// to the same underlying storage. This is critical for optimizer-module
/// interaction: the optimizer holds clones of the module's parameters, and
/// updates are visible to the module.
pub struct Parameter {
    /// Shared mutable tensor storage — only the optimizer should mutate this.
    /// Arc ensures cloned Parameters share the same backing storage.
    inner: Arc<UnsafeCell<Tensor>>,
}

// Safety: Parameter is used single-threaded (optimizer step is not parallel).
// The UnsafeCell is needed to allow mutation through a shared reference
// (since Module::parameters returns Vec<&Parameter>).
unsafe impl Send for Parameter {}
unsafe impl Sync for Parameter {}

impl Clone for Parameter {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl fmt::Debug for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Parameter(shape={:?})", self.tensor().shape())
    }
}

use std::fmt;

impl Parameter {
    /// Create a parameter from a Tensor (forces requires_grad=true).
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new(tensor: Tensor) -> Self {
        let t = if !tensor.requires_grad() {
            Tensor::new(tensor.data(), true)
        } else {
            tensor
        };
        Self {
            inner: Arc::new(UnsafeCell::new(t)),
        }
    }

    /// Create a parameter with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::new(Tensor::zeros(shape, true))
    }

    /// Create a parameter with He initialization.
    pub fn he_init(shape: &[usize], fan_in: usize) -> Self {
        let std_dev = (2.0 / fan_in as Float).sqrt();
        let mut rng = rand::rng();
        let size: usize = shape.iter().product();
        let data = cma_neural_network::init::randn_vec(size, std_dev, &mut rng);
        Self::new(Tensor::from_vec(data, shape, true))
    }

    /// Create a parameter with Xavier initialization.
    pub fn xavier_init(shape: &[usize], fan_in: usize, fan_out: usize) -> Self {
        let std_dev = (2.0 / (fan_in + fan_out) as Float).sqrt();
        let mut rng = rand::rng();
        let size: usize = shape.iter().product();
        let data = cma_neural_network::init::randn_vec(size, std_dev, &mut rng);
        Self::new(Tensor::from_vec(data, shape, true))
    }

    /// Access the underlying tensor (immutable).
    pub fn tensor(&self) -> &Tensor {
        // Safety: we only mutate via set_data which is called from optimizer step,
        // never concurrently with tensor reads.
        unsafe { &*self.inner.get() }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.tensor().numel()
    }

    /// Shape of the parameter.
    pub fn shape(&self) -> Vec<usize> {
        self.tensor().shape()
    }

    /// Get a clone of the data.
    pub fn data(&self) -> ArrayD<Float> {
        self.tensor().data()
    }

    /// Replace the underlying tensor with new data (used by optimizers).
    ///
    /// Creates a new leaf tensor with the same requires_grad=true.
    pub fn set_data(&self, data: ArrayD<Float>) {
        // Safety: optimizer step is single-threaded
        unsafe {
            *self.inner.get() = Tensor::new(data, true);
        }
    }

    /// Update data in-place via a closure (used by optimizers).
    pub fn update_data<F>(&self, f: F)
    where
        F: FnOnce(&mut ArrayD<Float>),
    {
        let mut data = self.data();
        f(&mut data);
        self.set_data(data);
    }

    /// Get the gradient.
    pub fn grad(&self) -> Option<ArrayD<Float>> {
        self.tensor().grad()
    }

    /// Reset gradient to zero.
    pub fn zero_grad(&self) {
        self.tensor().zero_grad();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Module trait
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for composite models (can contain sub-modules and parameters).
pub trait Module {
    /// Forward pass.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Returns all trainable parameters.
    fn parameters(&self) -> Vec<&Parameter>;

    /// Total number of trainable parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Zero all gradients.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TrainableLayer trait
// ═══════════════════════════════════════════════════════════════════════════

/// Marker trait for layers with trainable parameters.
///
/// The backward is handled automatically by the autograd engine via GradFn.
/// The layer just needs to use autograd-tracked operations in forward().
pub trait TrainableLayer: Module {
    /// Name of the layer.
    fn layer_name(&self) -> &'static str;
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear layer
// ═══════════════════════════════════════════════════════════════════════════

/// Fully connected linear layer: y = x @ W^T + b
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// Create a new linear layer with He initialization.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = Parameter::he_init(&[out_features, in_features], in_features);
        let bias = Some(Parameter::zeros(&[out_features]));
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Create a linear layer without bias.
    pub fn without_bias(in_features: usize, out_features: usize) -> Self {
        let weight = Parameter::he_init(&[out_features, in_features], in_features);
        Self {
            weight,
            bias: None,
            in_features,
            out_features,
        }
    }

    /// Input feature size.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output feature size.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_features]
        // weight: [out_features, in_features]
        // output = input @ weight^T → [batch, out_features]
        let wt = self.weight.tensor().t(); // [in_features, out_features]
        let mut output = input.matmul(&wt);

        if let Some(ref bias) = self.bias {
            output = &output + bias.tensor();
        }

        output
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }
}

impl TrainableLayer for Linear {
    fn layer_name(&self) -> &'static str {
        "Linear"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Conv2D module (autograd-based)
// ═══════════════════════════════════════════════════════════════════════════

/// 2D Convolution layer using im2col + matmul for autograd compatibility.
///
/// Gradients flow through the matmul operation automatically.
pub struct Conv2D {
    weight: Parameter,
    bias: Option<Parameter>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl Conv2D {
    /// Create a new Conv2D layer with He initialization.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        let weight = Parameter::he_init(
            &[out_channels, in_channels, kernel_size, kernel_size],
            fan_in,
        );
        let bias = Some(Parameter::zeros(&[out_channels]));

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Create Conv2D without bias.
    pub fn without_bias(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        let weight = Parameter::he_init(
            &[out_channels, in_channels, kernel_size, kernel_size],
            fan_in,
        );
        Self {
            weight,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// Access the weight parameter.
    pub fn weight(&self) -> &Parameter { &self.weight }
    /// Access the bias parameter (if any).
    pub fn bias(&self) -> Option<&Parameter> { self.bias.as_ref() }
    /// Input channels.
    pub fn in_channels(&self) -> usize { self.in_channels }
    /// Output channels.
    pub fn out_channels(&self) -> usize { self.out_channels }
    /// Kernel size.
    pub fn kernel_size(&self) -> usize { self.kernel_size }
    /// Stride.
    pub fn stride(&self) -> usize { self.stride }
    /// Padding.
    pub fn padding(&self) -> usize { self.padding }
}

impl Module for Conv2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, in_channels, H, W]
        // Uses cma-cnn's optimized im2col, then autograd matmul + GradFn wrapping
        let input_data = input.data();
        let input_shape = input_data.shape().to_vec();
        assert_eq!(
            input_shape.len(),
            4,
            "Conv2D input must be 4D [N, C, H, W]"
        );

        let batch = input_shape[0];
        let h = input_shape[2];
        let w = input_shape[3];

        let out_h = (h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        // im2col via cma-cnn's cache-optimized implementation: [batch * out_h * out_w, C_in * kH * kW]
        let col = crate::cnn_ops::im2col_optimized(
            &input_data,
            self.kernel_size,
            self.stride,
            self.padding,
        );

        // weight reshaped to 2D: [out_channels, C_in * kH * kW]
        // Use view instead of clone — no heap copy, weight tensor kept alive via self.weight
        let weight_data = self.weight.tensor().data();
        let col_size = self.in_channels * self.kernel_size * self.kernel_size;
        let weight_2d = weight_data
            .view()
            .into_shape_with_order(IxDyn(&[self.out_channels, col_size]))
            .unwrap();

        // col @ weight^T → [batch * out_h * out_w, out_channels]
        let col_2d = col.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let w_2d = weight_2d.view().into_dimensionality::<ndarray::Ix2>().unwrap();
        let output_2d = col_2d.dot(&w_2d.t()).into_dyn();

        // Reshape to BHWC then permute to BCHW — replaces 4-level scalar scatter loop
        // permuted_axes dispatches to cache-optimized axis reorder (BLAS-level), not scalar indexing
        let bchw_raw = output_2d
            .into_shape_with_order(IxDyn(&[batch, out_h, out_w, self.out_channels]))
            .unwrap();
        let mut bchw = bchw_raw
            .permuted_axes(IxDyn(&[0usize, 3, 1, 2]))
            .as_standard_layout()
            .into_owned();

        // Add bias: slice += scalar uses ndarray broadcast — eliminates 2 inner loops per (b,c)
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data();
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bv = bias_data[[c]];
                    let mut slice = bchw.slice_mut(ndarray::s![b, c, .., ..]);
                    slice += bv;
                }
            }
        }

        // Build autograd-tracked tensor with Conv2DBackward
        use crate::grad_fn::Conv2DBackward;
        use crate::tensor::is_grad_enabled;
        use std::sync::Arc;

        if is_grad_enabled()
            && (input.requires_grad() || self.weight.tensor().requires_grad())
        {
            let grad_fn = Arc::new(Conv2DBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                col_data: col,
                weight_2d_data: weight_2d.to_owned().into_dyn(),
                input_shape,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
            });
            Tensor::from_op(bchw, grad_fn)
        } else {
            Tensor::new(bchw, false)
        }
    }

    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }
}

impl TrainableLayer for Conv2D {
    fn layer_name(&self) -> &'static str {
        "Conv2D"
    }
}

// im2col is now provided by crate::cnn_ops::im2col_optimized (delegates to cma-cnn)
