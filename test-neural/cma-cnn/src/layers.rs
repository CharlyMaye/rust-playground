//! # CNN Layers
//!
//! Implementation of layers for convolutional neural networks:
//! - Conv2D: 2D Convolution
//! - MaxPool2D / AvgPool2D: Spatial pooling
//! - BatchNorm2D: Batch normalization
//! - Flatten: 4D to 2D conversion
//!
//! ## References
//!
//! - LeCun et al. (1998): Convolutions and pooling
//! - Ioffe & Szegedy (2015): Batch Normalization
//! - He et al. (2015): Initialization for ReLU

use crate::{Dim, Float};
use ndarray::{Array1, Array4};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::ops::{avgpool2d, conv2d_im2col, global_avgpool2d, maxpool2d};
use crate::tensor::{Tensor4D, TensorShape};

/// Layer type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Conv2D,
    DepthwiseConv2D,
    MaxPool2D,
    AvgPool2D,
    GlobalAvgPool2D,
    BatchNorm2D,
    Dropout2D,
    Flatten,
    Activation,
}

/// Common trait for all layers
pub trait Layer: Send + Sync {
    /// Forward propagation
    fn forward(&self, input: &Tensor4D) -> Tensor4D;

    /// Layer type
    fn layer_type(&self) -> LayerType;

    /// Trainable parameters count
    fn num_parameters(&self) -> usize;

    /// Output shape given an input shape
    fn output_shape(&self, input_shape: TensorShape) -> TensorShape;

    /// Description for debugging
    fn summary(&self) -> String;
}

// ═══════════════════════════════════════════════════════════════════════════
// Conv2D - Convolution 2D
// ═══════════════════════════════════════════════════════════════════════════

/// 2D Convolution Layer
///
/// # Architecture (LeCun et al., 1998)
///
/// Applies `out_channels` filters of size `kernel_size × kernel_size`
/// on the input with `in_channels` channels.
///
/// # Example
///
/// ```rust,ignore
/// // 1 channel input → 32 filters, kernel 3x3
/// let conv = Conv2D::new(1, 32, 3, 1, 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2D {
    /// Number of input channels
    pub in_channels: Dim,
    /// Number of filters (output channels)
    pub out_channels: Dim,
    /// Kernel size (square)
    pub kernel_size: Dim,
    /// Stride (step size)
    pub stride: Dim,
    /// Padding
    pub padding: Dim,
    /// Weights [out_channels, in_channels, kernel_h, kernel_w]
    pub weights: Array4<Float>,
    /// Bias [out_channels]
    pub bias: Array1<Float>,
    /// Whether to use bias
    pub use_bias: bool,
}

impl Conv2D {
    /// Creates a new Conv2D layer with He initialization
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels (1 for grayscale, 3 for RGB)
    /// * `out_channels` - Number of filters
    /// * `kernel_size` - Kernel size (e.g. 3 for 3x3)
    /// * `stride` - Step size (1 = every pixel)
    /// * `padding` - Zeros around the image
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let mut rng = rand::rng();

        // He initialization (for ReLU)
        // fan_in = in_channels * kernel_size * kernel_size
        let fan_in = in_channels * kernel_size * kernel_size;
        let std = (2.0 / fan_in as Float).sqrt();

        let weights_vec = cma_neural_network::init::randn_vec(
            out_channels * in_channels * kernel_size * kernel_size,
            std,
            &mut rng,
        );

        let weights = Array4::from_shape_vec(
            (out_channels, in_channels, kernel_size, kernel_size),
            weights_vec,
        )
        .unwrap();

        // Bias initialized to zero
        let bias = Array1::zeros(out_channels);

        Self {
            in_channels: in_channels as Dim,
            out_channels: out_channels as Dim,
            kernel_size: kernel_size as Dim,
            stride: stride as Dim,
            padding: padding as Dim,
            weights,
            bias,
            use_bias: true,
        }
    }

    /// Creates Conv2D without bias (useful before BatchNorm)
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }

    /// Creates Conv2D with "same" padding (preserves spatial size)
    pub fn same_padding(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let padding = kernel_size / 2;
        Self::new(in_channels, out_channels, kernel_size, 1, padding)
    }
}

impl Layer for Conv2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let bias = if self.use_bias {
            Some(&self.bias)
        } else {
            None
        };
        // Uses im2col + GEMM for ~10-100x faster convolution
        conv2d_im2col(input, &self.weights, bias, self.stride as usize, self.padding as usize)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Conv2D
    }

    fn num_parameters(&self) -> usize {
        let weights = self.out_channels as usize * self.in_channels as usize * self.kernel_size as usize * self.kernel_size as usize;
        let bias = if self.use_bias { self.out_channels as usize } else { 0 };
        weights + bias
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape.after_conv(
            self.out_channels as usize,
            self.kernel_size as usize,
            self.stride as usize,
            self.padding as usize,
        )
    }

    fn summary(&self) -> String {
        format!(
            "Conv2D({} → {}, {}x{}, stride={}, pad={})",
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DepthwiseConv2D - Depthwise (channel-wise) Convolution
// ═══════════════════════════════════════════════════════════════════════════

/// Depthwise 2D Convolution Layer
///
/// Applies one spatial filter per input channel (groups = channels).
/// Used in MobileNet, EfficientNet, and other efficient architectures.
///
/// # Weight layout
/// `[channels, 1, kernel_h, kernel_w]` — one filter per channel.
///
/// # Example
///
/// ```rust,ignore
/// // 32-channel depthwise conv with 3x3 kernel, stride 1, padding 1
/// let dw = DepthwiseConv2D::new(32, 3, 1, 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepthwiseConv2D {
    /// Number of input/output channels
    pub channels: Dim,
    /// Kernel size (square)
    pub kernel_size: Dim,
    /// Stride
    pub stride: Dim,
    /// Padding
    pub padding: Dim,
    /// Weights [channels, 1, kernel_h, kernel_w]
    pub weights: Array4<Float>,
    /// Bias [channels]
    pub bias: Array1<Float>,
    /// Whether to use bias
    pub use_bias: bool,
}

impl DepthwiseConv2D {
    /// Creates a new DepthwiseConv2D with He initialization.
    ///
    /// # Arguments
    /// * `channels` - Number of input (= output) channels
    /// * `kernel_size` - Kernel size (e.g. 3 for 3x3)
    /// * `stride` - Step size
    /// * `padding` - Zero-padding
    pub fn new(channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let mut rng = rand::rng();
        let fan_in = kernel_size * kernel_size; // single channel filter
        let std = (2.0 / fan_in as Float).sqrt();
        let weights_vec =
            cma_neural_network::init::randn_vec(channels * kernel_size * kernel_size, std, &mut rng);
        let weights =
            Array4::from_shape_vec((channels, 1, kernel_size, kernel_size), weights_vec).unwrap();
        let bias = Array1::zeros(channels);
        Self {
            channels: channels as Dim,
            kernel_size: kernel_size as Dim,
            stride: stride as Dim,
            padding: padding as Dim,
            weights,
            bias,
            use_bias: true,
        }
    }

    /// Disables bias (recommended before BatchNorm)
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }
}

impl Layer for DepthwiseConv2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let shape = input.shape();
        let data = input.data();
        let k = self.kernel_size as usize;
        let s = self.stride as usize;
        let p = self.padding as usize;
        let channels = self.channels as usize;

        let out_h = (shape.height + 2 * p - k) / s + 1;
        let out_w = (shape.width + 2 * p - k) / s + 1;

        let mut output = Array4::zeros((shape.batch, channels, out_h, out_w));

        for b in 0..shape.batch {
            for c in 0..channels {
                let bias_val = if self.use_bias { self.bias[c] } else { 0.0 };
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc: Float = bias_val;
                        for kh in 0..k {
                            for kw in 0..k {
                                let ih = (oh * s + kh).wrapping_sub(p);
                                let iw = (ow * s + kw).wrapping_sub(p);
                                // Bounds check (handles padding)
                                if oh * s + kh >= p
                                    && ow * s + kw >= p
                                    && ih < shape.height
                                    && iw < shape.width
                                {
                                    acc += data[[b, c, ih, iw]] * self.weights[[c, 0, kh, kw]];
                                }
                            }
                        }
                        output[[b, c, oh, ow]] = acc;
                    }
                }
            }
        }

        Tensor4D::from_array(output)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::DepthwiseConv2D
    }

    fn num_parameters(&self) -> usize {
        let weights = self.channels as usize * self.kernel_size as usize * self.kernel_size as usize;
        let bias = if self.use_bias { self.channels as usize } else { 0 };
        weights + bias
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape.after_conv(self.channels as usize, self.kernel_size as usize, self.stride as usize, self.padding as usize)
    }

    fn summary(&self) -> String {
        format!(
            "DepthwiseConv2D({} ch, {}x{}, stride={}, pad={})",
            self.channels, self.kernel_size, self.kernel_size, self.stride, self.padding
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MaxPool2D - Max Pooling
// ═══════════════════════════════════════════════════════════════════════════

/// 2D Max Pooling Layer
///
/// Reduces the spatial dimension by taking the maximum over each window.
/// Introduces invariance to small translations.
///
/// # Example
///
/// ```rust,ignore
/// // Pool 2x2 with stride 2 → halves the resolution
/// let pool = MaxPool2D::new(2, 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool2D {
    pub pool_size: Dim,
    pub stride: Dim,
}

impl MaxPool2D {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self { pool_size: pool_size as Dim, stride: stride as Dim }
    }

    /// Pool 2x2 stride 2 (most common)
    pub fn default_2x2() -> Self {
        Self::new(2, 2)
    }
}

impl Layer for MaxPool2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let (output, _indices) = maxpool2d(input, self.pool_size as usize, self.stride as usize);
        output
    }

    fn layer_type(&self) -> LayerType {
        LayerType::MaxPool2D
    }

    fn num_parameters(&self) -> usize {
        0 // No trainable parameters
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape.after_pool(self.pool_size as usize, self.stride as usize)
    }

    fn summary(&self) -> String {
        format!(
            "MaxPool2D({}x{}, stride={})",
            self.pool_size, self.pool_size, self.stride
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AvgPool2D - Average Pooling
// ═══════════════════════════════════════════════════════════════════════════

/// 2D Average Pooling Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvgPool2D {
    pub pool_size: Dim,
    pub stride: Dim,
}

impl AvgPool2D {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self { pool_size: pool_size as Dim, stride: stride as Dim }
    }
}

impl Layer for AvgPool2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        avgpool2d(input, self.pool_size as usize, self.stride as usize)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::AvgPool2D
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape.after_pool(self.pool_size as usize, self.stride as usize)
    }

    fn summary(&self) -> String {
        format!(
            "AvgPool2D({}x{}, stride={})",
            self.pool_size, self.pool_size, self.stride
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GlobalAvgPool2D - Global Average Pooling
// ═══════════════════════════════════════════════════════════════════════════

/// Global Average Pooling 2D
///
/// Reduces [batch, channels, H, W] → [batch, channels, 1, 1]
/// Used in modern architectures (ResNet, EfficientNet) instead of
/// fully-connected layers.
///
/// # Advantages
/// - No trainable parameters
/// - Implicit regularization
/// - Input size invariance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalAvgPool2D;

impl GlobalAvgPool2D {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GlobalAvgPool2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GlobalAvgPool2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        global_avgpool2d(input)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::GlobalAvgPool2D
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape.after_global_pool()
    }

    fn summary(&self) -> String {
        "GlobalAvgPool2D".to_string()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BatchNorm2D - Batch Normalization
// ═══════════════════════════════════════════════════════════════════════════

/// Batch Normalization 2D (Ioffe & Szegedy, 2015)
///
/// Normalizes activations per batch to stabilize training.
///
/// # Formula
/// ```text
/// y = γ * (x - μ) / √(σ² + ε) + β
/// ```
///
/// # Advantages
/// - Allows higher learning rates
/// - Reduces dependency on initialization
/// - Implicit regularization
#[derive(Debug, Serialize, Deserialize)]
pub struct BatchNorm2D {
    pub num_features: Dim,
    /// Learned parameters: scale (γ)
    pub gamma: Array1<Float>,
    /// Learned parameters: shift (β)
    pub beta: Array1<Float>,
    /// Running mean (for inference) — RwLock for thread-safe interior mutability
    /// Allows forward(&self) to update running stats during training
    #[serde(serialize_with = "serialize_rwlock", deserialize_with = "deserialize_rwlock")]
    pub running_mean: RwLock<Array1<Float>>,
    /// Running variance (for inference)
    #[serde(serialize_with = "serialize_rwlock", deserialize_with = "deserialize_rwlock")]
    pub running_var: RwLock<Array1<Float>>,
    /// Momentum for running stats (EMA: running = (1-m)*running + m*batch)
    pub momentum: Float,
    /// Epsilon for numerical stability
    pub eps: Float,
    /// Training mode (true) or eval (false)
    pub training: bool,
}

/// Manual Clone impl since RwLock doesn't derive Clone
impl Clone for BatchNorm2D {
    fn clone(&self) -> Self {
        Self {
            num_features: self.num_features,
            gamma: self.gamma.clone(),
            beta: self.beta.clone(),
            running_mean: RwLock::new(self.running_mean.read().unwrap().clone()),
            running_var: RwLock::new(self.running_var.read().unwrap().clone()),
            momentum: self.momentum,
            eps: self.eps,
            training: self.training,
        }
    }
}

/// Serde helper: serialize RwLock<Array1<Float>> transparently as Array1<Float>
fn serialize_rwlock<S>(val: &RwLock<Array1<Float>>, s: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    val.read().unwrap().serialize(s)
}

/// Serde helper: deserialize Array1<Float> into RwLock<Array1<Float>>
fn deserialize_rwlock<'de, D>(d: D) -> Result<RwLock<Array1<Float>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let arr = Array1::<Float>::deserialize(d)?;
    Ok(RwLock::new(arr))
}

impl BatchNorm2D {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features: num_features as Dim,
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            running_mean: RwLock::new(Array1::zeros(num_features)),
            running_var: RwLock::new(Array1::ones(num_features)),
            momentum: 0.1,
            eps: 1e-5,
            training: true,
        }
    }

    pub fn eval_mode(&mut self) {
        self.training = false;
    }

    pub fn train_mode(&mut self) {
        self.training = true;
    }

    /// Direct access to running_mean (for debug/tests)
    pub fn get_running_mean(&self) -> Array1<Float> {
        self.running_mean.read().unwrap().clone()
    }

    /// Direct access to running_var (for debug/tests)
    pub fn get_running_var(&self) -> Array1<Float> {
        self.running_var.read().unwrap().clone()
    }
}

impl Layer for BatchNorm2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        #[cfg(feature = "parallel")]
        return self.forward_parallel(input);

        #[cfg(not(feature = "parallel"))]
        self.forward_sequential(input)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::BatchNorm2D
    }

    fn num_parameters(&self) -> usize {
        // gamma and beta are learned
        self.num_features as usize * 2
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape // BatchNorm does not change the shape
    }

    fn summary(&self) -> String {
        format!("BatchNorm2D({})", self.num_features)
    }
}

impl BatchNorm2D {
    /// Sequential version of the forward pass
    ///
    /// In training mode: uses batch stats and updates running_mean/running_var via EMA
    /// In eval mode: uses stable running_mean/running_var
    #[allow(dead_code)]
    fn forward_sequential(&self, input: &Tensor4D) -> Tensor4D {
        let shape = input.shape();
        let data = input.data();

        let mut output =
            ndarray::Array4::zeros((shape.batch, shape.channels, shape.height, shape.width));

        let n = (shape.batch * shape.height * shape.width) as Float;
        let n_inv = 1.0 / n;

        for c in 0..shape.channels {
            // Channel extraction with ndarray slicing (vectorized)
            let channel_data = data.slice(ndarray::s![.., c, .., ..]);

            let (mean, var) = if self.training {
                // Optimization: uses vectorized iter().sum() and iter().map()
                let sum: Float = channel_data.iter().copied().sum();
                let mean = sum * n_inv;

                // Optimized variance: single pass
                let var: Float = channel_data
                    .iter()
                    .map(|&x| {
                        let diff = x - mean;
                        diff * diff
                    })
                    .sum::<Float>()
                    * n_inv;

                // EMA update of running stats (interior mutability via RwLock)
                // running = (1 - momentum) * running + momentum * batch
                let m = self.momentum;
                {
                    let mut rm = self.running_mean.write().unwrap();
                    rm[c] = (1.0 - m) * rm[c] + m * mean;
                }
                {
                    let mut rv = self.running_var.write().unwrap();
                    // Uses unbiased variance for running stats (Bessel's correction: n/(n-1))
                    let unbiased_var = if n > 1.0 { var * n / (n - 1.0) } else { var };
                    rv[c] = (1.0 - m) * rv[c] + m * unbiased_var;
                }

                (mean, var)
            } else {
                // Eval mode: uses accumulated running stats
                let rm = self.running_mean.read().unwrap();
                let rv = self.running_var.read().unwrap();
                (rm[c], rv[c])
            };

            let std_inv = 1.0 / (var + self.eps).sqrt();
            let gamma = self.gamma[c];
            let beta = self.beta[c];

            // Precompute constants to avoid repeated calculations
            let scale = gamma * std_inv;
            let shift = beta - mean * scale;

            // Optimized normalization: applied per slice
            let mut out_channel = output.slice_mut(ndarray::s![.., c, .., ..]);
            for (out_val, &in_val) in out_channel.iter_mut().zip(channel_data.iter()) {
                *out_val = in_val * scale + shift;
            }
        }

        Tensor4D::from_array(output)
    }

    /// Parallel version of the forward pass - parallelizes over channels
    #[cfg(feature = "parallel")]
    fn forward_parallel(&self, input: &Tensor4D) -> Tensor4D {
        let shape = input.shape();
        let data = input.data();

        let n = (shape.batch * shape.height * shape.width) as Float;
        let n_inv = 1.0 / n;

        // In eval mode, we can read running stats in a thread-safe manner
        let rm_snapshot: Option<Array1<Float>> = if !self.training {
            Some(self.running_mean.read().unwrap().clone())
        } else {
            None
        };
        let rv_snapshot: Option<Array1<Float>> = if !self.training {
            Some(self.running_var.read().unwrap().clone())
        } else {
            None
        };

        // Compute (scale, shift, batch_stats) per channel in parallel — no Vec<Float> per channel
        let channel_params: Vec<(usize, Float, Float, Option<(Float, Float)>)> =
            (0..shape.channels)
                .into_par_iter()
                .map(|c| {
                    let channel_data = data.slice(ndarray::s![.., c, .., ..]);

                    let (mean, var, batch_stats) = if self.training {
                        let sum: Float = channel_data.iter().copied().sum();
                        let mean = sum * n_inv;
                        let var: Float = channel_data
                            .iter()
                            .map(|&x| {
                                let diff = x - mean;
                                diff * diff
                            })
                            .sum::<Float>()
                            * n_inv;
                        (mean, var, Some((mean, var)))
                    } else {
                        let rm = rm_snapshot.as_ref().unwrap();
                        let rv = rv_snapshot.as_ref().unwrap();
                        (rm[c], rv[c], None)
                    };

                    let std_inv = 1.0 / (var + self.eps).sqrt();
                    let scale = self.gamma[c] * std_inv;
                    let shift = self.beta[c] - mean * scale;

                    (c, scale, shift, batch_stats)
                })
                .collect();

        // Rebuild output and update running stats sequentially
        // Direct Zip write: avoids per-channel Vec<Float> alloc from the par_iter phase
        let mut output =
            ndarray::Array4::zeros((shape.batch, shape.channels, shape.height, shape.width));

        let m = self.momentum;
        for (c, scale, shift, batch_stats) in channel_params {
            ndarray::Zip::from(output.slice_mut(ndarray::s![.., c, .., ..]))
                .and(data.slice(ndarray::s![.., c, .., ..]))
                .for_each(|o, &x| *o = x * scale + shift);

            // EMA update (sequential after par_iter)
            if let Some((batch_mean, batch_var)) = batch_stats {
                {
                    let mut rm = self.running_mean.write().unwrap();
                    rm[c] = (1.0 - m) * rm[c] + m * batch_mean;
                }
                {
                    let mut rv = self.running_var.write().unwrap();
                    let unbiased_var = if n > 1.0 {
                        batch_var * n / (n - 1.0)
                    } else {
                        batch_var
                    };
                    rv[c] = (1.0 - m) * rv[c] + m * unbiased_var;
                }
            }
        }

        Tensor4D::from_array(output)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dropout2D - Spatial Dropout
// ═══════════════════════════════════════════════════════════════════════════

/// Dropout 2D (Spatial Dropout)
///
/// Disables entire channels rather than individual pixels.
/// More effective for CNNs because adjacent pixels are correlated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dropout2D {
    pub rate: Float,
    pub training: bool,
}

impl Dropout2D {
    pub fn new(rate: Float) -> Self {
        Self {
            rate,
            training: true,
        }
    }

    pub fn eval_mode(&mut self) {
        self.training = false;
    }

    pub fn train_mode(&mut self) {
        self.training = true;
    }
}

impl Layer for Dropout2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        if !self.training || self.rate == 0.0 {
            return input.clone();
        }

        let mut rng = rand::rng();
        let shape = input.shape();
        let data = input.data();
        let scale = 1.0 / (1.0 - self.rate);

        let mut output = data.clone();

        // Per-channel dropout (spatial dropout) - optimized with slices
        for b in 0..shape.batch {
            for c in 0..shape.channels {
                let drop: bool = rng.random::<Float>() < self.rate;
                let mut channel_slice = output.slice_mut(ndarray::s![b, c, .., ..]);

                if drop {
                    // Optimization: fill(0.0) instead of loops
                    channel_slice.fill(0.0);
                } else {
                    // Optimization: mapv_inplace instead of loops
                    channel_slice.mapv_inplace(|x| x * scale);
                }
            }
        }

        Tensor4D::from_array(output)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Dropout2D
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape
    }

    fn summary(&self) -> String {
        format!("Dropout2D(p={})", self.rate)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten - 4D → 2D
// ═══════════════════════════════════════════════════════════════════════════

/// Flatten: Converts [batch, C, H, W] → vector for Dense layers
///
/// Used to connect CNN layers to fully-connected layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self
    }
}

impl Layer for Flatten {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        // Note: Flatten technically returns an Array2, not Tensor4D
        // But for the unified interface, we keep Tensor4D with H=1, W=flat_size
        let shape = input.shape();
        let flat_size = shape.channels * shape.height * shape.width;

        // Optimization: direct reshape if data is contiguous
        // Avoids double copy (flatten then reshape)
        if let Some(slice) = input.data().as_slice() {
            // Contiguous data: direct reshape in a single allocation
            let data =
                ndarray::Array4::from_shape_vec((shape.batch, 1, 1, flat_size), slice.to_vec())
                    .unwrap();
            Tensor4D::from_array(data)
        } else {
            // Fallback: non-contiguous data
            let mut data = ndarray::Array4::zeros((shape.batch, 1, 1, flat_size));
            for b in 0..shape.batch {
                let image = input.data().slice(ndarray::s![b, .., .., ..]);
                let out_slice = data.slice_mut(ndarray::s![b, 0, 0, ..]);
                for (dest, &src) in out_slice.into_iter().zip(image.iter()) {
                    *dest = src;
                }
            }
            Tensor4D::from_array(data)
        }
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Flatten
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        let flat_size = input_shape.channels * input_shape.height * input_shape.width;
        TensorShape::new(input_shape.batch, 1, 1, flat_size)
    }

    fn summary(&self) -> String {
        "Flatten".to_string()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Activation Layer Wrapper
// ═══════════════════════════════════════════════════════════════════════════

/// Activation layer (wrapper around cma_neural_network::Activation)
///
/// Reuses activations from cma-neural-network by applying them
/// element-wise on Tensor4D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationLayer {
    pub activation: cma_neural_network::Activation,
}

impl ActivationLayer {
    /// Creates a layer with the specified activation
    pub fn new(activation: cma_neural_network::Activation) -> Self {
        Self { activation }
    }

    pub fn relu() -> Self {
        Self::new(cma_neural_network::Activation::ReLU)
    }

    pub fn leaky_relu() -> Self {
        Self::new(cma_neural_network::Activation::LeakyReLU)
    }

    pub fn sigmoid() -> Self {
        Self::new(cma_neural_network::Activation::Sigmoid)
    }

    pub fn tanh() -> Self {
        Self::new(cma_neural_network::Activation::Tanh)
    }

    pub fn gelu() -> Self {
        Self::new(cma_neural_network::Activation::GELU)
    }

    pub fn mish() -> Self {
        Self::new(cma_neural_network::Activation::Mish)
    }

    pub fn swish() -> Self {
        Self::new(cma_neural_network::Activation::Swish)
    }

    pub fn softmax() -> Self {
        Self::new(cma_neural_network::Activation::Softmax)
    }

    pub fn elu() -> Self {
        Self::new(cma_neural_network::Activation::ELU)
    }

    pub fn selu() -> Self {
        Self::new(cma_neural_network::Activation::SELU)
    }
}

impl Layer for ActivationLayer {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        // Apply activation element-wise
        // Uses the same logic as cma_neural_network::Activation::apply
        // but on a scalar instead of an Array1
        input.map(|x| apply_activation_scalar(self.activation, x))
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Activation
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape
    }

    fn summary(&self) -> String {
        self.activation.name().to_string()
    }
}

/// Applies an activation on a scalar.
/// Delegates to `cma_neural_network::Activation::apply_scalar` to avoid duplication.
fn apply_activation_scalar(activation: cma_neural_network::Activation, x: Float) -> Float {
    activation.apply_scalar(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_creation() {
        let conv = Conv2D::new(1, 32, 3, 1, 1);
        assert_eq!(conv.in_channels, 1);
        assert_eq!(conv.out_channels, 32);
        assert_eq!(conv.kernel_size, 3);
        // He init: variance ≈ 2/9
        assert!(conv.weights.iter().all(|&w| w.abs() < 3.0));
    }

    #[test]
    fn test_conv2d_output_shape() {
        let conv = Conv2D::new(1, 32, 5, 1, 0);
        let input_shape = TensorShape::new(1, 1, 28, 28);
        let output_shape = conv.output_shape(input_shape);

        // 28 - 5 + 1 = 24
        assert_eq!(output_shape.height, 24);
        assert_eq!(output_shape.width, 24);
        assert_eq!(output_shape.channels, 32);
    }

    #[test]
    fn test_conv2d_same_padding() {
        let conv = Conv2D::same_padding(1, 32, 3);
        let input_shape = TensorShape::new(1, 1, 28, 28);
        let output_shape = conv.output_shape(input_shape);

        // Same padding preserves spatial size
        assert_eq!(output_shape.height, 28);
        assert_eq!(output_shape.width, 28);
    }

    #[test]
    fn test_maxpool2d() {
        let pool = MaxPool2D::default_2x2();
        let input_shape = TensorShape::new(1, 32, 24, 24);
        let output_shape = pool.output_shape(input_shape);

        // 24 / 2 = 12
        assert_eq!(output_shape.height, 12);
        assert_eq!(output_shape.width, 12);
        assert_eq!(output_shape.channels, 32);
    }

    #[test]
    fn test_batchnorm2d() {
        let bn = BatchNorm2D::new(32);
        assert_eq!(bn.num_parameters(), 64); // gamma + beta

        let input_shape = TensorShape::new(4, 32, 14, 14);
        assert_eq!(bn.output_shape(input_shape), input_shape);
    }

    #[test]
    fn test_batchnorm2d_running_stats_update() {
        let bn = BatchNorm2D::new(2);
        // Initial running stats: mean=0, var=1
        assert_eq!(bn.get_running_mean(), Array1::<Float>::zeros(2));
        assert_eq!(bn.get_running_var(), Array1::<Float>::ones(2));

        // Forward pass in training mode with non-zero data
        // Channel 0: all 2.0, Channel 1: all -1.0
        let mut data = ndarray::Array4::zeros((4, 2, 3, 3));
        data.slice_mut(ndarray::s![.., 0, .., ..]).fill(2.0);
        data.slice_mut(ndarray::s![.., 1, .., ..]).fill(-1.0);
        let input = Tensor4D::from_array(data);

        let _output = bn.forward(&input);

        // running_mean should have moved towards batch means (2.0 and -1.0)
        let rm = bn.get_running_mean();
        assert!(
            rm[0] > 0.0,
            "running_mean[0] should be positive after seeing 2.0s, got {}",
            rm[0]
        );
        assert!(
            rm[1] < 0.0,
            "running_mean[1] should be negative after seeing -1.0s, got {}",
            rm[1]
        );
        // With momentum=0.1: running_mean = 0.9*0 + 0.1*batch_mean = 0.1*2.0 = 0.2
        assert!((rm[0] - 0.2).abs() < 1e-5, "expected ~0.2, got {}", rm[0]);
        assert!((rm[1] - (-0.1)).abs() < 1e-5, "expected ~-0.1, got {}", rm[1]);
    }

    #[test]
    fn test_batchnorm2d_eval_uses_running_stats() {
        let mut bn = BatchNorm2D::new(1);

        // Manually set running stats
        *bn.running_mean.write().unwrap() = Array1::from_vec(vec![5.0]);
        *bn.running_var.write().unwrap() = Array1::from_vec(vec![4.0]);
        bn.eval_mode();

        // Input: constant 7.0
        let data = ndarray::Array4::from_elem((1, 1, 2, 2), 7.0);
        let input = Tensor4D::from_array(data);
        let output = bn.forward(&input);

        // Expected: gamma * (7 - 5) / sqrt(4 + 1e-5) + beta = 1.0 * 2 / 2.0 + 0 = 1.0
        let val = output.data()[[0, 0, 0, 0]];
        assert!(
            (val - 1.0).abs() < 1e-4,
            "eval mode should use running stats, expected ~1.0, got {}",
            val
        );
    }

    #[test]
    fn test_flatten_output() {
        let flatten = Flatten::new();
        let input_shape = TensorShape::new(4, 64, 7, 7);
        let output_shape = flatten.output_shape(input_shape);

        assert_eq!(output_shape.batch, 4);
        assert_eq!(output_shape.width, 64 * 7 * 7);
    }

    #[test]
    fn test_activation_layer() {
        let relu = ActivationLayer::relu();
        let input = Tensor4D::from_array(ndarray::Array4::from_elem((1, 1, 2, 2), -1.0));
        let output = relu.forward(&input);

        assert!(output.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_layer_parameters_count() {
        // Conv2D: 32 * 1 * 5 * 5 + 32 = 832
        let conv = Conv2D::new(1, 32, 5, 1, 0);
        assert_eq!(conv.num_parameters(), 832);

        // Without bias: 32 * 1 * 5 * 5 = 800
        let conv_no_bias = Conv2D::new(1, 32, 5, 1, 0).without_bias();
        assert_eq!(conv_no_bias.num_parameters(), 800);

        // Pool has no parameters
        let pool = MaxPool2D::new(2, 2);
        assert_eq!(pool.num_parameters(), 0);
    }
}
