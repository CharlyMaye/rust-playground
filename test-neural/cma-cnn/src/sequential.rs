//! # Sequential Container
//!
//! Container for stacking layers sequentially.
//!
//! ## Exemple
//!
//! ```rust,ignore
//! use cma_cnn::{Sequential, Conv2D, MaxPool2D, Flatten, ActivationLayer};
//!
//! let model = Sequential::new()
//!     .add_conv2d(Conv2D::new(1, 32, 5, 1, 0))
//!     .add_activation(ActivationLayer::relu())
//!     .add_pool(MaxPool2D::new(2, 2))
//!     .add_flatten(Flatten::new());
//! ```

use serde::{Deserialize, Serialize};

use crate::layers::{
    ActivationLayer, AvgPool2D, BatchNorm2D, Conv2D, Dropout2D, Flatten, GlobalAvgPool2D, Layer,
    MaxPool2D,
};
use crate::tensor::{Tensor4D, TensorShape};

/// Boxed layer for heterogeneous storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoxedLayer {
    Conv2D(Conv2D),
    MaxPool2D(MaxPool2D),
    AvgPool2D(AvgPool2D),
    GlobalAvgPool2D(GlobalAvgPool2D),
    BatchNorm2D(BatchNorm2D),
    Dropout2D(Dropout2D),
    Flatten(Flatten),
    Activation(ActivationLayer),
}

impl BoxedLayer {
    fn as_layer(&self) -> &dyn Layer {
        match self {
            BoxedLayer::Conv2D(l) => l,
            BoxedLayer::MaxPool2D(l) => l,
            BoxedLayer::AvgPool2D(l) => l,
            BoxedLayer::GlobalAvgPool2D(l) => l,
            BoxedLayer::BatchNorm2D(l) => l,
            BoxedLayer::Dropout2D(l) => l,
            BoxedLayer::Flatten(l) => l,
            BoxedLayer::Activation(l) => l,
        }
    }

    /// Returns the layer type name (e.g. "Conv2D", "MaxPool2D", "ReLU")
    pub fn type_name(&self) -> &str {
        match self {
            BoxedLayer::Conv2D(_) => "Conv2D",
            BoxedLayer::MaxPool2D(_) => "MaxPool2D",
            BoxedLayer::AvgPool2D(_) => "AvgPool2D",
            BoxedLayer::GlobalAvgPool2D(_) => "GlobalAvgPool2D",
            BoxedLayer::BatchNorm2D(_) => "BatchNorm2D",
            BoxedLayer::Dropout2D(_) => "Dropout2D",
            BoxedLayer::Flatten(_) => "Flatten",
            BoxedLayer::Activation(_) => "Activation",
        }
    }

    /// Returns a human-readable config string for this layer
    pub fn config_string(&self) -> String {
        match self {
            BoxedLayer::Conv2D(c) => format!(
                "{}→{}, {}×{}, s={}, p={}",
                c.in_channels, c.out_channels, c.kernel_size, c.kernel_size, c.stride, c.padding
            ),
            BoxedLayer::MaxPool2D(p) => format!("{}×{}, s={}", p.pool_size, p.pool_size, p.stride),
            BoxedLayer::AvgPool2D(p) => format!("{}×{}, s={}", p.pool_size, p.pool_size, p.stride),
            BoxedLayer::GlobalAvgPool2D(_) => "→1×1".to_string(),
            BoxedLayer::BatchNorm2D(bn) => format!("features={}", bn.num_features),
            BoxedLayer::Dropout2D(d) => format!("p={}", d.rate),
            BoxedLayer::Flatten(_) => String::new(),
            BoxedLayer::Activation(a) => a.activation.name().to_string(),
        }
    }
}

/// Sequential container for stacking layers
///
/// # Architecture
///
/// Layers are executed in the order they were added:
/// ```text
/// Input → Layer1 → Layer2 → ... → LayerN → Output
/// ```
///
/// # Exemple (LeNet-5 style)
///
/// ```rust,ignore
/// let model = Sequential::new()
///     // Block 1
///     .add_conv2d(Conv2D::new(1, 6, 5, 1, 0))    // 28x28 → 24x24
///     .add_activation(ActivationLayer::relu())
///     .add_pool(MaxPool2D::new(2, 2))            // 24x24 → 12x12
///     // Block 2
///     .add_conv2d(Conv2D::new(6, 16, 5, 1, 0))   // 12x12 → 8x8
///     .add_activation(ActivationLayer::relu())
///     .add_pool(MaxPool2D::new(2, 2))            // 8x8 → 4x4
///     // Flatten
///     .add_flatten(Flatten::new());              // 16*4*4 = 256
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequential {
    layers: Vec<BoxedLayer>,
    /// Model name (for debug/logging)
    name: String,
}

impl Sequential {
    /// Creates a new empty Sequential
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            name: "Sequential".to_string(),
        }
    }

    /// Creates a Sequential with a custom name
    pub fn named(name: &str) -> Self {
        Self {
            layers: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Adds a Conv2D layer
    pub fn add_conv2d(mut self, layer: Conv2D) -> Self {
        self.layers.push(BoxedLayer::Conv2D(layer));
        self
    }

    /// Adds a MaxPool2D layer
    pub fn add_maxpool(mut self, layer: MaxPool2D) -> Self {
        self.layers.push(BoxedLayer::MaxPool2D(layer));
        self
    }

    /// Adds an AvgPool2D layer
    pub fn add_avgpool(mut self, layer: AvgPool2D) -> Self {
        self.layers.push(BoxedLayer::AvgPool2D(layer));
        self
    }

    /// Adds a GlobalAvgPool2D layer
    pub fn add_global_avgpool(mut self) -> Self {
        self.layers
            .push(BoxedLayer::GlobalAvgPool2D(GlobalAvgPool2D::new()));
        self
    }

    /// Adds a BatchNorm2D layer
    pub fn add_batchnorm(mut self, layer: BatchNorm2D) -> Self {
        self.layers.push(BoxedLayer::BatchNorm2D(layer));
        self
    }

    /// Adds a Dropout2D layer
    pub fn add_dropout(mut self, layer: Dropout2D) -> Self {
        self.layers.push(BoxedLayer::Dropout2D(layer));
        self
    }

    /// Adds a Flatten layer
    pub fn add_flatten(mut self) -> Self {
        self.layers.push(BoxedLayer::Flatten(Flatten::new()));
        self
    }

    /// Adds an activation layer
    pub fn add_activation(mut self, layer: ActivationLayer) -> Self {
        self.layers.push(BoxedLayer::Activation(layer));
        self
    }

    /// Shortcut: Conv2D + ReLU
    pub fn add_conv_relu(
        self,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        self.add_conv2d(Conv2D::new(in_ch, out_ch, kernel, stride, padding))
            .add_activation(ActivationLayer::relu())
    }

    /// Shortcut: Conv2D + BatchNorm + ReLU
    pub fn add_conv_bn_relu(
        self,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        self.add_conv2d(Conv2D::new(in_ch, out_ch, kernel, stride, padding).without_bias())
            .add_batchnorm(BatchNorm2D::new(out_ch))
            .add_activation(ActivationLayer::relu())
    }

    /// Forward propagation (reference version, clone required)
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        self.forward_owned(input.clone())
    }

    /// Optimized forward propagation (takes ownership, avoids clone)
    pub fn forward_owned(&self, input: Tensor4D) -> Tensor4D {
        let mut x = input;
        for layer in &self.layers {
            x = layer.as_layer().forward(&x);
        }
        x
    }

    /// Forward propagation collecting intermediate outputs for each layer.
    ///
    /// Returns a Vec of `(layer_type, config, output_tensor)` for each layer.
    /// Used for CNN visualization (feature maps at each stage).
    pub fn forward_with_intermediates(&self, input: &Tensor4D) -> Vec<(String, String, Tensor4D)> {
        let mut x = input.clone();
        let mut intermediates = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            x = layer.as_layer().forward(&x);
            intermediates.push((
                layer.type_name().to_string(),
                layer.config_string(),
                x.clone(),
            ));
        }

        intermediates
    }

    /// Batch forward propagation with memory reuse
    ///
    /// Optimized for inference over multiple consecutive batches.
    /// Avoids repeated allocations by processing data in-place.
    ///
    /// # Arguments
    /// * `inputs` - Iterator over input tensors
    /// * `callback` - Function called with each result (avoids storing all results)
    ///
    /// # Exemple
    /// ```rust,ignore
    /// model.forward_batches(test_data.iter(), |batch_idx, output| {
    ///     // Process each output without storing all in memory
    ///     predictions.extend(output.data().iter().copied());
    /// });
    /// ```
    pub fn forward_batches<I, F>(&self, inputs: I, mut callback: F)
    where
        I: Iterator<Item = Tensor4D>,
        F: FnMut(usize, Tensor4D),
    {
        for (idx, input) in inputs.enumerate() {
            let output = self.forward_owned(input);
            callback(idx, output);
        }
    }

    /// Forward propagation with result collection
    ///
    /// Convenient version that returns all results.
    /// For very large datasets, prefer `forward_batches` with callback.
    pub fn forward_all<I>(&self, inputs: I) -> Vec<Tensor4D>
    where
        I: Iterator<Item = Tensor4D>,
    {
        inputs.map(|input| self.forward_owned(input)).collect()
    }

    /// Total parameter count
    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.as_layer().num_parameters())
            .sum()
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Computes the output shape for a given input shape
    pub fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        let mut shape = input_shape;
        for layer in &self.layers {
            shape = layer.as_layer().output_shape(shape);
        }
        shape
    }

    /// Prints a model summary (Keras style)
    pub fn summary(&self, input_shape: TensorShape) {
        println!("Model: {}", self.name);
        println!("{}", "=".repeat(70));
        println!(
            "{:<30} {:>20} {:>15}",
            "Layer (type)", "Output Shape", "Param #"
        );
        println!("{}", "=".repeat(70));

        let mut shape = input_shape;
        let mut total_params = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            let l = layer.as_layer();
            shape = l.output_shape(shape);
            let params = l.num_parameters();
            total_params += params;

            println!(
                "{:<30} {:>20} {:>15}",
                format!("{} ({})", l.summary(), i),
                format!("{}", shape),
                params
            );
        }

        println!("{}", "=".repeat(70));
        println!("Total params: {}", total_params);
        println!("Trainable params: {}", total_params);
        println!("Non-trainable params: 0");
        println!("{}", "=".repeat(70));
    }

    /// Switches to evaluation mode (disables dropout, uses running stats for BatchNorm)
    pub fn eval_mode(&mut self) {
        for layer in &mut self.layers {
            match layer {
                BoxedLayer::BatchNorm2D(bn) => bn.eval_mode(),
                BoxedLayer::Dropout2D(d) => d.eval_mode(),
                _ => {}
            }
        }
    }

    /// Switches to training mode
    pub fn train_mode(&mut self) {
        for layer in &mut self.layers {
            match layer {
                BoxedLayer::BatchNorm2D(bn) => bn.train_mode(),
                BoxedLayer::Dropout2D(d) => d.train_mode(),
                _ => {}
            }
        }
    }

    /// Access to layers
    pub fn layers(&self) -> &[BoxedLayer] {
        &self.layers
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_creation() {
        let model = Sequential::new()
            .add_conv2d(Conv2D::new(1, 32, 3, 1, 1))
            .add_activation(ActivationLayer::relu())
            .add_maxpool(MaxPool2D::new(2, 2));

        assert_eq!(model.num_layers(), 3);
    }

    #[test]
    fn test_sequential_output_shape() {
        let model = Sequential::new()
            .add_conv2d(Conv2D::new(1, 32, 5, 1, 0)) // 28→24
            .add_maxpool(MaxPool2D::new(2, 2)) // 24→12
            .add_conv2d(Conv2D::new(32, 64, 5, 1, 0)) // 12→8
            .add_maxpool(MaxPool2D::new(2, 2)) // 8→4
            .add_flatten(); // 64*4*4=1024

        let input_shape = TensorShape::new(1, 1, 28, 28);
        let output_shape = model.output_shape(input_shape);

        assert_eq!(output_shape.width, 64 * 4 * 4);
    }

    #[test]
    fn test_sequential_forward() {
        let model = Sequential::new()
            .add_conv2d(Conv2D::new(1, 8, 3, 1, 1))
            .add_activation(ActivationLayer::relu())
            .add_maxpool(MaxPool2D::new(2, 2));

        let input = Tensor4D::random(TensorShape::new(2, 1, 28, 28));
        let output = model.forward(&input);

        assert_eq!(output.shape().batch, 2);
        assert_eq!(output.shape().channels, 8);
        assert_eq!(output.shape().height, 14);
        assert_eq!(output.shape().width, 14);
    }

    #[test]
    fn test_sequential_params_count() {
        let model = Sequential::new()
            .add_conv2d(Conv2D::new(1, 6, 5, 1, 0)) // 1*6*5*5 + 6 = 156
            .add_conv2d(Conv2D::new(6, 16, 5, 1, 0)); // 6*16*5*5 + 16 = 2416

        assert_eq!(model.num_parameters(), 156 + 2416);
    }

    #[test]
    fn test_conv_bn_relu_shortcut() {
        let model = Sequential::new().add_conv_bn_relu(1, 32, 3, 1, 1);

        assert_eq!(model.num_layers(), 3); // Conv + BN + ReLU

        // Conv sans bias + BN (gamma + beta)
        // 32*1*3*3 = 288 + 64 = 352
        assert_eq!(model.num_parameters(), 288 + 64);
    }
}
