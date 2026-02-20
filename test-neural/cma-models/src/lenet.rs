//! # LeNet-5 (LeCun et al., 1998)
//!
//! First commercially successful CNN architecture, used for handwritten
//! digit recognition on bank checks.
//!
//! ## Original Paper
//!
//! **"Gradient-Based Learning Applied to Document Recognition"**
//! Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
//! Proceedings of the IEEE, 1998
//! http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
//!
//! ## Original Architecture (32x32 input)
//!
//! ```text
//! Input (32x32x1)
//!     │
//! ┌───▼───────────────────┐
//! │ C1: Conv 5x5, 6 maps  │ → 28x28x6
//! │ S2: AvgPool 2x2       │ → 14x14x6
//! └───┬───────────────────┘
//!     │
//! ┌───▼───────────────────┐
//! │ C3: Conv 5x5, 16 maps │ → 10x10x16
//! │ S4: AvgPool 2x2       │ → 5x5x16
//! └───┬───────────────────┘
//!     │
//! ┌───▼───────────────────┐
//! │ C5: Conv 5x5, 120     │ → 1x1x120
//! │ F6: FC 120 → 84       │
//! │ Output: FC 84 → 10    │
//! └───────────────────────┘
//! ```
//!
//! ## MNIST Adaptation (28x28 input)
//!
//! For MNIST, we use initial padding or adapt the sizes.
//! This implementation supports both modes.
//!
//! ## Key Innovations (1998)
//!
//! 1. **Weight sharing**: same filter applied everywhere
//! 2. **Subsampling**: reduces spatial dimension
//! 3. **Deep architecture**: 7 layers (revolutionary for the time)
//! 4. **End-to-end backpropagation**: differentiable training
//!
//! ## Exemple
//!
//! ```rust,ignore
//! use cma_models::lenet::{LeNet5, LeNet5Config};
//!
//! // Standard version for MNIST
//! let model = LeNet5::new(10);
//!
//! // Custom version
//! let model = LeNet5::with_config(LeNet5Config {
//!     num_classes: 10,
//!     input_size: 28,
//!     use_batch_norm: true,  // Modernization
//!     activation: "relu",     // ReLU instead of tanh
//! });
//! ```

use serde::{Deserialize, Serialize};

use cma_cnn::{ActivationLayer, AvgPool2D, BatchNorm2D, Conv2D, Sequential, Tensor4D, TensorShape};

/// LeNet-5 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeNet5Config {
    /// Number of output classes (10 for MNIST)
    pub num_classes: usize,
    /// Input image size (28 for MNIST, 32 for original)
    pub input_size: usize,
    /// Use BatchNorm (modernization, not in original paper)
    pub use_batch_norm: bool,
    /// Activation: "tanh" (original) or "relu" (modern)
    pub activation: String,
    /// Number of input channels (1 for grayscale)
    pub in_channels: usize,
}

impl Default for LeNet5Config {
    fn default() -> Self {
        Self {
            num_classes: 10,
            input_size: 28,
            use_batch_norm: false,
            activation: "tanh".to_string(), // Faithful to the paper
            in_channels: 1,
        }
    }
}

impl LeNet5Config {
    /// Config for MNIST (28x28)
    pub fn mnist() -> Self {
        Self::default()
    }

    /// Original paper config (32x32)
    pub fn original() -> Self {
        Self {
            num_classes: 10,
            input_size: 32,
            use_batch_norm: false,
            activation: "tanh".to_string(),
            in_channels: 1,
        }
    }

    /// Modern config with ReLU and BatchNorm
    pub fn modern() -> Self {
        Self {
            num_classes: 10,
            input_size: 28,
            use_batch_norm: true,
            activation: "relu".to_string(),
            in_channels: 1,
        }
    }
}

/// LeNet-5: Historic CNN Architecture (1998)
///
/// # Architecture
///
/// | Layer | Type | Output Shape | Params |
/// |--------|------|--------------|--------|
/// | Input | - | 1×28×28 | 0 |
/// | C1 | Conv 5×5, 6 | 6×24×24 | 156 |
/// | S2 | AvgPool 2×2 | 6×12×12 | 0 |
/// | C3 | Conv 5×5, 16 | 16×8×8 | 2,416 |
/// | S4 | AvgPool 2×2 | 16×4×4 | 0 |
/// | C5 | Conv 4×4, 120 | 120×1×1 | 30,840 |
/// | **Total** | | | **~33k** |
///
/// Note: The original paper used a partial connection table for C3,
/// this implementation uses full connections (modern standard).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeNet5 {
    /// Convolutional layers
    pub conv_layers: Sequential,
    /// Configuration
    pub config: LeNet5Config,
}

impl LeNet5 {
    /// Creates LeNet-5 with default configuration for MNIST
    ///
    /// # Arguments
    /// * `num_classes` - Number of classes (10 for MNIST)
    ///
    /// # Exemple
    /// ```rust,ignore
    /// let model = LeNet5::new(10);
    /// assert_eq!(model.num_parameters(), 33_456);
    /// ```
    pub fn new(num_classes: usize) -> Self {
        let mut config = LeNet5Config::mnist();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    /// Creates LeNet-5 with custom configuration
    pub fn with_config(config: LeNet5Config) -> Self {
        let activation = match config.activation.as_str() {
            "relu" => ActivationLayer::relu(),
            "sigmoid" => ActivationLayer::sigmoid(),
            _ => ActivationLayer::tanh(), // Default: tanh (original)
        };

        // Calculate C5 kernel size to get output 1x1
        // For input 28x28: after C1(5x5) → 24, S2(2x2) → 12, C3(5x5) → 8, S4(2x2) → 4
        // So C5 kernel = 4 for 28x28
        // For input 32x32: after C1 → 28, S2 → 14, C3 → 10, S4 → 5
        // So C5 kernel = 5 for 32x32
        let c5_kernel = if config.input_size == 32 { 5 } else { 4 };

        let mut conv_layers = Sequential::named("LeNet-5");

        // C1: Convolution Layer (6 feature maps, 5x5 kernel)
        conv_layers = conv_layers.add_conv2d(Conv2D::new(config.in_channels, 6, 5, 1, 0));
        if config.use_batch_norm {
            conv_layers = conv_layers.add_batchnorm(BatchNorm2D::new(6));
        }
        conv_layers = conv_layers.add_activation(activation.clone());

        // S2: Subsampling (Average Pooling 2x2)
        // Note: The original paper used a form of pooling with learned weights
        conv_layers = conv_layers.add_avgpool(AvgPool2D::new(2, 2));

        // C3: Convolution Layer (16 feature maps, 5x5 kernel)
        conv_layers = conv_layers.add_conv2d(Conv2D::new(6, 16, 5, 1, 0));
        if config.use_batch_norm {
            conv_layers = conv_layers.add_batchnorm(BatchNorm2D::new(16));
        }
        conv_layers = conv_layers.add_activation(activation.clone());

        // S4: Subsampling (Average Pooling 2x2)
        conv_layers = conv_layers.add_avgpool(AvgPool2D::new(2, 2));

        // C5: Convolution Layer (120 feature maps)
        // In the paper, C5 is actually a disguised fully-connected layer
        conv_layers = conv_layers.add_conv2d(Conv2D::new(16, 120, c5_kernel, 1, 0));
        if config.use_batch_norm {
            conv_layers = conv_layers.add_batchnorm(BatchNorm2D::new(120));
        }
        conv_layers = conv_layers.add_activation(activation);

        // Flatten for connection to FC layers
        conv_layers = conv_layers.add_flatten();

        Self {
            conv_layers,
            config,
        }
    }

    /// Forward pass (conv layers only)
    ///
    /// Returns flattened features ready for Dense layers.
    /// Use a Network from cma-neural-network for the FC layers.
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        self.conv_layers.forward(input)
    }

    /// Total number of parameters (conv layers)
    pub fn num_parameters(&self) -> usize {
        self.conv_layers.num_parameters()
    }

    /// Prints the model summary
    pub fn summary(&self) {
        let input_shape = TensorShape::new(
            1,
            self.config.in_channels,
            self.config.input_size,
            self.config.input_size,
        );
        self.conv_layers.summary(input_shape);
    }

    /// Output shape (flattened features)
    pub fn output_size(&self) -> usize {
        let input_shape = TensorShape::new(
            1,
            self.config.in_channels,
            self.config.input_size,
            self.config.input_size,
        );
        let output = self.conv_layers.output_shape(input_shape);
        output.width // After flatten
    }
}

/// Creates the FC classifier for LeNet-5
///
/// # FC Architecture (original paper)
/// - F6: 120 → 84 (tanh)
/// - Output: 84 → num_classes
///
/// # Exemple
/// ```rust,ignore
/// use cma_neural_network::NetworkBuilder;
/// use cma_models::lenet::{LeNet5, create_lenet5_classifier};
///
/// let cnn = LeNet5::new(10);
/// let classifier = create_lenet5_classifier(cnn.output_size(), 10);
/// ```
pub fn create_lenet5_classifier(input_size: usize, num_classes: usize) -> String {
    // Returns the recommended configuration for the NetworkBuilder
    format!(
        r#"// FC Classifier for LeNet-5
// Input: {} features (CNN output)
// Output: {} classes

use cma_neural_network::{{NetworkBuilder, Activation, LossFunction, OptimizerType}};

let classifier = NetworkBuilder::new({}, {})
    .hidden_layer(84, Activation::Tanh)  // F6 from the paper
    .output_activation(Activation::Softmax)
    .loss(LossFunction::CategoricalCrossEntropy)
    .optimizer(OptimizerType::adam(0.001))
    .build();"#,
        input_size, num_classes, input_size, num_classes
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lenet5_creation() {
        let model = LeNet5::new(10);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_lenet5_mnist_shape() {
        let model = LeNet5::new(10);
        let input = Tensor4D::zeros(TensorShape::new(1, 1, 28, 28));
        let output = model.forward(&input);

        // After C1(5x5) → 24, S2 → 12, C3(5x5) → 8, S4 → 4, C5(4x4) → 1
        // Flatten: 120 * 1 * 1 = 120
        assert_eq!(output.shape().width, 120);
    }

    #[test]
    fn test_lenet5_original_shape() {
        let model = LeNet5::with_config(LeNet5Config::original());
        let input = Tensor4D::zeros(TensorShape::new(1, 1, 32, 32));
        let output = model.forward(&input);

        // Pour 32x32: C1 → 28, S2 → 14, C3 → 10, S4 → 5, C5(5x5) → 1
        assert_eq!(output.shape().width, 120);
    }

    #[test]
    fn test_lenet5_modern() {
        let model = LeNet5::with_config(LeNet5Config::modern());
        // With BatchNorm, more parameters
        assert!(model.num_parameters() > LeNet5::new(10).num_parameters());
    }

    #[test]
    fn test_lenet5_batch() {
        let model = LeNet5::new(10);
        let batch = Tensor4D::random(TensorShape::new(32, 1, 28, 28));
        let output = model.forward(&batch);

        assert_eq!(output.shape().batch, 32);
        assert_eq!(output.shape().width, 120);
    }
}
