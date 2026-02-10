//! # VGG (Simonyan & Zisserman, 2014)
//!
//! Architecture characterized by the exclusive use of 3×3 convolutions,
//! demonstrating that depth is key to performance.
//!
//! ## Original Paper
//!
//! **"Very Deep Convolutional Networks for Large-Scale Image Recognition"**
//! Karen Simonyan, Andrew Zisserman
//! ICLR 2015 (arXiv 2014)
//! https://arxiv.org/abs/1409.1556
//!
//! ## Architecture VGG-16
//!
//! ```text
//! Input (224×224×3)
//!     │
//! ┌───▼────────────────┐
//! │ Conv 3×3, 64 (×2)  │ → 224×224×64
//! │ MaxPool 2×2        │ → 112×112×64
//! └───┬────────────────┘
//!     │
//! ┌───▼────────────────┐
//! │ Conv 3×3, 128 (×2) │ → 112×112×128
//! │ MaxPool 2×2        │ → 56×56×128
//! └───┬────────────────┘
//!     │
//! ┌───▼────────────────┐
//! │ Conv 3×3, 256 (×3) │ → 56×56×256
//! │ MaxPool 2×2        │ → 28×28×256
//! └───┬────────────────┘
//!     │
//! ┌───▼────────────────┐
//! │ Conv 3×3, 512 (×3) │ → 28×28×512
//! │ MaxPool 2×2        │ → 14×14×512
//! └───┬────────────────┘
//!     │
//! ┌───▼────────────────┐
//! │ Conv 3×3, 512 (×3) │ → 14×14×512
//! │ MaxPool 2×2        │ → 7×7×512
//! └───┬────────────────┘
//!     │
//! ┌───▼────────────────┐
//! │ Flatten            │ → 25088
//! │ FC 4096 (×2)       │
//! │ FC 1000            │
//! └────────────────────┘
//! ```
//!
//! ## Variants
//!
//! | Model | Conv Layers | Parameters |
//! |--------|--------------|------------|
//! | VGG-11 | 8 conv | 133M |
//! | VGG-13 | 10 conv | 133M |
//! | VGG-16 | 13 conv | 138M |
//! | VGG-19 | 16 conv | 144M |
//!
//! ## Key Innovations (2014)
//!
//! 1. **3×3 kernels only**: Two 3×3 = one 5×5 receptive field, but fewer params
//! 2. **Depth**: 16-19 layers (vs 8 for AlexNet)
//! 3. **Uniformity**: Very regular architecture, easy to understand
//! 4. **Pre-training**: Transferable weights for other tasks

use serde::{Deserialize, Serialize};

use cma_cnn::{ActivationLayer, BatchNorm2D, Conv2D, MaxPool2D, Sequential, Tensor4D, TensorShape};

/// VGG configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VGGConfig {
    /// Number of classes
    pub num_classes: usize,
    /// Input size (224 for ImageNet)
    pub input_size: usize,
    /// Input channels (3 for RGB)
    pub in_channels: usize,
    /// Use BatchNorm
    pub use_batch_norm: bool,
    /// Block configuration [num_convs, out_channels]
    pub blocks: Vec<(usize, usize)>,
}

impl Default for VGGConfig {
    fn default() -> Self {
        Self::vgg16()
    }
}

impl VGGConfig {
    /// VGG-16 configuration
    pub fn vgg16() -> Self {
        Self {
            num_classes: 1000,
            input_size: 224,
            in_channels: 3,
            use_batch_norm: true,
            blocks: vec![
                (2, 64),  // Block 1: 2 conv, 64 channels
                (2, 128), // Block 2: 2 conv, 128 channels
                (3, 256), // Block 3: 3 conv, 256 channels
                (3, 512), // Block 4: 3 conv, 512 channels
                (3, 512), // Block 5: 3 conv, 512 channels
            ],
        }
    }

    /// VGG-19 configuration
    pub fn vgg19() -> Self {
        Self {
            num_classes: 1000,
            input_size: 224,
            in_channels: 3,
            use_batch_norm: true,
            blocks: vec![
                (2, 64),
                (2, 128),
                (4, 256), // 4 conv instead of 3
                (4, 512), // 4 conv instead of 3
                (4, 512), // 4 conv instead of 3
            ],
        }
    }

    /// VGG-11 (lighter)
    pub fn vgg11() -> Self {
        Self {
            num_classes: 1000,
            input_size: 224,
            in_channels: 3,
            use_batch_norm: true,
            blocks: vec![(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)],
        }
    }

    /// Mini version for CIFAR-10
    pub fn cifar10() -> Self {
        Self {
            num_classes: 10,
            input_size: 32,
            in_channels: 3,
            use_batch_norm: true,
            blocks: vec![
                (2, 64),  // 32→16
                (2, 128), // 16→8
                (2, 256), // 8→4
                (2, 512), // 4→2
            ],
        }
    }
}

/// VGG-16: "Very Deep" Architecture (2014)
///
/// # Characteristics
///
/// - 13 convolutional layers
/// - All kernels are 3×3
/// - MaxPool 2×2 between blocks
/// - ~138M parameters (including 119M in FC layers)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VGG16 {
    pub features: Sequential,
    pub config: VGGConfig,
}

impl VGG16 {
    /// Creates VGG-16 for ImageNet
    pub fn new(num_classes: usize) -> Self {
        let mut config = VGGConfig::vgg16();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    /// Creates VGG-16 with custom configuration
    pub fn with_config(config: VGGConfig) -> Self {
        let features = Self::build_features(&config);
        Self { features, config }
    }

    fn build_features(config: &VGGConfig) -> Sequential {
        let mut features = Sequential::named("VGG-16");
        let mut in_channels = config.in_channels;

        for (num_convs, out_channels) in &config.blocks {
            for _ in 0..*num_convs {
                // Conv 3×3, same padding
                features = features.add_conv2d(Conv2D::new(in_channels, *out_channels, 3, 1, 1));

                if config.use_batch_norm {
                    features = features.add_batchnorm(BatchNorm2D::new(*out_channels));
                }

                features = features.add_activation(ActivationLayer::relu());
                in_channels = *out_channels;
            }

            // MaxPool after each block
            features = features.add_maxpool(MaxPool2D::new(2, 2));
        }

        features = features.add_flatten();
        features
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        self.features.forward(input)
    }

    /// Number of parameters
    pub fn num_parameters(&self) -> usize {
        self.features.num_parameters()
    }

    /// Prints the summary
    pub fn summary(&self) {
        let input_shape = TensorShape::new(
            1,
            self.config.in_channels,
            self.config.input_size,
            self.config.input_size,
        );
        self.features.summary(input_shape);
    }

    /// Output size
    pub fn output_size(&self) -> usize {
        let input_shape = TensorShape::new(
            1,
            self.config.in_channels,
            self.config.input_size,
            self.config.input_size,
        );
        let output = self.features.output_shape(input_shape);
        output.width
    }
}

/// VGG-19: Even deeper version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VGG19 {
    pub features: Sequential,
    pub config: VGGConfig,
}

impl VGG19 {
    /// Creates VGG-19 for ImageNet
    pub fn new(num_classes: usize) -> Self {
        let mut config = VGGConfig::vgg19();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    /// Creates VGG-19 with configuration
    pub fn with_config(config: VGGConfig) -> Self {
        let features = VGG16::build_features(&config);
        Self { features, config }
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        self.features.forward(input)
    }

    /// Number of parameters
    pub fn num_parameters(&self) -> usize {
        self.features.num_parameters()
    }
}

/// Creates a VGG block (n convolutions + maxpool)
pub fn vgg_block(
    in_channels: usize,
    out_channels: usize,
    num_convs: usize,
    use_bn: bool,
) -> Sequential {
    let mut block = Sequential::new();
    let mut ch = in_channels;

    for _ in 0..num_convs {
        block = block.add_conv2d(Conv2D::new(ch, out_channels, 3, 1, 1));
        if use_bn {
            block = block.add_batchnorm(BatchNorm2D::new(out_channels));
        }
        block = block.add_activation(ActivationLayer::relu());
        ch = out_channels;
    }

    block = block.add_maxpool(MaxPool2D::new(2, 2));
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vgg_block() {
        let block = vgg_block(3, 64, 2, true);
        let input = Tensor4D::zeros(TensorShape::new(1, 3, 32, 32));
        let output = block.forward(&input);

        // 32 → 16 after pool
        assert_eq!(output.shape().height, 16);
        assert_eq!(output.shape().channels, 64);
    }

    #[test]
    fn test_vgg16_cifar() {
        let model = VGG16::with_config(VGGConfig::cifar10());
        let input = Tensor4D::zeros(TensorShape::new(1, 3, 32, 32));
        let output = model.forward(&input);

        // 32 → 16 → 8 → 4 → 2, channels = 512
        // Flatten: 512 * 2 * 2 = 2048
        assert_eq!(output.shape().width, 2048);
    }

    #[test]
    fn test_vgg_configurations() {
        let vgg11 = VGGConfig::vgg11();
        let vgg16 = VGGConfig::vgg16();
        let vgg19 = VGGConfig::vgg19();

        // VGG-11: 1+1+2+2+2 = 8 conv
        let total_11: usize = vgg11.blocks.iter().map(|(n, _)| n).sum();
        assert_eq!(total_11, 8);

        // VGG-16: 2+2+3+3+3 = 13 conv
        let total_16: usize = vgg16.blocks.iter().map(|(n, _)| n).sum();
        assert_eq!(total_16, 13);

        // VGG-19: 2+2+4+4+4 = 16 conv
        let total_19: usize = vgg19.blocks.iter().map(|(n, _)| n).sum();
        assert_eq!(total_19, 16);
    }
}
