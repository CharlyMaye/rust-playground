//! # AlexNet (Krizhevsky et al., 2012)
//!
//! The architecture that sparked the Deep Learning revolution by winning
//! the ImageNet 2012 challenge by a significant margin.
//!
//! ## Original Paper
//!
//! **"ImageNet Classification with Deep Convolutional Neural Networks"**
//! Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
//! NIPS 2012
//! https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
//!
//! ## Original Architecture (227x227 RGB input)
//!
//! ```text
//! Input (227×227×3)
//!     │
//! ┌───▼────────────────────────┐
//! │ Conv1: 11×11, 96, stride 4 │ → 55×55×96
//! │ ReLU + LRN + MaxPool 3×3/2 │ → 27×27×96
//! └───┬────────────────────────┘
//!     │
//! ┌───▼────────────────────────┐
//! │ Conv2: 5×5, 256, pad 2     │ → 27×27×256
//! │ ReLU + LRN + MaxPool 3×3/2 │ → 13×13×256
//! └───┬────────────────────────┘
//!     │
//! ┌───▼────────────────────────┐
//! │ Conv3: 3×3, 384, pad 1     │ → 13×13×384
//! │ ReLU                       │
//! └───┬────────────────────────┘
//!     │
//! ┌───▼────────────────────────┐
//! │ Conv4: 3×3, 384, pad 1     │ → 13×13×384
//! │ ReLU                       │
//! └───┬────────────────────────┘
//!     │
//! ┌───▼────────────────────────┐
//! │ Conv5: 3×3, 256, pad 1     │ → 13×13×256
//! │ ReLU + MaxPool 3×3/2       │ → 6×6×256
//! └───┬────────────────────────┘
//!     │
//! ┌───▼────────────────────────┐
//! │ Flatten                    │ → 9216
//! │ FC1: 9216 → 4096 + ReLU    │
//! │ Dropout(0.5)               │
//! │ FC2: 4096 → 4096 + ReLU    │
//! │ Dropout(0.5)               │
//! │ FC3: 4096 → 1000           │
//! └────────────────────────────┘
//! ```
//!
//! ## Key Innovations (2012)
//!
//! 1. **ReLU**: First massive use, 6× faster than tanh
//! 2. **Dropout**: Revolutionary regularization (Hinton)
//! 3. **GPU Training**: Parallelization on 2 GPUs (GTX 580)
//! 4. **Data Augmentation**: Flip, crop, color jittering
//! 5. **LRN** (Local Response Normalization): replaced by BatchNorm today
//!
//! ## Results
//!
//! - **Top-5 Error**: 15.3% (vs 26.2% for 2nd place)
//! - **60M parameters** (revolutionary for the time)
//! - **5 days of training** on 2 GPUs
//!
//! ## Mini Adaptation (for smaller images)
//!
//! This implementation also provides a reduced version for CIFAR-10 (32x32).

use cma_cnn::{Dim, Float};
use serde::{Deserialize, Serialize};

use cma_cnn::{ActivationLayer, BatchNorm2D, Conv2D, MaxPool2D, Sequential, Tensor4D, TensorShape};

/// AlexNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlexNetConfig {
    /// Number of output classes (1000 for ImageNet)
    pub num_classes: Dim,
    /// Input image size (227 for ImageNet)
    pub input_size: Dim,
    /// Number of channels (3 for RGB)
    pub in_channels: Dim,
    /// Use BatchNorm instead of LRN
    pub use_batch_norm: bool,
    /// Dropout rate
    pub dropout_rate: Float,
}

impl Default for AlexNetConfig {
    fn default() -> Self {
        Self {
            num_classes: 1000,
            input_size: 227,
            in_channels: 3,
            use_batch_norm: true, // Modernization
            dropout_rate: 0.5,
        }
    }
}

impl AlexNetConfig {
    /// Config for ImageNet (original)
    pub fn imagenet() -> Self {
        Self::default()
    }

    /// Config for CIFAR-10 (32x32, 10 classes)
    /// Adapted reduced version
    pub fn cifar10() -> Self {
        Self {
            num_classes: 10,
            input_size: 32,
            in_channels: 3,
            use_batch_norm: true,
            dropout_rate: 0.5,
        }
    }

    /// Config for 64x64 images
    pub fn small(num_classes: usize) -> Self {
        Self {
            num_classes: num_classes as Dim,
            input_size: 64,
            in_channels: 3,
            use_batch_norm: true,
            dropout_rate: 0.5,
        }
    }
}

/// AlexNet: Revolutionary CNN Architecture (2012)
///
/// # Architecture (ImageNet version)
///
/// | Layer | Type | Output Shape | Params |
/// |--------|------|--------------|--------|
/// | Input | - | 3×227×227 | 0 |
/// | Conv1 | 11×11, 96, /4 | 96×55×55 | 34,944 |
/// | Pool1 | MaxPool 3×3/2 | 96×27×27 | 0 |
/// | Conv2 | 5×5, 256, p2 | 256×27×27 | 614,656 |
/// | Pool2 | MaxPool 3×3/2 | 256×13×13 | 0 |
/// | Conv3 | 3×3, 384, p1 | 384×13×13 | 885,120 |
/// | Conv4 | 3×3, 384, p1 | 384×13×13 | 1,327,488 |
/// | Conv5 | 3×3, 256, p1 | 256×13×13 | 884,992 |
/// | Pool5 | MaxPool 3×3/2 | 256×6×6 | 0 |
/// | **Total Conv** | | | **~3.7M** |
///
/// + FC layers: ~58M parameters = **~62M parameters total**
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlexNet {
    /// Convolutional layers (feature extractor)
    pub features: Sequential,
    /// Configuration
    pub config: AlexNetConfig,
}

impl AlexNet {
    /// Creates AlexNet for ImageNet (1000 classes)
    pub fn new(num_classes: usize) -> Self {
        let mut config = AlexNetConfig::imagenet();
        config.num_classes = num_classes as Dim;
        Self::with_config(config)
    }

    /// Creates AlexNet with custom configuration
    pub fn with_config(config: AlexNetConfig) -> Self {
        if config.input_size >= 200 {
            Self::build_full(config)
        } else if config.input_size >= 64 {
            Self::build_medium(config)
        } else {
            Self::build_mini(config)
        }
    }

    /// Full version for ImageNet (227x227)
    fn build_full(config: AlexNetConfig) -> Self {
        let mut features = Sequential::named("AlexNet");

        // Conv1: 11x11, stride 4
        features = features.add_conv2d(Conv2D::new(config.in_channels as usize, 96, 11, 4, 0));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(96));
        }
        features = features.add_activation(ActivationLayer::relu());
        features = features.add_maxpool(MaxPool2D::new(3, 2));

        // Conv2: 5x5, pad 2
        features = features.add_conv2d(Conv2D::new(96, 256, 5, 1, 2));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(256));
        }
        features = features.add_activation(ActivationLayer::relu());
        features = features.add_maxpool(MaxPool2D::new(3, 2));

        // Conv3: 3x3, pad 1
        features = features.add_conv2d(Conv2D::new(256, 384, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(384));
        }
        features = features.add_activation(ActivationLayer::relu());

        // Conv4: 3x3, pad 1
        features = features.add_conv2d(Conv2D::new(384, 384, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(384));
        }
        features = features.add_activation(ActivationLayer::relu());

        // Conv5: 3x3, pad 1
        features = features.add_conv2d(Conv2D::new(384, 256, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(256));
        }
        features = features.add_activation(ActivationLayer::relu());
        features = features.add_maxpool(MaxPool2D::new(3, 2));

        // Flatten
        features = features.add_flatten();

        Self { features, config }
    }

    /// Medium version for 64x64 images
    fn build_medium(config: AlexNetConfig) -> Self {
        let mut features = Sequential::named("AlexNet-Medium");

        // Adapted for 64x64
        features = features.add_conv2d(Conv2D::new(config.in_channels as usize, 64, 5, 1, 2));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(64));
        }
        features = features.add_activation(ActivationLayer::relu());
        features = features.add_maxpool(MaxPool2D::new(2, 2)); // 64→32

        features = features.add_conv2d(Conv2D::new(64, 128, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(128));
        }
        features = features.add_activation(ActivationLayer::relu());
        features = features.add_maxpool(MaxPool2D::new(2, 2)); // 32→16

        features = features.add_conv2d(Conv2D::new(128, 256, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(256));
        }
        features = features.add_activation(ActivationLayer::relu());

        features = features.add_conv2d(Conv2D::new(256, 256, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(256));
        }
        features = features.add_activation(ActivationLayer::relu());
        features = features.add_maxpool(MaxPool2D::new(2, 2)); // 16→8

        features = features.add_flatten();

        Self { features, config }
    }

    /// Mini version for CIFAR-10 (32x32)
    fn build_mini(config: AlexNetConfig) -> Self {
        let mut features = Sequential::named("AlexNet-Mini");

        // Adapted for 32x32
        // Block 1
        features = features.add_conv2d(Conv2D::new(config.in_channels as usize, 64, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(64));
        }
        features = features.add_activation(ActivationLayer::relu());
        features = features.add_maxpool(MaxPool2D::new(2, 2)); // 32→16

        // Block 2
        features = features.add_conv2d(Conv2D::new(64, 128, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(128));
        }
        features = features.add_activation(ActivationLayer::relu());
        features = features.add_maxpool(MaxPool2D::new(2, 2)); // 16→8

        // Block 3
        features = features.add_conv2d(Conv2D::new(128, 256, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(256));
        }
        features = features.add_activation(ActivationLayer::relu());

        // Block 4
        features = features.add_conv2d(Conv2D::new(256, 256, 3, 1, 1));
        if config.use_batch_norm {
            features = features.add_batchnorm(BatchNorm2D::new(256));
        }
        features = features.add_activation(ActivationLayer::relu());
        features = features.add_maxpool(MaxPool2D::new(2, 2)); // 8→4

        features = features.add_flatten();

        Self { features, config }
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        self.features.forward(input)
    }

    /// Number of parameters (conv layers)
    pub fn num_parameters(&self) -> usize {
        self.features.num_parameters()
    }

    /// Prints the summary
    pub fn summary(&self) {
        let input_shape = TensorShape::new(
            1,
            self.config.in_channels as usize,
            self.config.input_size as usize,
            self.config.input_size as usize,
        );
        self.features.summary(input_shape);
    }

    /// Output feature size
    pub fn output_size(&self) -> usize {
        let input_shape = TensorShape::new(
            1,
            self.config.in_channels as usize,
            self.config.input_size as usize,
            self.config.input_size as usize,
        );
        let output = self.features.output_shape(input_shape);
        output.width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alexnet_cifar() {
        let model = AlexNet::with_config(AlexNetConfig::cifar10());
        let input = Tensor4D::zeros(TensorShape::new(1, 3, 32, 32));
        let output = model.forward(&input);

        // 32 → 16 → 8 → 4, channels = 256
        // Flatten: 256 * 4 * 4 = 4096
        assert_eq!(output.shape().width, 4096);
    }

    #[test]
    fn test_alexnet_medium() {
        let model = AlexNet::with_config(AlexNetConfig::small(100));
        let input = Tensor4D::zeros(TensorShape::new(1, 3, 64, 64));
        let output = model.forward(&input);

        // 64 → 32 → 16 → 8, channels = 256
        // Flatten: 256 * 8 * 8 = 16384
        assert_eq!(output.shape().width, 16384);
    }

    #[test]
    fn test_alexnet_batch() {
        let model = AlexNet::with_config(AlexNetConfig::cifar10());
        let batch = Tensor4D::random(TensorShape::new(16, 3, 32, 32));
        let output = model.forward(&batch);

        assert_eq!(output.shape().batch, 16);
    }
}
