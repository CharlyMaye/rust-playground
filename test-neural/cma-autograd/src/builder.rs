//! # CNN Builder — Configurable CNN construction with presets
//!
//! Provides a builder pattern for constructing autograd CNN models,
//! with presets for common architectures (LeNet-5, AlexNet-Mini, VGG-Tiny, ResNet-Plain).
//!
//! ## Quick Start with Presets
//!
//! ```rust,ignore
//! use cma_autograd::builder::CnnBuilder;
//!
//! // Build a ResNet-style model for MNIST
//! let model = CnnBuilder::resnet_mnist(10);
//!
//! // Build a LeNet-5 model
//! let model = CnnBuilder::lenet5(10);
//! ```
//!
//! ## Custom Architecture with Generic Builder
//!
//! ```rust,ignore
//! use cma_autograd::builder::CnnBuilder;
//!
//! let model = CnnBuilder::new()
//!     .conv_bn_relu(1, 32, 3, 1, 1)   // Conv+BN+ReLU block
//!     .maxpool(2, 2)
//!     .conv_relu(32, 64, 3, 1, 1)      // Conv+ReLU block (no BN)
//!     .maxpool(2, 2)
//!     .global_avg_pool()
//!     .flatten()
//!     .linear(64, 10)
//!     .build();
//! ```

use crate::layers::{
    AvgPool2D, BatchNorm2D, Dropout, Flatten, GlobalAvgPool2D, MaxPool2D, ReLU,
};
use crate::module::{Conv2D, Linear};
use crate::sequential::Sequential;

/// Builder for constructing autograd CNN models with a fluent API.
///
/// Supports both custom architectures via chained method calls and
/// pre-built architecture presets for common models.
pub struct CnnBuilder {
    model: Sequential,
}

impl CnnBuilder {
    /// Create a new empty CNN builder.
    pub fn new() -> Self {
        Self {
            model: Sequential::new(),
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Individual Layer Methods
    // ═══════════════════════════════════════════════════════════════════

    /// Add a Conv2D layer.
    pub fn conv2d(
        mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        self.model
            .push(Conv2D::new(in_channels, out_channels, kernel_size, stride, padding));
        self
    }

    /// Add a Conv2D layer without bias.
    pub fn conv2d_no_bias(
        mut self,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        self.model.push(Conv2D::without_bias(
            in_channels, out_channels, kernel_size, stride, padding,
        ));
        self
    }

    /// Add a BatchNorm2D layer.
    pub fn batch_norm(mut self, num_channels: usize) -> Self {
        self.model.push(BatchNorm2D::new(num_channels));
        self
    }

    /// Add a ReLU activation.
    pub fn relu(mut self) -> Self {
        self.model.push(ReLU::new());
        self
    }

    /// Add a MaxPool2D layer.
    pub fn maxpool(mut self, kernel_size: usize, stride: usize) -> Self {
        self.model.push(MaxPool2D::new(kernel_size, stride));
        self
    }

    /// Add an AvgPool2D layer.
    pub fn avgpool(mut self, kernel_size: usize, stride: usize) -> Self {
        self.model.push(AvgPool2D::new(kernel_size, stride));
        self
    }

    /// Add a GlobalAvgPool2D layer.
    pub fn global_avg_pool(mut self) -> Self {
        self.model.push(GlobalAvgPool2D::new());
        self
    }

    /// Add a Flatten layer.
    pub fn flatten(mut self) -> Self {
        self.model.push(Flatten::new());
        self
    }

    /// Add a Linear (fully connected) layer.
    pub fn linear(mut self, in_features: usize, out_features: usize) -> Self {
        self.model.push(Linear::new(in_features, out_features));
        self
    }

    /// Add a Dropout layer.
    pub fn dropout(mut self, p: crate::Float) -> Self {
        self.model.push(Dropout::new(p));
        self
    }

    // ═══════════════════════════════════════════════════════════════════
    // Composite Block Methods
    // ═══════════════════════════════════════════════════════════════════

    /// Add a Conv2D + ReLU block.
    pub fn conv_relu(
        self,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        self.conv2d(in_ch, out_ch, kernel, stride, padding).relu()
    }

    /// Add a Conv2D + BatchNorm2D + ReLU block.
    pub fn conv_bn_relu(
        self,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        self.conv2d(in_ch, out_ch, kernel, stride, padding)
            .batch_norm(out_ch)
            .relu()
    }

    /// Add a Conv2D + BatchNorm2D + ReLU + MaxPool2D block.
    pub fn conv_bn_relu_pool(
        self,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        pool_size: usize,
        pool_stride: usize,
    ) -> Self {
        self.conv_bn_relu(in_ch, out_ch, kernel, stride, padding)
            .maxpool(pool_size, pool_stride)
    }

    // ═══════════════════════════════════════════════════════════════════
    // Build
    // ═══════════════════════════════════════════════════════════════════

    /// Consume the builder and return the Sequential model.
    pub fn build(self) -> Sequential {
        self.model
    }

    // ═══════════════════════════════════════════════════════════════════
    // Architecture Presets
    // ═══════════════════════════════════════════════════════════════════

    /// LeNet-5 architecture (LeCun et al., 1998) for 28×28 grayscale images.
    ///
    /// Architecture:
    /// - C1: Conv(1→6, 5×5, p=2) → ReLU → AvgPool(2,2)    28→14
    /// - C3: Conv(6→16, 5×5) → ReLU → AvgPool(2,2)         10→5
    /// - C5: Conv(16→120, 5×5) → ReLU                       1×1
    /// - Flatten → Linear(120→num_classes)
    pub fn lenet5(num_classes: usize) -> Sequential {
        CnnBuilder::new()
            // C1: 28×28→28×28→14×14
            .conv_relu(1, 6, 5, 1, 2)
            .avgpool(2, 2)
            // C3: 14×14→10×10→5×5
            .conv_relu(6, 16, 5, 1, 0)
            .avgpool(2, 2)
            // C5: 5×5→1×1
            .conv_relu(16, 120, 5, 1, 0)
            // Head
            .flatten()
            .linear(120, num_classes)
            .build()
    }

    /// AlexNet-Mini architecture (Krizhevsky et al., 2012 style) for 28×28 images.
    ///
    /// Architecture:
    /// - Block 1: Conv(1→64, 3×3, p=1) → BN → ReLU → MaxPool(2,2)    28→14
    /// - Block 2: Conv(64→128, 3×3, p=1) → BN → ReLU → MaxPool(2,2)  14→7
    /// - Block 3: Conv(128→256, 3×3, p=1) → BN → ReLU                 7
    /// - Block 4: Conv(256→256, 3×3, p=1) → BN → ReLU → MaxPool(2,2)  7→3
    /// - Flatten → Linear(2304→num_classes)
    pub fn alexnet_mnist(num_classes: usize) -> Sequential {
        CnnBuilder::new()
            .conv_bn_relu_pool(1, 64, 3, 1, 1, 2, 2)    // 28→14
            .conv_bn_relu_pool(64, 128, 3, 1, 1, 2, 2)   // 14→7
            .conv_bn_relu(128, 256, 3, 1, 1)              // 7
            .conv_bn_relu(256, 256, 3, 1, 1)              // 7
            .maxpool(2, 2)                                 // 7→3
            .flatten()
            .linear(256 * 3 * 3, num_classes)
            .build()
    }

    /// VGG-Tiny architecture (Simonyan & Zisserman, 2014 style) for 28×28 images.
    ///
    /// Architecture:
    /// - Block 1: 2× Conv(3×3, 32) → ReLU → MaxPool(2,2)    28→14
    /// - Block 2: 2× Conv(3×3, 64) → ReLU → MaxPool(2,2)    14→7
    /// - Flatten → Linear(3136→num_classes)
    pub fn vgg_mnist(num_classes: usize) -> Sequential {
        CnnBuilder::new()
            // Block 1
            .conv_relu(1, 32, 3, 1, 1)
            .conv_relu(32, 32, 3, 1, 1)
            .maxpool(2, 2)
            // Block 2
            .conv_relu(32, 64, 3, 1, 1)
            .conv_relu(64, 64, 3, 1, 1)
            .maxpool(2, 2)
            // Head
            .flatten()
            .linear(64 * 7 * 7, num_classes)
            .build()
    }

    /// ResNet-Plain architecture (He et al., 2015 style) for 28×28 images.
    ///
    /// A plain network without skip connections (6 conv blocks + head):
    /// - Stem:     Conv(1→16, 3×3, p=1) → BN → ReLU           28×28
    /// - Stage 1:  Conv(16→16, 3×3, p=1) → BN → ReLU          28×28
    /// - Stage 2a: Conv(16→32, 3×3, s=2, p=1) → BN → ReLU     14×14
    /// - Stage 2b: Conv(32→32, 3×3, p=1) → BN → ReLU          14×14
    /// - Stage 3a: Conv(32→64, 3×3, s=2, p=1) → BN → ReLU     7×7
    /// - Stage 3b: Conv(64→64, 3×3, p=1) → BN → ReLU          7×7
    /// - GlobalAvgPool → Flatten → Linear(64→num_classes)
    pub fn resnet_mnist(num_classes: usize) -> Sequential {
        CnnBuilder::new()
            // Stem
            .conv_bn_relu(1, 16, 3, 1, 1)
            // Stage 1
            .conv_bn_relu(16, 16, 3, 1, 1)
            // Stage 2 (downsample stride=2)
            .conv_bn_relu(16, 32, 3, 2, 1)
            .conv_bn_relu(32, 32, 3, 1, 1)
            // Stage 3 (downsample stride=2)
            .conv_bn_relu(32, 64, 3, 2, 1)
            .conv_bn_relu(64, 64, 3, 1, 1)
            // Head
            .global_avg_pool()
            .flatten()
            .linear(64, num_classes)
            .build()
    }
}

impl Default for CnnBuilder {
    fn default() -> Self {
        Self::new()
    }
}
