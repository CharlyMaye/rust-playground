//! # ResNet (He et al., 2015)
//!
//! Revolutionary architecture introducing residual connections (skip connections)
//! enabling training of extremely deep networks (152+ layers).
//!
//! ## Original Paper
//!
//! **"Deep Residual Learning for Image Recognition"**
//! Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
//! CVPR 2016 (arXiv 2015)
//! https://arxiv.org/abs/1512.03385
//!
//! ## Key Innovation: Skip Connections
//!
//! ```text
//!        ┌──────────────────────────┐
//!        │                          │
//!        │  ┌─────────────────┐     │
//! x ─────┼─►│  Conv + BN + ReLU│     │
//!        │  │  Conv + BN      │     │
//!        │  └────────┬────────┘     │
//!        │           │              │
//!        │           ▼              │
//!        │     ┌─────────┐          │
//!        └────►│   ADD   │◄─────────┘
//!              └────┬────┘
//!                   │
//!                   ▼
//!                 ReLU
//!                   │
//!              F(x) + x  ← "Residual"
//! ```
//!
//! Instead of learning H(x), the network learns F(x) = H(x) - x
//! It is easier to learn a small correction than the complete function.
//!
//! ## Variantes
//!
//! | Model | Layers | Params | Top-1 Acc |
//! |--------|---------|--------|-----------|
//! | ResNet-18 | 18 | 11.7M | 69.8% |
//! | ResNet-34 | 34 | 21.8M | 73.3% |
//! | ResNet-50 | 50 | 25.6M | 76.1% |
//! | ResNet-101 | 101 | 44.5M | 77.4% |
//! | ResNet-152 | 152 | 60.2M | 78.3% |
//!
//! ## Block Types
//!
//! - **BasicBlock** (ResNet-18/34): 2 conv 3×3
//! - **Bottleneck** (ResNet-50+): 1×1 → 3×3 → 1×1 (dimension reduction)

use serde::{Deserialize, Serialize};

use cma_cnn::{
    ActivationLayer, BatchNorm2D, Conv2D, Flatten, GlobalAvgPool2D, MaxPool2D, Sequential, Tensor4D,
};

/// ResNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNetConfig {
    /// Number of classes
    pub num_classes: usize,
    /// Input size (224 for ImageNet)
    pub input_size: usize,
    /// Input channels (3 for RGB)
    pub in_channels: usize,
    /// Number of blocks per stage [stage1, stage2, stage3, stage4]
    pub blocks_per_stage: Vec<usize>,
    /// Use Bottleneck (true for ResNet-50+)
    pub use_bottleneck: bool,
}

impl ResNetConfig {
    /// ResNet-18: [2, 2, 2, 2] BasicBlocks
    pub fn resnet18() -> Self {
        Self {
            num_classes: 1000,
            input_size: 224,
            in_channels: 3,
            blocks_per_stage: vec![2, 2, 2, 2],
            use_bottleneck: false,
        }
    }

    /// ResNet-34: [3, 4, 6, 3] BasicBlocks
    pub fn resnet34() -> Self {
        Self {
            num_classes: 1000,
            input_size: 224,
            in_channels: 3,
            blocks_per_stage: vec![3, 4, 6, 3],
            use_bottleneck: false,
        }
    }

    /// ResNet-50: [3, 4, 6, 3] Bottleneck
    pub fn resnet50() -> Self {
        Self {
            num_classes: 1000,
            input_size: 224,
            in_channels: 3,
            blocks_per_stage: vec![3, 4, 6, 3],
            use_bottleneck: true,
        }
    }

    /// ResNet-101: [3, 4, 23, 3] Bottleneck
    pub fn resnet101() -> Self {
        Self {
            num_classes: 1000,
            input_size: 224,
            in_channels: 3,
            blocks_per_stage: vec![3, 4, 23, 3],
            use_bottleneck: true,
        }
    }

    /// Version for CIFAR-10 (32×32)
    pub fn cifar10() -> Self {
        Self {
            num_classes: 10,
            input_size: 32,
            in_channels: 3,
            blocks_per_stage: vec![2, 2, 2, 2], // 4 stages for ResNet-18
            use_bottleneck: false,
        }
    }
}

/// Basic residual block (ResNet-18/34)
///
/// ```text
/// x ──┬──► Conv3×3 → BN → ReLU → Conv3×3 → BN ──┬──► ReLU → out
///     │                                          │
///     └───────────────── (identity) ─────────────┘
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualBlock {
    /// First convolution
    pub conv1: Conv2D,
    pub bn1: BatchNorm2D,
    /// Second convolution
    pub conv2: Conv2D,
    pub bn2: BatchNorm2D,
    /// Convolution for projection (when dimensions change)
    pub downsample: Option<(Conv2D, BatchNorm2D)>,
    /// Stride (1 ou 2)
    pub stride: usize,
}

impl ResidualBlock {
    /// Creates a residual block
    ///
    /// # Arguments
    /// * `in_channels` - Input channels
    /// * `out_channels` - Output channels
    /// * `stride` - Stride (2 for downsampling)
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let conv1 = Conv2D::new(in_channels, out_channels, 3, stride, 1).without_bias();
        let bn1 = BatchNorm2D::new(out_channels);

        let conv2 = Conv2D::new(out_channels, out_channels, 3, 1, 1).without_bias();
        let bn2 = BatchNorm2D::new(out_channels);

        // Downsample if stride > 1 or channel change
        let downsample = if stride != 1 || in_channels != out_channels {
            Some((
                Conv2D::new(in_channels, out_channels, 1, stride, 0).without_bias(),
                BatchNorm2D::new(out_channels),
            ))
        } else {
            None
        };

        Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
            stride,
        }
    }

    /// Forward pass with skip connection
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        // Main branch
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(&out);
        let out = out.relu();

        let out = self.conv2.forward(&out);
        let out = self.bn2.forward(&out);

        // Skip connection
        let identity = if let Some((ref conv, ref bn)) = self.downsample {
            let x = conv.forward(input);
            bn.forward(&x)
        } else {
            input.clone()
        };

        // Add and ReLU
        let sum = &out + &identity;
        sum.relu()
    }

    /// Number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut params = 0;
        params += self.conv1.num_parameters();
        params += self.bn1.num_parameters();
        params += self.conv2.num_parameters();
        params += self.bn2.num_parameters();

        if let Some((ref conv, ref bn)) = self.downsample {
            params += conv.num_parameters();
            params += bn.num_parameters();
        }

        params
    }

    /// Sets all BatchNorm layers to evaluation mode
    pub fn eval_mode(&mut self) {
        self.bn1.eval_mode();
        self.bn2.eval_mode();
        if let Some((_, ref mut bn)) = self.downsample {
            bn.eval_mode();
        }
    }

    /// Sets all BatchNorm layers to training mode
    pub fn train_mode(&mut self) {
        self.bn1.train_mode();
        self.bn2.train_mode();
        if let Some((_, ref mut bn)) = self.downsample {
            bn.train_mode();
        }
    }
}

use cma_cnn::Layer;

// ═══════════════════════════════════════════════════════════════════════════
// ResNet Builder - Flexible ResNet construction
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for flexible ResNet construction
///
/// Allows customization of:
/// - Input channels (1 for grayscale, 3 for RGB)
/// - Input size (28 for MNIST, 32 for CIFAR, 224 for ImageNet)
/// - Channels per stage (e.g., [16, 32, 64] for small images)
/// - Blocks per stage (e.g., [2, 2, 2])
/// - Stem configuration (with/without pooling)
///
/// # Example
/// ```rust,ignore
/// // ResNet for MNIST (28x28 grayscale)
/// let resnet = ResNetBuilder::new()
///     .input_channels(1)
///     .input_size(28)
///     .channels(&[16, 32, 64])
///     .blocks(&[2, 2, 2])
///     .build();
///
/// // ResNet for CIFAR-10 (32x32 RGB)
/// let resnet = ResNetBuilder::cifar10();
///
/// // ResNet for MNIST (preset)
/// let resnet = ResNetBuilder::mnist();
/// ```
#[derive(Debug, Clone)]
pub struct ResNetBuilder {
    in_channels: usize,
    input_size: usize,
    stage_channels: Vec<usize>,
    blocks_per_stage: Vec<usize>,
    stem_channels: usize,
    use_stem_pooling: bool,
}

impl Default for ResNetBuilder {
    fn default() -> Self {
        Self {
            in_channels: 3,
            input_size: 224,
            stage_channels: vec![64, 128, 256, 512],
            blocks_per_stage: vec![2, 2, 2, 2],
            stem_channels: 64,
            use_stem_pooling: true,
        }
    }
}

impl ResNetBuilder {
    /// Create a new ResNet builder with default ImageNet settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set input channels (1 for grayscale, 3 for RGB)
    pub fn input_channels(mut self, channels: usize) -> Self {
        self.in_channels = channels;
        self
    }

    /// Set input image size
    pub fn input_size(mut self, size: usize) -> Self {
        self.input_size = size;
        self
    }

    /// Set channels for each stage
    pub fn channels(mut self, channels: &[usize]) -> Self {
        self.stage_channels = channels.to_vec();
        self
    }

    /// Set number of blocks per stage
    pub fn blocks(mut self, blocks: &[usize]) -> Self {
        self.blocks_per_stage = blocks.to_vec();
        self
    }

    /// Set stem (initial) channels
    pub fn stem_channels(mut self, channels: usize) -> Self {
        self.stem_channels = channels;
        self
    }

    /// Enable or disable stem pooling (disable for small images)
    pub fn stem_pooling(mut self, enabled: bool) -> Self {
        self.use_stem_pooling = enabled;
        self
    }

    /// Preset for MNIST (28×28 grayscale)
    ///
    /// Architecture:
    /// - Stem: 1→16, 3×3 (no pooling)
    /// - Stage 1: 16→16, 2 blocks
    /// - Stage 2: 16→32, 2 blocks (stride 2)
    /// - Stage 3: 32→64, 2 blocks (stride 2)
    /// - Output: 64 features after GAP
    pub fn mnist() -> Self {
        Self {
            in_channels: 1,
            input_size: 28,
            stage_channels: vec![16, 32, 64],
            blocks_per_stage: vec![2, 2, 2],
            stem_channels: 16,
            use_stem_pooling: false,
        }
    }

    /// Preset for CIFAR-10/100 (32×32 RGB)
    ///
    /// Architecture:
    /// - Stem: 3→16, 3×3 (no pooling)
    /// - Stage 1: 16→16, 2 blocks
    /// - Stage 2: 16→32, 2 blocks (stride 2)
    /// - Stage 3: 32→64, 2 blocks (stride 2)
    /// - Output: 64 features after GAP
    pub fn cifar() -> Self {
        Self {
            in_channels: 3,
            input_size: 32,
            stage_channels: vec![16, 32, 64],
            blocks_per_stage: vec![2, 2, 2],
            stem_channels: 16,
            use_stem_pooling: false,
        }
    }

    /// Preset for ImageNet (224×224 RGB)
    pub fn imagenet() -> Self {
        Self::default()
    }

    /// Build the ResNet model
    pub fn build(self) -> ResNet {
        ResNet::from_builder(self)
    }

    /// Get the output feature dimension (channels after last stage)
    pub fn output_features(&self) -> usize {
        *self.stage_channels.last().unwrap_or(&64)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ResNet - Flexible ResNet implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Flexible ResNet supporting any number of stages and channel configurations
///
/// Unlike ResNet18/34/50 which are fixed architectures, this struct can be
/// configured for any image size and channel configuration using ResNetBuilder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet {
    /// Stem convolution
    pub stem_conv: Conv2D,
    pub stem_bn: BatchNorm2D,
    /// Whether stem has max pooling
    pub has_stem_pool: bool,
    /// All stages (variable number)
    pub stages: Vec<Vec<ResidualBlock>>,
    /// Channel configuration for reference
    pub stage_channels: Vec<usize>,
}

impl ResNet {
    /// Create from builder configuration
    pub fn from_builder(builder: ResNetBuilder) -> Self {
        // Stem: 3×3 or 7×7 conv depending on input size
        let stem_kernel = if builder.input_size >= 200 { 7 } else { 3 };
        let stem_stride = if builder.input_size >= 200 { 2 } else { 1 };
        let stem_pad = if builder.input_size >= 200 { 3 } else { 1 };

        let stem_conv = Conv2D::new(
            builder.in_channels,
            builder.stem_channels,
            stem_kernel,
            stem_stride,
            stem_pad,
        )
        .without_bias();
        let stem_bn = BatchNorm2D::new(builder.stem_channels);

        // Build stages
        let mut stages = Vec::new();
        let mut in_ch = builder.stem_channels;

        for (i, &out_ch) in builder.stage_channels.iter().enumerate() {
            let num_blocks = builder.blocks_per_stage.get(i).copied().unwrap_or(2);
            let stride = if i == 0 { 1 } else { 2 }; // First stage no downsample

            let mut stage = Vec::new();
            for j in 0..num_blocks {
                let block_stride = if j == 0 { stride } else { 1 };
                let block_in = if j == 0 { in_ch } else { out_ch };
                stage.push(ResidualBlock::new(block_in, out_ch, block_stride));
            }

            stages.push(stage);
            in_ch = out_ch;
        }

        Self {
            stem_conv,
            stem_bn,
            has_stem_pool: builder.use_stem_pooling,
            stages,
            stage_channels: builder.stage_channels,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        // Stem
        let mut x = self.stem_conv.forward(input);
        x = self.stem_bn.forward(&x);
        x = x.relu();

        // Max pool for large images
        if self.has_stem_pool {
            let pool = MaxPool2D::new(3, 2);
            x = pool.forward(&x);
        }

        // All stages
        for stage in &self.stages {
            for block in stage {
                x = block.forward(&x);
            }
        }

        // Global Average Pooling + Flatten
        let gap = GlobalAvgPool2D::new();
        let x = gap.forward(&x);
        Flatten::new().forward(&x)
    }

    /// Number of output features (for FC layer sizing)
    pub fn output_features(&self) -> usize {
        *self.stage_channels.last().unwrap_or(&64)
    }

    /// Total number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut params = self.stem_conv.num_parameters();
        params += self.stem_bn.num_parameters();

        for stage in &self.stages {
            for block in stage {
                params += block.num_parameters();
            }
        }
        params
    }

    /// Print architecture summary
    pub fn summary(&self) {
        println!("ResNet Architecture:");
        println!(
            "├── Stem: Conv→BN→ReLU{}",
            if self.has_stem_pool { "→MaxPool" } else { "" }
        );

        for (i, stage) in self.stages.iter().enumerate() {
            let ch = self.stage_channels.get(i).unwrap_or(&0);
            let stride = if i == 0 { 1 } else { 2 };
            println!(
                "├── Stage {}: {} × ResidualBlock(→{}, stride={})",
                i + 1,
                stage.len(),
                ch,
                stride
            );
        }

        println!("├── GlobalAvgPool → {}", self.output_features());
        println!("└── Total params: {}", self.num_parameters());
    }

    /// Sets the entire network to evaluation mode (BatchNorm uses running stats)
    pub fn eval_mode(&mut self) {
        self.stem_bn.eval_mode();
        for stage in &mut self.stages {
            for block in stage {
                block.eval_mode();
            }
        }
    }

    /// Sets the entire network to training mode (BatchNorm uses batch stats)
    pub fn train_mode(&mut self) {
        self.stem_bn.train_mode();
        for stage in &mut self.stages {
            for block in stage {
                block.train_mode();
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Legacy ResNet-18/34/50 (unchanged for backward compatibility)
// ═══════════════════════════════════════════════════════════════════════════

/// ResNet-18: 18 layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet18 {
    /// Stem: Initial conv
    pub stem: Sequential,
    /// Stage 1: 2 blocks, 64 channels
    pub layer1: Vec<ResidualBlock>,
    /// Stage 2: 2 blocks, 128 channels
    pub layer2: Vec<ResidualBlock>,
    /// Stage 3: 2 blocks, 256 channels
    pub layer3: Vec<ResidualBlock>,
    /// Stage 4: 2 blocks, 512 channels
    pub layer4: Vec<ResidualBlock>,
    /// Configuration
    pub config: ResNetConfig,
}

impl ResNet18 {
    /// Creates ResNet-18 for ImageNet
    pub fn new(num_classes: usize) -> Self {
        let mut config = ResNetConfig::resnet18();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    /// Creates ResNet-18 with configuration
    pub fn with_config(config: ResNetConfig) -> Self {
        // Stem: Conv7×7 stride 2 + MaxPool
        let stem = if config.input_size >= 200 {
            // ImageNet: 7×7 conv + maxpool
            Sequential::named("ResNet-Stem")
                .add_conv2d(Conv2D::new(config.in_channels, 64, 7, 2, 3).without_bias())
                .add_batchnorm(BatchNorm2D::new(64))
                .add_activation(ActivationLayer::relu())
                .add_maxpool(MaxPool2D::new(3, 2))
        } else {
            // CIFAR: 3×3 conv only
            Sequential::named("ResNet-Stem")
                .add_conv2d(Conv2D::new(config.in_channels, 64, 3, 1, 1).without_bias())
                .add_batchnorm(BatchNorm2D::new(64))
                .add_activation(ActivationLayer::relu())
        };

        // Layer 1: 64 channels, stride 1
        let layer1 = Self::make_layer(64, 64, config.blocks_per_stage[0], 1);

        // Layer 2: 128 channels, stride 2
        let layer2 = Self::make_layer(64, 128, config.blocks_per_stage[1], 2);

        // Layer 3: 256 channels, stride 2
        let layer3 = Self::make_layer(128, 256, config.blocks_per_stage[2], 2);

        // Layer 4: 512 channels, stride 2
        let layer4 = Self::make_layer(256, 512, config.blocks_per_stage[3], 2);

        Self {
            stem,
            layer1,
            layer2,
            layer3,
            layer4,
            config,
        }
    }

    fn make_layer(
        in_channels: usize,
        out_channels: usize,
        num_blocks: usize,
        stride: usize,
    ) -> Vec<ResidualBlock> {
        let mut blocks = Vec::new();

        // First block with stride (potentially downsample)
        blocks.push(ResidualBlock::new(in_channels, out_channels, stride));

        // Subsequent blocks stride 1
        for _ in 1..num_blocks {
            blocks.push(ResidualBlock::new(out_channels, out_channels, 1));
        }

        blocks
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        // Stem
        let mut x = self.stem.forward(input);

        // Layer 1
        for block in &self.layer1 {
            x = block.forward(&x);
        }

        // Layer 2
        for block in &self.layer2 {
            x = block.forward(&x);
        }

        // Layer 3
        for block in &self.layer3 {
            x = block.forward(&x);
        }

        // Layer 4
        for block in &self.layer4 {
            x = block.forward(&x);
        }

        // Global Average Pooling
        let gap = GlobalAvgPool2D::new();
        let x = gap.forward(&x);

        // Flatten
        let flatten = Flatten::new();
        flatten.forward(&x)
    }

    /// Number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut params = self.stem.num_parameters();

        for block in &self.layer1 {
            params += block.num_parameters();
        }
        for block in &self.layer2 {
            params += block.num_parameters();
        }
        for block in &self.layer3 {
            params += block.num_parameters();
        }
        for block in &self.layer4 {
            params += block.num_parameters();
        }

        params
    }

    /// Prints the summary
    pub fn summary(&self) {
        println!("ResNet-18");
        println!("{}", "=".repeat(50));
        println!("Stem: {} params", self.stem.num_parameters());
        println!(
            "Layer1: {} blocks, {} params",
            self.layer1.len(),
            self.layer1
                .iter()
                .map(|b| b.num_parameters())
                .sum::<usize>()
        );
        println!(
            "Layer2: {} blocks, {} params",
            self.layer2.len(),
            self.layer2
                .iter()
                .map(|b| b.num_parameters())
                .sum::<usize>()
        );
        println!(
            "Layer3: {} blocks, {} params",
            self.layer3.len(),
            self.layer3
                .iter()
                .map(|b| b.num_parameters())
                .sum::<usize>()
        );
        println!(
            "Layer4: {} blocks, {} params",
            self.layer4.len(),
            self.layer4
                .iter()
                .map(|b| b.num_parameters())
                .sum::<usize>()
        );
        println!("{}", "=".repeat(50));
        println!("Total params: {}", self.num_parameters());
    }
}

/// ResNet-34: 34 layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet34 {
    pub stem: Sequential,
    pub layer1: Vec<ResidualBlock>,
    pub layer2: Vec<ResidualBlock>,
    pub layer3: Vec<ResidualBlock>,
    pub layer4: Vec<ResidualBlock>,
    pub config: ResNetConfig,
}

impl ResNet34 {
    /// Creates ResNet-34 for ImageNet
    pub fn new(num_classes: usize) -> Self {
        let mut config = ResNetConfig::resnet34();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    pub fn with_config(config: ResNetConfig) -> Self {
        let stem = if config.input_size >= 200 {
            Sequential::named("ResNet-Stem")
                .add_conv2d(Conv2D::new(config.in_channels, 64, 7, 2, 3).without_bias())
                .add_batchnorm(BatchNorm2D::new(64))
                .add_activation(ActivationLayer::relu())
                .add_maxpool(MaxPool2D::new(3, 2))
        } else {
            Sequential::named("ResNet-Stem")
                .add_conv2d(Conv2D::new(config.in_channels, 64, 3, 1, 1).without_bias())
                .add_batchnorm(BatchNorm2D::new(64))
                .add_activation(ActivationLayer::relu())
        };

        let layer1 = ResNet18::make_layer(64, 64, config.blocks_per_stage[0], 1);
        let layer2 = ResNet18::make_layer(64, 128, config.blocks_per_stage[1], 2);
        let layer3 = ResNet18::make_layer(128, 256, config.blocks_per_stage[2], 2);
        let layer4 = ResNet18::make_layer(256, 512, config.blocks_per_stage[3], 2);

        Self {
            stem,
            layer1,
            layer2,
            layer3,
            layer4,
            config,
        }
    }

    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let mut x = self.stem.forward(input);
        for block in &self.layer1 {
            x = block.forward(&x);
        }
        for block in &self.layer2 {
            x = block.forward(&x);
        }
        for block in &self.layer3 {
            x = block.forward(&x);
        }
        for block in &self.layer4 {
            x = block.forward(&x);
        }

        let gap = GlobalAvgPool2D::new();
        let x = gap.forward(&x);
        let flatten = Flatten::new();
        flatten.forward(&x)
    }

    pub fn num_parameters(&self) -> usize {
        let mut params = self.stem.num_parameters();
        for block in &self.layer1 {
            params += block.num_parameters();
        }
        for block in &self.layer2 {
            params += block.num_parameters();
        }
        for block in &self.layer3 {
            params += block.num_parameters();
        }
        for block in &self.layer4 {
            params += block.num_parameters();
        }
        params
    }
}

/// ResNet-50: 50 layers with Bottleneck
///
/// Note: This implementation uses BasicBlock for simplicity.
/// A real ResNet-50 would use Bottleneck blocks (1×1 → 3×3 → 1×1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet50 {
    pub stem: Sequential,
    pub layer1: Vec<ResidualBlock>,
    pub layer2: Vec<ResidualBlock>,
    pub layer3: Vec<ResidualBlock>,
    pub layer4: Vec<ResidualBlock>,
    pub config: ResNetConfig,
}

impl ResNet50 {
    /// Creates ResNet-50 for ImageNet
    pub fn new(num_classes: usize) -> Self {
        let mut config = ResNetConfig::resnet50();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    pub fn with_config(config: ResNetConfig) -> Self {
        // Note: Real ResNet-50 uses Bottleneck with expansion=4
        // Here we use BasicBlock for simplicity
        let stem = Sequential::named("ResNet-Stem")
            .add_conv2d(Conv2D::new(config.in_channels, 64, 7, 2, 3).without_bias())
            .add_batchnorm(BatchNorm2D::new(64))
            .add_activation(ActivationLayer::relu())
            .add_maxpool(MaxPool2D::new(3, 2));

        // For a real ResNet-50, channels would be 256, 512, 1024, 2048
        // with expansion=4 in Bottleneck
        let layer1 = ResNet18::make_layer(64, 64, config.blocks_per_stage[0], 1);
        let layer2 = ResNet18::make_layer(64, 128, config.blocks_per_stage[1], 2);
        let layer3 = ResNet18::make_layer(128, 256, config.blocks_per_stage[2], 2);
        let layer4 = ResNet18::make_layer(256, 512, config.blocks_per_stage[3], 2);

        Self {
            stem,
            layer1,
            layer2,
            layer3,
            layer4,
            config,
        }
    }

    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let mut x = self.stem.forward(input);
        for block in &self.layer1 {
            x = block.forward(&x);
        }
        for block in &self.layer2 {
            x = block.forward(&x);
        }
        for block in &self.layer3 {
            x = block.forward(&x);
        }
        for block in &self.layer4 {
            x = block.forward(&x);
        }

        let gap = GlobalAvgPool2D::new();
        let x = gap.forward(&x);
        let flatten = Flatten::new();
        flatten.forward(&x)
    }

    pub fn num_parameters(&self) -> usize {
        let mut params = self.stem.num_parameters();
        for block in &self.layer1 {
            params += block.num_parameters();
        }
        for block in &self.layer2 {
            params += block.num_parameters();
        }
        for block in &self.layer3 {
            params += block.num_parameters();
        }
        for block in &self.layer4 {
            params += block.num_parameters();
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cma_cnn::TensorShape;

    #[test]
    fn test_residual_block() {
        let block = ResidualBlock::new(64, 64, 1);
        let input = Tensor4D::random(TensorShape::new(1, 64, 56, 56));
        let output = block.forward(&input);

        // Stride 1: same shape
        assert_eq!(output.shape().height, 56);
        assert_eq!(output.shape().channels, 64);
    }

    #[test]
    fn test_residual_block_downsample() {
        let block = ResidualBlock::new(64, 128, 2);
        let input = Tensor4D::random(TensorShape::new(1, 64, 56, 56));
        let output = block.forward(&input);

        // Stride 2: half size, double channels
        assert_eq!(output.shape().height, 28);
        assert_eq!(output.shape().channels, 128);
    }

    #[test]
    fn test_resnet18_cifar() {
        let mut config = ResNetConfig::resnet18();
        config.input_size = 32;
        config.num_classes = 10;
        config.blocks_per_stage = vec![2, 2, 2, 2];

        let model = ResNet18::with_config(config);
        let input = Tensor4D::random(TensorShape::new(1, 3, 32, 32));
        let output = model.forward(&input);

        // Global avg pool + flatten → 512
        assert_eq!(output.shape().width, 512);
    }

    #[test]
    fn test_resnet_params() {
        // ResNet-18 ≈ 11.7M params (sans FC final)
        // Our version is lighter because it only has the conv layers
        let model = ResNet18::with_config(ResNetConfig::cifar10());
        assert!(model.num_parameters() > 0);
    }

    // ===== ResNetBuilder tests =====

    #[test]
    fn test_resnet_builder_mnist() {
        let resnet = ResNetBuilder::mnist().build();

        // MNIST: 28x28x1 input
        let input = Tensor4D::random(TensorShape::new(1, 1, 28, 28));
        let output = resnet.forward(&input);

        // Output should be 64 features (last stage channels)
        assert_eq!(output.shape().width, 64);
        assert_eq!(resnet.output_features(), 64);
        assert_eq!(resnet.stages.len(), 3); // 3 stages for MNIST
    }

    #[test]
    fn test_resnet_builder_cifar() {
        let resnet = ResNetBuilder::cifar().build();

        // CIFAR: 32x32x3 input
        let input = Tensor4D::random(TensorShape::new(1, 3, 32, 32));
        let output = resnet.forward(&input);

        // Output should be 64 features
        assert_eq!(output.shape().width, 64);
        assert_eq!(resnet.stages.len(), 3);
    }

    #[test]
    fn test_resnet_builder_custom() {
        // Custom ResNet with 4 stages and more channels
        let resnet = ResNetBuilder::new()
            .input_channels(3)
            .input_size(64)
            .channels(&[32, 64, 128, 256])
            .blocks(&[2, 2, 2, 2])
            .stem_channels(32)
            .stem_pooling(false)
            .build();

        let input = Tensor4D::random(TensorShape::new(1, 3, 64, 64));
        let output = resnet.forward(&input);

        // Output should be 256 features (last stage)
        assert_eq!(output.shape().width, 256);
        assert_eq!(resnet.stages.len(), 4);
    }

    #[test]
    fn test_resnet_builder_output_features() {
        let builder = ResNetBuilder::mnist();
        assert_eq!(builder.output_features(), 64);

        let builder = ResNetBuilder::new().channels(&[128, 256, 512]);
        assert_eq!(builder.output_features(), 512);
    }

    #[test]
    fn test_resnet_num_parameters() {
        let resnet = ResNetBuilder::mnist().build();
        // Should have reasonable number of parameters
        assert!(resnet.num_parameters() > 10_000);
        assert!(resnet.num_parameters() < 1_000_000); // Not too many for MNIST
    }
}
