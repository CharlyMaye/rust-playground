//! # EfficientNet (Tan & Le, 2019)
//!
//! Architecture état de l'art utilisant le "compound scaling" pour équilibrer
//! profondeur, largeur et résolution de manière optimale.
//!
//! ## Paper Original
//!
//! **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"**
//! Mingxing Tan, Quoc V. Le
//! ICML 2019
//! https://arxiv.org/abs/1905.11946
//!
//! ## Innovation Clé: Compound Scaling
//!
//! Scaling traditionnel vs EfficientNet:
//!
//! ```text
//! Traditionnel:          EfficientNet:
//! ┌─────────┐            ┌─────────────────────┐
//! │ Deeper  │            │                     │
//! │   OR    │     vs     │ Depth × Width ×     │
//! │ Wider   │            │ Resolution (φ)      │
//! │   OR    │            │                     │
//! │ Higher  │            │ Balanced scaling    │
//! │  res    │            │                     │
//! └─────────┘            └─────────────────────┘
//! ```
//!
//! Formules de scaling:
//! - depth: d = α^φ
//! - width: w = β^φ  
//! - resolution: r = γ^φ
//!
//! Avec α × β² × γ² ≈ 2 (pour doubler les FLOPs)
//!
//! ## Famille EfficientNet
//!
//! | Modèle | Resolution | Params | Top-1 Acc |
//! |--------|------------|--------|-----------|
//! | B0 | 224 | 5.3M | 77.3% |
//! | B1 | 240 | 7.8M | 79.2% |
//! | B2 | 260 | 9.2M | 80.3% |
//! | B3 | 300 | 12M | 81.6% |
//! | B4 | 380 | 19M | 82.9% |
//! | B5 | 456 | 30M | 83.6% |
//! | B6 | 528 | 43M | 84.0% |
//! | B7 | 600 | 66M | 84.4% |
//!
//! ## Architecture EfficientNet-B0
//!
//! ```text
//! Input (224×224×3)
//!     │
//! ┌───▼─────────────────────────┐
//! │ Stem: Conv3×3, 32, /2       │ → 112×112×32
//! └───┬─────────────────────────┘
//!     │
//! ┌───▼─────────────────────────┐
//! │ Stage 1: MBConv1, k3×3, 16  │ × 1
//! │ Stage 2: MBConv6, k3×3, 24  │ × 2, /2
//! │ Stage 3: MBConv6, k5×5, 40  │ × 2, /2
//! │ Stage 4: MBConv6, k3×3, 80  │ × 3, /2
//! │ Stage 5: MBConv6, k5×5, 112 │ × 3
//! │ Stage 6: MBConv6, k5×5, 192 │ × 4, /2
//! │ Stage 7: MBConv6, k3×3, 320 │ × 1
//! └───┬─────────────────────────┘
//!     │
//! ┌───▼─────────────────────────┐
//! │ Head: Conv1×1, 1280         │
//! │ GlobalAvgPool               │
//! │ FC 1280 → 1000              │
//! └─────────────────────────────┘
//! ```
//!
//! ## MBConv Block (Mobile Inverted Bottleneck)
//!
//! ```text
//! x ──► Expand 1×1 ──► DepthwiseConv ──► SE ──► Project 1×1 ──┬──► out
//!  │                                                           │
//!  └────────────────── (skip if stride=1) ────────────────────┘
//! ```

use cma_cnn::Float;
use serde::{Deserialize, Serialize};

use cma_cnn::{
    ActivationLayer, BatchNorm2D, Conv2D, Flatten, GlobalAvgPool2D, Layer, MaxPool2D, Sequential,
    Tensor4D, TensorShape,
};

/// Configuration EfficientNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficientNetConfig {
    /// Nombre de classes
    pub num_classes: usize,
    /// Résolution d'entrée
    pub input_size: usize,
    /// Canaux d'entrée
    pub in_channels: usize,
    /// Coefficient de largeur (width multiplier)
    pub width_mult: Float,
    /// Coefficient de profondeur (depth multiplier)
    pub depth_mult: Float,
    /// Dropout rate
    pub dropout_rate: Float,
}

impl EfficientNetConfig {
    /// EfficientNet-B0: baseline
    pub fn b0() -> Self {
        Self {
            num_classes: 1000,
            input_size: 224,
            in_channels: 3,
            width_mult: 1.0,
            depth_mult: 1.0,
            dropout_rate: 0.2,
        }
    }

    /// EfficientNet-B1
    pub fn b1() -> Self {
        Self {
            num_classes: 1000,
            input_size: 240,
            in_channels: 3,
            width_mult: 1.0,
            depth_mult: 1.1,
            dropout_rate: 0.2,
        }
    }

    /// EfficientNet-B2
    pub fn b2() -> Self {
        Self {
            num_classes: 1000,
            input_size: 260,
            in_channels: 3,
            width_mult: 1.1,
            depth_mult: 1.2,
            dropout_rate: 0.3,
        }
    }

    /// Version mini pour CIFAR-10
    pub fn cifar10() -> Self {
        Self {
            num_classes: 10,
            input_size: 32,
            in_channels: 3,
            width_mult: 0.5,
            depth_mult: 0.5,
            dropout_rate: 0.2,
        }
    }

    /// Applique le width multiplier
    fn scale_width(&self, channels: usize) -> usize {
        ((channels as Float * self.width_mult).ceil() as usize).max(1)
    }

    /// Applique le depth multiplier
    fn scale_depth(&self, num_layers: usize) -> usize {
        ((num_layers as Float * self.depth_mult).ceil() as usize).max(1)
    }
}

/// MBConv Block (Mobile Inverted Bottleneck Convolution)
///
/// Utilisé dans MobileNetV2 et EfficientNet.
///
/// # Architecture
///
/// 1. **Expansion**: Conv 1×1 pour augmenter les channels (si expand_ratio > 1)
/// 2. **Depthwise**: Conv dépthwise (k×k par channel)
/// 3. **Squeeze-Excitation**: Attention spatiale
/// 4. **Projection**: Conv 1×1 pour réduire les channels
/// 5. **Skip connection**: si stride=1 et in_ch == out_ch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MBConvBlock {
    /// Expansion conv 1×1 (si expand_ratio > 1)
    pub expand_conv: Option<Sequential>,
    /// Depthwise conv
    pub depthwise_conv: Sequential,
    /// Squeeze-and-Excitation
    pub se: Option<SqueezeExcitation>,
    /// Projection conv 1×1
    pub project_conv: Sequential,
    /// Skip connection?
    pub use_skip: bool,
    /// Configuration
    pub in_channels: usize,
    pub out_channels: usize,
    pub expand_ratio: usize,
    pub stride: usize,
}

impl MBConvBlock {
    /// Crée un bloc MBConv
    ///
    /// # Arguments
    /// * `in_channels` - Canaux d'entrée
    /// * `out_channels` - Canaux de sortie
    /// * `kernel_size` - Taille du kernel depthwise (3 ou 5)
    /// * `stride` - Stride (1 ou 2)
    /// * `expand_ratio` - Ratio d'expansion (1 ou 6)
    /// * `use_se` - Utiliser Squeeze-and-Excitation
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        expand_ratio: usize,
        use_se: bool,
    ) -> Self {
        let expanded_channels = in_channels * expand_ratio;
        let padding = kernel_size / 2;

        // Expansion phase
        let expand_conv = if expand_ratio > 1 {
            Some(
                Sequential::new()
                    .add_conv2d(Conv2D::new(in_channels, expanded_channels, 1, 1, 0).without_bias())
                    .add_batchnorm(BatchNorm2D::new(expanded_channels))
                    .add_activation(ActivationLayer::swish()),
            )
        } else {
            None
        };

        // Depthwise phase
        // Note: On simule depthwise avec groups=expanded_channels
        // Pour simplifier, on utilise une conv normale ici
        let depthwise_conv = Sequential::new()
            .add_conv2d(
                Conv2D::new(
                    expanded_channels,
                    expanded_channels,
                    kernel_size,
                    stride,
                    padding,
                )
                .without_bias(),
            )
            .add_batchnorm(BatchNorm2D::new(expanded_channels))
            .add_activation(ActivationLayer::swish());

        // Squeeze-and-Excitation
        let se = if use_se {
            Some(SqueezeExcitation::new(
                expanded_channels,
                expanded_channels / 4,
            ))
        } else {
            None
        };

        // Projection phase (no activation)
        let project_conv = Sequential::new()
            .add_conv2d(Conv2D::new(expanded_channels, out_channels, 1, 1, 0).without_bias())
            .add_batchnorm(BatchNorm2D::new(out_channels));

        // Skip connection si stride=1 et même channels
        let use_skip = stride == 1 && in_channels == out_channels;

        Self {
            expand_conv,
            depthwise_conv,
            se,
            project_conv,
            use_skip,
            in_channels,
            out_channels,
            expand_ratio,
            stride,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let mut x = input.clone();

        // Expansion
        if let Some(ref conv) = self.expand_conv {
            x = conv.forward(&x);
        }

        // Depthwise
        x = self.depthwise_conv.forward(&x);

        // SE
        if let Some(ref se) = self.se {
            x = se.forward(&x);
        }

        // Projection
        x = self.project_conv.forward(&x);

        // Skip connection
        if self.use_skip {
            x = &x + input;
        }

        x
    }

    /// Nombre de paramètres
    pub fn num_parameters(&self) -> usize {
        let mut params = 0;

        if let Some(ref conv) = self.expand_conv {
            params += conv.num_parameters();
        }
        params += self.depthwise_conv.num_parameters();
        if let Some(ref se) = self.se {
            params += se.num_parameters();
        }
        params += self.project_conv.num_parameters();

        params
    }
}

/// Squeeze-and-Excitation Block
///
/// Attention mechanism qui recalibre les feature maps par channel.
///
/// ```text
/// x ──► GlobalAvgPool ──► FC ──► ReLU ──► FC ──► Sigmoid ──► Scale ──► out
///  │                                                           ▲
///  └───────────────────────────────────────────────────────────┘
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqueezeExcitation {
    /// FC pour réduction
    pub fc1: Conv2D, // Implémenté comme Conv 1×1
    /// FC pour expansion
    pub fc2: Conv2D,
    pub channels: usize,
    pub reduction: usize,
}

impl SqueezeExcitation {
    pub fn new(channels: usize, reduced_channels: usize) -> Self {
        // Utilisé comme 1×1 conv sur 1×1 feature map
        let fc1 = Conv2D::new(channels, reduced_channels, 1, 1, 0);
        let fc2 = Conv2D::new(reduced_channels, channels, 1, 1, 0);

        Self {
            fc1,
            fc2,
            channels,
            reduction: reduced_channels,
        }
    }

    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let shape = input.shape();

        // Global average pooling
        let gap = GlobalAvgPool2D::new();
        let mut x = gap.forward(input);

        // FC1 + ReLU
        x = self.fc1.forward(&x);
        x = x.relu();

        // FC2 + Sigmoid
        x = self.fc2.forward(&x);
        x = x.map(|v| 1.0 / (1.0 + (-v).exp())); // Sigmoid

        // Broadcast and multiply
        let scale = x.data();
        let mut output = input.data().clone();

        for b in 0..shape.batch {
            for c in 0..shape.channels {
                let s = scale[[b, c, 0, 0]];
                for h in 0..shape.height {
                    for w in 0..shape.width {
                        output[[b, c, h, w]] *= s;
                    }
                }
            }
        }

        Tensor4D::from_array(output)
    }

    pub fn num_parameters(&self) -> usize {
        self.fc1.num_parameters() + self.fc2.num_parameters()
    }
}

/// EfficientNet-B0: Architecture baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficientNetB0 {
    /// Stem conv
    pub stem: Sequential,
    /// MBConv stages
    pub stages: Vec<Vec<MBConvBlock>>,
    /// Head conv
    pub head: Sequential,
    /// Configuration
    pub config: EfficientNetConfig,
}

impl EfficientNetB0 {
    /// Crée EfficientNet-B0 pour ImageNet
    pub fn new(num_classes: usize) -> Self {
        let mut config = EfficientNetConfig::b0();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    /// Crée avec configuration
    pub fn with_config(config: EfficientNetConfig) -> Self {
        // Stem: Conv3×3, stride 2
        let stem_channels = config.scale_width(32);
        let stem = Sequential::named("EfficientNet-Stem")
            .add_conv2d(Conv2D::new(config.in_channels, stem_channels, 3, 2, 1).without_bias())
            .add_batchnorm(BatchNorm2D::new(stem_channels))
            .add_activation(ActivationLayer::swish());

        // Stages configuration: (expand_ratio, out_channels, num_layers, kernel, stride)
        let stage_configs = [
            (1, 16, 1, 3, 1),  // Stage 1
            (6, 24, 2, 3, 2),  // Stage 2
            (6, 40, 2, 5, 2),  // Stage 3
            (6, 80, 3, 3, 2),  // Stage 4
            (6, 112, 3, 5, 1), // Stage 5
            (6, 192, 4, 5, 2), // Stage 6
            (6, 320, 1, 3, 1), // Stage 7
        ];

        let mut stages = Vec::new();
        let mut in_channels = stem_channels;

        for (expand_ratio, out_channels, num_layers, kernel, stride) in stage_configs {
            let scaled_out = config.scale_width(out_channels);
            let scaled_layers = config.scale_depth(num_layers);

            let mut stage = Vec::new();

            for i in 0..scaled_layers {
                let s = if i == 0 { stride } else { 1 };
                let block = MBConvBlock::new(
                    in_channels,
                    scaled_out,
                    kernel,
                    s,
                    expand_ratio,
                    true, // use_se
                );
                stage.push(block);
                in_channels = scaled_out;
            }

            stages.push(stage);
        }

        // Head: Conv1×1, GlobalAvgPool
        let head_channels = config.scale_width(1280);
        let head = Sequential::named("EfficientNet-Head")
            .add_conv2d(Conv2D::new(in_channels, head_channels, 1, 1, 0).without_bias())
            .add_batchnorm(BatchNorm2D::new(head_channels))
            .add_activation(ActivationLayer::swish())
            .add_global_avgpool()
            .add_flatten();

        Self {
            stem,
            stages,
            head,
            config,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let mut x = self.stem.forward(input);

        for stage in &self.stages {
            for block in stage {
                x = block.forward(&x);
            }
        }

        self.head.forward(&x)
    }

    /// Nombre de paramètres
    pub fn num_parameters(&self) -> usize {
        let mut params = self.stem.num_parameters();

        for stage in &self.stages {
            for block in stage {
                params += block.num_parameters();
            }
        }

        params += self.head.num_parameters();
        params
    }

    /// Affiche le résumé
    pub fn summary(&self) {
        println!("EfficientNet-B0");
        println!("{}", "=".repeat(50));
        println!("Stem: {} params", self.stem.num_parameters());

        for (i, stage) in self.stages.iter().enumerate() {
            let stage_params: usize = stage.iter().map(|b| b.num_parameters()).sum();
            println!(
                "Stage {}: {} blocks, {} params",
                i + 1,
                stage.len(),
                stage_params
            );
        }

        println!("Head: {} params", self.head.num_parameters());
        println!("{}", "=".repeat(50));
        println!("Total params: {}", self.num_parameters());
    }

    /// Taille de sortie (features)
    pub fn output_size(&self) -> usize {
        self.config.scale_width(1280)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_se_block() {
        let se = SqueezeExcitation::new(64, 16);
        let input = Tensor4D::random(TensorShape::new(1, 64, 14, 14));
        let output = se.forward(&input);

        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_mbconv_block() {
        let block = MBConvBlock::new(32, 16, 3, 1, 1, true);
        let input = Tensor4D::random(TensorShape::new(1, 32, 112, 112));
        let output = block.forward(&input);

        assert_eq!(output.shape().channels, 16);
        assert_eq!(output.shape().height, 112); // stride 1
    }

    #[test]
    fn test_mbconv_block_stride2() {
        let block = MBConvBlock::new(24, 40, 5, 2, 6, true);
        let input = Tensor4D::random(TensorShape::new(1, 24, 56, 56));
        let output = block.forward(&input);

        assert_eq!(output.shape().channels, 40);
        assert_eq!(output.shape().height, 28); // stride 2
    }

    #[test]
    fn test_efficientnet_b0_cifar() {
        let model = EfficientNetB0::with_config(EfficientNetConfig::cifar10());
        let input = Tensor4D::random(TensorShape::new(1, 3, 32, 32));
        let output = model.forward(&input);

        // Output should be flattened features
        assert!(output.shape().width > 0);
    }

    #[test]
    fn test_config_scaling() {
        let config = EfficientNetConfig::b0();
        assert_eq!(config.scale_width(32), 32);
        assert_eq!(config.scale_depth(3), 3);

        let config_b2 = EfficientNetConfig::b2();
        assert!(config_b2.scale_width(32) >= 32); // width_mult = 1.1
        assert!(config_b2.scale_depth(3) >= 3); // depth_mult = 1.2
    }
}
