//! # ResNet (He et al., 2015)
//!
//! Architecture révolutionnaire introduisant les connexions résiduelles (skip connections)
//! permettant l'entraînement de réseaux extrêmement profonds (152+ couches).
//!
//! ## Paper Original
//!
//! **"Deep Residual Learning for Image Recognition"**
//! Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
//! CVPR 2016 (arXiv 2015)
//! https://arxiv.org/abs/1512.03385
//!
//! ## Innovation Clé: Skip Connections
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
//! Au lieu d'apprendre H(x), le réseau apprend F(x) = H(x) - x
//! Il est plus facile d'apprendre une petite correction que la fonction complète.
//!
//! ## Variantes
//!
//! | Modèle | Couches | Params | Top-1 Acc |
//! |--------|---------|--------|-----------|
//! | ResNet-18 | 18 | 11.7M | 69.8% |
//! | ResNet-34 | 34 | 21.8M | 73.3% |
//! | ResNet-50 | 50 | 25.6M | 76.1% |
//! | ResNet-101 | 101 | 44.5M | 77.4% |
//! | ResNet-152 | 152 | 60.2M | 78.3% |
//!
//! ## Types de Blocs
//!
//! - **BasicBlock** (ResNet-18/34): 2 conv 3×3
//! - **Bottleneck** (ResNet-50+): 1×1 → 3×3 → 1×1 (réduction de dimension)

use serde::{Deserialize, Serialize};

use cma_cnn::{
    ActivationLayer, AvgPool2D, BatchNorm2D, Conv2D, Flatten, GlobalAvgPool2D, MaxPool2D,
    Sequential, Tensor4D, TensorShape,
};

/// Configuration ResNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNetConfig {
    /// Nombre de classes
    pub num_classes: usize,
    /// Taille d'entrée (224 pour ImageNet)
    pub input_size: usize,
    /// Canaux d'entrée (3 pour RGB)
    pub in_channels: usize,
    /// Nombre de blocs par stage [stage1, stage2, stage3, stage4]
    pub blocks_per_stage: Vec<usize>,
    /// Utiliser Bottleneck (true pour ResNet-50+)
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

    /// Version pour CIFAR-10 (32×32)
    pub fn cifar10() -> Self {
        Self {
            num_classes: 10,
            input_size: 32,
            in_channels: 3,
            blocks_per_stage: vec![2, 2, 2], // 3 stages seulement
            use_bottleneck: false,
        }
    }
}

/// Bloc résiduel basique (ResNet-18/34)
///
/// ```text
/// x ──┬──► Conv3×3 → BN → ReLU → Conv3×3 → BN ──┬──► ReLU → out
///     │                                          │
///     └───────────────── (identity) ─────────────┘
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualBlock {
    /// Première convolution
    pub conv1: Conv2D,
    pub bn1: BatchNorm2D,
    /// Deuxième convolution
    pub conv2: Conv2D,
    pub bn2: BatchNorm2D,
    /// Convolution pour projection (si changement de dimension)
    pub downsample: Option<(Conv2D, BatchNorm2D)>,
    /// Stride (1 ou 2)
    pub stride: usize,
}

impl ResidualBlock {
    /// Crée un bloc résiduel
    ///
    /// # Arguments
    /// * `in_channels` - Canaux d'entrée
    /// * `out_channels` - Canaux de sortie
    /// * `stride` - Stride (2 pour downsampling)
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let conv1 = Conv2D::new(in_channels, out_channels, 3, stride, 1).without_bias();
        let bn1 = BatchNorm2D::new(out_channels);

        let conv2 = Conv2D::new(out_channels, out_channels, 3, 1, 1).without_bias();
        let bn2 = BatchNorm2D::new(out_channels);

        // Downsample si stride > 1 ou changement de canaux
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

    /// Forward pass avec skip connection
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        // Branche principale
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

        // Add et ReLU
        let sum = &out + &identity;
        sum.relu()
    }

    /// Nombre de paramètres
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
}

use cma_cnn::Layer;

/// ResNet-18: 18 couches
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNet18 {
    /// Stem: Conv initiale
    pub stem: Sequential,
    /// Stage 1: 2 blocs, 64 channels
    pub layer1: Vec<ResidualBlock>,
    /// Stage 2: 2 blocs, 128 channels
    pub layer2: Vec<ResidualBlock>,
    /// Stage 3: 2 blocs, 256 channels
    pub layer3: Vec<ResidualBlock>,
    /// Stage 4: 2 blocs, 512 channels
    pub layer4: Vec<ResidualBlock>,
    /// Configuration
    pub config: ResNetConfig,
}

impl ResNet18 {
    /// Crée ResNet-18 pour ImageNet
    pub fn new(num_classes: usize) -> Self {
        let mut config = ResNetConfig::resnet18();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    /// Crée ResNet-18 avec configuration
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
            // CIFAR: 3×3 conv seulement
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

        // Premier bloc avec stride (potentiellement downsample)
        blocks.push(ResidualBlock::new(in_channels, out_channels, stride));

        // Blocs suivants stride 1
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

    /// Nombre de paramètres
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

    /// Affiche le résumé
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

/// ResNet-34: 34 couches
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
    /// Crée ResNet-34 pour ImageNet
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

/// ResNet-50: 50 couches avec Bottleneck
///
/// Note: Cette implémentation utilise BasicBlock pour simplifier.
/// Une vraie ResNet-50 utiliserait des Bottleneck blocks (1×1 → 3×3 → 1×1).
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
    /// Crée ResNet-50 pour ImageNet
    pub fn new(num_classes: usize) -> Self {
        let mut config = ResNetConfig::resnet50();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    pub fn with_config(config: ResNetConfig) -> Self {
        // Note: Vraie ResNet-50 utilise Bottleneck avec expansion=4
        // Ici on utilise BasicBlock pour simplifier
        let stem = Sequential::named("ResNet-Stem")
            .add_conv2d(Conv2D::new(config.in_channels, 64, 7, 2, 3).without_bias())
            .add_batchnorm(BatchNorm2D::new(64))
            .add_activation(ActivationLayer::relu())
            .add_maxpool(MaxPool2D::new(3, 2));

        // Pour une vraie ResNet-50, les channels seraient 256, 512, 1024, 2048
        // avec expansion=4 dans Bottleneck
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

    #[test]
    fn test_residual_block() {
        let block = ResidualBlock::new(64, 64, 1);
        let input = Tensor4D::random(TensorShape::new(1, 64, 56, 56));
        let output = block.forward(&input);

        // Stride 1: même shape
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
        // Notre version est plus légère car elle n'a que les conv
        let model = ResNet18::with_config(ResNetConfig::cifar10());
        assert!(model.num_parameters() > 0);
    }
}
