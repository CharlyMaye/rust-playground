//! # ResNet Example: Deep Residual Learning (He et al., 2015)
//!
//! This example reproduces the ResNet architecture that revolutionized
//! the training of very deep networks through skip connections.
//!
//! ## Paper Citation
//!
//! ```text
//! @inproceedings{he2016deep,
//!   title={Deep Residual Learning for Image Recognition},
//!   author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
//!   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
//!   pages={770--778},
//!   year={2016}
//! }
//! ```
//!
//! ## The Depth Problem
//!
//! Before ResNet, deeper networks had WORSE accuracy:
//!
//! ```text
//! Training Error (CIFAR-10):
//!
//! Error │                     
//!   %   │ ───────────────────  20-layer
//!   10  │                      
//!       │       ─────────────  56-layer (PIRE!)
//!    5  │                      
//!       │                      
//!       └──────────────────────────────
//!                   Iterations
//!
//! "Adding more layers to a suitably deep model leads to higher
//!  training error" - He et al., 2015
//! ```
//!
//! ## The Solution: Skip Connections
//!
//! ```text
//! Standard Block:           Residual Block:
//! ┌─────────────┐         ┌─────────────┐
//! │   x (input) │         │   x (input) │──────────────┐
//! └──────┬──────┘         └──────┬──────┘              │
//!        │                       │                      │
//!        ▼                       ▼                      │
//! ┌─────────────┐         ┌─────────────┐              │
//! │  Conv + BN  │         │  Conv + BN  │              │
//! │    ReLU     │         │    ReLU     │              │
//! └──────┬──────┘         └──────┬──────┘              │
//!        │                       │                      │
//!        ▼                       ▼                      │
//! ┌─────────────┐         ┌─────────────┐              │
//! │  Conv + BN  │         │  Conv + BN  │              │
//! │             │         │             │              │
//! └──────┬──────┘         └──────┬──────┘              │
//!        │                       │                      │
//!        │                       ▼                      │
//!        │                 ┌─────────────┐              │
//!        │                 │     ADD     │◄─────────────┘
//!        │                 └──────┬──────┘
//!        │                        │
//!        ▼                        ▼
//!   H(x) = ?                 F(x) + x
//!                           (Residual!)
//!
//! The network learns F(x) = H(x) - x
//! If the ideal transformation is close to identity,
//! it is easier to learn F(x) ≈ 0 than H(x) ≈ x
//! ```
//!
//! ## Impact
//!
//! - **1st** at ILSVRC 2015 challenge (classification, detection, localization)
//! - **152 layers** (vs 22 for VGG, 8 for AlexNet)
//! - **Top-5 error**: 3.57% (surpasses human level ~5%)
//! - **Over 100,000 citations** (one of the most cited papers)

use cma_cnn::{Tensor4D, TensorShape};
use cma_models::resnet::{ResNet, ResNetBuilder, ResidualBlock};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  ResNet: Deep Residual Learning for Image Recognition");
    println!("  He, Zhang, Ren, Sun (CVPR 2016)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Residual Block: The Key Innovation
    // ═══════════════════════════════════════════════════════════════════════

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Residual Block: The Key Innovation                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // Block without dimension change
    let block_identity = ResidualBlock::new(64, 64, 1);
    let input_64 = Tensor4D::random(TensorShape::new(1, 64, 56, 56));
    let output_64 = block_identity.forward(&input_64);

    println!("Identity Block (stride=1, in_ch == out_ch):");
    println!("  Input:  {:?}", input_64.shape());
    println!("  Output: {:?}", output_64.shape());
    println!("  Params: {}", block_identity.num_parameters());
    println!("  Skip:   x → F(x) + x (direct identity)");
    println!();

    // Block with downsampling
    let block_downsample = ResidualBlock::new(64, 128, 2);
    let output_128 = block_downsample.forward(&input_64);

    println!("Downsample Block (stride=2, in_ch ≠ out_ch):");
    println!("  Input:  {:?}", input_64.shape());
    println!("  Output: {:?}", output_128.shape());
    println!("  Params: {}", block_downsample.num_parameters());
    println!("  Skip:   x → F(x) + Conv1×1(x) (projection)");

    // ═══════════════════════════════════════════════════════════════════════
    // ResNet-18 for CIFAR-10
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ ResNet-18 for CIFAR-10 (32×32)                                  │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // ResNet-18 equivalent using the unified builder
    let resnet18 = ResNetBuilder::new()
        .input_channels(3)
        .input_size(32)
        .channels(&[64, 128, 256, 512])
        .blocks(&[2, 2, 2, 2])
        .stem_channels(64)
        .stem_pooling(false)
        .build();

    println!("ResNet-18 (CIFAR-10 config):");
    println!("  Parameters: {}", resnet18.num_parameters());
    println!("  Output features: {}", resnet18.output_features());


    // ═══════════════════════════════════════════════════════════════════════
    // Variant Comparison
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ ResNet Variants (Table 1 from the paper)                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("┌─────────────┬────────────────────────────────────────────────────┐");
    println!("│ Layer       │ Output     18-layer    34-layer    50-layer       │");
    println!("├─────────────┼────────────────────────────────────────────────────┤");
    println!("│ conv1       │ 112×112    7×7, 64, stride 2                       │");
    println!("├─────────────┼────────────────────────────────────────────────────┤");
    println!("│             │ 56×56      3×3 max pool, stride 2                  │");
    println!("├─────────────┼────────────────────────────────────────────────────┤");
    println!("│ conv2_x     │ 56×56      [3×3, 64]    [3×3, 64]   [1×1, 64]     │");
    println!("│             │            [3×3, 64]×2  [3×3, 64]×3 [3×3, 64]×3   │");
    println!("│             │                                     [1×1, 256]    │");
    println!("├─────────────┼────────────────────────────────────────────────────┤");
    println!("│ conv3_x     │ 28×28      [3×3, 128]   [3×3, 128]  [1×1, 128]    │");
    println!("│             │            [3×3, 128]×2 [3×3, 128]×4[3×3, 128]×4  │");
    println!("│             │                                     [1×1, 512]    │");
    println!("├─────────────┼────────────────────────────────────────────────────┤");
    println!("│ conv4_x     │ 14×14      [3×3, 256]   [3×3, 256]  [1×1, 256]    │");
    println!("│             │            [3×3, 256]×2 [3×3, 256]×6[3×3, 256]×6  │");
    println!("│             │                                     [1×1, 1024]   │");
    println!("├─────────────┼────────────────────────────────────────────────────┤");
    println!("│ conv5_x     │ 7×7        [3×3, 512]   [3×3, 512]  [1×1, 512]    │");
    println!("│             │            [3×3, 512]×2 [3×3, 512]×3[3×3, 512]×3  │");
    println!("│             │                                     [1×1, 2048]   │");
    println!("├─────────────┼────────────────────────────────────────────────────┤");
    println!("│             │ 1×1        global average pool                     │");
    println!("│             │            1000-d fc, softmax                      │");
    println!("├─────────────┼────────────────────────────────────────────────────┤");
    println!("│ FLOPs       │            1.8×10⁹     3.6×10⁹     3.8×10⁹        │");
    println!("│ Params      │            11.7M       21.8M       25.6M          │");
    println!("└─────────────┴────────────────────────────────────────────────────┘");

    // ═══════════════════════════════════════════════════════════════════════
    // BasicBlock vs Bottleneck
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ BasicBlock vs Bottleneck (Figure 5 du paper)                    │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("BasicBlock (ResNet-18/34):      Bottleneck (ResNet-50+):");
    println!("┌─────────────────────┐         ┌─────────────────────┐");
    println!("│ x                   │         │ x                   │");
    println!("└─────────┬───────────┘         └─────────┬───────────┘");
    println!("          │                               │");
    println!("          ▼                               ▼");
    println!("┌─────────────────────┐         ┌─────────────────────┐");
    println!("│ Conv 3×3, 64        │         │ Conv 1×1, 64        │ ← Reduce");
    println!("│ BN + ReLU           │         │ BN + ReLU           │");
    println!("└─────────┬───────────┘         └─────────┬───────────┘");
    println!("          │                               │");
    println!("          ▼                               ▼");
    println!("┌─────────────────────┐         ┌─────────────────────┐");
    println!("│ Conv 3×3, 64        │         │ Conv 3×3, 64        │");
    println!("│ BN                  │         │ BN + ReLU           │");
    println!("└─────────┬───────────┘         └─────────┬───────────┘");
    println!("          │                               │");
    println!("          ▼                               ▼");
    println!("    (Add + ReLU)               ┌─────────────────────┐");
    println!("                               │ Conv 1×1, 256       │ ← Expand");
    println!("                               │ BN                  │");
    println!("                               └─────────┬───────────┘");
    println!("                                         │");
    println!("                                         ▼");
    println!("                                   (Add + ReLU)");
    println!();
    println!("Bottleneck reduces computation: 3×3×64×64 → 1×1×64 + 3×3×64 + 1×1×256");

    // ═══════════════════════════════════════════════════════════════════════
    // Paper Results
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Paper Results (ILSVRC-2015)                                     │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────────────┬──────────┬──────────┬─────────────────────┐");
    println!("│ Model               │ Layers   │ Top-5 Err│ Notes               │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ VGG-16              │ 16       │ 7.32%    │ Baseline 2014       │");
    println!("│ VGG-19              │ 19       │ 7.10%    │                     │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ ResNet-34           │ 34       │ 5.71%    │ BasicBlock          │");
    println!("│ ResNet-50           │ 50       │ 5.25%    │ Bottleneck          │");
    println!("│ ResNet-101          │ 101      │ 4.60%    │                     │");
    println!("│ ResNet-152          │ 152      │ 4.49%    │ Single model        │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ ResNet-152 ensemble │ -        │ 3.57%    │ WINNER ILSVRC-2015  │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ Human performance   │ -        │ ~5.1%    │ Andrej Karpathy     │");
    println!("└─────────────────────┴──────────┴──────────┴─────────────────────┘");
    println!();
    println!("ResNet-152 surpasses human performance on ImageNet!");

    // ═══════════════════════════════════════════════════════════════════════
    // Why It Works
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Why Skip Connections Work                                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("1. GRADIENT FLOW:");
    println!("   Without skip: ∂L/∂x = ∂L/∂F × ∂F/∂x");
    println!("   With skip: ∂L/∂x = ∂L/∂F × ∂F/∂x + ∂L/∂x");
    println!("                                        ^^^^^ highway!");
    println!("   Gradients can \"bypass\" the layers.");
    println!();
    println!("2. IDENTITY MAPPING:");
    println!("   If F(x) ≈ 0, then y = x + F(x) ≈ x");
    println!("   Learning F(x) = 0 is easier than H(x) = x");
    println!();
    println!("3. IMPLICIT ENSEMBLING:");
    println!("   A ResNet with N blocks = ensemble of 2^N paths");
    println!("   Each path = sub-network of variable depth");

    // ═══════════════════════════════════════════════════════════════════════
    // FC Classifier
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ FC Classifier (cma-neural-network)                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("```rust");
    println!(
        "use cma_neural_network::{{NetworkBuilder, Activation, LossFunction, OptimizerType}};"
    );
    println!();
    println!("// ResNet produces 512 features (after GAP)");
    println!("let classifier = NetworkBuilder::new(512, 10)");
    println!("    .output_activation(Activation::Softmax)");
    println!("    .loss(LossFunction::CategoricalCrossEntropy)");
    println!("    .optimizer(OptimizerType::adam(0.001))");
    println!("    .build();");
    println!("```");
    println!();
    println!("Note: ResNet uses Global Average Pooling → no FC hidden layers.");
    println!("      The final classifier is simply a linear projection.");

    // ═══════════════════════════════════════════════════════════════════════
    // Legacy
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ ResNet's Legacy                                                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("ResNet inspired:");
    println!("- DenseNet (2017): All layers connected to each other");
    println!("- ResNeXt (2017): Aggregation of parallel branches");
    println!("- SE-ResNet (2018): Squeeze-and-Excitation attention");
    println!("- EfficientNet (2019): Compound scaling");
    println!("- Vision Transformers (2020): Skip connections in attention");
    println!();
    println!("The concept of skip connection is now UNIVERSAL:");
    println!("- Transformers (attention residual)");
    println!("- U-Net (segmentation)");
    println!("- Diffusion models (denoising)");
    println!("- LLMs (layer normalization with residual)");

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  End of ResNet example");
    println!("═══════════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residual_block() {
        let block = ResidualBlock::new(64, 64, 1);
        let input = Tensor4D::random(TensorShape::new(1, 64, 56, 56));
        let output = block.forward(&input);
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_resnet18() {
        let model = ResNetBuilder::new()
            .input_channels(3)
            .input_size(32)
            .channels(&[64, 128, 256, 512])
            .blocks(&[2, 2, 2, 2])
            .stem_channels(64)
            .stem_pooling(false)
            .build();
        let input = Tensor4D::random(TensorShape::new(1, 3, 32, 32));
        let output = model.forward(&input);
        assert_eq!(output.shape().width, 512);
    }
}
