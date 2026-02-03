//! # Exemple ResNet: Deep Residual Learning (He et al., 2015)
//!
//! Cet exemple reproduit l'architecture ResNet qui a révolutionné
//! l'entraînement de réseaux très profonds grâce aux skip connections.
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
//! ## Le Problème de la Profondeur
//!
//! Avant ResNet, les réseaux plus profonds avaient une accuracy PIRE:
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
//! ## La Solution: Skip Connections
//!
//! ```text
//! Bloc Standard:          Bloc Résiduel:
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
//! Le réseau apprend F(x) = H(x) - x
//! Si la transformation idéale est proche de l'identité,
//! il est plus facile d'apprendre F(x) ≈ 0 que H(x) ≈ x
//! ```
//!
//! ## Impact
//!
//! - **1er** au challenge ILSVRC 2015 (classification, détection, localisation)
//! - **152 couches** (vs 22 pour VGG, 8 pour AlexNet)
//! - **Top-5 error**: 3.57% (surpasse le niveau humain ~5%)
//! - **Plus de 100,000 citations** (un des papers les plus cités)

use cma_cnn::{Tensor4D, TensorShape};
use cma_models::resnet::{ResNet18, ResNet34, ResNet50, ResNetConfig, ResidualBlock};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  ResNet: Deep Residual Learning for Image Recognition");
    println!("  He, Zhang, Ren, Sun (CVPR 2016)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Bloc Résiduel: L'Innovation Clé
    // ═══════════════════════════════════════════════════════════════════════

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Bloc Résiduel: L'Innovation Clé                                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // Bloc sans changement de dimension
    let block_identity = ResidualBlock::new(64, 64, 1);
    let input_64 = Tensor4D::random(TensorShape::new(1, 64, 56, 56));
    let output_64 = block_identity.forward(&input_64);

    println!("Bloc Identity (stride=1, in_ch == out_ch):");
    println!("  Input:  {:?}", input_64.shape());
    println!("  Output: {:?}", output_64.shape());
    println!("  Params: {}", block_identity.num_parameters());
    println!("  Skip:   x → F(x) + x (identité directe)");
    println!();

    // Bloc avec downsampling
    let block_downsample = ResidualBlock::new(64, 128, 2);
    let output_128 = block_downsample.forward(&input_64);

    println!("Bloc Downsample (stride=2, in_ch ≠ out_ch):");
    println!("  Input:  {:?}", input_64.shape());
    println!("  Output: {:?}", output_128.shape());
    println!("  Params: {}", block_downsample.num_parameters());
    println!("  Skip:   x → F(x) + Conv1×1(x) (projection)");

    // ═══════════════════════════════════════════════════════════════════════
    // ResNet-18 pour CIFAR-10
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ ResNet-18 pour CIFAR-10 (32×32)                                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let mut config = ResNetConfig::resnet18();
    config.input_size = 32;
    config.num_classes = 10;

    let resnet18 = ResNet18::with_config(config);

    resnet18.summary();

    let batch = Tensor4D::random(TensorShape::new(32, 3, 32, 32));
    let features = resnet18.forward(&batch);

    println!();
    println!("Forward pass:");
    println!("  Input:  [32, 3, 32, 32]");
    println!("  Output: {:?} (512 features après GAP)", features.shape());

    // ═══════════════════════════════════════════════════════════════════════
    // Comparaison des Variantes
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Variantes ResNet (Table 1 du paper)                             │");
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
    println!("Bottleneck réduit les calculs: 3×3×64×64 → 1×1×64 + 3×3×64 + 1×1×256");

    // ═══════════════════════════════════════════════════════════════════════
    // Résultats du Paper
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Résultats du Paper (ILSVRC-2015)                                │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────────────┬──────────┬──────────┬─────────────────────┐");
    println!("│ Modèle              │ Couches  │ Top-5 Err│ Notes               │");
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
    println!("ResNet-152 surpasse la performance humaine sur ImageNet!");

    // ═══════════════════════════════════════════════════════════════════════
    // Pourquoi ça Marche
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Pourquoi les Skip Connections Marchent                          │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("1. GRADIENT FLOW:");
    println!("   Sans skip: ∂L/∂x = ∂L/∂F × ∂F/∂x");
    println!("   Avec skip: ∂L/∂x = ∂L/∂F × ∂F/∂x + ∂L/∂x");
    println!("                                        ^^^^^ highway!");
    println!("   Les gradients peuvent \"bypasser\" les couches.");
    println!();
    println!("2. IDENTITY MAPPING:");
    println!("   Si F(x) ≈ 0, alors y = x + F(x) ≈ x");
    println!("   Apprendre F(x) = 0 est plus facile que H(x) = x");
    println!();
    println!("3. IMPLICIT ENSEMBLING:");
    println!("   Un ResNet de N blocs = ensemble de 2^N chemins");
    println!("   Chaque chemin = sous-réseau de profondeur variable");

    // ═══════════════════════════════════════════════════════════════════════
    // Classifieur FC
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Classifieur FC (cma-neural-network)                             │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("```rust");
    println!(
        "use cma_neural_network::{{NetworkBuilder, Activation, LossFunction, OptimizerType}};"
    );
    println!();
    println!("// ResNet produit 512 features (après GAP)");
    println!("let classifier = NetworkBuilder::new(512, 10)");
    println!("    .output_activation(Activation::Softmax)");
    println!("    .loss(LossFunction::CategoricalCrossEntropy)");
    println!("    .optimizer(OptimizerType::adam(0.001))");
    println!("    .build();");
    println!("```");
    println!();
    println!("Note: ResNet utilise Global Average Pooling → pas de FC hidden layers.");
    println!("      Le classifieur final est simplement une projection linéaire.");

    // ═══════════════════════════════════════════════════════════════════════
    // Héritage
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Héritage de ResNet                                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("ResNet a inspiré:");
    println!("- DenseNet (2017): Toutes les couches connectées entre elles");
    println!("- ResNeXt (2017): Agrégation de branches parallèles");
    println!("- SE-ResNet (2018): Squeeze-and-Excitation attention");
    println!("- EfficientNet (2019): Compound scaling");
    println!("- Vision Transformers (2020): Skip connections dans attention");
    println!();
    println!("Le concept de skip connection est maintenant UNIVERSEL:");
    println!("- Transformers (attention residual)");
    println!("- U-Net (segmentation)");
    println!("- Diffusion models (denoising)");
    println!("- LLMs (layer normalization with residual)");

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Fin de l'exemple ResNet");
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
        let mut config = ResNetConfig::resnet18();
        config.input_size = 32;
        let model = ResNet18::with_config(config);
        let input = Tensor4D::random(TensorShape::new(1, 3, 32, 32));
        let output = model.forward(&input);
        assert_eq!(output.shape().width, 512);
    }
}
