//! # Exemple EfficientNet: Compound Scaling (Tan & Le, 2019)
//!
//! Cet exemple reproduit l'architecture EfficientNet qui a révolutionné
//! l'efficacité des CNN grâce au compound scaling.
//!
//! ## Paper Citation
//!
//! ```text
//! @inproceedings{tan2019efficientnet,
//!   title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
//!   author={Tan, Mingxin and Le, Quoc V.},
//!   booktitle={International Conference on Machine Learning},
//!   pages={6105--6114},
//!   year={2019},
//!   organization={PMLR}
//! }
//! ```
//!
//! ## Le Problème du Scaling
//!
//! ```text
//! Comment rendre un réseau plus performant ?
//!
//! Option 1: Plus LARGE (width)          ResNeXt
//!           ┌───┐         ┌───────────┐
//!           │64 │   →     │   128     │
//!           └───┘         └───────────┘
//!
//! Option 2: Plus PROFOND (depth)        ResNet-152
//!           ┌───┐         ┌───┐
//!           │   │         │   │
//!           └───┘   →     ├───┤
//!                         │   │
//!                         └───┘
//!
//! Option 3: Plus RÉSOLU (resolution)    High-res input
//!           ┌───┐         ┌─────────┐
//!           │224│   →     │   448   │
//!           └───┘         └─────────┘
//!
//! Problème: Chaque dimension a des rendements décroissants!
//! ```
//!
//! ## La Solution: Compound Scaling
//!
//! ```text
//! EfficientNet scale les 3 dimensions ENSEMBLE:
//!
//! depth:      d = α^φ
//! width:      w = β^φ
//! resolution: r = γ^φ
//!
//! Contrainte: α × β² × γ² ≈ 2 (pour doubler les FLOPs)
//!
//! Valeurs optimales (trouvées par grid search sur B0):
//!   α = 1.2   (depth)
//!   β = 1.1   (width)
//!   γ = 1.15  (resolution)
//!
//! EfficientNet-B0: φ = 0 (baseline)
//! EfficientNet-B1: φ = 1
//! EfficientNet-B2: φ = 2
//! ...
//! EfficientNet-B7: φ = 7
//! ```

use cma_cnn::{Tensor4D, TensorShape};
use cma_models::efficientnet::{
    EfficientNetB0, EfficientNetConfig, MBConvBlock, SqueezeExcitation,
};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  EfficientNet: Rethinking Model Scaling for CNNs");
    println!("  Tan & Le (ICML 2019)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Mobile Inverted Bottleneck (MBConv)
    // ═══════════════════════════════════════════════════════════════════════

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ MBConv Block: L'Unité de Base                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("MBConv = Mobile Inverted Bottleneck Convolution");
    println!("(Originaire de MobileNetV2, utilisé dans EfficientNet)");
    println!();
    println!("```");
    println!("Standard Bottleneck (ResNet):     Inverted Bottleneck (MobileNet):");
    println!("┌─────────────────────┐           ┌─────────────────────┐");
    println!("│ x (high dim)        │           │ x (low dim)         │");
    println!("└─────────┬───────────┘           └─────────┬───────────┘");
    println!("          ▼                                 ▼");
    println!("┌─────────────────────┐           ┌─────────────────────┐");
    println!("│ Conv 1×1 (reduce)   │           │ Conv 1×1 (expand)   │← ×6");
    println!("└─────────┬───────────┘           │ BN + Swish          │");
    println!("          ▼                       └─────────┬───────────┘");
    println!("┌─────────────────────┐                     ▼");
    println!("│ Conv 3×3            │           ┌─────────────────────┐");
    println!("└─────────┬───────────┘           │ DepthwiseConv 3×3/5×5│");
    println!("          ▼                       │ BN + Swish          │");
    println!("┌─────────────────────┐           └─────────┬───────────┘");
    println!("│ Conv 1×1 (expand)   │                     ▼");
    println!("└─────────┬───────────┘           ┌─────────────────────┐");
    println!("          ▼                       │ Squeeze-Excitation  │");
    println!("     (Add + ReLU)                 └─────────┬───────────┘");
    println!("                                            ▼");
    println!("                                  ┌─────────────────────┐");
    println!("                                  │ Conv 1×1 (project)  │← low dim");
    println!("                                  │ BN (no activation!) │");
    println!("                                  └─────────┬───────────┘");
    println!("                                            ▼");
    println!("                                       (Add residual)");
    println!("```");
    println!();

    // Démonstration MBConv
    let mbconv = MBConvBlock::new(32, 16, 3, 1, 6, true); // use_se = true
    let input = Tensor4D::random(TensorShape::new(1, 32, 56, 56));
    let output = mbconv.forward(&input);

    println!("MBConv6 (expand_ratio=6, kernel=3×3, SE=true):");
    println!("  Input:  {:?}", input.shape());
    println!("  Output: {:?}", output.shape());
    println!("  Expand: 32 → 192 → 16");

    // ═══════════════════════════════════════════════════════════════════════
    // Squeeze-and-Excitation
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Squeeze-and-Excitation (Hu et al., 2018)                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("```");
    println!("SE Block: Channel Attention");
    println!();
    println!("Input: [B, C, H, W]");
    println!("          │");
    println!("          ▼");
    println!("┌─────────────────────────────────────┐");
    println!("│ Global Average Pooling              │");
    println!("│ [B, C, H, W] → [B, C, 1, 1]         │");
    println!("└───────────────┬─────────────────────┘");
    println!("                │");
    println!("                ▼ Squeeze");
    println!("┌─────────────────────────────────────┐");
    println!("│ FC (C → C/r) + ReLU                 │ r = 4 (reduction)");
    println!("└───────────────┬─────────────────────┘");
    println!("                │");
    println!("                ▼ Excitation");
    println!("┌─────────────────────────────────────┐");
    println!("│ FC (C/r → C) + Sigmoid              │");
    println!("└───────────────┬─────────────────────┘");
    println!("                │");
    println!("                ▼ Scale (channel-wise multiply)");
    println!("         [B, C, 1, 1] × [B, C, H, W]");
    println!("                │");
    println!("                ▼");
    println!("Output: [B, C, H, W] (recalibrated)");
    println!("```");
    println!();

    let se = SqueezeExcitation::new(64, 4);
    let se_input = Tensor4D::random(TensorShape::new(1, 64, 28, 28));
    let se_output = se.forward(&se_input);

    println!("SE Block (reduction_ratio=4):");
    println!("  Input:  {:?}", se_input.shape());
    println!("  Output: {:?}", se_output.shape());
    println!("  Path:   64 → 16 → 64 (attention weights)");

    // ═══════════════════════════════════════════════════════════════════════
    // Architecture EfficientNet-B0
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ EfficientNet-B0 Architecture (Baseline)                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let config = EfficientNetConfig::b0();
    let efficientnet = EfficientNetB0::with_config(config);

    efficientnet.summary();

    let batch = Tensor4D::random(TensorShape::new(1, 3, 224, 224));
    let features = efficientnet.forward(&batch);

    println!();
    println!("Forward pass:");
    println!("  Input:  [1, 3, 224, 224]");
    println!(
        "  Output: {:?} (1280 features après head)",
        features.shape()
    );
    println!("  Params: {} (~5.3M)", efficientnet.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // Compound Scaling
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Compound Scaling: B0 → B7                                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("```");
    println!("Model       Resolution  Width   Depth   Params    Top-1 Acc");
    println!("──────────────────────────────────────────────────────────────");
    println!("EfficientNet-B0  224      1.0     1.0    5.3M      77.1%");
    println!("EfficientNet-B1  240      1.0     1.1    7.8M      79.1%");
    println!("EfficientNet-B2  260      1.1     1.2    9.2M      80.1%");
    println!("EfficientNet-B3  300      1.2     1.4    12M       81.6%");
    println!("EfficientNet-B4  380      1.4     1.8    19M       82.9%");
    println!("EfficientNet-B5  456      1.6     2.2    30M       83.6%");
    println!("EfficientNet-B6  528      1.8     2.6    43M       84.0%");
    println!("EfficientNet-B7  600      2.0     3.1    66M       84.3%");
    println!("──────────────────────────────────────────────────────────────");
    println!("ResNet-50         224      -       -      26M      76.3%");
    println!("ResNet-152        224      -       -      60M      77.8%");
    println!("```");
    println!();
    println!("EfficientNet-B0 est meilleur que ResNet-50 avec 5× moins de params!");
    println!("EfficientNet-B7 atteint 84.3% avec moins de params que GPipe (84.3%, 557M)");

    // ═══════════════════════════════════════════════════════════════════════
    // Swish Activation
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Swish: L'Activation Magique                                     │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("```");
    println!("Swish(x) = x × σ(x) = x × (1 / (1 + e^(-x)))");
    println!();
    println!("        │");
    println!("   1.0  │          ....────");
    println!("        │       ...");
    println!("   0.5  │     ..          Swish");
    println!("        │   ..            (smooth)");
    println!("   0.0  ├─.───────────────────────");
    println!("        │..");
    println!("  -0.5  │ .                ReLU");
    println!("        │                  (sharp)");
    println!("        └─────────────────────────");
    println!("           -2   0   2   4");
    println!();
    println!("Propriétés:");
    println!("• Smooth (dérivée continue)");
    println!("• Non-monotone (légère négativité)");
    println!("• Self-gated (x contrôle son propre gate)");
    println!("```");

    // ═══════════════════════════════════════════════════════════════════════
    // CIFAR-10 Configuration
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ EfficientNet pour CIFAR-10                                      │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let cifar_config = EfficientNetConfig::cifar10();
    let efficientnet_cifar = EfficientNetB0::with_config(cifar_config);

    println!("Configuration adaptée:");
    println!("  Input: 32×32 (vs 224×224 original)");
    println!("  Classes: 10 (vs 1000)");
    println!("  Stride initial: 1 (vs 2) pour préserver la résolution");
    println!();

    let cifar_batch = Tensor4D::random(TensorShape::new(32, 3, 32, 32));
    let cifar_features = efficientnet_cifar.forward(&cifar_batch);

    println!("Forward pass (CIFAR-10):");
    println!("  Input:  [32, 3, 32, 32]");
    println!("  Output: {:?}", cifar_features.shape());

    // ═══════════════════════════════════════════════════════════════════════
    // Comparaison Efficacité
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Comparaison d'Efficacité                                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────────────┬──────────┬──────────┬─────────────────────┐");
    println!("│ Modèle              │ Params   │ FLOPs    │ ImageNet Top-1      │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ ResNet-50           │ 26M      │ 3.8B     │ 76.3%               │");
    println!("│ ResNet-152          │ 60M      │ 11.3B    │ 77.8%               │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ DenseNet-201        │ 20M      │ 4.3B     │ 77.4%               │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ EfficientNet-B0     │ 5.3M     │ 0.39B    │ 77.1%               │");
    println!("│ EfficientNet-B1     │ 7.8M     │ 0.70B    │ 79.1%               │");
    println!("│ EfficientNet-B3     │ 12M      │ 1.8B     │ 81.6%               │");
    println!("│ EfficientNet-B7     │ 66M      │ 37B      │ 84.3%               │");
    println!("└─────────────────────┴──────────┴──────────┴─────────────────────┘");
    println!();
    println!("EfficientNet-B0 ≈ ResNet-50 accuracy avec 5× moins de params!");

    // ═══════════════════════════════════════════════════════════════════════
    // Neural Architecture Search (NAS)
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Neural Architecture Search (NAS)                                │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("L'architecture de base (B0) a été trouvée par NAS:");
    println!();
    println!("1. SEARCH SPACE:");
    println!("   • Type de bloc: MBConv3 ou MBConv6");
    println!("   • Kernel size: 3×3 ou 5×5");
    println!("   • Channels: 16, 24, 40, 80, 112, 192, 320");
    println!("   • Layers per stage: 1-7");
    println!();
    println!("2. OPTIMIZATION:");
    println!("   Objectif multi-critère:");
    println!("   • Maximiser accuracy");
    println!("   • Minimiser FLOPs");
    println!("   • Minimiser latence");
    println!();
    println!("3. RÉSULTAT:");
    println!("   EfficientNet-B0: 7 stages avec des configs optimales");
    println!("   Puis compound scaling pour B1-B7");

    // ═══════════════════════════════════════════════════════════════════════
    // Usage avec cma-neural-network
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
    println!("// EfficientNet-B0 produit 1280 features (après head)");
    println!("let classifier = NetworkBuilder::new(1280, 10)");
    println!("    .output_activation(Activation::Softmax)");
    println!("    .loss(LossFunction::CategoricalCrossEntropy)");
    println!("    .optimizer(OptimizerType::adam(0.001))");
    println!("    .build();");
    println!();
    println!("// Note: EfficientNet utilise déjà Global Average Pooling");
    println!("// et un head Conv1×1 → 1280, donc le classifieur est simple");
    println!("```");

    // ═══════════════════════════════════════════════════════════════════════
    // Évolutions
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Évolutions d'EfficientNet                                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("2019: EfficientNet (original)");
    println!("      • Compound scaling");
    println!("      • MBConv + SE");
    println!();
    println!("2020: EfficientNet-Lite");
    println!("      • Pour mobile/edge devices");
    println!("      • Swish → ReLU6");
    println!("      • SE simplifié");
    println!();
    println!("2021: EfficientNetV2");
    println!("      • Training-aware NAS");
    println!("      • Fused-MBConv (conv 3×3 au lieu de depthwise)");
    println!("      • Progressive learning");
    println!("      • 11× faster training que V1");
    println!();
    println!("2023: EfficientViT");
    println!("      • Hybrid CNN-Transformer");
    println!("      • Efficient attention");
    println!("      • State-of-the-art mobile");

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Fin de l'exemple EfficientNet");
    println!("═══════════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbconv_block() {
        let block = MBConvBlock::new(32, 16, 3, 1, 6);
        let input = Tensor4D::random(TensorShape::new(1, 32, 28, 28));
        let output = block.forward(&input);
        assert_eq!(output.shape().channels, 16);
    }

    #[test]
    fn test_squeeze_excitation() {
        let se = SqueezeExcitation::new(64, 4);
        let input = Tensor4D::random(TensorShape::new(1, 64, 14, 14));
        let output = se.forward(&input);
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_efficientnet_b0() {
        let config = EfficientNetConfig::cifar10();
        let model = EfficientNetB0::with_config(config);
        let input = Tensor4D::random(TensorShape::new(1, 3, 32, 32));
        let output = model.forward(&input);
        assert!(output.shape().width > 0);
    }
}
