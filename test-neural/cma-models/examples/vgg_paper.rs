//! # Exemple VGG: Very Deep Convolutional Networks (Simonyan & Zisserman, 2014)
//!
//! Cet exemple reproduit l'architecture VGG qui a démontré l'importance
//! de la profondeur et la puissance des petits filtres 3×3.
//!
//! ## Paper Citation
//!
//! ```text
//! @article{simonyan2014very,
//!   title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
//!   author={Simonyan, Karen and Zisserman, Andrew},
//!   journal={arXiv preprint arXiv:1409.1556},
//!   year={2014}
//! }
//! ```
//!
//! ## Philosophie VGG
//!
//! ```text
//! "We address an important aspect of ConvNet architecture design –
//!  its depth. [...] we push the depth to 16-19 weight layers,
//!  which is substantially deeper than what has been used before"
//!                                    - Simonyan & Zisserman
//!
//! La clé: utiliser UNIQUEMENT des filtres 3×3
//!
//! Pourquoi 3×3 ?
//! ━━━━━━━━━━━━━━
//! 2 convolutions 3×3 = champ réceptif 5×5
//! 3 convolutions 3×3 = champ réceptif 7×7
//!
//! Mais avec MOINS de paramètres:
//! • 3 × (3×3×C×C) = 27C²
//! • 1 × (7×7×C×C) = 49C²
//!
//! Et PLUS de non-linéarités (ReLU entre chaque conv)
//! ```
//!
//! ## Architecture VGG-16
//!
//! ```text
//!     Input: 224×224×3
//!     ┌───────────────────────────────────────────────────────────────────┐
//!     │ [Conv 3×3, 64] × 2  →  MaxPool 2×2  →  112×112×64                │
//!     │ [Conv 3×3, 128] × 2 →  MaxPool 2×2  →  56×56×128                 │
//!     │ [Conv 3×3, 256] × 3 →  MaxPool 2×2  →  28×28×256                 │
//!     │ [Conv 3×3, 512] × 3 →  MaxPool 2×2  →  14×14×512                 │
//!     │ [Conv 3×3, 512] × 3 →  MaxPool 2×2  →  7×7×512                   │
//!     │ Flatten → 7×7×512 = 25,088                                        │
//!     │ FC 4096 → ReLU → Dropout(0.5)                                     │
//!     │ FC 4096 → ReLU → Dropout(0.5)                                     │
//!     │ FC 1000 → Softmax                                                 │
//!     └───────────────────────────────────────────────────────────────────┘
//!     Total: ~138M paramètres (dont 123M dans les FC!)
//! ```

use cma_cnn::{Tensor4D, TensorShape};
use cma_models::vgg::{VGG16, VGG19, VGGConfig};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  VGG: Very Deep Convolutional Networks for Large-Scale Image");
    println!("       Recognition");
    println!("  Simonyan & Zisserman (ICLR 2015)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Configurations VGG (Table 1 du paper)
    // ═══════════════════════════════════════════════════════════════════════

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Configurations VGG (Table 1 du paper)                           │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────┬───────────┬───────────┬───────────┬──────────────┐");
    println!("│   Layer     │    A      │  A-LRN    │    B      │   C (VGG-16) │");
    println!("│             │ (11 lyrs) │ (11 lyrs) │ (13 lyrs) │   (16 lyrs)  │");
    println!("├─────────────┼───────────┼───────────┼───────────┼──────────────┤");
    println!("│ conv3-64    │     1     │    1+LRN  │     2     │      2       │");
    println!("│ maxpool     │     ✓     │    ✓      │     ✓     │      ✓       │");
    println!("├─────────────┼───────────┼───────────┼───────────┼──────────────┤");
    println!("│ conv3-128   │     1     │    1      │     2     │      2       │");
    println!("│ maxpool     │     ✓     │    ✓      │     ✓     │      ✓       │");
    println!("├─────────────┼───────────┼───────────┼───────────┼──────────────┤");
    println!("│ conv3-256   │     2     │    2      │     2     │   2 + 1×1    │");
    println!("│ maxpool     │     ✓     │    ✓      │     ✓     │      ✓       │");
    println!("├─────────────┼───────────┼───────────┼───────────┼──────────────┤");
    println!("│ conv3-512   │     2     │    2      │     2     │   2 + 1×1    │");
    println!("│ maxpool     │     ✓     │    ✓      │     ✓     │      ✓       │");
    println!("├─────────────┼───────────┼───────────┼───────────┼──────────────┤");
    println!("│ conv3-512   │     2     │    2      │     2     │   2 + 1×1    │");
    println!("│ maxpool     │     ✓     │    ✓      │     ✓     │      ✓       │");
    println!("├─────────────┼───────────┼───────────┼───────────┼──────────────┤");
    println!("│ FC-4096     │     ✓     │    ✓      │     ✓     │      ✓       │");
    println!("│ FC-4096     │     ✓     │    ✓      │     ✓     │      ✓       │");
    println!("│ FC-1000     │     ✓     │    ✓      │     ✓     │      ✓       │");
    println!("├─────────────┼───────────┼───────────┼───────────┼──────────────┤");
    println!("│ Params      │   133M    │   133M    │   133M    │     138M     │");
    println!("└─────────────┴───────────┴───────────┴───────────┴──────────────┘");

    // ═══════════════════════════════════════════════════════════════════════
    // VGG-16 pour ImageNet
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ VGG-16 Configuration D (Configuration classique)                │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let config = VGGConfig::vgg16();
    let vgg16 = VGG16::with_config(config);

    vgg16.summary();

    let batch = Tensor4D::random(TensorShape::new(1, 3, 224, 224));
    let features = vgg16.forward(&batch);

    println!();
    println!("Forward pass (ImageNet):");
    println!("  Input:  [1, 3, 224, 224]");
    println!("  Output: {:?} (4096 features après FC)", features.shape());
    println!("  Params: {} (~138M)", vgg16.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // VGG pour CIFAR-10
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ VGG-16 adapté pour CIFAR-10 (32×32)                             │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let cifar_config = VGGConfig::cifar10();
    let vgg16_cifar = VGG16::with_config(cifar_config);

    vgg16_cifar.summary();

    let cifar_batch = Tensor4D::random(TensorShape::new(32, 3, 32, 32));
    let cifar_features = vgg16_cifar.forward(&cifar_batch);

    println!();
    println!("Forward pass (CIFAR-10):");
    println!("  Input:  [32, 3, 32, 32]");
    println!("  Output: {:?}", cifar_features.shape());
    println!(
        "  Params: {} (beaucoup moins car FC plus petit)",
        vgg16_cifar.num_parameters()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // VGG-19
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ VGG-19 Configuration E (le plus profond)                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let vgg19_config = VGGConfig::vgg19();
    let vgg19 = VGG19::with_config(vgg19_config);

    vgg19.features.summary(TensorShape::new(1, 3, 224, 224));

    // ═══════════════════════════════════════════════════════════════════════
    // Analyse des Paramètres
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Analyse des Paramètres VGG-16                                   │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("```");
    println!("Layer            Params       Cumul      % Total");
    println!("──────────────────────────────────────────────────");
    println!("Conv1_1 (3→64)   1,792        1,792      0.001%");
    println!("Conv1_2 (64→64)  36,928       38,720     0.03%");
    println!("Conv2_1 (64→128) 73,856       112,576    0.08%");
    println!("Conv2_2          147,584      260,160    0.19%");
    println!("Conv3_1          295,168      555,328    0.40%");
    println!("Conv3_2          590,080      1,145,408  0.83%");
    println!("Conv3_3          590,080      1,735,488  1.26%");
    println!("Conv4_1          1,180,160    2,915,648  2.11%");
    println!("Conv4_2          2,359,808    5,275,456  3.82%");
    println!("Conv4_3          2,359,808    7,635,264  5.53%");
    println!("Conv5_1          2,359,808    9,995,072  7.24%");
    println!("Conv5_2          2,359,808    12,354,880 8.95%");
    println!("Conv5_3          2,359,808    14,714,688 10.66%");
    println!("──────────────────────────────────────────────────");
    println!("FC1 (25088→4096) 102,764,544  117,479,232 85.1%  ← !!!");
    println!("FC2 (4096→4096)  16,781,312   134,260,544 97.3%");
    println!("FC3 (4096→1000)  4,097,000    138,357,544 100%");
    println!("──────────────────────────────────────────────────");
    println!("Total Conv:      14.7M (10.6%)");
    println!("Total FC:        123.6M (89.4%) ← Problème!");
    println!("```");
    println!();
    println!("Les couches FC contiennent 89% des paramètres!");
    println!("C'est pourquoi ResNet utilise Global Average Pooling.");

    // ═══════════════════════════════════════════════════════════════════════
    // Champ Réceptif
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Champ Réceptif: Pourquoi 3×3 × N > 7×7 × 1                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("```");
    println!("2 couches 3×3:                 1 couche 5×5:");
    println!("                               ");
    println!("Couche 2:    [3×3]             [5×5]");
    println!("               ↑                 ↑");
    println!("Couche 1:  [1×1×3×3]          [1×1×5×5]");
    println!("               ↑                 ↑");
    println!("Input:   [1×1×5×5]            [1×1×5×5]");
    println!("         = 5×5 RF             = 5×5 RF");
    println!();
    println!("Params: 2 × (3×3×C×C) = 18C²  25C²");
    println!("ReLUs:  2                      1");
    println!("```");
    println!();
    println!("Conclusion: 2 couches 3×3 ont:");
    println!("• Même champ réceptif que 1 couche 5×5");
    println!("• 28% moins de paramètres");
    println!("• 2× plus de non-linéarités");

    // ═══════════════════════════════════════════════════════════════════════
    // Résultats ILSVRC
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Résultats ILSVRC-2014                                           │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────────────┬──────────┬──────────┬─────────────────────┐");
    println!("│ Modèle              │ Top-1 Err│ Top-5 Err│ Notes               │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ AlexNet (2012)      │ 40.7%    │ 18.2%    │ Baseline            │");
    println!("│ ZFNet (2013)        │ 36.0%    │ 14.8%    │                     │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ VGG-11 (A)          │ 29.6%    │ 10.4%    │                     │");
    println!("│ VGG-13 (B)          │ 28.7%    │  9.9%    │                     │");
    println!("│ VGG-16 (D)          │ 28.5%    │  9.9%    │                     │");
    println!("│ VGG-19 (E)          │ 28.7%    │  9.9%    │                     │");
    println!("├─────────────────────┼──────────┼──────────┼─────────────────────┤");
    println!("│ GoogLeNet           │ -        │  6.67%   │ WINNER 2014         │");
    println!("│ VGGNet ensemble     │ 25.5%    │  7.32%   │ 2nd place 2014      │");
    println!("└─────────────────────┴──────────┴──────────┴─────────────────────┘");
    println!();
    println!("VGG a fini 2ème mais est devenu plus populaire que GoogLeNet");
    println!("grâce à sa simplicité architecturale.");

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
    println!("// VGG produit 4096 features (après les FC internes)");
    println!("// Pour un classifieur, il suffit de la dernière couche:");
    println!("let classifier = NetworkBuilder::new(4096, 1000)");
    println!("    .output_activation(Activation::Softmax)");
    println!("    .loss(LossFunction::CategoricalCrossEntropy)");
    println!("    .optimizer(OptimizerType::sgd_with_momentum(0.01, 0.9))");
    println!("    .build();");
    println!();
    println!("// Ou version sans les FC internes de VGG:");
    println!("let classifier = NetworkBuilder::new(512 * 7 * 7, 10)");
    println!("    .hidden_layers(&[4096, 4096])");
    println!("    .hidden_activation(Activation::ReLU)");
    println!("    .dropout(0.5)");
    println!("    .output_activation(Activation::Softmax)");
    println!("    .loss(LossFunction::CategoricalCrossEntropy)");
    println!("    .optimizer(OptimizerType::sgd_with_momentum(0.01, 0.9))");
    println!("    .build();");
    println!("```");

    // ═══════════════════════════════════════════════════════════════════════
    // Héritage de VGG
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Héritage de VGG                                                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("VGG a établi des principes fondamentaux:");
    println!();
    println!("1. PROFONDEUR COMPTE");
    println!("   Plus profond = meilleur (jusqu'à 19 couches)");
    println!("   Limite: degradation problem → résolu par ResNet");
    println!();
    println!("2. PETITS FILTRES");
    println!("   3×3 est devenu le standard de facto");
    println!("   Utilisé dans ResNet, DenseNet, etc.");
    println!();
    println!("3. STRUCTURE EN BLOCS");
    println!("   [Conv-ReLU] × N → MaxPool");
    println!("   Doubler les channels à chaque étage");
    println!();
    println!("4. TRANSFER LEARNING");
    println!("   VGG pre-trained est encore utilisé pour:");
    println!("   • Extraction de features");
    println!("   • Style transfer (Gatys et al., 2015)");
    println!("   • Perceptual loss (Johnson et al., 2016)");

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Fin de l'exemple VGG");
    println!("═══════════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vgg16_cifar() {
        let config = VGGConfig::cifar10();
        let vgg = VGG16::with_config(config);
        let input = Tensor4D::random(TensorShape::new(1, 3, 32, 32));
        let output = vgg.forward(&input);
        assert!(output.shape().width > 0);
    }

    #[test]
    fn test_vgg19() {
        let config = VGGConfig::vgg19();
        let vgg = VGG19::with_config(config);
        assert!(vgg.num_parameters() > 0);
    }
}
