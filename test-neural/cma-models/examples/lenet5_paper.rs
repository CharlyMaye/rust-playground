//! # Exemple LeNet-5: Implémentation Historique (LeCun et al., 1998)
//!
//! Cet exemple reproduit l'architecture LeNet-5 du paper original
//! "Gradient-Based Learning Applied to Document Recognition" de Yann LeCun.
//!
//! ## Paper Citation
//!
//! ```text
//! @article{lecun1998gradient,
//!   title={Gradient-based learning applied to document recognition},
//!   author={LeCun, Yann and Bottou, L{\'e}on and Bengio, Yoshua and Haffner, Patrick},
//!   journal={Proceedings of the IEEE},
//!   volume={86},
//!   number={11},
//!   pages={2278--2324},
//!   year={1998}
//! }
//! ```
//!
//! ## Architecture (Reproduction Fidèle)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ INPUT: 32×32 grayscale image (MNIST padding à 32×32, original du paper) │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ C1: Convolution 5×5, 6 feature maps                                     │
//! │     Output: 28×28×6                                                     │
//! │     Paramètres: (5×5×1 + 1) × 6 = 156                                   │
//! │     Activation: Tanh (original) ou ReLU (moderne)                       │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ S2: Subsampling (Average Pooling) 2×2                                   │
//! │     Output: 14×14×6                                                     │
//! │     Note: Le paper original utilisait une forme de pooling avec poids   │
//! │           appris. Nous utilisons AvgPool standard (équivalent moderne). │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ C3: Convolution 5×5, 16 feature maps                                    │
//! │     Output: 10×10×16                                                    │
//! │     Paramètres: (5×5×6 + 1) × 16 = 2,416                                │
//! │     Note: Le paper original utilisait une table de connexion partielle. │
//! │           Nous utilisons des connexions complètes (standard moderne).   │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ S4: Subsampling (Average Pooling) 2×2                                   │
//! │     Output: 5×5×16                                                      │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ C5: Convolution 5×5, 120 feature maps                                   │
//! │     Output: 1×1×120 (équivalent à fully-connected)                      │
//! │     Paramètres: (5×5×16 + 1) × 120 = 48,120                             │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ F6: Fully Connected, 84 unités                                          │
//! │     Paramètres: (120 + 1) × 84 = 10,164                                 │
//! │     Activation: Tanh                                                    │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ OUTPUT: 10 unités (Euclidean Radial Basis Function dans le paper)       │
//! │         Nous utilisons Softmax (standard moderne)                       │
//! │         Paramètres: (84 + 1) × 10 = 850                                 │
//! └─────────────────────────────────────────────────────────────────────────┘
//!
//! TOTAL PARAMÈTRES: ~60,000 (selon le paper)
//! Notre implémentation: ~62,000 (connexions complètes)
//! ```
//!
//! ## Contexte Historique
//!
//! LeNet-5 a été développé pour la reconnaissance de chèques bancaires.
//! Il traitait des millions de chèques par jour aux États-Unis dans les années 90.
//!
//! Innovations clés:
//! - Premier CNN entraîné avec backpropagation end-to-end
//! - Introduction du concept de "feature maps"
//! - Démonstration de l'efficacité du partage de poids
//! - Architecture qui a inspiré AlexNet (2012) et tous les CNN modernes

use cma_cnn::{Tensor4D, TensorShape};
use cma_models::lenet::{LeNet5, LeNet5Config};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  LeNet-5: Gradient-Based Learning Applied to Document Recognition");
    println!("  LeCun, Bottou, Bengio, Haffner (1998)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Configuration 1: Architecture Originale du Paper (32×32 input)
    // ═══════════════════════════════════════════════════════════════════════

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Configuration 1: Architecture Originale (32×32)                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let config_original = LeNet5Config::original();
    let lenet_original = LeNet5::with_config(config_original);

    println!();
    println!("Architecture (fidèle au paper):");
    lenet_original.summary();

    // Simulation d'une image 32×32 (comme dans le paper)
    let input_32x32 = Tensor4D::random(TensorShape::new(1, 1, 32, 32));
    let features = lenet_original.forward(&input_32x32);

    println!();
    println!("Forward pass:");
    println!("  Input:  [1, 1, 32, 32] (batch=1, grayscale, 32×32 pixels)");
    println!(
        "  Output: {:?} (120 features pour le classifieur FC)",
        features.shape()
    );
    println!("  Paramètres (conv): {}", lenet_original.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // Configuration 2: Adaptation MNIST (28×28 input)
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Configuration 2: Adaptation MNIST (28×28)                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let lenet_mnist = LeNet5::new(10);

    println!();
    println!("Architecture (adaptée pour MNIST 28×28):");
    lenet_mnist.summary();

    // Simulation d'un batch de 32 images MNIST
    let batch = Tensor4D::random(TensorShape::new(32, 1, 28, 28));
    let features = lenet_mnist.forward(&batch);

    println!();
    println!("Forward pass (batch de 32):");
    println!("  Input:  [32, 1, 28, 28]");
    println!("  Output: {:?}", features.shape());

    // ═══════════════════════════════════════════════════════════════════════
    // Configuration 3: Version Moderne avec BatchNorm et ReLU
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Configuration 3: Version Moderne (BatchNorm + ReLU)             │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let config_modern = LeNet5Config::modern();
    let lenet_modern = LeNet5::with_config(config_modern);

    println!();
    println!("Améliorations modernes:");
    println!("  - ReLU au lieu de Tanh (convergence plus rapide)");
    println!("  - BatchNorm après chaque conv (stabilité, LR plus élevé)");
    println!(
        "  - Paramètres: {} (+ BatchNorm γ/β)",
        lenet_modern.num_parameters()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // Comparaison avec le Classifieur FC
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Classifieur FC (cma-neural-network)                             │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!();
    println!("Pour compléter LeNet-5, ajoutez les couches FC de cma-neural-network:");
    println!();
    println!("```rust");
    println!(
        "use cma_neural_network::{{NetworkBuilder, Activation, LossFunction, OptimizerType}};"
    );
    println!();
    println!("// Les 120 features de LeNet-5 → classifieur FC");
    println!("let classifier = NetworkBuilder::new(120, 10)");
    println!("    .hidden_layer(84, Activation::Tanh)  // F6 du paper");
    println!("    .output_activation(Activation::Softmax)");
    println!("    .loss(LossFunction::CategoricalCrossEntropy)");
    println!("    .optimizer(OptimizerType::adam(0.001))");
    println!("    .build();");
    println!("```");

    // ═══════════════════════════════════════════════════════════════════════
    // Résultats Attendus
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Résultats Attendus (MNIST)                                      │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌────────────────┬──────────┬──────────────┬──────────────────────┐");
    println!("│ Configuration  │ Params   │ Test Error   │ Notes                │");
    println!("├────────────────┼──────────┼──────────────┼──────────────────────┤");
    println!("│ Original (1998)│ ~60k     │ 0.95%        │ Paper LeCun et al.   │");
    println!("│ Notre implem.  │ ~62k     │ ~0.8-1.0%    │ Connexions complètes │");
    println!("│ Avec BatchNorm │ ~63k     │ ~0.7%        │ Convergence rapide   │");
    println!("│ FC seul        │ ~110k    │ ~2-3%        │ Sans convolutions    │");
    println!("└────────────────┴──────────┴──────────────┴──────────────────────┘");
    println!();
    println!("Note: Le paper original rapporte 0.95% d'erreur avec des techniques");
    println!("      additionnelles (elastic distortions, voting). Notre baseline");
    println!("      propre devrait atteindre ~0.8-1.0% avec data augmentation.");

    // ═══════════════════════════════════════════════════════════════════════
    // Différences avec le Paper Original
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Différences avec le Paper Original                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("1. CONNEXIONS C3:");
    println!("   - Paper: Table de connexion partielle (réduire params)");
    println!("   - Notre: Connexions complètes (standard moderne)");
    println!();
    println!("2. SUBSAMPLING (S2, S4):");
    println!("   - Paper: Averaging + poids appris + bias + sigmoid");
    println!("   - Notre: Simple Average Pooling (équivalent fonctionnel)");
    println!();
    println!("3. SORTIE:");
    println!("   - Paper: Euclidean Radial Basis Function (distance au prototype)");
    println!("   - Notre: Softmax + Cross-Entropy (standard moderne)");
    println!();
    println!("4. ACTIVATION:");
    println!("   - Paper: Tanh escalé: A * tanh(S * x)");
    println!("   - Notre: Tanh standard ou ReLU");

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Fin de l'exemple LeNet-5");
    println!("═══════════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lenet5_original() {
        let model = LeNet5::with_config(LeNet5Config::original());
        let input = Tensor4D::zeros(TensorShape::new(1, 1, 32, 32));
        let output = model.forward(&input);
        assert_eq!(output.shape().width, 120);
    }

    #[test]
    fn test_lenet5_mnist() {
        let model = LeNet5::new(10);
        let input = Tensor4D::zeros(TensorShape::new(1, 1, 28, 28));
        let output = model.forward(&input);
        assert_eq!(output.shape().width, 120);
    }
}
