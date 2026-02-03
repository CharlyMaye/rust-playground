//! # Exemple AlexNet: Révolution Deep Learning (Krizhevsky et al., 2012)
//!
//! Cet exemple reproduit l'architecture AlexNet qui a déclenché la révolution
//! du Deep Learning en gagnant le challenge ImageNet LSVRC-2012.
//!
//! ## Paper Citation
//!
//! ```text
//! @inproceedings{krizhevsky2012imagenet,
//!   title={ImageNet Classification with Deep Convolutional Neural Networks},
//!   author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
//!   booktitle={Advances in Neural Information Processing Systems},
//!   volume={25},
//!   year={2012}
//! }
//! ```
//!
//! ## Impact Historique
//!
//! AlexNet a réduit le top-5 error de 26% à 15.3% sur ImageNet, une amélioration
//! sans précédent qui a convaincu la communauté de la puissance du deep learning.
//!
//! ## Architecture Originale
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ INPUT: 227×227×3 RGB image (redimensionnée depuis 256×256)              │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV1: 96 kernels 11×11, stride 4                                       │
//! │        Output: 55×55×96                                                 │
//! │        Paramètres: 11×11×3×96 + 96 = 34,944                             │
//! │        + ReLU (première utilisation massive!)                           │
//! │        + LRN (Local Response Normalization)                             │
//! │        + MaxPool 3×3, stride 2 → 27×27×96                               │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV2: 256 kernels 5×5, padding 2                                       │
//! │        Output: 27×27×256                                                │
//! │        Paramètres: 5×5×96×256 + 256 = 614,656                           │
//! │        Note: Dans le paper, split sur 2 GPUs (48 channels chacun)       │
//! │        + ReLU + LRN + MaxPool 3×3/2 → 13×13×256                         │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV3: 384 kernels 3×3, padding 1                                       │
//! │        Output: 13×13×384                                                │
//! │        Paramètres: 3×3×256×384 + 384 = 885,120                          │
//! │        + ReLU (pas de pooling après Conv3)                              │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV4: 384 kernels 3×3, padding 1                                       │
//! │        Output: 13×13×384                                                │
//! │        Paramètres: 3×3×384×384 + 384 = 1,327,488                        │
//! │        + ReLU                                                           │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV5: 256 kernels 3×3, padding 1                                       │
//! │        Output: 13×13×256                                                │
//! │        Paramètres: 3×3×384×256 + 256 = 884,992                          │
//! │        + ReLU + MaxPool 3×3/2 → 6×6×256                                 │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ FLATTEN: 6×6×256 = 9,216                                                │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ FC6: 9,216 → 4,096                                                      │
//! │      Paramètres: 9,216×4,096 + 4,096 = 37,752,832                       │
//! │      + ReLU + Dropout(0.5)                                              │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ FC7: 4,096 → 4,096                                                      │
//! │      Paramètres: 4,096×4,096 + 4,096 = 16,781,312                       │
//! │      + ReLU + Dropout(0.5)                                              │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ FC8: 4,096 → 1,000 (ImageNet classes)                                   │
//! │      Paramètres: 4,096×1,000 + 1,000 = 4,097,000                        │
//! │      + Softmax                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//!
//! TOTAL: ~62 millions de paramètres
//! - Conv layers: ~3.7M (6%)
//! - FC layers: ~58.6M (94%)
//! ```
//!
//! ## Innovations Clés (2012)
//!
//! 1. **ReLU**: Première utilisation massive, 6× plus rapide que tanh
//! 2. **Dropout**: Première régularisation efficace contre l'overfitting
//! 3. **GPU Training**: Parallélisation sur 2 NVIDIA GTX 580 (3GB chacun)
//! 4. **Data Augmentation**: Random crops, flips, color augmentation
//! 5. **Local Response Normalization**: Normalisation locale (remplacé par BatchNorm)

use cma_cnn::{Tensor4D, TensorShape};
use cma_models::alexnet::{AlexNet, AlexNetConfig};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  AlexNet: ImageNet Classification with Deep CNNs");
    println!("  Krizhevsky, Sutskever, Hinton (NIPS 2012)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Version CIFAR-10 (32×32) - Utilisable sur CPU
    // ═══════════════════════════════════════════════════════════════════════

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ AlexNet-Mini: Adaptation CIFAR-10 (32×32)                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let alexnet_cifar = AlexNet::with_config(AlexNetConfig::cifar10());

    println!("Architecture (adaptée pour 32×32):");
    alexnet_cifar.summary();

    // Forward pass
    let batch = Tensor4D::random(TensorShape::new(16, 3, 32, 32));
    let features = alexnet_cifar.forward(&batch);

    println!();
    println!("Forward pass:");
    println!("  Input:  [16, 3, 32, 32] (batch=16, RGB, 32×32)");
    println!(
        "  Output: {:?} (features pour classifieur FC)",
        features.shape()
    );
    println!("  Paramètres (conv): {}", alexnet_cifar.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // Version Medium (64×64)
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ AlexNet-Medium: Pour images 64×64                               │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let alexnet_medium = AlexNet::with_config(AlexNetConfig::small(100));

    let input_64 = Tensor4D::random(TensorShape::new(4, 3, 64, 64));
    let features_64 = alexnet_medium.forward(&input_64);

    println!("Forward pass:");
    println!("  Input:  [4, 3, 64, 64]");
    println!("  Output: {:?}", features_64.shape());
    println!("  Paramètres: {}", alexnet_medium.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // Innovations du Paper
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Innovations Clés du Paper (2012)                                │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("1. ReLU (Section 3.1 du paper):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   \"Deep convolutional neural networks with ReLUs train");
    println!("    several times faster than their equivalents with tanh units.\"");
    println!();
    println!("   f(x) = max(0, x)");
    println!();
    println!("   Avantages:");
    println!("   - Pas de saturation pour x > 0");
    println!("   - Gradient constant = 1 (pas de vanishing)");
    println!("   - Calcul très simple (pas d'exp)");
    println!();

    println!("2. Dropout (Section 4.2 du paper):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   \"Dropout consists of setting to zero the output of each");
    println!("    hidden neuron with probability 0.5.\"");
    println!();
    println!("   Proposé par Hinton, co-auteur du paper.");
    println!("   Appliqué dans FC6 et FC7 (pas dans les conv).");
    println!("   Réduit l'overfitting sans augmenter le dataset.");
    println!();

    println!("3. Training sur 2 GPUs (Section 3.2 du paper):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   \"Spreading the net across two GPUs...allows us to train");
    println!("    larger networks in the same time.\"");
    println!();
    println!("   - GTX 580: 3GB RAM, ~1.5 TFLOPS");
    println!("   - Split des feature maps entre les 2 GPUs");
    println!("   - Communication inter-GPU seulement à certaines couches");
    println!();

    println!("4. Local Response Normalization (Section 3.3):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   Remplacé par BatchNorm dans les architectures modernes.");
    println!("   Notre implémentation utilise BatchNorm (plus efficace).");
    println!();

    println!("5. Data Augmentation (Section 4.1):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   - Random crops 224×224 depuis 256×256");
    println!("   - Horizontal flips");
    println!("   - PCA color augmentation (\"Fancy PCA\")");

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
    println!(
        "// Pour AlexNet-Mini (CIFAR-10): {} features → 10 classes",
        features.shape().width
    );
    println!(
        "let classifier = NetworkBuilder::new({}, 10)",
        features.shape().width
    );
    println!("    .hidden_layer(512, Activation::ReLU)");
    println!("    .dropout(0.5)");
    println!("    .hidden_layer(512, Activation::ReLU)");
    println!("    .dropout(0.5)");
    println!("    .output_activation(Activation::Softmax)");
    println!("    .loss(LossFunction::CategoricalCrossEntropy)");
    println!("    .optimizer(OptimizerType::adam(0.001))");
    println!("    .build();");
    println!("```");

    // ═══════════════════════════════════════════════════════════════════════
    // Résultats du Paper
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Résultats du Paper (ILSVRC-2012)                                │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌────────────────────┬────────────┬────────────┬─────────────────┐");
    println!("│ Méthode            │ Top-1 Err  │ Top-5 Err  │ Notes           │");
    println!("├────────────────────┼────────────┼────────────┼─────────────────┤");
    println!("│ AlexNet (1 CNN)    │ 40.7%      │ 18.2%      │ Single model    │");
    println!("│ AlexNet (5 CNNs)   │ 38.1%      │ 16.4%      │ Ensemble        │");
    println!("│ AlexNet (7 CNNs)*  │ 36.7%      │ 15.3%      │ WINNER          │");
    println!("├────────────────────┼────────────┼────────────┼─────────────────┤");
    println!("│ 2ème place (2012)  │ -          │ 26.2%      │ Non-DL method   │");
    println!("└────────────────────┴────────────┴────────────┴─────────────────┘");
    println!();
    println!("* Avec multi-crop averaging et ensemble de modèles");
    println!();
    println!("Amélioration: 26.2% → 15.3% = réduction de 41% de l'erreur!");

    // ═══════════════════════════════════════════════════════════════════════
    // Héritage
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Héritage d'AlexNet                                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("AlexNet a influencé directement:");
    println!("- VGG (2014): Plus profond, kernels 3×3 uniquement");
    println!("- GoogLeNet (2014): Inception modules");
    println!("- ResNet (2015): Skip connections");
    println!("- Tous les CNN modernes");
    println!();
    println!("Le succès d'AlexNet a:");
    println!("- Relancé l'intérêt pour le deep learning");
    println!("- Démontré l'importance du GPU computing");
    println!("- Établi les CNN comme standard en vision");
    println!("- Conduit à l'ère actuelle de l'IA");

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Fin de l'exemple AlexNet");
    println!("═══════════════════════════════════════════════════════════════════");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alexnet_cifar() {
        let model = AlexNet::with_config(AlexNetConfig::cifar10());
        let input = Tensor4D::zeros(TensorShape::new(1, 3, 32, 32));
        let output = model.forward(&input);
        assert!(output.shape().width > 0);
    }
}
