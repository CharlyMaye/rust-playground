//! # AlexNet Example: Deep Learning Revolution (Krizhevsky et al., 2012)
//!
//! This example reproduces the AlexNet architecture that triggered the revolution
//! of Deep Learning by winning the ImageNet LSVRC-2012 challenge.
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
//! ## Historical Impact
//!
//! AlexNet reduced the top-5 error from 26% to 15.3% on ImageNet, an unprecedented
//! improvement that convinced the community of the power of deep learning.
//!
//! ## Original Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ INPUT: 227×227×3 RGB image (resized from 256×256)                       │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV1: 96 kernels 11×11, stride 4                                       │
//! │        Output: 55×55×96                                                 │
//! │        Parameters: 11×11×3×96 + 96 = 34,944                            │
//! │        + ReLU (first massive use!)                                      │
//! │        + LRN (Local Response Normalization)                             │
//! │        + MaxPool 3×3, stride 2 → 27×27×96                               │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV2: 256 kernels 5×5, padding 2                                       │
//! │        Output: 27×27×256                                                │
//! │        Parameters: 5×5×96×256 + 256 = 614,656                          │
//! │        Note: In the paper, split across 2 GPUs (48 channels each)      │
//! │        + ReLU + LRN + MaxPool 3×3/2 → 13×13×256                         │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV3: 384 kernels 3×3, padding 1                                       │
//! │        Output: 13×13×384                                                │
//! │        Parameters: 3×3×256×384 + 384 = 885,120                         │
//! │        + ReLU (no pooling after Conv3)                                  │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV4: 384 kernels 3×3, padding 1                                       │
//! │        Output: 13×13×384                                                │
//! │        Parameters: 3×3×384×384 + 384 = 1,327,488                       │
//! │        + ReLU                                                           │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ CONV5: 256 kernels 3×3, padding 1                                       │
//! │        Output: 13×13×256                                                │
//! │        Parameters: 3×3×384×256 + 256 = 884,992                         │
//! │        + ReLU + MaxPool 3×3/2 → 6×6×256                                 │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ FLATTEN: 6×6×256 = 9,216                                                │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ FC6: 9,216 → 4,096                                                      │
//! │      Parameters: 9,216×4,096 + 4,096 = 37,752,832                      │
//! │      + ReLU + Dropout(0.5)                                              │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ FC7: 4,096 → 4,096                                                      │
//! │      Parameters: 4,096×4,096 + 4,096 = 16,781,312                      │
//! │      + ReLU + Dropout(0.5)                                              │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ FC8: 4,096 → 1,000 (ImageNet classes)                                   │
//! │      Parameters: 4,096×1,000 + 1,000 = 4,097,000                       │
//! │      + Softmax                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//!
//! TOTAL: ~62 million parameters
//! - Conv layers: ~3.7M (6%)
//! - FC layers: ~58.6M (94%)
//! ```
//!
//! ## Key Innovations (2012)
//!
//! 1. **ReLU**: First massive use, 6× faster than tanh
//! 2. **Dropout**: First effective regularization against overfitting
//! 3. **GPU Training**: Parallelization across 2 NVIDIA GTX 580 (3GB each)
//! 4. **Data Augmentation**: Random crops, flips, color augmentation
//! 5. **Local Response Normalization**: Local normalization (replaced by BatchNorm)

use cma_cnn::{Tensor4D, TensorShape};
use cma_models::alexnet::{AlexNet, AlexNetConfig};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  AlexNet: ImageNet Classification with Deep CNNs");
    println!("  Krizhevsky, Sutskever, Hinton (NIPS 2012)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // CIFAR-10 version (32×32) - Usable on CPU
    // ═══════════════════════════════════════════════════════════════════════

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ AlexNet-Mini: Adaptation CIFAR-10 (32×32)                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let alexnet_cifar = AlexNet::with_config(AlexNetConfig::cifar10());

    println!("Architecture (adapted for 32×32):");
    alexnet_cifar.summary();

    // Forward pass
    let batch = Tensor4D::random(TensorShape::new(16, 3, 32, 32));
    let features = alexnet_cifar.forward(&batch);

    println!();
    println!("Forward pass:");
    println!("  Input:  [16, 3, 32, 32] (batch=16, RGB, 32×32)");
    println!(
        "  Output: {:?} (features for FC classifier)",
        features.shape()
    );
    println!("  Parameters (conv): {}", alexnet_cifar.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // Medium version (64×64)
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ AlexNet-Medium: For 64×64 images                                │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let alexnet_medium = AlexNet::with_config(AlexNetConfig::small(100));

    let input_64 = Tensor4D::random(TensorShape::new(4, 3, 64, 64));
    let features_64 = alexnet_medium.forward(&input_64);

    println!("Forward pass:");
    println!("  Input:  [4, 3, 64, 64]");
    println!("  Output: {:?}", features_64.shape());
    println!("  Parameters: {}", alexnet_medium.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // Paper Innovations
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Key Innovations of the Paper (2012)                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    println!("1. ReLU (Section 3.1 of the paper):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   \"Deep convolutional neural networks with ReLUs train");
    println!("    several times faster than their equivalents with tanh units.\"");
    println!();
    println!("   f(x) = max(0, x)");
    println!();
    println!("   Advantages:");
    println!("   - No saturation for x > 0");
    println!("   - Constant gradient = 1 (no vanishing)");
    println!("   - Very simple computation (no exp)");
    println!();

    println!("2. Dropout (Section 4.2 of the paper):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   \"Dropout consists of setting to zero the output of each");
    println!("    hidden neuron with probability 0.5.\"");
    println!();
    println!("   Proposed by Hinton, co-author of the paper.");
    println!("   Applied in FC6 and FC7 (not in conv layers).");
    println!("   Reduces overfitting without augmenting the dataset.");
    println!();

    println!("3. Training on 2 GPUs (Section 3.2 of the paper):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   \"Spreading the net across two GPUs...allows us to train");
    println!("    larger networks in the same time.\"");
    println!();
    println!("   - GTX 580: 3GB RAM, ~1.5 TFLOPS");
    println!("   - Feature maps split between the 2 GPUs");
    println!("   - Inter-GPU communication only at certain layers");
    println!();

    println!("4. Local Response Normalization (Section 3.3):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   Replaced by BatchNorm in modern architectures.");
    println!("   Our implementation uses BatchNorm (more efficient).");
    println!();

    println!("5. Data Augmentation (Section 4.1):");
    println!("   ─────────────────────────────────────────────────────");
    println!("   - Random crops 224×224 from 256×256");
    println!("   - Horizontal flips");
    println!("   - PCA color augmentation (\"Fancy PCA\")");

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
    println!(
        "// For AlexNet-Mini (CIFAR-10): {} features → 10 classes",
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
    // Paper Results
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Paper Results (ILSVRC-2012)                                     │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌────────────────────┬────────────┬────────────┬─────────────────┐");
    println!("│ Method             │ Top-1 Err  │ Top-5 Err  │ Notes           │");
    println!("├────────────────────┼────────────┼────────────┼─────────────────┤");
    println!("│ AlexNet (1 CNN)    │ 40.7%      │ 18.2%      │ Single model    │");
    println!("│ AlexNet (5 CNNs)   │ 38.1%      │ 16.4%      │ Ensemble        │");
    println!("│ AlexNet (7 CNNs)*  │ 36.7%      │ 15.3%      │ WINNER          │");
    println!("├────────────────────┼────────────┼────────────┼─────────────────┤");
    println!("│ 2nd place (2012)   │ -          │ 26.2%      │ Non-DL method   │");
    println!("└────────────────────┴────────────┴────────────┴─────────────────┘");
    println!();
    println!("* With multi-crop averaging and model ensemble");
    println!();
    println!("Improvement: 26.2% → 15.3% = 41% error reduction!");

    // ═══════════════════════════════════════════════════════════════════════
    // Legacy
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ AlexNet's Legacy                                                │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("AlexNet directly influenced:");
    println!("- VGG (2014): Deeper, 3×3 kernels only");
    println!("- GoogLeNet (2014): Inception modules");
    println!("- ResNet (2015): Skip connections");
    println!("- All modern CNNs");
    println!();
    println!("AlexNet's success has:");
    println!("- Renewed interest in deep learning");
    println!("- Demonstrated the importance of GPU computing");
    println!("- Established CNNs as the standard in computer vision");
    println!("- Led to the current era of AI");

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  End of AlexNet example");
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
