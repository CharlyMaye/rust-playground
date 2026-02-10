//! # VGG Example: Very Deep Convolutional Networks (Simonyan & Zisserman, 2014)
//!
//! This example reproduces the VGG architecture that demonstrated the importance
//! of depth and the power of small 3×3 filters.
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
//! ## VGG Philosophy
//!
//! ```text
//! "We address an important aspect of ConvNet architecture design –
//!  its depth. [...] we push the depth to 16-19 weight layers,
//!  which is substantially deeper than what has been used before"
//!                                    - Simonyan & Zisserman
//!
//! The key: use ONLY 3×3 filters
//!
//! Why 3×3?
//! ━━━━━━━━━━━━━━
//! 2 convolutions 3×3 = receptive field 5×5
//! 3 convolutions 3×3 = receptive field 7×7
//!
//! But with FEWER parameters:
//! • 3 × (3×3×C×C) = 27C²
//! • 1 × (7×7×C×C) = 49C²
//!
//! And MORE non-linearities (ReLU between each conv)
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
//!     Total: ~138M parameters (of which 123M in the FC layers!)
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
    // VGG Configurations (Table 1 from the paper)
    // ═══════════════════════════════════════════════════════════════════════

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ VGG Configurations (Table 1 from the paper)                     │");
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
    // VGG-16 for ImageNet
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ VGG-16 Configuration D (Classic configuration)                  │");
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
    println!("  Output: {:?} (4096 features after FC)", features.shape());
    println!("  Params: {} (~138M)", vgg16.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // VGG for CIFAR-10
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ VGG-16 adapted for CIFAR-10 (32×32)                              │");
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
        "  Params: {} (much fewer since FC is smaller)",
        vgg16_cifar.num_parameters()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // VGG-19
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ VGG-19 Configuration E (the deepest)                            │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let vgg19_config = VGGConfig::vgg19();
    let vgg19 = VGG19::with_config(vgg19_config);

    vgg19.features.summary(TensorShape::new(1, 3, 224, 224));

    // ═══════════════════════════════════════════════════════════════════════
    // Parameter Analysis
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ VGG-16 Parameter Analysis                                       │");
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
    println!("Total FC:        123.6M (89.4%) ← Problem!");
    println!("```");
    println!();
    println!("The FC layers contain 89% of the parameters!");
    println!("This is why ResNet uses Global Average Pooling.");

    // ═══════════════════════════════════════════════════════════════════════
    // Receptive Field
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Receptive Field: Why 3×3 × N > 7×7 × 1                            │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("```");
    println!("2 layers 3×3:                 1 layer 5×5:");
    println!("                               ");
    println!("Layer 2:     [3×3]             [5×5]");
    println!("               ↑                 ↑");
    println!("Layer 1:   [1×1×3×3]          [1×1×5×5]");
    println!("               ↑                 ↑");
    println!("Input:   [1×1×5×5]            [1×1×5×5]");
    println!("         = 5×5 RF             = 5×5 RF");
    println!();
    println!("Params: 2 × (3×3×C×C) = 18C²  25C²");
    println!("ReLUs:  2                      1");
    println!("```");
    println!();
    println!("Conclusion: 2 layers of 3×3 have:");
    println!("• Same receptive field as 1 layer of 5×5");
    println!("• 28% fewer parameters");
    println!("• 2× more non-linearities");

    // ═══════════════════════════════════════════════════════════════════════
    // ILSVRC Results
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ ILSVRC-2014 Results                                             │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌─────────────────────┬──────────┬──────────┬─────────────────────┐");
    println!("│ Model               │ Top-1 Err│ Top-5 Err│ Notes               │");
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
    println!("VGG finished 2nd but became more popular than GoogLeNet");
    println!("thanks to its architectural simplicity.");

    // ═══════════════════════════════════════════════════════════════════════
    // Usage with cma-neural-network
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
    println!("// VGG produces 4096 features (after the internal FCs)");
    println!("// For a classifier, just the last layer is needed:");
    println!("let classifier = NetworkBuilder::new(4096, 1000)");
    println!("    .output_activation(Activation::Softmax)");
    println!("    .loss(LossFunction::CategoricalCrossEntropy)");
    println!("    .optimizer(OptimizerType::sgd_with_momentum(0.01, 0.9))");
    println!("    .build();");
    println!();
    println!("// Or version without VGG's internal FC layers:");
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
    // VGG's Legacy
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ VGG's Legacy                                                    │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("VGG established fundamental principles:");
    println!();
    println!("1. DEPTH MATTERS");
    println!("   Deeper = better (up to 19 layers)");
    println!("   Limit: degradation problem → solved by ResNet");
    println!();
    println!("2. SMALL FILTERS");
    println!("   3×3 became the de facto standard");
    println!("   Used in ResNet, DenseNet, etc.");
    println!();
    println!("3. BLOCK STRUCTURE");
    println!("   [Conv-ReLU] × N → MaxPool");
    println!("   Double the channels at each stage");
    println!();
    println!("4. TRANSFER LEARNING");
    println!("   VGG pre-trained is still used for:");
    println!("   • Feature extraction");
    println!("   • Style transfer (Gatys et al., 2015)");
    println!("   • Perceptual loss (Johnson et al., 2016)");

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  End of VGG example");
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
