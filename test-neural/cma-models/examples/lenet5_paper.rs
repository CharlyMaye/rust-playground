//! # LeNet-5 Example: Historical Implementation (LeCun et al., 1998)
//!
//! This example reproduces the LeNet-5 architecture from the original paper
//! "Gradient-Based Learning Applied to Document Recognition" by Yann LeCun.
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
//! ## Architecture (Faithful Reproduction)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ INPUT: 32×32 grayscale image (MNIST padded to 32×32, as in the paper)  │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ C1: Convolution 5×5, 6 feature maps                                     │
//! │     Output: 28×28×6                                                     │
//! │     Parameters: (5×5×1 + 1) × 6 = 156                                   │
//! │     Activation: Tanh (original) or ReLU (modern)                       │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ S2: Subsampling (Average Pooling) 2×2                                   │
//! │     Output: 14×14×6                                                     │
//! │     Note: The original paper used a form of pooling with learned       │
//! │           weights. We use standard AvgPool (modern equivalent).        │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ C3: Convolution 5×5, 16 feature maps                                    │
//! │     Output: 10×10×16                                                    │
//! │     Parameters: (5×5×6 + 1) × 16 = 2,416                                │
//! │     Note: The original paper used a partial connection table.           │
//! │           We use full connections (modern standard).                   │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ S4: Subsampling (Average Pooling) 2×2                                   │
//! │     Output: 5×5×16                                                      │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ C5: Convolution 5×5, 120 feature maps                                   │
//! │     Output: 1×1×120 (equivalent to fully-connected)                      │
//! │     Parameters: (5×5×16 + 1) × 120 = 48,120                             │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ F6: Fully Connected, 84 units                                          │
//! │     Parameters: (120 + 1) × 84 = 10,164                                 │
//! │     Activation: Tanh                                                    │
//! └───────────────────────────────┬─────────────────────────────────────────┘
//!                                 ▼
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │ OUTPUT: 10 units (Euclidean Radial Basis Function in the paper)        │
//! │         We use Softmax (modern standard)                               │
//! │         Parameters: (84 + 1) × 10 = 850                                 │
//! └─────────────────────────────────────────────────────────────────────────┘
//!
//! TOTAL PARAMETERS: ~60,000 (according to the paper)
//! Our implementation: ~62,000 (full connections)
//! ```
//!
//! ## Historical Context
//!
//! LeNet-5 was developed for bank check recognition.
//! It processed millions of checks per day in the United States during the 1990s.
//!
//! Key innovations:
//! - First CNN trained with end-to-end backpropagation
//! - Introduction of the "feature maps" concept
//! - Demonstration of weight sharing effectiveness
//! - Architecture that inspired AlexNet (2012) and all modern CNNs

use cma_cnn::{Tensor4D, TensorShape};
use cma_models::lenet::{LeNet5, LeNet5Config};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  LeNet-5: Gradient-Based Learning Applied to Document Recognition");
    println!("  LeCun, Bottou, Bengio, Haffner (1998)");
    println!("═══════════════════════════════════════════════════════════════════");
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // Configuration 1: Original Paper Architecture (32×32 input)
    // ═══════════════════════════════════════════════════════════════════════

    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Configuration 1: Original Architecture (32×32)                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let config_original = LeNet5Config::original();
    let lenet_original = LeNet5::with_config(config_original);

    println!();
    println!("Architecture (faithful to the paper):");
    lenet_original.summary();

    // Simulation of a 32×32 image (as in the paper)
    let input_32x32 = Tensor4D::random(TensorShape::new(1, 1, 32, 32));
    let features = lenet_original.forward(&input_32x32);

    println!();
    println!("Forward pass:");
    println!("  Input:  [1, 1, 32, 32] (batch=1, grayscale, 32×32 pixels)");
    println!(
        "  Output: {:?} (120 features for the FC classifier)",
        features.shape()
    );
    println!("  Parameters (conv): {}", lenet_original.num_parameters());

    // ═══════════════════════════════════════════════════════════════════════
    // Configuration 2: MNIST Adaptation (28×28 input)
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Configuration 2: MNIST Adaptation (28×28)                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let lenet_mnist = LeNet5::new(10);

    println!();
    println!("Architecture (adapted for MNIST 28×28):");
    lenet_mnist.summary();

    // Simulation of a batch of 32 MNIST images
    let batch = Tensor4D::random(TensorShape::new(32, 1, 28, 28));
    let features = lenet_mnist.forward(&batch);

    println!();
    println!("Forward pass (batch of 32):");
    println!("  Input:  [32, 1, 28, 28]");
    println!("  Output: {:?}", features.shape());

    // ═══════════════════════════════════════════════════════════════════════
    // Configuration 3: Modern Version with BatchNorm and ReLU
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Configuration 3: Modern Version (BatchNorm + ReLU)             │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    let config_modern = LeNet5Config::modern();
    let lenet_modern = LeNet5::with_config(config_modern);

    println!();
    println!("Modern improvements:");
    println!("  - ReLU instead of Tanh (faster convergence)");
    println!("  - BatchNorm after each conv (stability, higher LR)");
    println!(
        "  - Parameters: {} (+ BatchNorm γ/β)",
        lenet_modern.num_parameters()
    );

    // ═══════════════════════════════════════════════════════════════════════
    // Comparison with FC Classifier
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ FC Classifier (cma-neural-network)                              │");
    println!("└─────────────────────────────────────────────────────────────────┘");

    println!();
    println!("To complete LeNet-5, add the FC layers from cma-neural-network:");
    println!();
    println!("```rust");
    println!(
        "use cma_neural_network::{{NetworkBuilder, Activation, LossFunction, OptimizerType}};"
    );
    println!();
    println!("// The 120 features from LeNet-5 → FC classifier");
    println!("let classifier = NetworkBuilder::new(120, 10)");
    println!("    .hidden_layer(84, Activation::Tanh)  // F6 from the paper");
    println!("    .output_activation(Activation::Softmax)");
    println!("    .loss(LossFunction::CategoricalCrossEntropy)");
    println!("    .optimizer(OptimizerType::adam(0.001))");
    println!("    .build();");
    println!("```");

    // ═══════════════════════════════════════════════════════════════════════
    // Expected Results
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Expected Results (MNIST)                                        │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("┌────────────────┬──────────┬──────────────┬──────────────────────┐");
    println!("│ Configuration  │ Params   │ Test Error   │ Notes                │");
    println!("├────────────────┼──────────┼──────────────┼──────────────────────┤");
    println!("│ Original (1998)│ ~60k     │ 0.95%        │ Paper LeCun et al.   │");
    println!("│ Our impl.       │ ~62k     │ ~0.8-1.0%    │ Full connections    │");
    println!("│ With BatchNorm  │ ~63k     │ ~0.7%        │ Fast convergence    │");
    println!("│ FC only         │ ~110k    │ ~2-3%        │ Without convolutions│");
    println!("└────────────────┴──────────┴──────────────┴──────────────────────┘");
    println!();
    println!("Note: The original paper reports 0.95% error with additional techniques");
    println!("      (elastic distortions, voting). Our clean baseline should reach");
    println!("      ~0.8-1.0% with data augmentation.");

    // ═══════════════════════════════════════════════════════════════════════
    // Differences from the Original Paper
    // ═══════════════════════════════════════════════════════════════════════

    println!();
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ Differences from the Original Paper                             │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();
    println!("1. C3 CONNECTIONS:");
    println!("   - Paper: Partial connection table (to reduce params)");
    println!("   - Ours: Full connections (modern standard)");
    println!();
    println!("2. SUBSAMPLING (S2, S4):");
    println!("   - Paper: Averaging + learned weights + bias + sigmoid");
    println!("   - Ours: Simple Average Pooling (functional equivalent)");
    println!();
    println!("3. OUTPUT:");
    println!("   - Paper: Euclidean Radial Basis Function (distance to prototype)");
    println!("   - Ours: Softmax + Cross-Entropy (modern standard)");
    println!();
    println!("4. ACTIVATION:");
    println!("   - Paper: Scaled tanh: A * tanh(S * x)");
    println!("   - Ours: Standard Tanh or ReLU");

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  End of LeNet-5 example");
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
