//! CNN inference with cma-cnn
//!
//! Demonstrates:
//! - Building a Sequential CNN pipeline with the fluent builder API
//! - Tensor4D creation and NCHW layout
//! - summary() and num_parameters()
//! - forward() with a single image and a batch
//! - forward_with_intermediates() for debugging feature maps
//! - Train/eval mode switching
//!
//! Run: cargo run --example cnn_inference

use cma_cnn::tensor::{Tensor4D, TensorShape};
use cma_cnn::{
    ActivationLayer, BatchNorm2D, Conv2D, MaxPool2D, Sequential,
};

fn main() {
    println!("=== CNN Inference Demo (cma-cnn) ===\n");

    // ─── 1. Shape arithmetic ──────────────────────────────────────────────
    println!("1. Shape arithmetic\n");

    let input_shape = TensorShape::new(1, 1, 28, 28);
    println!("   Input:         {}", input_shape);

    let after_conv1 = input_shape.after_conv(32, 3, 1, 1); // same-padding
    println!("   After conv1:   {}", after_conv1);

    let after_pool1 = after_conv1.after_pool(2, 2);
    println!("   After pool1:   {}", after_pool1);

    let after_conv2 = after_pool1.after_conv(64, 3, 1, 1);
    println!("   After conv2:   {}", after_conv2);

    let after_pool2 = after_conv2.after_pool(2, 2);
    println!("   After pool2:   {}", after_pool2);

    let after_gap = after_pool2.after_global_pool();
    println!("   After GlobalAvgPool: {}\n", after_gap);

    // ─── 2. Build a small CNN ─────────────────────────────────────────────
    println!("2. Building a LeNet-like CNN (manual)\n");

    let mut model = Sequential::named("SmallCNN")
        // Block 1: 1→32, 3×3, same padding, stride 1
        .add_conv2d(Conv2D::new(1, 32, 3, 1, 1))
        .add_batchnorm(BatchNorm2D::new(32))
        .add_activation(ActivationLayer::relu())
        .add_maxpool(MaxPool2D::new(2, 2))
        // Block 2: 32→64
        .add_conv2d(Conv2D::new(32, 64, 3, 1, 1))
        .add_batchnorm(BatchNorm2D::new(64))
        .add_activation(ActivationLayer::relu())
        .add_maxpool(MaxPool2D::new(2, 2))
        // 7×7×64 → GAP → 64-d vector
        .add_global_avgpool()
        .add_flatten();

    model.summary(TensorShape::new(1, 1, 28, 28));
    println!("   Total parameters: {}\n", model.num_parameters());

    // ─── 3. Shortcut builder API ──────────────────────────────────────────
    println!("3. Same network via shortcut methods\n");

    let model2 = Sequential::named("SmallCNN-v2")
        .add_conv_bn_relu(1, 32, 3, 1, 1)
        .add_maxpool(MaxPool2D::new(2, 2))
        .add_conv_bn_relu(32, 64, 3, 1, 1)
        .add_maxpool(MaxPool2D::new(2, 2))
        .add_global_avgpool()
        .add_flatten();

    println!("   Parameters: {} (should match above)\n", model2.num_parameters());

    // ─── 4. Single forward pass ───────────────────────────────────────────
    println!("4. Single-image forward pass\n");

    model.eval_mode(); // freeze BatchNorm running stats

    let img = Tensor4D::random(TensorShape::new(1, 1, 28, 28));
    println!("   Input shape:  {}", img.shape());

    let features = model.forward(&img);
    println!("   Output shape: {}  (64-d feature vector)\n", features.shape());

    // ─── 5. Batch forward pass ────────────────────────────────────────────
    println!("5. Batch forward pass (8 images)\n");

    let batch = Tensor4D::random(TensorShape::new(8, 1, 28, 28));
    let batch_out = model.forward(&batch);
    println!("   Batch input:  {}", batch.shape());
    println!("   Batch output: {}\n", batch_out.shape());

    // ─── 6. Intermediate feature maps ────────────────────────────────────
    println!("6. Intermediate feature maps (forward_with_intermediates)\n");

    let steps = model.forward_with_intermediates(&img);
    for (layer_name, config_str, feat_map) in &steps {
        println!("   {:>20}  ({})  ->  {}", layer_name, config_str, feat_map.shape());
    }
    println!();

    // ─── 7. output_shape() without running a forward pass ─────────────────
    println!("7. Static output shape query\n");
    let static_out = model.output_shape(TensorShape::new(1, 1, 28, 28));
    println!("   Predicted output shape: {}", static_out);
    println!("   Actual output shape:    {}  (should match)\n", features.shape());

    // ─── 8. Custom Conv2D options ─────────────────────────────────────────
    println!("8. Conv2D variants\n");

    let _conv_standard = Conv2D::new(3, 64, 3, 1, 1);
    let conv_same     = Conv2D::same_padding(3, 64, 3);   // padding = kernel/2
    let conv_no_bias  = Conv2D::new(3, 64, 3, 1, 0).without_bias(); // common before BN

    println!("   conv_standard:   in={}, out={}, k={}", 3, 64, 3);
    println!("   conv_same:       weights shape = {:?}", conv_same.weights.shape());
    println!("   conv_no_bias:    use_bias = {}\n", conv_no_bias.use_bias);

    // ─── 9. Using GlobalAvgPool instead of Flatten+FC ────────────────────
    println!("9. GlobalAvgPool2D directly\n");

    let mut gap_model = Sequential::named("GAPNet")
        .add_conv_bn_relu(1, 10, 3, 1, 1) // 10 classes → 10 output channels
        .add_global_avgpool()               // [N, 10, H, W] → [N, 10, 1, 1]
        .add_flatten();                     // [N, 10, 1, 1] → [N, 10, 1, 1]

    gap_model.eval_mode();
    let gap_out = gap_model.forward(&Tensor4D::random(TensorShape::new(1, 1, 28, 28)));
    println!("   GAP output shape: {}  (direct channel scores, no FC needed)\n", gap_out.shape());

    println!("=== Demo Complete ===");
    println!("See docs/guides/03-cnn-inference.md for full API reference.");
}
