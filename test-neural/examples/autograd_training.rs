//! CNN training with cma-autograd
//!
//! Demonstrates end-to-end CNN training via automatic differentiation:
//! - Tensor creation with requires_grad
//! - CnnBuilder presets (lenet5, alexnet_mnist)
//! - CnnTrainer fluent builder (.train_data, .epochs, .fit)
//! - Adam optimizer
//! - Manual training loop variant
//! - no_grad() for inference
//!
//! Run: cargo run --example autograd_training

use cma_autograd::prelude::*;
use ndarray::{ArrayD, IxDyn};

fn main() {
    println!("=== Autograd Training Demo (cma-autograd) ===\n");

    // ─── 1. Tensor basics ─────────────────────────────────────────────────
    println!("1. Tensor basics\n");

    let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2], false);
    let w = Tensor::randn(&[2, 2], true); // requires_grad = true

    let y = (&x * &w).sum(); // scalar
    y.backward();

    println!("   x shape:     {:?}", x.shape());
    println!("   w shape:     {:?}", w.shape());
    println!("   y (scalar):  {:.4}", y.item());
    println!("   w.grad():    {:?}", w.grad().map(|g| g.shape().to_vec()));
    println!();

    // ─── 2. Simple MLP with autograd (XOR) ───────────────────────────────
    println!("2. MLP for XOR (Sequential + Adam)\n");

    let model_mlp = Sequential::new()
        .add(Linear::new(2, 8))
        .add(ReLULayer::new())
        .add(Linear::new(8, 1));

    model_mlp.summary();
    println!();

    // XOR dataset as ArrayD (shape [N, features])
    let xor_inputs: Vec<ArrayD<Float>> = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0_f32, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0_f32, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0_f32, 1.0]).unwrap(),
    ];
    let xor_targets: Vec<ArrayD<Float>> = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0_f32]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0_f32]).unwrap(),
    ];

    let mlp_params: Vec<Parameter> = model_mlp.parameters().into_iter().cloned().collect();
    let mut mlp_opt = Adam::new(mlp_params, 0.05_f32);

    let history = model_mlp
        .trainer(&mut mlp_opt)
        .train_data(&xor_inputs, &xor_targets)
        .loss_fn(mse_loss)
        .epochs(200)
        .batch_size(4)
        .verbose(false)
        .fit();

    println!("   Epochs run:       {}", history.len());
    println!("   Final train loss: {:.4}", history.last().unwrap().train_loss);
    println!();

    // ─── 3. LeNet-5 forward pass and architecture ────────────────────────
    println!("3. LeNet-5 (CnnBuilder preset) for 28×28 grayscale\n");

    let lenet = CnnBuilder::lenet5(10);   // 10 output classes
    lenet.summary();
    println!("   Parameters: {}\n", lenet.num_parameters());

    // ─── 4. CnnTrainer with synthetic image dataset ───────────────────────
    println!("4. CnnTrainer: training LeNet-5 on synthetic 28×28 images\n");

    let n_train = 64;
    let n_val   = 16;
    let n_total = n_train + n_val;

    // Generate random images: each sample is [1, 28, 28] (C×H×W)
    let all_images: Vec<ArrayD<Float>> = (0..n_total)
        .map(|_| {
            let data: Vec<Float> = (0..784).map(|_| rand::random::<Float>()).collect();
            ArrayD::from_shape_vec(IxDyn(&[1, 28, 28]), data).unwrap()
        })
        .collect();

    // Random one-hot labels (10 classes)
    let all_labels: Vec<ArrayD<Float>> = (0..n_total)
        .map(|i| {
            let class = i % 10;
            let mut label = vec![0.0_f32; 10];
            label[class] = 1.0;
            ArrayD::from_shape_vec(IxDyn(&[10]), label).unwrap()
        })
        .collect();

    let (train_images, val_images)   = all_images.split_at(n_train);
    let (train_labels, val_labels)   = all_labels.split_at(n_train);

    let lenet_train = CnnBuilder::lenet5(10);
    let lenet_params: Vec<Parameter> = lenet_train.parameters().into_iter().cloned().collect();
    let mut lenet_opt = Adam::new(lenet_params, 0.001_f32);

    println!("   Training {} images, validating {}", n_train, n_val);
    println!("   (Random data — loss should oscillate, not converge)\n");

    let history = lenet_train
        .trainer(&mut lenet_opt)
        .train_data(train_images, train_labels)
        .validation_data(val_images, val_labels)
        .loss_fn(cross_entropy_loss)
        .epochs(5)
        .batch_size(16)
        .verbose(true)
        .early_stopping(0) // disabled
        .fit();

    println!();
    println!("   Final train loss:  {:.4}", history.last().unwrap().train_loss);
    if let Some(vl) = history.last().unwrap().val_loss {
        println!("   Final val loss:    {:.4}", vl);
    }
    println!();

    // ─── 5. no_grad() for inference ───────────────────────────────────────
    println!("5. Inference with no_grad()\n");

    let mut lenet_infer = CnnBuilder::lenet5(10);
    lenet_infer.eval();

    let img = ArrayD::from_shape_vec(
        IxDyn(&[1, 28, 28]),
        (0..784).map(|_| rand::random::<Float>()).collect::<Vec<_>>(),
    ).unwrap();

    let logits = no_grad(|| {
        let t = Tensor::from_vec(img.iter().cloned().collect(), &[1, 1, 28, 28], false);
        lenet_infer.forward(&t)
    });
    println!("   Logits shape: {:?}", logits.shape());
    println!("   Grad tracking disabled inside no_grad block.\n");

    // ─── 6. Manual training loop ──────────────────────────────────────────
    println!("6. Manual training loop (2 steps)\n");

    let manual_model = Sequential::new()
        .add(Linear::new(4, 8))
        .add(ReLULayer::new())
        .add(Linear::new(8, 2));

    let manual_params: Vec<Parameter> = manual_model.parameters().into_iter().cloned().collect();
    let mut manual_opt = Adam::new(manual_params.clone(), 0.01_f32);

    for step in 0..2 {
        let input  = Tensor::from_vec(vec![0.1_f32, 0.5, -0.3, 0.8], &[1, 4], false);
        let target = Tensor::from_vec(vec![1.0_f32, 0.0], &[1, 2], false);

        // Forward
        let output = manual_model.forward(&input);
        let loss   = mse_loss(&output, &target);

        // Backward
        manual_opt.zero_grad();
        loss.backward();
        manual_opt.step();

        println!("   Step {}: loss = {:.4}", step + 1, loss.item());
    }

    println!();
    println!("=== Demo Complete ===");
    println!("See docs/guides/04-autograd-training.md for full API reference.");
}
