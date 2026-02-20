//! Integration tests for Sequential container and training helper.
//!
//! Tests cover:
//! - Sequential forward pass (MLP, CNN)
//! - Parameter collection across layers
//! - Train/eval mode switching
//! - Training loop convergence (XOR, spiral, mini-CNN)
//! - LeNet-5 architecture construction and forward

use cma_autograd::prelude::*;
use cma_autograd::Float;
use ndarray::{ArrayD, IxDyn};

// ═══════════════════════════════════════════════════════════════════════════
// Sequential basics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sequential_empty() {
    let model = Sequential::new();
    assert!(model.is_empty());
    assert_eq!(model.len(), 0);
    assert_eq!(model.num_parameters(), 0);
}

#[test]
fn test_sequential_mlp_forward() {
    let model = Sequential::new()
        .add(Linear::new(4, 8))
        .add(ReLULayer::new())
        .add(Linear::new(8, 3));

    assert_eq!(model.len(), 3);

    let input = Tensor::from_vec((0..8).map(|i| i as Float * 0.1).collect(), &[2, 4], true);
    let output = model.forward(&input);
    assert_eq!(output.data().shape(), &[2, 3]);
}

#[test]
fn test_sequential_parameters_collected() {
    let model = Sequential::new()
        .add(Linear::new(4, 8))   // weight [8,4] + bias [8] = 40
        .add(ReLULayer::new())    // 0
        .add(Linear::new(8, 3));  // weight [3,8] + bias [3] = 27

    let params = model.parameters();
    // Linear(4→8): 32 + 8 = 40, Linear(8→3): 24 + 3 = 27
    assert_eq!(model.num_parameters(), 40 + 27);
    assert_eq!(params.len(), 4); // 2 weights + 2 biases
}

#[test]
fn test_sequential_backward() {
    let model = Sequential::new()
        .add(Linear::new(4, 8))
        .add(ReLULayer::new())
        .add(Linear::new(8, 2));

    let input = Tensor::from_vec((0..8).map(|i| i as Float * 0.1).collect(), &[2, 4], true);
    let output = model.forward(&input);
    let target = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2], false);
    let loss = mse_loss(&output, &target);
    loss.backward();

    // All parameters should have gradients
    for param in model.parameters() {
        assert!(
            param.tensor().grad().is_some(),
            "All params should have gradients after backward"
        );
    }
}

#[test]
fn test_sequential_zero_grad() {
    let model = Sequential::new()
        .add(Linear::new(4, 2));

    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], true);
    let output = model.forward(&input);
    output.sum().backward();

    // Gradients should exist
    for param in model.parameters() {
        assert!(param.tensor().grad().is_some());
    }

    // Zero them
    model.zero_grad();
    for param in model.parameters() {
        if let Some(g) = param.tensor().grad() {
            assert!(g.iter().all(|v| *v == 0.0), "Gradients should be zeroed");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sequential CNN
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sequential_cnn_forward() {
    // Simple CNN: Conv → ReLU → Pool → Flatten → FC
    let model = Sequential::new()
        .add(Conv2DModule::new(1, 4, 3, 1, 0)) // [1,1,8,8] → [1,4,6,6]
        .add(ReLULayer::new())
        .add(MaxPool2DLayer::new(2, 2))         // → [1,4,3,3]
        .add(Flatten::new())                    // → [1,36]
        .add(Linear::new(36, 10));              // → [1,10]

    let input = Tensor::from_vec(
        (0..64).map(|i| i as Float * 0.01).collect(),
        &[1, 1, 8, 8],
        true,
    );
    let output = model.forward(&input);
    assert_eq!(output.data().shape(), &[1, 10]);
}

#[test]
fn test_sequential_cnn_backward() {
    let model = Sequential::new()
        .add(Conv2DModule::new(1, 2, 3, 1, 0))  // [1,1,6,6] → [1,2,4,4]
        .add(ReLULayer::new())
        .add(AvgPool2D::new(2, 2))              // → [1,2,2,2]
        .add(Flatten::new())                    // → [1,8]
        .add(Linear::new(8, 3));                // → [1,3]

    let input = Tensor::from_vec(
        (0..36).map(|i| i as Float * 0.05 - 0.5).collect(),
        &[1, 1, 6, 6],
        true,
    );
    let output = model.forward(&input);
    let target = Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3], false);
    let loss = mse_loss(&output, &target);
    loss.backward();

    assert!(input.grad().is_some());
    // Conv + FC params should all have gradients
    let params = model.parameters();
    assert!(params.len() >= 4); // conv_w, conv_b, fc_w, fc_b
    for p in &params {
        assert!(p.tensor().grad().is_some());
    }
}

#[test]
fn test_sequential_cnn_with_batchnorm() {
    // CNN with BatchNorm: Conv → BN → ReLU → Pool → Flatten → FC
    let model = Sequential::new()
        .add(Conv2DModule::new(1, 4, 3, 1, 0))  // → [2,4,4,4]
        .add(BatchNorm2D::new(4))
        .add(ReLULayer::new())
        .add(AvgPool2D::new(2, 2))              // → [2,4,2,2]
        .add(Flatten::new())                    // → [2,16]
        .add(Linear::new(16, 3));               // → [2,3]

    // Batch size > 1 needed for BatchNorm
    let input = Tensor::from_vec(
        (0..72).map(|i| i as Float * 0.02 - 0.5).collect(),
        &[2, 1, 6, 6],
        true,
    );
    let output = model.forward(&input);
    assert_eq!(output.data().shape(), &[2, 3]);

    let target = Tensor::from_vec(
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[2, 3],
        false,
    );
    let loss = mse_loss(&output, &target);
    loss.backward();

    // BN adds gamma + beta = 4 + 4 = 8 more parameters
    assert!(model.num_parameters() > 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Train/Eval mode
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sequential_train_eval_mode() {
    let mut model = Sequential::new()
        .add(Linear::new(4, 8))
        .add(Dropout::new(0.5))
        .add(Linear::new(8, 2));

    // In train mode, dropout should change output
    let input = Tensor::from_vec(vec![1.0; 4], &[1, 4], false);

    model.train();
    let out_train1 = model.forward(&input);
    let out_train2 = model.forward(&input);
    // Outputs may differ due to dropout randomness (not guaranteed but likely)

    // In eval mode, dropout is identity — outputs should be deterministic
    model.eval();
    let out_eval1 = model.forward(&input);
    let out_eval2 = model.forward(&input);
    let d1 = out_eval1.data();
    let d2 = out_eval2.data();
    for (a, b) in d1.iter().zip(d2.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Eval mode should be deterministic"
        );
    }

    // Ensure train mode can be re-enabled
    model.train();
    let _ = model.forward(&input); // should not panic
}

// ═══════════════════════════════════════════════════════════════════════════
// LeNet-5 architecture
// ═══════════════════════════════════════════════════════════════════════════

/// Build a LeNet-5-style architecture for 28×28 input images.
fn build_lenet5() -> Sequential {
    Sequential::new()
        // Layer 1: Conv 1→6, 5×5, padding=2 → [N,6,28,28]
        .add(Conv2DModule::new(1, 6, 5, 1, 2))
        .add(ReLULayer::new())
        .add(AvgPool2D::new(2, 2))  // → [N,6,14,14]
        // Layer 2: Conv 6→16, 5×5 → [N,16,10,10]
        .add(Conv2DModule::new(6, 16, 5, 1, 0))
        .add(ReLULayer::new())
        .add(AvgPool2D::new(2, 2))  // → [N,16,5,5]
        // Classifier
        .add(Flatten::new())         // → [N,400]
        .add(Linear::new(400, 120))
        .add(ReLULayer::new())
        .add(Linear::new(120, 84))
        .add(ReLULayer::new())
        .add(Linear::new(84, 10))
}

#[test]
fn test_lenet5_forward_shape() {
    let model = build_lenet5();

    // Single image 28×28
    let input = Tensor::from_vec(
        vec![0.0; 28 * 28],
        &[1, 1, 28, 28],
        false,
    );
    let output = model.forward(&input);
    assert_eq!(output.data().shape(), &[1, 10]);
}

#[test]
fn test_lenet5_batch_forward() {
    let model = build_lenet5();

    // Batch of 4 images
    let input = Tensor::from_vec(
        vec![0.5; 4 * 28 * 28],
        &[4, 1, 28, 28],
        false,
    );
    let output = model.forward(&input);
    assert_eq!(output.data().shape(), &[4, 10]);
}

#[test]
fn test_lenet5_num_parameters() {
    let model = build_lenet5();
    let n = model.num_parameters();
    // Conv1: 6*1*5*5+6 = 156
    // Conv2: 16*6*5*5+16 = 2416
    // FC1:   120*400+120 = 48120
    // FC2:   84*120+84 = 10164
    // FC3:   10*84+10 = 850
    // Total: ~61706
    assert_eq!(n, 156 + 2416 + 48120 + 10164 + 850);
    model.summary();
}

#[test]
fn test_lenet5_backward() {
    let model = build_lenet5();

    let input = Tensor::from_vec(
        (0..784).map(|i| (i as Float / 784.0) - 0.5).collect(),
        &[1, 1, 28, 28],
        true,
    );
    let output = model.forward(&input);
    let target = Tensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        &[1, 10],
        false,
    );
    let loss = cross_entropy_loss(&output, &target);
    loss.backward();

    assert!(input.grad().is_some(), "LeNet5 should propagate gradients to input");

    // All parameters should have gradients
    for (i, param) in model.parameters().iter().enumerate() {
        assert!(
            param.tensor().grad().is_some(),
            "Param {} should have gradient",
            i
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Training convergence: XOR with Sequential
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_train_xor_sequential() {
    let model = Sequential::new()
        .add(Linear::new(2, 16))
        .add(ReLULayer::new())
        .add(Linear::new(16, 1));

    let params: Vec<Parameter> = model.parameters().into_iter().cloned().collect();
    let mut optimizer = Adam::new(params, 0.01);

    let inputs: Vec<ArrayD<Float>> = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 1.0]).unwrap(),
    ];
    let targets: Vec<ArrayD<Float>> = vec![
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![1.0]).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0]).unwrap(),
    ];

    let history = model
        .trainer(&mut optimizer)
        .train_data(&inputs, &targets)
        .loss_fn(mse_loss)
        .epochs(300)
        .batch_size(4)
        .verbose(false)
        .fit();

    let last = history.last().unwrap();
    assert!(
        last.train_loss < 0.05,
        "XOR should converge: loss={:.4}",
        last.train_loss
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Training convergence: Mini-CNN classification
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_train_mini_cnn_classification() {
    // 2-class synthetic classification on 5×5 images.
    // Class 0: higher values in top-left corner
    // Class 1: higher values in bottom-right corner
    let model = Sequential::new()
        .add(Conv2DModule::new(1, 2, 3, 1, 0))  // → [N,2,3,3]
        .add(ReLULayer::new())
        .add(Flatten::new())                     // → [N,18]
        .add(Linear::new(18, 2));                // → [N,2]

    let params: Vec<Parameter> = model.parameters().into_iter().cloned().collect();
    let mut optimizer = Adam::new(params, 0.005);

    // Generate synthetic data
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for i in 0..16 {
        let mut data = vec![0.0f32; 25];
        if i % 2 == 0 {
            // Class 0: bright top-left
            for r in 0..3 {
                for c in 0..3 {
                    data[r * 5 + c] = 1.0 + (i as f32 * 0.1);
                }
            }
            targets.push(ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, 0.0]).unwrap());
        } else {
            // Class 1: bright bottom-right
            for r in 2..5 {
                for c in 2..5 {
                    data[r * 5 + c] = 1.0 + (i as f32 * 0.1);
                }
            }
            targets.push(ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![0.0, 1.0]).unwrap());
        }
        inputs.push(ArrayD::from_shape_vec(IxDyn(&[1, 1, 5, 5]), data).unwrap());
    }

    let history = model
        .trainer(&mut optimizer)
        .train_data(&inputs, &targets)
        .loss_fn(mse_loss)
        .epochs(50)
        .batch_size(4)
        .verbose(false)
        .fit();

    let first_loss = history[0].train_loss;
    let last_loss = history.last().unwrap().train_loss;
    assert!(
        last_loss < first_loss,
        "CNN training should decrease loss: first={:.4}, last={:.4}",
        first_loss,
        last_loss
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Model summary
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_model_summary() {
    let model = Sequential::new()
        .add(Conv2DModule::new(1, 6, 5, 1, 2))
        .add(ReLULayer::new())
        .add(AvgPool2D::new(2, 2))
        .add(Flatten::new())
        .add(Linear::new(1176, 10)); // approximate

    // Just verify summary doesn't panic
    model.summary();
    assert_eq!(model.len(), 5);
}

// ═══════════════════════════════════════════════════════════════════════════
// Training with validation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_train_with_validation() {
    let model = Sequential::new()
        .add(Linear::new(2, 8))
        .add(ReLULayer::new())
        .add(Linear::new(8, 1));

    let params: Vec<Parameter> = model.parameters().into_iter().cloned().collect();
    let mut optimizer = Adam::new(params, 0.01);

    // Simple regression: y = x1 + x2
    let inputs: Vec<ArrayD<Float>> = (0..8)
        .map(|i| {
            let x1 = (i as Float) * 0.25;
            let x2 = 1.0 - x1;
            ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![x1, x2]).unwrap()
        })
        .collect();
    let targets: Vec<ArrayD<Float>> = inputs
        .iter()
        .map(|x| {
            let sum = x.iter().sum::<Float>();
            ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![sum]).unwrap()
        })
        .collect();

    // Split: 6 train, 2 val
    let (train_x, val_x) = inputs.split_at(6);
    let (train_t, val_t) = targets.split_at(6);

    let history = model
        .trainer(&mut optimizer)
        .train_data(train_x, train_t)
        .validation_data(val_x, val_t)
        .loss_fn(mse_loss)
        .epochs(50)
        .batch_size(6)
        .verbose(false)
        .fit();

    // History should have val_loss entries
    assert!(history.last().unwrap().val_loss.is_some());
}

// ═══════════════════════════════════════════════════════════════════════════
// Early stopping
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_early_stopping() {
    let model = Sequential::new()
        .add(Linear::new(2, 4))
        .add(ReLULayer::new())
        .add(Linear::new(4, 1));

    let params: Vec<Parameter> = model.parameters().into_iter().cloned().collect();
    let mut optimizer = Adam::new(params, 0.01);

    let inputs: Vec<ArrayD<Float>> = (0..4)
        .map(|i| ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![i as Float, 0.0]).unwrap())
        .collect();
    let targets: Vec<ArrayD<Float>> = (0..4)
        .map(|i| ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![i as Float]).unwrap())
        .collect();

    let history = model
        .trainer(&mut optimizer)
        .train_data(&inputs, &targets)
        .validation_data(&inputs, &targets)
        .loss_fn(mse_loss)
        .epochs(1000)
        .batch_size(4)
        .verbose(false)
        .early_stopping(10)
        .fit();

    // Should stop before 1000 epochs
    assert!(
        history.len() < 1000,
        "Early stopping should trigger, ran {} epochs",
        history.len()
    );
}
