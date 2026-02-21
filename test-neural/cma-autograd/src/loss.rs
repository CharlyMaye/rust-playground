//! # Loss Functions
//!
//! MSE and Cross-Entropy with autograd support.

use crate::Float;
use crate::ops;
use crate::tensor::Tensor;
use ndarray::{ArrayD, IxDyn};

/// Mean Squared Error loss.
///
/// MSE = mean((prediction - target)²)
///
/// Both inputs must have the same shape.
pub fn mse_loss(prediction: &Tensor, target: &Tensor) -> Tensor {
    let diff = prediction - target;
    let sq = diff.powf(2.0);
    sq.mean()
}

/// Cross-Entropy loss for multi-class classification.
///
/// Combines log-softmax and negative log-likelihood:
///
/// L = -Σ target_i * log(softmax(prediction_i))
///
/// # Arguments
/// - `logits`: raw scores [batch, num_classes] (NOT softmax)
/// - `targets`: one-hot encoded targets [batch, num_classes]
///
/// Returns a scalar loss tensor.
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    // Numerically stable log-softmax + NLL computed on raw arrays,
    // then a custom CrossEntropyBackward GradFn for correct gradient propagation.
    //
    // ∂L/∂logits = (softmax(logits) - targets) / batch_size
    let logits_data = logits.data();
    let shape = logits_data.shape().to_vec();
    assert_eq!(shape.len(), 2, "cross_entropy expects 2D logits [batch, classes]");

    let batch = shape[0];
    let classes = shape[1];
    let targets_data = targets.data();

    // Compute softmax and loss in one pass
    let mut softmax = ArrayD::zeros(IxDyn(&shape));
    let mut total_loss: Float = 0.0;

    for b in 0..batch {
        // Max for numerical stability
        let mut max_val = Float::NEG_INFINITY;
        for c in 0..classes {
            let v = logits_data[[b, c]];
            if v > max_val {
                max_val = v;
            }
        }

        // exp(logit - max) and sum
        let mut sum_exp: Float = 0.0;
        for c in 0..classes {
            let e = (logits_data[[b, c]] - max_val).exp();
            softmax[[b, c]] = e;
            sum_exp += e;
        }

        // Normalize to softmax and accumulate NLL
        for c in 0..classes {
            softmax[[b, c]] /= sum_exp;
            if targets_data[[b, c]] > 0.0 {
                total_loss -= targets_data[[b, c]] * (softmax[[b, c]] + 1e-8).ln();
            }
        }
    }

    let loss_val = total_loss / batch as Float;

    use crate::tensor::is_grad_enabled;
    use std::sync::Arc;

    if is_grad_enabled() && logits.requires_grad() {
        let grad_fn = Arc::new(CrossEntropyBackward {
            logits: logits.clone(),
            softmax,
            targets: targets_data,
            batch_size: batch,
        });
        Tensor::from_op(
            ArrayD::from_elem(IxDyn(&[]), loss_val),
            grad_fn,
        )
    } else {
        Tensor::scalar(loss_val, false)
    }
}

/// Backward for cross-entropy loss.
///
/// ∂L/∂logits = (softmax(logits) - targets) / batch_size
#[derive(Debug)]
struct CrossEntropyBackward {
    logits: Tensor,
    softmax: ArrayD<Float>,
    targets: ArrayD<Float>,
    batch_size: usize,
}

impl crate::grad_fn::GradFn for CrossEntropyBackward {
    fn backward(&self, _grad_output: &ArrayD<Float>) -> Vec<ArrayD<Float>> {
        // grad_output is scalar (1.0)
        // ∂L/∂logits = (softmax - targets) / batch
        let mut grad = &self.softmax - &self.targets;
        grad.mapv_inplace(|v| v / self.batch_size as Float);
        vec![grad]
    }

    fn inputs(&self) -> Vec<Tensor> {
        vec![self.logits.clone()]
    }

    fn name(&self) -> &'static str {
        "CrossEntropyBackward"
    }
}

/// Binary Cross-Entropy loss.
///
/// BCE = -mean(target * log(pred) + (1 - target) * log(1 - pred))
///
/// `prediction` should be in range (0, 1) (after sigmoid).
pub fn binary_cross_entropy_loss(prediction: &Tensor, target: &Tensor) -> Tensor {
    let eps = 1e-7;

    // Clamp prediction to avoid log(0) — using in-graph op to preserve gradients
    let pred_clamped = ops::clamp(prediction, eps, 1.0 - eps);

    // -target * log(pred) - (1 - target) * log(1 - pred)
    let log_pred = pred_clamped.log();
    let one = Tensor::from_vec(
        vec![1.0; prediction.numel()],
        &prediction.shape(),
        false,
    );
    let one_minus_pred = &one - &pred_clamped;
    let log_one_minus_pred = one_minus_pred.log();
    let one_minus_target = &one - target;

    let term1 = ops::mul(target, &log_pred);
    let term2 = ops::mul(&one_minus_target, &log_one_minus_pred);
    let sum = &term1 + &term2;
    let neg_sum = ops::neg(&sum);
    neg_sum.mean()
}
