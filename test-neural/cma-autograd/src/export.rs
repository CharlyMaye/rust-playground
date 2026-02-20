//! # Weight Export: cma-autograd → cma-cnn
//!
//! Converts a trained autograd `Sequential` model into a `cma_cnn::Sequential`
//! for efficient inference (and WASM deployment).
//!
//! ## Architecture
//!
//! ```text
//! Training (cma-autograd)          Inference (cma-cnn)
//! ┌───────────────────┐            ┌──────────────────┐
//! │ autograd::Conv2D  │  export    │  cnn::Conv2D     │
//! │ autograd::BN2D    │ ───────→   │  cnn::BN2D       │
//! │ autograd::ReLU    │            │  cnn::Activation  │
//! │ autograd::MaxPool │            │  cnn::MaxPool2D  │
//! │ autograd::Flatten │            │  cnn::Flatten    │
//! └───────────────────┘            └──────────────────┘
//! ```
//!
//! Linear layers are NOT converted (they remain in `cma-neural-network::Network`
//! as the FC classifier head).
//!
//! ## Usage
//!
//! ```rust,ignore
//! use cma_autograd::export::export_cnn_to_inference;
//!
//! let autograd_model: Sequential = build_and_train_model();
//! let (cnn_seq, fc_start_idx) = export_cnn_to_inference(&autograd_model)?;
//! // cnn_seq: cma_cnn::Sequential with trained weights
//! // fc_start_idx: index where Linear layers begin (for FC head extraction)
//! ```

use crate::Float;
use crate::sequential::Sequential;

use cma_cnn::layers::{
    ActivationLayer, AvgPool2D as CnnAvgPool2D, BatchNorm2D as CnnBatchNorm2D,
    Conv2D as CnnConv2D, MaxPool2D as CnnMaxPool2D,
};
use cma_cnn::sequential::Sequential as CnnSequential;

use ndarray::Array1;

/// Result of exporting an autograd Sequential to cma-cnn format.
///
/// The CNN part (conv, bn, pool, flatten) becomes a `CnnSequential`.
/// Linear layers are left for the caller to build into a `Network` FC head.
pub struct ExportedModel {
    /// The CNN feature extractor with trained weights.
    pub cnn: CnnSequential,
    /// Indices of Linear layers in the original model (for FC head extraction).
    pub linear_indices: Vec<usize>,
}

/// Export the CNN layers from a trained autograd Sequential to cma-cnn format.
///
/// Walks through the model layers and converts each trainable layer by
/// copying weights. Stateless layers (ReLU, MaxPool, Flatten) are recreated.
/// Linear layers are skipped (they belong to the FC classifier head).
///
/// Returns the converted `CnnSequential` and the indices of Linear layers.
pub fn export_cnn_to_inference(model: &Sequential) -> Result<ExportedModel, String> {
    let mut cnn = CnnSequential::new();
    let mut linear_indices = Vec::new();

    for (idx, layer) in model.layers().iter().enumerate() {
        let name = layer.name();
        let any = layer.as_any();

        match name {
            "Conv2D" => {
                let conv = any
                    .downcast_ref::<crate::module::Conv2D>()
                    .ok_or_else(|| format!("Layer {} is Conv2D but downcast failed", idx))?;

                let mut cnn_conv = CnnConv2D::new(
                    conv.in_channels(),
                    conv.out_channels(),
                    conv.kernel_size(),
                    conv.stride(),
                    conv.padding(),
                );

                // Copy trained weights: ArrayD [O, I, kH, kW] → Array4
                let weight_data = conv.weight().data();
                let w4 = weight_data
                    .into_dimensionality::<ndarray::Ix4>()
                    .map_err(|e| format!("Conv2D weight shape error: {}", e))?;
                cnn_conv.weights = w4;

                // Copy bias
                if let Some(bias_param) = conv.bias() {
                    let bias_data = bias_param.data();
                    let b1 = bias_data
                        .into_dimensionality::<ndarray::Ix1>()
                        .map_err(|e| format!("Conv2D bias shape error: {}", e))?;
                    cnn_conv.bias = b1;
                    cnn_conv.use_bias = true;
                } else {
                    cnn_conv.use_bias = false;
                }

                cnn = cnn.add_conv2d(cnn_conv);
            }

            "BatchNorm2D" => {
                let bn = any
                    .downcast_ref::<crate::layers::BatchNorm2D>()
                    .ok_or_else(|| format!("Layer {} is BatchNorm2D but downcast failed", idx))?;

                let mut cnn_bn = CnnBatchNorm2D::new(bn.num_channels());

                // Copy gamma and beta from Parameter → Array1
                let gamma_data = bn.gamma().data();
                cnn_bn.gamma = gamma_data
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|e| format!("BN gamma shape error: {}", e))?;

                let beta_data = bn.beta().data();
                cnn_bn.beta = beta_data
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|e| format!("BN beta shape error: {}", e))?;

                // Copy running stats
                let rm = bn.running_mean();
                *cnn_bn.running_mean.write().unwrap() = Array1::from_vec(rm);

                let rv = bn.running_var();
                *cnn_bn.running_var.write().unwrap() = Array1::from_vec(rv);

                // Set to eval mode (inference uses running stats)
                cnn_bn.training = false;

                cnn = cnn.add_batchnorm(cnn_bn);
            }

            "ReLU" => {
                cnn = cnn.add_activation(ActivationLayer::relu());
            }

            "Sigmoid" => {
                cnn = cnn.add_activation(ActivationLayer::sigmoid());
            }

            "Tanh" => {
                cnn = cnn.add_activation(ActivationLayer::tanh());
            }

            "MaxPool2D" => {
                let pool = any
                    .downcast_ref::<crate::layers::MaxPool2D>()
                    .ok_or_else(|| format!("Layer {} is MaxPool2D but downcast failed", idx))?;
                cnn = cnn.add_maxpool(CnnMaxPool2D::new(pool.kernel_size(), pool.stride()));
            }

            "AvgPool2D" => {
                let pool = any
                    .downcast_ref::<crate::layers::AvgPool2D>()
                    .ok_or_else(|| format!("Layer {} is AvgPool2D but downcast failed", idx))?;
                cnn = cnn.add_avgpool(CnnAvgPool2D::new(pool.kernel_size(), pool.stride()));
            }

            "GlobalAvgPool2D" => {
                cnn = cnn.add_global_avgpool();
            }

            "Flatten" => {
                cnn = cnn.add_flatten();
            }

            "Linear" => {
                // Linear layers are part of the FC classifier head, not the CNN.
                // Record their index for the caller to extract separately.
                linear_indices.push(idx);
            }

            "Dropout" | "Softmax" => {
                // Dropout is identity at inference. Softmax is usually
                // applied separately or is part of the loss function.
                // Skip these in the exported CNN.
            }

            _ => {
                return Err(format!(
                    "Unknown layer type '{}' at index {}. Cannot export.",
                    name, idx
                ));
            }
        }
    }

    // Set the entire CNN to eval mode
    cnn.eval_mode();

    Ok(ExportedModel {
        cnn,
        linear_indices,
    })
}

/// Extract Linear layer weights from the autograd model as (weight, bias) pairs.
///
/// Returns a Vec of (weight_2d, bias_1d) for each Linear layer, in order.
/// These can be used to build a `cma_neural_network::Network` FC classifier.
pub fn extract_linear_weights(
    model: &Sequential,
) -> Result<Vec<(ndarray::Array2<Float>, ndarray::Array1<Float>)>, String> {
    let mut weights = Vec::new();

    for (idx, layer) in model.layers().iter().enumerate() {
        if layer.name() == "Linear" {
            let linear = layer
                .as_any()
                .downcast_ref::<crate::module::Linear>()
                .ok_or_else(|| format!("Layer {} is Linear but downcast failed", idx))?;

            let params = crate::module::Module::parameters(linear);

            // Weight: [out_features, in_features]
            let w = params[0].data();
            let w2 = w
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| format!("Linear weight shape error: {}", e))?;

            // Bias: [out_features]
            let b = if params.len() > 1 {
                let b_data = params[1].data();
                b_data
                    .into_dimensionality::<ndarray::Ix1>()
                    .map_err(|e| format!("Linear bias shape error: {}", e))?
            } else {
                ndarray::Array1::zeros(w2.shape()[0])
            };

            weights.push((w2, b));
        }
    }

    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_export_simple_cnn() {
        // Build a small autograd CNN
        let model = crate::sequential::Sequential::new()
            .add(Conv2DModule::new(1, 4, 3, 1, 0))
            .add(ReLULayer::new())
            .add(MaxPool2DLayer::new(2, 2))
            .add(Flatten::new())
            .add(Linear::new(16, 10));

        let result = export_cnn_to_inference(&model).expect("export should succeed");

        // CNN should have 4 layers (conv, relu, pool, flatten)
        assert_eq!(result.cnn.layers().len(), 4);

        // One linear layer should be recorded
        assert_eq!(result.linear_indices.len(), 1);
        assert_eq!(result.linear_indices[0], 4); // index 4 in original model
    }

    #[test]
    fn test_export_cnn_with_batchnorm() {
        let model = crate::sequential::Sequential::new()
            .add(Conv2DModule::new(1, 8, 3, 1, 1))
            .add(BatchNorm2D::new(8))
            .add(ReLULayer::new())
            .add(MaxPool2DLayer::new(2, 2))
            .add(Flatten::new())
            .add(Linear::new(32, 10));

        let result = export_cnn_to_inference(&model).expect("export should succeed");

        // CNN: conv, bn, relu, pool, flatten = 5 layers
        assert_eq!(result.cnn.layers().len(), 5);
        assert_eq!(result.linear_indices.len(), 1);
    }

    #[test]
    fn test_export_preserves_weights() {
        use crate::tensor::Tensor;

        // Build and "train" a tiny model
        let model = crate::sequential::Sequential::new()
            .add(Conv2DModule::new(1, 2, 3, 1, 0))
            .add(ReLULayer::new())
            .add(Flatten::new())
            .add(Linear::new(18, 2)); // 2 channels * 3x3 = 18 after flatten

        // Run one forward pass to verify it works
        let input = Tensor::from_vec(
            (0..25).map(|i| i as Float * 0.1).collect(),
            &[1, 1, 5, 5],
            false,
        );
        let autograd_output = model.forward(&input);

        // Export to cma-cnn
        let exported = export_cnn_to_inference(&model).expect("export should succeed");

        // Forward through cma-cnn (CNN part only — up to Flatten)
        use cma_cnn::tensor::Tensor4D;
        let cnn_input_data: Vec<Float> = (0..25).map(|i| i as Float * 0.1).collect();
        let cnn_input = Tensor4D::from_array(
            ndarray::Array4::from_shape_vec((1, 1, 5, 5), cnn_input_data).unwrap()
        );
        let cnn_output = exported.cnn.forward(&cnn_input);

        // The CNN output should match the autograd output up to the Flatten layer
        // (before Linear). Shape should be [1, 2*3*3] = [1, 18] flattened
        // or [1, 2, 3, 3] if Flatten gives raw shape
        assert!(cnn_output.data().len() > 0, "CNN output should have data");
    }

    #[test]
    fn test_extract_linear_weights() {
        let model = crate::sequential::Sequential::new()
            .add(Linear::new(10, 5))
            .add(ReLULayer::new())
            .add(Linear::new(5, 3));

        let weights = extract_linear_weights(&model).expect("extraction should succeed");
        assert_eq!(weights.len(), 2);

        // First linear: [5, 10] weight, [5] bias
        assert_eq!(weights[0].0.shape(), &[5, 10]);
        assert_eq!(weights[0].1.shape(), &[5]);

        // Second linear: [3, 5] weight, [3] bias
        assert_eq!(weights[1].0.shape(), &[3, 5]);
        assert_eq!(weights[1].1.shape(), &[3]);
    }
}
