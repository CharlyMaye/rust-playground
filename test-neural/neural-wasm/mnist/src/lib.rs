//! WebAssembly Neural Network for MNIST
//!
//! This module exposes a pre-trained MNIST neural network via WebAssembly.
//! Uses cma_neural_network for all neural network operations.

use cma_neural_network::network::Network;
use cma_neural_network::Float;
use ndarray::Array1;
use neural_wasm_shared::{
    build_prediction_result, build_test_result, get_mnist_test_samples, load_model_from_bytes,
    ActivationsResponse, ArchitectureSummary, LayerActivation, LayerInfo, LayerSummary, ModelInfo,
    NormalizationStats, TestResult, WeightsInfo,
};
use wasm_bindgen::prelude::*;

// Embed the pre-trained model at compile time (binary format for smaller size)
const MODEL_BIN: &[u8] = include_bytes!("mnist_model.bin");

// Class names for MNIST digits
const CLASS_NAMES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

// ===== Main Network Struct =====

/// MNIST Neural Network exposed to JavaScript
#[wasm_bindgen]
pub struct MnistNetwork {
    network: Network,
    accuracy: Float,
    test_samples: usize,
    trained_at: String,
    normalization: Option<NormalizationStats>,
}

#[wasm_bindgen]
impl MnistNetwork {
    /// Create a new MNIST network by loading the embedded model
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<MnistNetwork, JsValue> {
        let model = load_model_from_bytes(MODEL_BIN)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        Ok(MnistNetwork {
            network: model.network,
            accuracy: model.metadata.accuracy,
            test_samples: model.metadata.test_samples,
            trained_at: model.metadata.trained_at,
            normalization: model.metadata.normalization,
        })
    }

    /// Predict MNIST digit from pixel array
    /// Accepts 784 pixels (28x28 image) or normalized values
    /// Returns JSON with digit prediction (0-9), probabilities, and confidence
    #[wasm_bindgen]
    pub fn predict(&self, pixels: &[Float]) -> String {
        if pixels.len() != 784 {
            return serde_json::json!({
                "error": format!("Expected 784 pixels, got {}", pixels.len())
            })
            .to_string();
        }

        let normalized = self.normalize_input(pixels);
        let input = Array1::from_vec(normalized);
        let output = self.network.predict(&input);
        let probs = output.to_vec();
        let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();

        let result = build_prediction_result(&probs, &class_names);
        serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Normalize input pixels using stored normalization statistics
    fn normalize_input(&self, pixels: &[Float]) -> Vec<Float> {
        if let Some(ref norm) = self.normalization {
            norm.normalize(pixels)
        } else {
            pixels.to_vec()
        }
    }

    /// Get class probabilities for 784 pixels
    #[wasm_bindgen]
    pub fn get_probabilities(&self, pixels: &[Float]) -> String {
        if pixels.len() != 784 {
            return serde_json::json!({"error": "Expected 784 pixels"}).to_string();
        }

        let normalized = self.normalize_input(pixels);
        let input = Array1::from_vec(normalized);
        let output = self.network.predict(&input);
        serde_json::to_string(&output.to_vec()).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get class names (digits 0-9)
    #[wasm_bindgen]
    pub fn get_class_names(&self) -> String {
        serde_json::to_string(&vec!["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
            .unwrap_or_else(|_| "[]".to_string())
    }

    // Private helper methods
    fn predict_probs(&self, pixels: &[Float]) -> Vec<Float> {
        if pixels.len() != 784 {
            return vec![0.0; 10];
        }
        let normalized = self.normalize_input(pixels);
        let input = Array1::from_vec(normalized);
        let output = self.network.predict(&input);
        output.to_vec()
    }

    /// Test with sample MNIST digits
    /// Returns results with digit predictions (0-9)
    #[wasm_bindgen]
    pub fn test_all(&self) -> String {
        let test_samples = get_mnist_test_samples();
        let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();

        let results: Vec<TestResult> = test_samples
            .iter()
            .map(|(pixels, expected)| {
                let probs = self.predict_probs(pixels);
                build_test_result(pixels.clone(), *expected as usize, &probs, &class_names)
            })
            .collect();

        serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get model info with accuracy and metadata
    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        let info = ModelInfo {
            name: "MNIST Classifier".to_string(),
            architecture: self.network.architecture_string(),
            accuracy: self.accuracy * 100.0,
            description: "Binary classification using MNIST dataset".to_string(),
            test_samples: self.test_samples,
            trained_at: self.trained_at.clone(),
        };
        serde_json::to_string(&info).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get all weights and biases as JSON
    #[wasm_bindgen]
    pub fn get_weights(&self) -> String {
        let layers = self.network.get_layers_info();
        let response = WeightsInfo {
            layers: layers
                .iter()
                .map(|(weights, biases, activation_name)| {
                    let weights_2d: Vec<Vec<Float>> =
                        weights.rows().into_iter().map(|row| row.to_vec()).collect();

                    LayerInfo {
                        weights: weights_2d,
                        biases: biases.to_vec(),
                        activation: activation_name.to_string(),
                        shape: [weights.nrows(), weights.ncols()],
                    }
                })
                .collect(),
        };

        serde_json::to_string(&response).unwrap_or_else(|_| r#"{"layers":[]}"#.to_string())
    }

    /// Run inference and return all neuron activations for visualization
    #[wasm_bindgen]
    pub fn get_activations(&self, pixels: &[Float]) -> String {
        if pixels.len() != 784 {
            return r#"{"inputs":[],"layers":[],"output":[]}"#.to_string();
        }

        let normalized = self.normalize_input(pixels);
        let input = Array1::from_vec(normalized);
        let activations = self.network.get_all_activations(&input);

        let output_probs = activations
            .last()
            .map(|(_, post, _)| post.to_vec())
            .unwrap_or_else(|| vec![0.0; 10]);

        let response = ActivationsResponse {
            inputs: pixels.to_vec(),
            layers: activations
                .iter()
                .map(|(pre, post, activation_name)| LayerActivation {
                    pre_activation: pre.to_vec(),
                    activation: post.to_vec(),
                    function: activation_name.to_string(),
                })
                .collect(),
            output: output_probs,
        };

        serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"inputs":[],"layers":[],"output":[]}"#.to_string())
    }

    /// Get CNN intermediate activations (not available for FC-only models)
    #[wasm_bindgen]
    pub fn get_cnn_activations(&self, _pixels: &[Float]) -> String {
        serde_json::json!({"error": "This model has no CNN layers"}).to_string()
    }

    /// Get architecture summary
    #[wasm_bindgen]
    pub fn get_architecture(&self) -> String {
        let layers_info = self.network.get_layers_info();
        let layers: Vec<LayerSummary> = layers_info
            .iter()
            .enumerate()
            .map(|(i, (weights, _, activation))| LayerSummary {
                name: format!("FC{}", i + 1),
                config: format!("{}→{} ({})", weights.ncols(), weights.nrows(), activation),
            })
            .collect();

        let num_params: usize = layers_info.iter().map(|(w, b, _)| w.len() + b.len()).sum();

        let summary = ArchitectureSummary {
            name: "MNIST FC Classifier".to_string(),
            model_type: "fc".to_string(),
            input_shape: vec![784],
            output_features: 10,
            num_parameters: num_params,
            layers,
        };

        serde_json::to_string(&summary).unwrap_or_else(|_| "{}".to_string())
    }
}

/// Initialize the module
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}
