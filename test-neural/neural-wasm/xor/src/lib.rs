//! WebAssembly Neural Network for XOR
//!
//! This module exposes a pre-trained XOR neural network via WebAssembly.
//! Uses cma_neural_network for all neural network operations.

use cma_neural_network::network::Network;
use cma_neural_network::Float;
use ndarray::array;
use neural_wasm_shared::{
    build_test_result, load_model_from_bytes, ActivationsResponse, ArchitectureSummary,
    LayerActivation, LayerInfo, LayerSummary, ModelInfo, PredictionResult, TestResult, WeightsInfo,
};
use wasm_bindgen::prelude::*;

// Embed the pre-trained model at compile time (binary format for smaller size)
const MODEL_BIN: &[u8] = include_bytes!("xor_model.bin");

// Class names for XOR
const CLASS_NAMES: [&str; 2] = ["0", "1"];

// ===== Main Network Struct =====

/// XOR Neural Network exposed to JavaScript
#[wasm_bindgen]
pub struct XorNetwork {
    network: Network,
    accuracy: Float,
    test_samples: usize,
    trained_at: String,
}

#[wasm_bindgen]
impl XorNetwork {
    /// Create a new XOR network by loading the embedded model
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<XorNetwork, JsValue> {
        let model = load_model_from_bytes(MODEL_BIN)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        Ok(XorNetwork {
            network: model.network,
            accuracy: model.metadata.accuracy,
            test_samples: model.metadata.test_samples as usize,
            trained_at: model.metadata.trained_at,
        })
    }

    /// Predict XOR result for two binary inputs
    /// Returns JSON with prediction details
    #[wasm_bindgen]
    pub fn predict(&self, x1: Float, x2: Float) -> String {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        let raw = output[0];
        let probs = vec![1.0 - raw, raw];

        let result = PredictionResult {
            class_name: if raw > 0.5 {
                "1".to_string()
            } else {
                "0".to_string()
            },
            class_index: if raw > 0.5 { 1 } else { 0 },
            probabilities: probs,
            confidence: (raw - 0.5).abs() * 2.0, // ratio 0-1, frontend multiplies by 100 for display
        };

        serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get class probabilities
    #[wasm_bindgen]
    pub fn get_probabilities(&self, x1: Float, x2: Float) -> String {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        let raw = output[0];
        let probs = vec![1.0 - raw, raw];
        serde_json::to_string(&probs).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get class names
    #[wasm_bindgen]
    pub fn get_class_names(&self) -> String {
        serde_json::to_string(&vec!["0", "1"]).unwrap_or_else(|_| "[]".to_string())
    }

    // Private helper methods
    fn predict_probs(&self, x1: Float, x2: Float) -> Vec<Float> {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        let raw = output[0];
        vec![1.0 - raw, raw]
    }

    /// Test all XOR combinations and return results as JSON string
    #[wasm_bindgen]
    pub fn test_all(&self) -> String {
        let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();

        let results: Vec<TestResult> = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            .iter()
            .map(|&(a, b)| {
                let expected = if (a as u8) ^ (b as u8) == 1 { 1 } else { 0 };
                let probs = self.predict_probs(a, b);

                build_test_result(vec![a, b], expected, &probs, &class_names)
            })
            .collect();

        serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get model info with accuracy and metadata
    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        let info = ModelInfo {
            name: "XOR Logic Gate Classifier".to_string(),
            architecture: self.network.architecture_string(),
            accuracy: self.accuracy * 100.0,
            description: "Binary classification using XOR logic gate".to_string(),
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
    pub fn get_activations(&self, x1: Float, x2: Float) -> String {
        let input = array![x1, x2];
        let activations = self.network.get_all_activations(&input);

        // Get the actual output layer activation (not transformed probabilities)
        let output = activations
            .last()
            .map(|(_, post, _)| post.to_vec())
            .unwrap_or_else(|| vec![0.5]);

        let response = ActivationsResponse {
            inputs: vec![x1, x2],
            layers: activations
                .iter()
                .map(|(pre, post, activation_name)| LayerActivation {
                    pre_activation: pre.to_vec(),
                    activation: post.to_vec(),
                    function: activation_name.to_string(),
                })
                .collect(),
            output,
        };

        serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"inputs":[],"layers":[],"output":[]}"#.to_string())
    }

    /// Get CNN intermediate activations (not available for FC-only models)
    #[wasm_bindgen]
    pub fn get_cnn_activations(&self, _x1: Float, _x2: Float) -> String {
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
            name: "XOR Network".to_string(),
            model_type: "fc".to_string(),
            input_shape: vec![2],
            output_features: 1,
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
