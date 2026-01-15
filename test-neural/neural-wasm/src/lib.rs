//! WebAssembly Neural Network for XOR
//!
//! This module exposes a pre-trained XOR neural network via WebAssembly.
//! Uses cma_neural_network for all neural network operations.

use wasm_bindgen::prelude::*;
use cma_neural_network::network::Network;
use ndarray::array;

// Embed the pre-trained model at compile time
const MODEL_JSON: &str = include_str!("xor_model.json");

/// XOR Neural Network exposed to JavaScript
#[wasm_bindgen]
pub struct XorNetwork {
    network: Network,
}

#[wasm_bindgen]
impl XorNetwork {
    /// Create a new XOR network by loading the embedded model
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<XorNetwork, JsValue> {
        let network: Network = serde_json::from_str(MODEL_JSON)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;
        
        Ok(XorNetwork { network })
    }

    /// Predict XOR result for two binary inputs
    /// Returns 0 or 1 (rounded prediction)
    #[wasm_bindgen]
    pub fn predict(&self, x1: f64, x2: f64) -> u8 {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        if output[0] > 0.5 { 1 } else { 0 }
    }

    /// Get raw prediction value (0.0 to 1.0)
    #[wasm_bindgen]
    pub fn predict_raw(&self, x1: f64, x2: f64) -> f64 {
        let input = array![x1, x2];
        let output = self.network.predict(&input);
        output[0]
    }

    /// Get confidence percentage (0-100)
    #[wasm_bindgen]
    pub fn confidence(&self, x1: f64, x2: f64) -> f64 {
        let raw = self.predict_raw(x1, x2);
        let distance_from_uncertain = (raw - 0.5).abs();
        distance_from_uncertain * 2.0 * 100.0
    }

    /// Test all XOR combinations and return results as JSON string
    #[wasm_bindgen]
    pub fn test_all(&self) -> String {
        let results: Vec<_> = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
            .iter()
            .map(|&(a, b)| {
                let expected = if (a as u8) ^ (b as u8) == 1 { 1 } else { 0 };
                let prediction = self.predict(a, b);
                let raw = self.predict_raw(a, b);
                let confidence = self.confidence(a, b);
                format!(
                    r#"{{"a":{},"b":{},"expected":{},"prediction":{},"raw":{:.4},"confidence":{:.1}}}"#,
                    a as u8, b as u8, expected, prediction, raw, confidence
                )
            })
            .collect();
        
        format!("[{}]", results.join(","))
    }

    /// Get model info
    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        format!("XOR Network: {}", self.network.architecture_string())
    }

    /// Get all weights and biases as JSON
    #[wasm_bindgen]
    pub fn get_weights(&self) -> String {
        let layers = self.network.get_layers_info();
        let layer_data: Vec<String> = layers.iter().map(|(weights, biases, activation_name)| {
            let shape = weights.shape();
            
            format!(
                r#"{{"weights":[{}],"biases":[{}],"activation":"{}","shape":[{},{}]}}"#,
                weights.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(","),
                biases.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(","),
                activation_name,
                shape[0],
                shape[1]
            )
        }).collect();
        
        format!(r#"{{"layers":[{}]}}"#, layer_data.join(","))
    }

    /// Run inference and return all neuron activations for visualization
    #[wasm_bindgen]
    pub fn get_activations(&self, x1: f64, x2: f64) -> String {
        let input = array![x1, x2];
        let activations = self.network.get_all_activations(&input);
        
        let inputs_json = format!("[{},{}]", x1, x2);
        
        let layers_json: Vec<String> = activations.iter().map(|(pre, post, activation_name)| {
            format!(
                r#"{{"pre_activation":[{}],"activation":[{}],"function":"{}"}}"#,
                pre.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(","),
                post.iter().map(|v| format!("{:.6}", v)).collect::<Vec<_>>().join(","),
                activation_name
            )
        }).collect();
        
        let output = activations.last().map(|(_, post, _)| post[0]).unwrap_or(0.0);
        
        format!(
            r#"{{"inputs":{},"layers":[{}],"output":{:.6}}}"#,
            inputs_json,
            layers_json.join(","),
            output
        )
    }
}

impl Default for XorNetwork {
    fn default() -> Self {
        Self::new().expect("Failed to create default XorNetwork")
    }
}

/// Initialize the module
#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Quick predict function
#[wasm_bindgen]
pub fn xor_predict(x1: f64, x2: f64) -> u8 {
    let network = XorNetwork::default();
    network.predict(x1, x2)
}

/// Quick raw predict function
#[wasm_bindgen]
pub fn xor_predict_raw(x1: f64, x2: f64) -> f64 {
    let network = XorNetwork::default();
    network.predict_raw(x1, x2)
}