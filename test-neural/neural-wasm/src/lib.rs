//! WebAssembly Neural Network for XOR
//!
//! This module exposes a pre-trained XOR neural network via WebAssembly.
//! The model is embedded at compile time from a JSON file.

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// Embed the pre-trained model at compile time
const MODEL_JSON: &str = include_str!("xor_model.json");

/// Simplified layer structure for WASM (no need for full Network complexity)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ArrayData {
    data: Vec<f64>,
    dim: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Layer {
    weights: ArrayData,
    biases: ArrayData,
    activation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NetworkData {
    layers: Vec<Layer>,
}

/// XOR Neural Network exposed to JavaScript
#[wasm_bindgen]
pub struct XorNetwork {
    network: NetworkData,
}

#[wasm_bindgen]
impl XorNetwork {
    /// Create a new XOR network by loading the embedded model
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<XorNetwork, JsValue> {
        let network: NetworkData = serde_json::from_str(MODEL_JSON)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;
        
        Ok(XorNetwork { network })
    }

    /// Predict XOR result for two binary inputs
    /// Returns 0 or 1 (rounded prediction)
    #[wasm_bindgen]
    pub fn predict(&self, x1: f64, x2: f64) -> u8 {
        let raw = self.predict_raw(x1, x2);
        if raw > 0.5 { 1 } else { 0 }
    }

    /// Get raw prediction value (0.0 to 1.0)
    /// Useful for seeing the confidence of the prediction
    #[wasm_bindgen]
    pub fn predict_raw(&self, x1: f64, x2: f64) -> f64 {
        let mut current = vec![x1, x2];

        for layer in &self.network.layers {
            current = self.forward_layer(&current, layer);
        }

        current[0]
    }

    /// Get confidence percentage (0-100)
    #[wasm_bindgen]
    pub fn confidence(&self, x1: f64, x2: f64) -> f64 {
        let raw = self.predict_raw(x1, x2);
        // Confidence is how far from 0.5 (uncertainty) we are
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
        let layer_sizes: Vec<String> = self.network.layers
            .iter()
            .map(|l| format!("{}", l.weights.dim[0]))
            .collect();
        
        format!("XOR Network: 2 → [{}] → 1", layer_sizes.join(", "))
    }
}

impl XorNetwork {
    /// Forward pass through a single layer
    fn forward_layer(&self, input: &[f64], layer: &Layer) -> Vec<f64> {
        let out_size = layer.weights.dim[0];
        let in_size = layer.weights.dim[1];
        
        let mut output = vec![0.0; out_size];
        
        // Matrix multiplication: output = weights * input + biases
        for i in 0..out_size {
            let mut sum = layer.biases.data[i];
            for j in 0..in_size {
                sum += layer.weights.data[i * in_size + j] * input[j];
            }
            output[i] = self.activate(sum, &layer.activation);
        }
        
        output
    }

    /// Apply activation function
    fn activate(&self, x: f64, activation: &str) -> f64 {
        match activation {
            "Tanh" => x.tanh(),
            "Sigmoid" => 1.0 / (1.0 + (-x).exp()),
            "ReLU" => x.max(0.0),
            "LeakyReLU" => if x > 0.0 { x } else { 0.01 * x },
            "Linear" | _ => x,
        }
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

// Convenience functions for direct use without creating an instance

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