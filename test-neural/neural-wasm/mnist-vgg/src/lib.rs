//! WebAssembly VGG-Tiny CNN for MNIST
//!
//! Uses trained CNN weights exported from cma_autograd via cma_cnn Sequential.
//! Model format: CnnModelWithMetadata (CNN feature extractor + FC classifier).

use cma_cnn::sequential::Sequential as CnnSequential;
use cma_cnn::Float;
use cma_cnn::Tensor4D;
use cma_neural_network::network::Network;
use ndarray::Array1;
use neural_wasm_shared::{
    build_prediction_result, build_test_result, load_cnn_model_from_bytes, ArchitectureSummary,
    LayerInfo, LayerSummary, ModelInfo, NormalizationStats, TestResult, WeightsInfo,
};
use wasm_bindgen::prelude::*;

#[cfg(not(feature = "training"))]
const MODEL_BIN: &[u8] = include_bytes!("vgg_model.bin");

#[cfg(feature = "training")]
const MODEL_BIN: &[u8] = &[];

const CLASS_NAMES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

#[wasm_bindgen]
pub struct MnistVGGNetwork {
    cnn: CnnSequential,
    classifier: Network,
    accuracy: Float,
    test_samples: usize,
    trained_at: String,
    normalization: Option<NormalizationStats>,
}

#[wasm_bindgen]
impl MnistVGGNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<MnistVGGNetwork, JsValue> {
        let model = load_cnn_model_from_bytes(MODEL_BIN)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        Ok(MnistVGGNetwork {
            cnn: model.cnn,
            classifier: model.classifier,
            accuracy: model.metadata.accuracy,
            test_samples: model.metadata.test_samples,
            trained_at: model.metadata.trained_at,
            normalization: model.metadata.normalization,
        })
    }

    #[wasm_bindgen]
    pub fn predict(&self, pixels: &[Float]) -> String {
        if pixels.len() != 784 {
            return serde_json::json!({"error": format!("Expected 784 pixels, got {}", pixels.len())}).to_string();
        }
        let probs = self.forward(pixels);
        let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();
        let result = build_prediction_result(&probs, &class_names);
        serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
    }

    fn forward(&self, pixels: &[Float]) -> Vec<Float> {
        let normalized = self.normalize_input(pixels);
        let tensor = Tensor4D::from_array(
            ndarray::Array4::from_shape_vec((1, 1, 28, 28), normalized).expect("reshape failed"),
        );
        let features = self.cnn.forward(&tensor);
        let flat = features.flatten();
        let fc_input = Array1::from_vec(flat.row(0).to_vec());
        self.classifier.predict(&fc_input).to_vec()
    }

    fn normalize_input(&self, pixels: &[Float]) -> Vec<Float> {
        if let Some(ref norm) = self.normalization {
            norm.normalize(pixels)
        } else {
            pixels.to_vec()
        }
    }

    #[wasm_bindgen]
    pub fn get_probabilities(&self, pixels: &[Float]) -> String {
        if pixels.len() != 784 {
            return serde_json::json!({"error": "Expected 784 pixels"}).to_string();
        }
        serde_json::to_string(&self.forward(pixels)).unwrap_or_else(|_| "[]".to_string())
    }

    #[wasm_bindgen]
    pub fn get_class_names(&self) -> String {
        serde_json::to_string(&CLASS_NAMES.to_vec()).unwrap_or_else(|_| "[]".to_string())
    }

    #[wasm_bindgen]
    pub fn test_all(&self) -> String {
        let test_samples = get_mnist_test_samples();
        let class_names: Vec<String> = CLASS_NAMES.iter().map(|s| s.to_string()).collect();
        let results: Vec<TestResult> = test_samples
            .iter()
            .map(|(pixels, expected)| {
                let probs = self.forward(pixels);
                build_test_result(pixels.clone(), *expected as usize, &probs, &class_names)
            })
            .collect();
        serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string())
    }

    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        let num_cnn_layers = self.cnn.layers().len();
        let info = ModelInfo {
            name: "VGG-Tiny MNIST Classifier".to_string(),
            architecture: format!("CNN: {} layers → FC(→10)", num_cnn_layers),
            accuracy: self.accuracy * 100.0,
            description: "VGG-Tiny CNN trained end-to-end with autograd (Simonyan & Zisserman, 2014)".to_string(),
            test_samples: self.test_samples,
            trained_at: self.trained_at.clone(),
        };
        serde_json::to_string(&info).unwrap_or_else(|_| "{}".to_string())
    }

    #[wasm_bindgen]
    pub fn get_weights(&self) -> String {
        let layers = self.classifier.get_layers_info();
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

    #[wasm_bindgen]
    pub fn get_architecture(&self) -> String {
        use cma_cnn::sequential::BoxedLayer;

        let layers: Vec<LayerSummary> = self
            .cnn
            .layers()
            .iter()
            .map(|layer| {
                let (name, config) = match layer {
                    BoxedLayer::Conv2D(_) => ("Conv2D", ""),
                    BoxedLayer::BatchNorm2D(_) => ("BatchNorm2D", ""),
                    BoxedLayer::MaxPool2D(_) => ("MaxPool2D", "2×2"),
                    BoxedLayer::AvgPool2D(_) => ("AvgPool2D", ""),
                    BoxedLayer::GlobalAvgPool2D(_) => ("GlobalAvgPool2D", "→1×1"),
                    BoxedLayer::Activation(_) => ("Activation", "ReLU"),
                    BoxedLayer::Flatten(_) => ("Flatten", ""),
                    BoxedLayer::Dropout2D(_) => ("Dropout2D", ""),
                };
                LayerSummary {
                    name: name.to_string(),
                    config: config.to_string(),
                }
            })
            .collect();

        let summary = ArchitectureSummary {
            name: "VGG-Tiny".to_string(),
            model_type: "cnn".to_string(),
            input_shape: vec![1, 1, 28, 28],
            output_features: 3136,
            num_parameters: 0,
            layers,
        };
        serde_json::to_string(&summary).unwrap_or_else(|_| "{}".to_string())
    }
}

#[wasm_bindgen(start)]
pub fn main() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

fn get_mnist_test_samples() -> Vec<(Vec<Float>, u8)> {
    vec![
        (vec![0.0; 784], 0),
        (vec![0.0; 784], 1),
        (vec![0.0; 784], 2),
    ]
}
