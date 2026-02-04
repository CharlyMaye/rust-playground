//! WebAssembly ResNet CNN for MNIST
//!
//! Proper ResNet (He et al., 2015) with residual blocks and skip connections.
//! Uses the flexible ResNet/ResNetBuilder from cma_models library.

use cma_cnn::Float;
use cma_cnn::Tensor4D;
use cma_models::resnet::{ResNet, ResNetBuilder};
use cma_neural_network::network::Network;
use ndarray::Array1;
use neural_wasm_shared::{
    build_prediction_result, build_test_result, load_model_from_bytes, ArchitectureSummary,
    LayerInfo, LayerSummary, ModelInfo, NormalizationStats, TestResult, WeightsInfo,
};
use wasm_bindgen::prelude::*;

#[cfg(not(feature = "training"))]
const MODEL_BIN: &[u8] = include_bytes!("resnet_model.bin");

#[cfg(feature = "training")]
const MODEL_BIN: &[u8] = &[];

const CLASS_NAMES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

#[wasm_bindgen]
pub struct MnistResNetNetwork {
    resnet: ResNet,
    classifier: Network,
    accuracy: Float,
    test_samples: usize,
    trained_at: String,
    normalization: Option<NormalizationStats>,
}

#[wasm_bindgen]
impl MnistResNetNetwork {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<MnistResNetNetwork, JsValue> {
        let model = load_model_from_bytes(MODEL_BIN)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        let resnet = ResNetBuilder::mnist().build();

        Ok(MnistResNetNetwork {
            resnet,
            classifier: model.network,
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
        let features = self.resnet.forward(&tensor);
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
        let info = ModelInfo {
            name: "ResNet-MNIST Classifier".to_string(),
            architecture: format!(
                "ResNet: Stem→{} stages→GAP→FC({}→10)",
                self.resnet.stages.len(),
                self.resnet.output_features()
            ),
            accuracy: self.accuracy * 100.0,
            description: "ResNet CNN with residual blocks (He et al., 2015)".to_string(),
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
        let mut layers = vec![LayerSummary {
            name: "Stem".to_string(),
            config: "Conv(1→16, 3x3) + BN + ReLU".to_string(),
        }];

        for (i, stage) in self.resnet.stages.iter().enumerate() {
            let ch = self.resnet.stage_channels.get(i).unwrap_or(&0);
            let stride = if i == 0 { 1 } else { 2 };
            layers.push(LayerSummary {
                name: format!("Stage{}", i + 1),
                config: format!("{}× BasicBlock(→{}, stride={})", stage.len(), ch, stride),
            });
        }

        layers.push(LayerSummary {
            name: "GlobalAvgPool".to_string(),
            config: "→1x1".to_string(),
        });
        layers.push(LayerSummary {
            name: "Output".to_string(),
            config: format!("→{}", self.resnet.output_features()),
        });

        let summary = ArchitectureSummary {
            name: "ResNet-MNIST".to_string(),
            model_type: "resnet".to_string(),
            input_shape: vec![1, 1, 28, 28],
            output_features: self.resnet.output_features(),
            num_parameters: self.resnet.num_parameters(),
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
