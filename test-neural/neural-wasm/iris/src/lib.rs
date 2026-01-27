use cma_neural_network::network::Network;
use ndarray::array;
use neural_wasm_shared::{
    softmax, LayerInfo, ModelInfo, ModelWithMetadata, NormalizationStats, WeightsInfo,
};
use serde::Serialize;
use wasm_bindgen::prelude::*;

const MODEL_JSON: &str = include_str!("iris_model.json");

#[derive(Serialize)]
pub struct IrisPrediction {
    pub class: String,
    pub class_idx: usize,
    pub probabilities: Vec<f64>,
    pub confidence: f64,
}

#[derive(Serialize)]
pub struct IrisTestResult {
    pub sepal_length: f64,
    pub sepal_width: f64,
    pub petal_length: f64,
    pub petal_width: f64,
    pub predicted: String,
    pub predicted_idx: usize,
    pub expected: String,
    pub expected_idx: usize,
    pub probabilities: Vec<f64>,
    pub is_correct: bool,
}

#[wasm_bindgen]
pub struct IrisClassifier {
    network: Network,
    classes: Vec<String>,
    accuracy: f64,
    test_samples: usize,
    trained_at: String,
    normalization: Option<NormalizationStats>,
}

#[wasm_bindgen]
impl IrisClassifier {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<IrisClassifier, JsValue> {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();

        let model: ModelWithMetadata = serde_json::from_str(MODEL_JSON)
            .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;

        let classes = vec![
            "Setosa".to_string(),
            "Versicolor".to_string(),
            "Virginica".to_string(),
        ];

        Ok(IrisClassifier {
            network: model.network,
            classes,
            accuracy: model.metadata.accuracy,
            test_samples: model.metadata.test_samples,
            trained_at: model.metadata.trained_at,
            normalization: model.metadata.normalization,
        })
    }

    /// Normalize input features using stored statistics
    fn normalize_input(
        &self,
        sepal_length: f64,
        sepal_width: f64,
        petal_length: f64,
        petal_width: f64,
    ) -> [f64; 4] {
        if let Some(ref norm) = self.normalization {
            let raw = [sepal_length, sepal_width, petal_length, petal_width];
            let normalized = norm.normalize(&raw);
            [normalized[0], normalized[1], normalized[2], normalized[3]]
        } else {
            // No normalization stats - use raw values (backward compatibility)
            [sepal_length, sepal_width, petal_length, petal_width]
        }
    }

    /// Predict iris species from measurements
    /// Parameters: sepal_length, sepal_width, petal_length, petal_width (in cm)
    #[wasm_bindgen]
    pub fn predict(
        &self,
        sepal_length: f64,
        sepal_width: f64,
        petal_length: f64,
        petal_width: f64,
    ) -> String {
        let normalized = self.normalize_input(sepal_length, sepal_width, petal_length, petal_width);
        let input = array![normalized[0], normalized[1], normalized[2], normalized[3]];
        let output = self.network.predict(&input);

        // Network already uses Softmax output activation - output IS probabilities
        let probs = output.to_vec();

        let (max_idx, _) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let result = IrisPrediction {
            class: self.classes[max_idx].clone(),
            class_idx: max_idx,
            probabilities: probs.clone(),
            confidence: probs[max_idx] * 100.0,
        };

        serde_json::to_string(&result).unwrap()
    }

    /// Get class probabilities for a prediction
    #[wasm_bindgen]
    pub fn get_probabilities(
        &self,
        sepal_length: f64,
        sepal_width: f64,
        petal_length: f64,
        petal_width: f64,
    ) -> String {
        let normalized = self.normalize_input(sepal_length, sepal_width, petal_length, petal_width);
        let input = array![normalized[0], normalized[1], normalized[2], normalized[3]];
        let output = self.network.predict(&input);
        // Network already uses Softmax output activation - output IS probabilities
        serde_json::to_string(&output.to_vec()).unwrap()
    }

    /// Test all samples from the dataset
    #[wasm_bindgen]
    pub fn test_all(&self) -> String {
        let test_data = get_iris_test_samples();
        let mut results = Vec::new();

        for (inputs, expected_idx) in test_data {
            let input = array![inputs[0], inputs[1], inputs[2], inputs[3]];
            let output = self.network.predict(&input);
            // Network already uses Softmax output activation - output IS probabilities
            let probs = output.to_vec();

            let (predicted_idx, _) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            results.push(IrisTestResult {
                sepal_length: inputs[0],
                sepal_width: inputs[1],
                petal_length: inputs[2],
                petal_width: inputs[3],
                predicted: self.classes[predicted_idx].clone(),
                predicted_idx,
                expected: self.classes[expected_idx].clone(),
                expected_idx,
                probabilities: probs,
                is_correct: predicted_idx == expected_idx,
            });
        }

        serde_json::to_string(&results).unwrap()
    }

    #[wasm_bindgen]
    pub fn model_info(&self) -> String {
        // Accuracy is loaded from the model metadata (saved during training)
        let info = ModelInfo {
            name: "Iris Species Classifier".to_string(),
            architecture: self.network.architecture_string(),
            accuracy: self.accuracy * 100.0,
            description: "Classifies iris flowers into three species: Setosa, Versicolor, and Virginica based on sepal and petal measurements".to_string(),
            test_samples: self.test_samples,
            trained_at: self.trained_at.clone(),
        };
        serde_json::to_string(&info).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_weights(&self) -> String {
        let layers_info = self.network.get_layers_info();
        let layers: Vec<LayerInfo> = layers_info
            .iter()
            .map(|(weights, biases, activation_name)| {
                let weights_2d: Vec<Vec<f64>> =
                    weights.rows().into_iter().map(|row| row.to_vec()).collect();

                LayerInfo {
                    weights: weights_2d,
                    biases: biases.to_vec(),
                    activation: activation_name.to_string(),
                    shape: [weights.nrows(), weights.ncols()],
                }
            })
            .collect();

        let weights_info = WeightsInfo { layers };

        serde_json::to_string(&weights_info).unwrap()
    }

    #[wasm_bindgen]
    pub fn get_class_names(&self) -> String {
        serde_json::to_string(&self.classes).unwrap()
    }

    /// Get layer-by-layer activations for visualization
    #[wasm_bindgen]
    pub fn get_activations(
        &self,
        sepal_length: f64,
        sepal_width: f64,
        petal_length: f64,
        petal_width: f64,
    ) -> String {
        let input = array![sepal_length, sepal_width, petal_length, petal_width];
        let activations = self.network.get_all_activations(&input);

        let output = self.network.predict(&input);
        let probs = softmax(&output.to_vec());

        #[derive(Serialize)]
        struct LayerActivation {
            pre_activation: Vec<f64>,
            activation: Vec<f64>,
            function: String,
        }

        #[derive(Serialize)]
        struct ActivationsResponse {
            inputs: [f64; 4],
            layers: Vec<LayerActivation>,
            output: Vec<f64>,
        }

        let response = ActivationsResponse {
            inputs: [sepal_length, sepal_width, petal_length, petal_width],
            layers: activations
                .iter()
                .map(|(pre, post, activation_name)| LayerActivation {
                    pre_activation: pre.iter().cloned().collect(),
                    activation: post.iter().cloned().collect(),
                    function: activation_name.to_string(),
                })
                .collect(),
            output: probs,
        };

        serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"inputs":[0,0,0,0],"layers":[],"output":[]}"#.to_string())
    }
}

/// Sample iris data for testing
fn get_iris_test_samples() -> Vec<([f64; 4], usize)> {
    vec![
        // Setosa samples (class 0)
        ([5.1, 3.5, 1.4, 0.2], 0),
        ([4.9, 3.0, 1.4, 0.2], 0),
        ([5.0, 3.6, 1.4, 0.2], 0),
        ([4.6, 3.1, 1.5, 0.2], 0),
        // Versicolor samples (class 1)
        ([7.0, 3.2, 4.7, 1.4], 1),
        ([6.4, 3.2, 4.5, 1.5], 1),
        ([6.9, 3.1, 4.9, 1.5], 1),
        ([5.5, 2.3, 4.0, 1.3], 1),
        // Virginica samples (class 2)
        ([6.3, 3.3, 6.0, 2.5], 2),
        ([5.8, 2.7, 5.1, 1.9], 2),
        ([7.1, 3.0, 5.9, 2.1], 2),
        ([6.5, 3.0, 5.8, 2.2], 2),
    ]
}
