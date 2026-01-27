use ndarray::array;
use neural_wasm_shared::ModelWithMetadata;

fn main() {
    let model_json = include_str!("../../neural-wasm/iris/src/iris_model.json");
    let model: ModelWithMetadata = serde_json::from_str(model_json).unwrap();

    println!("Model accuracy: {:.2}%", model.metadata.accuracy * 100.0);

    if let Some(ref norm) = model.metadata.normalization {
        println!("✅ Normalization stats found!");
        println!("   Means: {:?}", norm.means);
        println!("   Stds: {:?}", norm.stds);

        // Test avec Setosa typique
        let raw = [5.0_f64, 3.5, 1.4, 0.2];
        let normalized = norm.normalize(&raw);
        println!("\n🌸 Setosa test (5.0, 3.5, 1.4, 0.2):");
        println!(
            "   Normalized: [{:.3}, {:.3}, {:.3}, {:.3}]",
            normalized[0], normalized[1], normalized[2], normalized[3]
        );

        // Prédiction
        let input = array![normalized[0], normalized[1], normalized[2], normalized[3]];
        let output = model.network.predict(&input);
        println!(
            "   Output: [{:.4}, {:.4}, {:.4}]",
            output[0], output[1], output[2]
        );
        println!(
            "   Prediction: {} (confidence: {:.1}%)",
            if output[0] > output[1] && output[0] > output[2] {
                "Setosa"
            } else if output[1] > output[2] {
                "Versicolor"
            } else {
                "Virginica"
            },
            output.iter().cloned().fold(0.0_f64, f64::max) * 100.0
        );
    } else {
        println!("❌ ERROR: No normalization stats found!");
    }
}
