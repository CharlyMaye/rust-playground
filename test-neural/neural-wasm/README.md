# Neural Network WebAssembly Modules

This directory contains multiple WebAssembly-compiled neural network models, each in its own subdirectory.

## 📁 Structure

````
neural-wasm/
├── shared/              # Common code shared across all WASM modules
│   ├── Cargo.toml
│   └── src/lib.rs       # ModelInfo, LayerInfo, softmax, etc.
│
├── xor/                 # XOR binary classification
│   ├── Cargo.toml
│   ├── build.sh
│   ├── src/
│   │   ├── lib.rs       # XorNetwork WASM bindings
│   │   ├── train_xor.rs # Training script
│   │   └── xor_model.json
│   └── pkg/             # Built WASM output
│
├── iris/                # Iris multi-class classification
│   ├── Cargo.toml
│   ├── build.sh
│   ├── src/
│   │   ├── lib.rs       # IrisClassifier WASM bindings
│   │   ├── train_iris.rs
│   │   └── iris_model.json
│   └── pkg/
│
└── build_all.sh         # Build all modules at once
````

## 🚀 Building

### Build All Modules

````bash
./build_all.sh
````

This will:
1. Build the shared library
2. Build each module (XOR, Iris, etc.)
3. Copy WASM outputs to `../www/pkg/{module}_wasm/`

### Build Individual Module

````bash
cd xor
./build.sh
````

or

````bash
cd iris
./build.sh
````

## 📦 Adding a New Model

1. **Create module directory:**
   ````bash
   mkdir -p neural-wasm/my_model/src
   ````

2. **Create `Cargo.toml`:**
   ````toml
   [package]
   name = "neural-wasm-my-model"
   version = "0.1.0"
   edition = "2021"

   [lib]
   crate-type = ["cdylib"]

   [[bin]]
   name = "train_my_model"
   path = "src/train_my_model.rs"

   [dependencies]
   wasm-bindgen = "0.2"
   cma-neural-network = { path = "../../cma-neural-network" }
   neural-wasm-shared = { path = "../shared" }
   ndarray = { version = "0.17.1", features = ["serde"] }
   serde = { version = "1.0", features = ["derive"] }
   serde_json = "1.0"
   getrandom = { version = "0.3", features = ["wasm_js"] }
   console_error_panic_hook = { version = "0.1", optional = true }

   [features]
   default = ["console_error_panic_hook"]

   [profile.release]
   opt-level = "z"
   lto = true
   ````

3. **Create `src/lib.rs`:**
   ````rust
   use wasm_bindgen::prelude::*;
   use cma_neural_network::network::Network;
   use neural_wasm_shared::ModelInfo;

   const MODEL_JSON: &str = include_str!("my_model.json");

   #[wasm_bindgen]
   pub struct MyModelClassifier {
       network: Network,
   }

   #[wasm_bindgen]
   impl MyModelClassifier {
       #[wasm_bindgen(constructor)]
       pub fn new() -> Result<MyModelClassifier, JsValue> {
           let network: Network = serde_json::from_str(MODEL_JSON)
               .map_err(|e| JsValue::from_str(&format!("Failed to load model: {}", e)))?;
           Ok(MyModelClassifier { network })
       }

       #[wasm_bindgen]
       pub fn predict(&self, input: &[f64]) -> String {
           // Your prediction logic
           todo!()
       }
   }
   ````

4. **Create training script `src/train_my_model.rs`:**
   ````rust
   use cma_neural_network::builder::NetworkBuilder;
   use cma_neural_network::network::{Activation, LossFunction};
   use cma_neural_network::optimizer::OptimizerType;
   use cma_neural_network::dataset::Dataset;
   use std::fs;

   fn main() {
       // Build network
       let mut network = NetworkBuilder::new(input_size, output_size)
           .hidden_layer(hidden_size, Activation::ReLU)
           .output_activation(Activation::Sigmoid)
           .loss(LossFunction::BinaryCrossEntropy)
           .optimizer(OptimizerType::adam(0.01))
           .build();

       // Train network
       // ...

       // Save model
       let model_json = serde_json::to_string_pretty(&network)
           .expect("Failed to serialize network");
       fs::write("src/my_model.json", model_json)
           .expect("Failed to write model file");
   }
   ````

5. **Create `build.sh`:**
   ````bash
   #!/bin/bash
   set -e
   cd "$(dirname "$0")"

   MODEL_PATH="src/my_model.json"

   if [ ! -f "$MODEL_PATH" ]; then
       echo "🧠 Training model..."
       cargo run --bin train_my_model --release
   fi

   echo "📦 Building WebAssembly..."
   wasm-pack build --target web --out-dir pkg
   ````

6. **Update root `Cargo.toml`:**
   ````toml
   [workspace]
   members = [
       "cma-neural-network",
       "neural-wasm/shared",
       "neural-wasm/xor",
       "neural-wasm/iris",
       "neural-wasm/my_model",  # Add this line
   ]
   ````

7. **Update `build_all.sh`:**
   ````bash
   MODELS=("xor" "iris" "my_model")  # Add your model
   ````

8. **Create web page in `www/my_model.html`**

9. **Update `www/index.html`** to add link to your new model

## 🧪 Testing

````bash
# Build all
cd neural-wasm
./build_all.sh

# Start web server
cd ../www
npx http-server -p 8080 -c-1

# Open http://localhost:8080
````

## 📊 Current Models

- **XOR** (`/xor/`): Simple binary classification (2 → [8] → 1)
- **Iris** (`/iris/`): Multi-class classification (4 → [8] → 3)

## 🔧 Architecture Benefits

### Shared Code
- Common types (`ModelInfo`, `LayerInfo`, etc.)
- Utility functions (`softmax`, `confidence_to_percentage`)
- Reduces duplication

### Independent Modules
- Each model is self-contained
- Can be built separately
- Different architectures and datasets
- Individual WASM bundles (smaller downloads)

### Scalable
- Easy to add new models
- No cross-dependencies between models
- Clean separation of concerns

## 📝 Notes

- Each module creates its own WASM file
- WASM files are optimized with `opt-level = "z"` and `lto = true`
- Models are embedded in the WASM (no network requests)
- All computation happens in the browser
