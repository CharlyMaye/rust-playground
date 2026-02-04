#!/bin/bash
set -e

echo "🔧 Building AlexNet-Mini MNIST WASM..."

# First train the model if needed
if [ ! -f "src/alexnet_model.bin" ]; then
    echo "📦 Training AlexNet-Mini model first..."
    cargo run --bin train_alexnet --release --features training
fi

# Build the WASM module
wasm-pack build --target web --out-dir pkg

echo "✅ AlexNet-Mini WASM built successfully!"
echo "📦 Output: pkg/"
