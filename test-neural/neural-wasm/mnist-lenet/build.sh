#!/bin/bash
# Build LeNet-5 MNIST WASM module

set -e

echo "🔧 Building LeNet-5 MNIST WASM..."

# First train the model if needed
if [ ! -f "src/lenet_model.bin" ]; then
    echo "📦 Training LeNet-5 model first..."
    cargo run --bin train_lenet --release
fi

# Build WASM
wasm-pack build --target web --out-dir pkg

echo "✅ LeNet-5 WASM built successfully!"
echo "📦 Output: pkg/"
