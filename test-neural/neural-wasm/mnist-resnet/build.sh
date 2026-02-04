#!/bin/bash
# Build ResNet MNIST WASM module

set -e

echo "🔧 Building ResNet MNIST WASM..."

# First train the model if needed
if [ ! -f "src/resnet_model.bin" ]; then
    echo "📦 Training ResNet model first..."
    cargo run --bin train_resnet --release
fi

# Build WASM
wasm-pack build --target web --out-dir pkg

echo "✅ ResNet WASM built successfully!"
echo "📦 Output: pkg/"
