#!/bin/bash
set -e
echo "🔧 Building VGG-Tiny MNIST WASM..."

# First train the model if needed
if [ ! -f "src/vgg_model.bin" ]; then
    echo "📦 Training VGG-Tiny model first..."
    cargo run --bin train_vgg --release --features training 2>&1 | tee src/training.log
fi

wasm-pack build --target web --out-dir pkg
echo "✅ VGG-Tiny WASM built successfully!"
