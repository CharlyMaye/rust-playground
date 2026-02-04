#!/bin/bash
set -e

echo "🔧 Building AlexNet-Mini MNIST WASM..."

# Build the WASM module
wasm-pack build --target web --out-dir pkg

echo "✅ AlexNet-Mini WASM built successfully!"
echo "📦 Output: pkg/"
