#!/bin/bash
set -e
echo "🔧 Building VGG-Tiny MNIST WASM..."
wasm-pack build --target web --out-dir pkg
echo "✅ VGG-Tiny WASM built successfully!"
