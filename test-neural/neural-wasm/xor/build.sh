#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Neural WASM Build Script                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

MODEL_BIN_PATH="src/xor_model.bin"
MODEL_JSON_PATH="src/xor_model.json"
cd "$(dirname "$0")"

# Step 1: Check if model exists
echo ""
echo "🔍 Checking for pre-trained model..."

if [ -f "$MODEL_BIN_PATH" ]; then
    echo "   ✅ Model found: $MODEL_BIN_PATH"
    echo "   📊 Using existing trained model"
else
    echo "   ⚠️  No model found at $MODEL_BIN_PATH"
    echo ""
    echo "🧠 Training XOR neural network..."
    echo ""
    
    # Build and run the training script
    cargo run --bin train_xor --release --features training
    
    if [ -f "$MODEL_BIN_PATH" ]; then
        echo "   ✅ Model trained and saved successfully!"
    else
        echo "   ❌ Error: Model training failed!"
        exit 1
    fi
fi

# Step 2: Build WASM
echo ""
echo "📦 Building WebAssembly module..."
wasm-pack build --target web --out-dir pkg

# Step 3: Copy to ai-web-app/pkg/xor_wasm/ (do NOT modify historical www/)
echo ""
echo "📋 Copying WASM files to ai-web-app/pkg/xor_wasm/..."
mkdir -p ../../ai-web-app/pkg/xor_wasm
cp pkg/*.js pkg/*.wasm pkg/*.ts ../../ai-web-app/pkg/xor_wasm/ 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Build Complete! 🎉                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Output: neural-wasm/pkg/                                    ║"
echo "║  Test:   Open www/index.html in a browser                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"