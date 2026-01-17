#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Iris Neural WASM Build Script                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

MODEL_PATH="src/iris_model.json"
cd "$(dirname "$0")"

# Step 1: Check if model exists
echo ""
echo "🔍 Checking for pre-trained model..."

if [ -f "$MODEL_PATH" ]; then
    echo "   ✅ Model found: $MODEL_PATH"
    echo "   📊 Using existing trained model"
else
    echo "   ⚠️  No model found at $MODEL_PATH"
    echo ""
    echo "🧠 Training Iris classification network..."
    echo ""
    
    # Build and run the training script
    cargo run --bin train_iris --release
    
    if [ -f "$MODEL_PATH" ]; then
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

# Step 3: Copy to www/pkg/iris_wasm/
echo ""
echo "📋 Copying WASM files to www/pkg/iris_wasm/..."
mkdir -p ../../www/pkg/iris_wasm
cp pkg/*.js pkg/*.wasm pkg/*.ts ../../www/pkg/iris_wasm/ 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Build Complete! 🎉                        ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Output: iris/pkg/                                           ║"
echo "║  Test:   Open www/iris.html in a browser                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
