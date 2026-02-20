#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        Neural Network WebAssembly - Build All                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

cd "$(dirname "$0")"

# Build shared library first
echo ""
echo "📦 Building shared library..."
cd shared
cargo build --release
cd ..

echo ""
echo "✅ Shared library built"

# Build each model - ordered by training time (fastest first)
# XOR: ~1 second
# Iris: ~5 seconds
# MNIST (MLP): ~30 seconds
# MNIST-LeNet: ~2 minutes
# MNIST-ResNet: ~5-10 minutes (with residual blocks)
# MNIST-VGG: ~10-15 minutes
# MNIST-AlexNet: ~15-20 minutes
MODELS=("xor" "iris" "mnist" "mnist-lenet" "mnist-resnet" "mnist-vgg" "mnist-alexnet")
SUCCESS_COUNT=0
FAIL_COUNT=0

for model in "${MODELS[@]}"; do
    if [ -d "$model" ]; then
        echo ""
        echo "╔══════════════════════════════════════════════════════════════╗"
        echo "║  Building: $model"
        echo "╚══════════════════════════════════════════════════════════════╝"
        
        cd "$model"
        if ./build.sh; then
            echo ""
            echo "✅ $model built successfully"
            
            # Determine package name (convert - to _)
            PKG_NAME="${model//-/_}_wasm"
            
            # Copy to Angular app pkg
            echo "📋 Copying $model to ai-web-app/pkg/${PKG_NAME}/..."
            mkdir -p "../../ai-web-app/pkg/${PKG_NAME}"
            cp -r pkg/* "../../ai-web-app/pkg/${PKG_NAME}/"
            
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo ""
            echo "❌ $model build failed"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
        cd ..
    else
        echo ""
        echo "⚠️  Skipping $model (directory not found)"
    fi
done

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Build Summary                             ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  ✅ Successful: $SUCCESS_COUNT                                         ║"
echo "║  ❌ Failed:     $FAIL_COUNT                                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"

if [ $FAIL_COUNT -eq 0 ]; then
    echo "║  🎉 All modules built successfully!                          ║"
    echo "║                                                              ║"
    echo "║  🌐 Start Angular app:                                       ║"
    echo "║     cd ../ai-web-app && npm start                            ║"
    echo "║                                                              ║"
    echo "║  📱 Then open: http://localhost:4200                         ║"
else
    echo "║  ⚠️  Some modules failed to build                            ║"
fi

echo "╚══════════════════════════════════════════════════════════════╝"

exit $FAIL_COUNT
