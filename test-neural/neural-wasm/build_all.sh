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

# Build each model
MODELS=("xor" "iris")
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
            
            # Copy to www
            echo "📋 Copying $model to www/pkg/${model}_wasm/..."
            mkdir -p "../../www/pkg/${model}_wasm"
            cp -r pkg/* "../../www/pkg/${model}_wasm/"
            
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
echo "║  ✅ Successful: $SUCCESS_COUNT                                           ║"
echo "║  ❌ Failed:     $FAIL_COUNT                                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"

if [ $FAIL_COUNT -eq 0 ]; then
    echo "║  🎉 All modules built successfully!                         ║"
    echo "║                                                              ║"
    echo "║  🌐 Start a web server:                                     ║"
    echo "║     cd ../www && npx http-server -p 8080 -c-1               ║"
    echo "║                                                              ║"
    echo "║  📱 Then open: http://localhost:8080                        ║"
else
    echo "║  ⚠️  Some modules failed to build                           ║"
fi

echo "╚══════════════════════════════════════════════════════════════╝"

exit $FAIL_COUNT
