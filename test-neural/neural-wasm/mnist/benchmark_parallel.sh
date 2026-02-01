#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          MNIST Parallel vs Sequential Benchmark              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

EPOCHS=20

echo "📊 Test 1: Sequential (Cpu)"
echo "════════════════════════════════════════════════════════════════"
rm -f src/mnist_model.bin src/mnist_model.json
sed -i 's/.parallel()/\/\/.parallel()/' src/train_mnist.rs
sed -i "s/let epochs = [0-9_]*;/let epochs = $EPOCHS;/" src/train_mnist.rs
cargo build --bin train_mnist --release 2>&1 >/dev/null
echo "Starting..."
time cargo run --bin train_mnist --release 2>&1 | tail -8
echo ""

echo "📊 Test 2: Parallel (CpuParallel)"
echo "════════════════════════════════════════════════════════════════"
rm -f src/mnist_model.bin src/mnist_model.json
sed -i 's/\/\/.parallel()/.parallel()/' src/train_mnist.rs
cargo build --bin train_mnist --release 2>&1 >/dev/null
echo "Starting..."
time cargo run --bin train_mnist --release 2>&1 | tail -8
echo ""

rm -f src/mnist_model.bin src/mnist_model.json
sed -i "s/let epochs = $EPOCHS;/let epochs = 2_000;/" src/train_mnist.rs
echo "✅ Done!"
