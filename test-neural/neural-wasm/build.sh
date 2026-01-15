#!/bin/bash
set -e
echo "Building neural-wasm..."
wasm-pack build --target web --out-dir pkg
echo "Done!"
