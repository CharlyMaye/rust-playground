# 🧠 Neural Networks - Web Demos

Interactive web interfaces for testing neural network models via WebAssembly.

## 🎯 Available Demos

- **[XOR Logic Gate](xor.html)** - Binary classification (2 inputs → 1 output)
- **[Iris Classifier](iris.html)** - Multi-class classification (4 inputs → 3 classes)
- **MNIST Digits** - Coming soon

## 🚀 Quick Start

### 1. Build WASM Modules

````bash
cd ../neural-wasm
./build_all.sh
````

This builds all models and copies them to `www/pkg/`.

### 2. Start Web Server

````bash
cd ../www
npx http-server -p 8080 -c-1 --host 0.0.0.0
````

| Option | Description |
|--------|-------------|
| `-p 8080` | Server port |
| `-c-1` | Disable caching (useful for development) |
| `--host 0.0.0.0` | Listen on all interfaces (required for containers) |

### 3. Open Browser

- **Local**: http://localhost:8080
- **Network**: http://\<IP\>:8080

## 📂 Structure

````
www/
├── index.html          # Homepage with demo cards
├── xor.html            # XOR demo
├── iris.html           # Iris demo
├── shared/
│   └── styles.css      # Common styles
└── pkg/
    ├── xor_wasm/       # XOR WebAssembly module
    │   ├── neural_wasm_xor.js
    │   └── neural_wasm_xor_bg.wasm
    └── iris_wasm/      # Iris WebAssembly module
        ├── neural_wasm_iris.js
        └── neural_wasm_iris_bg.wasm
````

## � API Examples

### XOR Demo

````javascript
import init, { XorNetwork } from './pkg/xor_wasm/neural_wasm_xor.js';

await init();
const network = new XorNetwork();

// Binary prediction (0 or 1)
network.predict(0, 1);  // → 1

// Raw value (0.0 - 1.0)
network.predict_raw(0, 1);  // → 0.9987

// Confidence percentage
network.confidence(0, 1);  // → 99.7

// Test all combinations
const results = JSON.parse(network.test_all());

// Model info
network.model_info();  // → "XOR Network: 2 → [8] → 1"
````

### Iris Demo

````javascript
import init, { IrisClassifier } from './pkg/iris_wasm/neural_wasm_iris.js';

await init();
const classifier = new IrisClassifier();

// Predict iris species
const result = JSON.parse(
    classifier.predict(5.1, 3.5, 1.4, 0.2)
);
console.log(result);
// {
//   class: "Setosa",
//   class_idx: 0,
//   probabilities: [0.98, 0.01, 0.01],
//   confidence: 98.0
// }

// Test all samples
const testResults = JSON.parse(classifier.test_all());

// Model info
const info = JSON.parse(classifier.model_info());
````

## 🎨 Features

### XOR Demo
- ⚡ Interactive binary inputs
- 📊 Truth table visualization
- 🔬 Network visualization with activations
- 🎯 Real-time confidence display

### Iris Demo
- 🌸 Interactive measurement inputs
- 📊 Probability bars for each class
- 🧪 Test all samples button
- 🎯 Quick preset buttons (Setosa, Versicolor, Virginica)

## 🛠️ Development

### Hot Reload

Use `-c-1` flag to disable caching during development:

````bash
npx http-server -p 8080 -c-1
````

### Rebuilding WASM

After modifying Rust code:

````bash
cd ../neural-wasm
./build_all.sh  # or build individual module
````

The WASM files are automatically copied to `www/pkg/`.

## 📦 Deployment to GitHub Pages

See the main repository README for GitHub Actions workflow to deploy to GitHub Pages.

## 🌐 Browser Support

- Chrome/Edge (recommended)
- Firefox
- Safari
- Any modern browser with WebAssembly support

## 📝 Notes

- All models run entirely in the browser
- No backend required
- WASM files are ~220KB each (optimized)
- Models are embedded in WASM (no additional network requests)
