// Stub module for AlexNet-Mini MNIST WASM
// This is a placeholder that will be replaced when the model is trained

/**
 * @typedef {Object} InitOutput
 * @property {WebAssembly.Memory} memory
 */

/**
 * AlexNet-Mini MNIST Network (stub)
 */
class MnistAlexNetNetwork {
  constructor() {
    throw new Error('AlexNet-Mini model not trained yet. Run: cd neural-wasm/mnist-alexnet && cargo run --bin train_alexnet --features training --release');
  }
  
  predict(_pixels) {
    return '{"error": "Model not trained"}';
  }
  
  model_info() {
    return '{"name": "AlexNet-Mini (Not Trained)", "accuracy": 0, "description": "Model not trained yet"}';
  }
  
  get_cnn_summary() {
    return '{"error": "Model not trained"}';
  }
  
  get_weights() {
    return '{"layers": []}';
  }
  
  test_all() {
    return '[]';
  }
}

/**
 * Initialize the WASM module
 * @param {string|URL|Request|RequestInfo} [_input]
 * @returns {Promise<InitOutput>}
 */
async function init(_input) {
  throw new Error('AlexNet-Mini WASM module not built. Train the model first.');
}

export { MnistAlexNetNetwork };
export default init;
