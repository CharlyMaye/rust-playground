// Stub module for VGG-Tiny MNIST WASM
// This is a placeholder that will be replaced when the model is trained

/**
 * @typedef {Object} InitOutput
 * @property {WebAssembly.Memory} memory
 */

/**
 * VGG-Tiny MNIST Network (stub)
 */
class MnistVggNetwork {
  constructor() {
    throw new Error('VGG-Tiny model not trained yet. Run: cd neural-wasm/mnist-vgg && cargo run --bin train_vgg --features training --release');
  }
  
  predict(_pixels) {
    return '{"error": "Model not trained"}';
  }
  
  model_info() {
    return '{"name": "VGG-Tiny (Not Trained)", "accuracy": 0, "description": "Model not trained yet"}';
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
  throw new Error('VGG-Tiny WASM module not built. Train the model first.');
}

export { MnistVggNetwork };
export default init;
