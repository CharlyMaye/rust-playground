# CNN Visualization — Implementation Plan

## Goal

Add a `get_cnn_activations(pixels)` API to **all** WASM modules with a **unified signature**.
CNN modules return intermediate feature maps; non-CNN modules return an error.

---

## Phase 1 — Rust / WASM API

### 1.1 Add shared types in `neural-wasm/shared`

- [x] `CnnLayerActivation` — one CNN layer's output metadata + data
  ```rust
  pub struct CnnLayerActivation {
      pub layer_type: String,    // "Conv2D", "ReLU", "MaxPool2D", "BatchNorm2D", "Flatten"
      pub config: String,        // "1→32, 3×3, s=1, p=1" for Conv2D, "2×2, s=2" for Pool, etc.
      pub shape: Vec<usize>,     // [channels, height, width] (or [1, 1, flat_size] after Flatten)
      pub activations: Vec<f32>, // flattened NCHW data for ONE sample (C×H×W values)
  }
  ```
- [x] `CnnActivationsResponse` — full CNN forward pass result
  ```rust
  pub struct CnnActivationsResponse {
      pub input_shape: Vec<usize>,       // [1, 28, 28]
      pub layers: Vec<CnnLayerActivation>,
      pub output_shape: Vec<usize>,      // shape of last CNN layer output
  }
  ```
- [x] Helper `build_cnn_activations(cnn: &CnnSequential, input: &Tensor4D) -> CnnActivationsResponse`
  - Runs forward pass layer by layer
  - Collects intermediate output at each layer
  - Extracts layer metadata (type, config) from `BoxedLayer`

### 1.2 Add `forward_with_intermediates` to `cma-cnn::Sequential`

- [x] New method on `Sequential`:

  ```rust
  pub fn forward_with_intermediates(&self, input: &Tensor4D) -> Vec<(String, Tensor4D)>
  ```

  - Returns vec of `(layer_summary, output_tensor)` for each layer
  - Enables the shared helper to collect activations without re-implementing forward pass

### 1.3 Unified `get_cnn_activations` on all WASM modules

Identical signature: `pub fn get_cnn_activations(&self, pixels: &[Float]) -> String`

- [x] **CNN modules** (LeNet, AlexNet, VGG, ResNet):
  - Reshape `pixels` to `Tensor4D [1, 1, 28, 28]`, normalize if needed
  - Call `build_cnn_activations(&self.cnn, &tensor)`
  - Serialize to JSON, return
- [x] **Non-CNN modules** (XOR, Iris, MNIST FC):
  - Return JSON error: `{"error": "This model has no CNN layers"}`

### 1.4 Enrich `get_architecture` on CNN modules

- [x] Extract **real config** from `BoxedLayer` fields instead of hardcoded strings
  - Conv2D: `"in_ch→out_ch, kernel×kernel, stride=s, pad=p"`
  - Pool: `"pool×pool, stride=s"`
  - BatchNorm: `"features=N"`
  - Activation: actual activation name from `ActivationLayer.activation.name()`
  - Flatten: `"C×H×W → flat_size"`
- [x] Compute and set `num_parameters` from `cnn.num_parameters() + classifier.num_parameters()`

---

## Phase 2 — Angular / Frontend

- [x] Add `CnnActivationsResponse` TypeScript type in `model-info.ts`
- [x] Build CNN feature maps component (`app-cnn-feature-maps`)
- [x] Integrate into CNN pages (LeNet, AlexNet, VGG, ResNet)

---

## Files to modify

| File                                   | Change                                                                        |
| -------------------------------------- | ----------------------------------------------------------------------------- |
| `cma-cnn/src/sequential.rs`            | Add `forward_with_intermediates()`                                            |
| `neural-wasm/shared/src/lib.rs`        | Add `CnnLayerActivation`, `CnnActivationsResponse`, `build_cnn_activations()` |
| `neural-wasm/mnist-lenet/src/lib.rs`   | Add `get_cnn_activations()`, enrich `get_architecture()`                      |
| `neural-wasm/mnist-alexnet/src/lib.rs` | Add `get_cnn_activations()`, enrich `get_architecture()`                      |
| `neural-wasm/mnist-vgg/src/lib.rs`     | Add `get_cnn_activations()`, enrich `get_architecture()`                      |
| `neural-wasm/mnist-resnet/src/lib.rs`  | Add `get_cnn_activations()`, enrich `get_architecture()`                      |
| `neural-wasm/xor/src/lib.rs`           | Add `get_cnn_activations()` → error                                           |
| `neural-wasm/iris/src/lib.rs`          | Add `get_cnn_activations()` → error                                           |
| `neural-wasm/mnist/src/lib.rs`         | Add `get_cnn_activations()` → error                                           |
