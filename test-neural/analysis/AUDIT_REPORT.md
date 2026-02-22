# Full Codebase Audit Report

> Generated: 2026-02-21 — Branch: `cma/cleanup`

## Summary

| Category | Severity | Count |
|---|---|---|
| **Bug** (gradient disconnection) | Critical | 1 |
| **WASM compat** (`std::time::Instant` unguarded) | Critical | 1 |
| **Code duplication** | Critical | ~2500 lines |
| **French text** | High | ~35 locations |
| **Functions with 4+ args** | High | ~20 functions |
| **Dead code** | High | ~10 items |
| **Performance** (scalar loops, clones, allocations) | Medium | ~30 sites |
| **Missing docs on public items** | Medium | ~60+ items |
| **Missing unit tests** | Medium | ~10 modules with 0 tests |
| **f64 instead of Float** | Low | ~5 sites |

---

## Step 1 — Fix the correctness bug

- **File**: `cma-autograd/src/loss.rs` (~L121-128)
- **Issue**: `binary_cross_entropy_loss` creates a new leaf `Tensor` from clamped data, disconnecting it from the computation graph. Gradients do **not** flow back through prediction.
- **Fix**: Use an in-graph clamp operation or preserve the grad chain.

---

## Step 2 — Fix WASM compatibility

- **File**: `cma-neural-network/src/callbacks.rs` (~L618, L630, L648)
- **Issue**: `ProgressBar` uses `std::time::Instant` without a `#[cfg(not(target_arch = "wasm32"))]` guard.
- **Fix**: Wrap all `Instant` usage or provide a WASM-compatible fallback.

---

## Step 3 — Eliminate massive code duplication

### 3a. neural-wasm CNN lib.rs (~900 duplicated lines)

`neural-wasm/mnist-lenet/src/lib.rs`, `neural-wasm/mnist-alexnet/src/lib.rs`, `neural-wasm/mnist-resnet/src/lib.rs`, `neural-wasm/mnist-vgg/src/lib.rs` are ~95% identical.

Only differences:

| What varies | LeNet | AlexNet | ResNet | VGG |
|---|---|---|---|---|
| Struct name | `MnistLeNetNetwork` | `MnistAlexNetNetwork` | `MnistResNetNetwork` | `MnistVGGNetwork` |
| Model binary | `lenet_model.bin` | `alexnet_model.bin` | `resnet_model.bin` | `vgg_model.bin` |
| Display name | `"LeNet-5 MNIST"` | `"AlexNet-Mini MNIST"` | `"ResNet-MNIST"` | `"VGG-Tiny MNIST"` |
| `output_features` | 120 | 2304 | 64 | 3136 |

**Fix**: Create a generic `CnnMnistNetwork` in `neural-wasm/shared` parameterized by model metadata, or use a declarative macro.

### 3b. neural-wasm CNN training scripts (~800 duplicated lines)

`train_lenet.rs`, `train_alexnet.rs`, `train_resnet.rs`, `train_vgg.rs` share identical data loading, normalization, shuffle, training loop, export, evaluation, and `extract_cnn_features()`.

**Fix**: Extract a `train_cnn_mnist(config: TrainConfig)` function in shared.

### 3c. `normalize_features_with_stats` — triplicated

Exists in 3 places:
1. `neural-wasm/shared/src/lib.rs` (~L527)
2. `neural-wasm/iris/src/train_iris.rs` (~L162)
3. `neural-wasm/mnist/src/train_mnist.rs` (~L164)

**Fix**: Remove the two copies, use the shared version.

### 3d. `load_mnist_from_csv` — duplicated

In `shared/src/lib.rs` (~L497) and `mnist/src/train_mnist.rs` (~L127).

**Fix**: Remove the train_mnist copy.

### 3e. Box-Muller random sampling — 7 copies across 3 crates

| Location | Function |
|---|---|
| `cma-neural-network/src/network.rs` | `WeightInit::Xavier` (~L124) |
| `cma-neural-network/src/network.rs` | `WeightInit::He` (~L135) |
| `cma-neural-network/src/network.rs` | `WeightInit::LeCun` (~L146) |
| `cma-cnn/src/layers.rs` | `Conv2D::new()` (~L122) |
| `cma-autograd/src/module.rs` | `Parameter::he_init()` (~L85) |
| `cma-autograd/src/module.rs` | `Parameter::xavier_init()` (~L101) |
| `cma-autograd/src/tensor.rs` | `Tensor::randn()` (~L127) |

**Fix**: Extract a `randn(shape, std_dev)` utility into `cma-neural-network` and reuse everywhere.

### 3f. OptimizerState1D / OptimizerState2D duplication (~150 lines)

In `cma-neural-network/src/optimizer.rs`: `step()` is copy-pasted for `Array1` and `Array2`.

**Fix**: Use a generic function or macro over ndarray dimensions.

### 3g. Legacy ResNet18/34/50 structs (~470 lines)

In `cma-models/src/resnet.rs` (~L557-L1023): Three individual structs superseded by the generic `ResNet` + `ResNetBuilder`.

**Fix**: Remove entirely and update call sites.

### 3h. Activation scalar logic duplicated

`cma-cnn/src/layers.rs` `apply_activation_scalar()` (~L920-L974) duplicates `cma-neural-network`'s `Activation::apply()`.

**Fix**: Add `Activation::apply_scalar(x: Float) -> Float` and call from cma-cnn.

### 3i. `get_mnist_test_samples` — 5 copies

In `neural-wasm/mnist/src/lib.rs` and all 4 CNN lib.rs files.

**Fix**: Move to shared.

### 3j. `get_weights()` boilerplate — 7 copies

Identical in all 7 WASM crates.

**Fix**: Move to a shared utility function.

---

## Step 4 — Translate all French text to English

~35 locations across 4 crates:

### cma-cnn (~8 locations)

| File | Line | French | English |
|---|---|---|---|
| `src/ops.rs` | ~L18 | `Mode de padding pour les convolutions` | `Padding mode for convolutions` |
| `src/ops.rs` | ~L44 | `# Principe (LeCun et al., 1998)` | `# Principle (LeCun et al., 1998)` |
| `src/ops.rs` | ~L131 | `// Safe car on check ih_valid` | `// Safe because we check ih_valid` |
| `src/ops.rs` | ~L371 | `// Im2col pour cette image` | `// Im2col for this image` |
| `src/layers.rs` | ~L76 | `/// # Exemple` | `/// # Example` |
| `src/sequential.rs` | ~L6 | `/// ## Exemple` | `/// ## Example` |
| `src/sequential.rs` | ~L100 | `/// # Exemple (LeNet-5 style)` | `/// # Example (LeNet-5 style)` |
| `src/sequential.rs` | ~L262 | `/// # Exemple` | `/// # Example` |

### cma-models (~12 locations)

| File | Line | French | English |
|---|---|---|---|
| `src/lenet.rs` | ~L53 | `/// ## Exemple` | `/// ## Example` |
| `src/lenet.rs` | ~L168 | `/// # Exemple` | `/// # Example` |
| `src/lenet.rs` | ~L277 | `/// # Exemple` | `/// # Example` |
| `src/lenet.rs` (test) | ~L309 | `// Pour 32x32:` | `// For 32x32:` |
| `src/resnet.rs` | ~L38 | `/// ## Variantes` | `/// ## Variants` |
| `src/resnet.rs` | ~L178 | `/// Stride (1 ou 2)` | `/// Stride (1 or 2)` |
| `src/efficientnet.rs` | ~L16 | `Scaling traditionnel vs EfficientNet` | `Traditional scaling vs EfficientNet` |
| `src/efficientnet.rs` | ~L33 | `Formules de scaling:` / `Avec α × β²...` | `Scaling formulas:` / `With α × β²...` |
| `examples/efficientnet_paper.rs` | ~L355 | `"Classifieur FC"` | `"FC Classifier"` |
| `examples/efficientnet_paper.rs` | ~L363 | `"Type de bloc"` / `"ou"` | `"Block type"` / `"or"` |
| `examples/resnet_paper.rs` | ~L208 | `"Figure 5 du paper"` | `"Figure 5 of the paper"` |

### cma-autograd (~2 locations)

| File | Line | French | English |
|---|---|---|---|
| `src/engine.rs` | ~L3-7 | `Orchestration du backward pass`, `Tri topologique du graphe de calcul`, `Propagation des gradients`, `Accumulation des gradients dans les tenseurs feuilles` | Translate all to English |
| `src/loss.rs` | ~L4 | `MSE et Cross-Entropy avec support autograd.` | `MSE and Cross-Entropy with autograd support.` |

### cma-neural-network (~1 location)

| File | Line | French | English |
|---|---|---|---|
| `src/optimizer.rs` | ~L139 | `// In-place: m *= beta, puis m += gradient` | `// In-place: m *= beta, then m += gradient` |

---

## Step 5 — Introduce Builder Patterns for 4+ arg functions

By priority (most-called first):

| Function | File | Args | Recommended fix |
|---|---|---|---|
| `Conv2D::new()` | `cma-cnn/src/layers.rs`, `cma-autograd/src/module.rs` | 5 | `Conv2DBuilder` |
| `MBConvBlock::new()` | `cma-models/src/efficientnet.rs` | 6 | `MBConvBlockBuilder` |
| `CnnBuilder::conv_bn_relu_pool()` | `cma-autograd/src/builder.rs` | 7 | Conv config struct |
| `save_cnn_model_binary()` | `neural-wasm/shared/src/lib.rs` | 6 | `ModelSaver` builder |
| `Network::new_deep_with_init()` | `cma-neural-network/src/network.rs` | 9 | Internal config struct |
| `Network::fit()` | `cma-neural-network/src/network.rs` | 8 | Internal config struct |
| `col2im()` | `cma-autograd/src/grad_fn.rs` | 8 | `Col2ImConfig` struct |
| `conv2d_im2col()` | `cma-cnn/src/ops.rs` | 5 | Conv config struct |
| `conv2d_naive()` | `cma-cnn/src/ops.rs` | 5 | Same config struct |
| `im2col_single()` | `cma-cnn/src/ops.rs` | 5 | Im2col config struct |
| `col2im()` | `cma-cnn/src/ops.rs` | 5 | Same struct |
| `one_cycle_lr()` | `cma-neural-network/src/callbacks.rs` | 5 | Internal method (acceptable) |
| Additional `CnnBuilder::conv2d*` variants | `cma-autograd/src/builder.rs` | 5 each | Conv config struct |

---

## Step 6 — Remove dead code

| Item | Location | Action |
|---|---|---|
| `forward`, `forward_with_stored_rng`, `forward_full` | `cma-neural-network/src/network.rs` (~L725-L754) | Delete (marked `#[allow(dead_code)]`) |
| `BatchGradients` struct | `cma-neural-network/src/trainer.rs` (~L28-L33) | Delete |
| `im2col()` function | `cma-cnn/src/ops.rs` (~L153) | Delete (marked `#[allow(dead_code)]`) |
| `Flatten::input_shape` field | `cma-cnn/src/layers.rs` (~L835) | Delete (always `None`) |
| `create_alexnet_classifier()` | `cma-models/src/alexnet.rs` (~L345) | Delete (returns code-as-string) |
| `create_lenet5_classifier()` | `cma-models/src/lenet.rs` (~L260) | Delete (returns code-as-string) |
| `use_bottleneck` field | `cma-models/src/resnet.rs` | Delete (never checked) |
| `parallel` feature (unused) | `cma-autograd/Cargo.toml` | Remove or implement |
| `test_iris_norm.rs` | `src/bin/test_iris_norm.rs` | Delete (references deleted JSON format) |

---

## Step 7 — Performance improvements

### High impact

| File | Location | Issue | Fix |
|---|---|---|---|
| `cma-autograd/src/tensor.rs` | ~L158 | `data()` clones entire `ArrayD` on every call | Return `Arc<ArrayD>` or read guard |
| `cma-autograd/src/tensor.rs` | ~L171 | `shape()` returns `Vec<usize>` (allocates) | Return `&[usize]` |
| `cma-neural-network/src/network.rs` | ~L226-L254 | Loss `compute()` uses scalar loops | Use `ndarray::Zip` or `mapv` |
| `cma-neural-network/src/network.rs` | ~L289-L311 | Loss `derivative()` uses manual indexing | Use `ndarray::Zip` |
| `cma-neural-network/src/dataset.rs` | ~L119-L133 | `shuffle_with_indices` clones all data | In-place permutation |
| `cma-autograd/src/layers.rs` | ~L378-L465 | `BatchNorm2D::forward` triple-nested scalar loops | Delegate to `cma-cnn` optimized impl |
| `cma-autograd/src/cnn_ops.rs` | ~L31-L41 | `arrayd_to_array4` double allocation (clone + into_dimensionality) | Single conversion |

### Medium impact

| File | Location | Issue | Fix |
|---|---|---|---|
| `cma-neural-network/src/network.rs` | ~L912 | `evaluate()` takes `&Vec<T>` | Change to `&[T]` |
| `cma-neural-network/src/dataset.rs` | ~L81, L86 | `inputs()`/`targets()` return `&Vec<T>` | Return `&[T]` |
| `cma-neural-network/src/io.rs` | ~L174-L183 | `get_serialized_size` allocates full JSON just to measure length | Use `serde_json::to_writer(io::sink())` |
| `cma-neural-network/src/metrics.rs` | ~L72-L95 | `accuracy` uses scalar loops | Vectorize with ndarray |
| `cma-autograd/src/engine.rs` | ~L44-L47 | `grad_output` cloned from HashMap even when last use | Use `.remove()` on last use |
| Neural-wasm crates | Multiple | `class_names` Vec recreated on every `predict()` call | Cache in struct at construction |
| Neural-wasm training | Multiple | `iter().map().clone().collect()` for split | Use `unzip()` |

---

## Step 8 — Add missing documentation and tests

### Documentation (~60+ public items without `///`)

| Crate | Items |
|---|---|
| `cma-autograd` | All layer `forward()`, `name()`, `as_any()` methods; `Linear::new`, `Conv2D::new`, `Sequential::push`, `SGD::with_momentum` |
| `cma-cnn` | `MaxPool2D::new`, `AvgPool2D::new`, `BatchNorm2D::eval_mode/train_mode`, `Dropout2D::eval_mode/train_mode` |
| `cma-neural-network` | `Activation` enum variants, `BinaryMetrics` fields, `ForwardResult` fields |
| `cma-models` | `ResNet34/50` methods, `VGG19::with_config` |
| `neural-wasm` | All `#[wasm_bindgen]` structs and methods, `ModelInfo`, `LayerInfo`, `WeightsInfo` |

### Tests (modules with 0 unit tests)

| Module | Lines | Priority |
|---|---|---|
| `cma-neural-network/src/network.rs` | 1115 | High |
| `cma-neural-network/src/trainer.rs` | 443 | High |
| `neural-wasm/xor` | — | Medium |
| `neural-wasm/iris` | — | Medium |
| `neural-wasm/mnist` | — | Medium |
| `neural-wasm/mnist-lenet` | — | Low |
| `neural-wasm/mnist-alexnet` | — | Low |
| `neural-wasm/mnist-resnet` | — | Low |
| `neural-wasm/mnist-vgg` | — | Low |
| `cma-cnn`: `col2im`, `Dropout2D`, non-ReLU activations | — | Medium |
| `cma-models`: `VGG19` forward, `ResNetBuilder::imagenet()` | — | Medium |

---

## Other Notable Issues

| Issue | Location | Details |
|---|---|---|
| **Unsafe code** | `cma-autograd/src/module.rs` (~L38-44) | `UnsafeCell` with manual `unsafe impl Send + Sync`. Fragile if parallelism is added. Consider `RwLock`. |
| **Softmax non-differentiable** | `cma-autograd/src/layers.rs` (~L276-285) | Returns `requires_grad=false`, silently breaking autograd. |
| **Missing feature forwarding** | `cma-models/Cargo.toml` | Doesn't forward `parallel` to `cma-cnn`. |
| **String as enum** | `cma-models/src/lenet.rs` (`LeNet5Config::activation`) | `String` matching is fragile. Should be an activation enum. |
| **Fake bottleneck** | `cma-models/src/resnet.rs` (`ResNet50`) | `use_bottleneck` field is never checked. No BottleneckBlock impl exists. |
| **Fake depthwise conv** | `cma-models/src/efficientnet.rs` (~L240) | Claims "depthwise" but uses standard conv (no groups). |
| **Cross-crate BatchNorm2D** | `cma-autograd` vs `cma-cnn` | Fully duplicated, different quality. Autograd version should delegate forward to cma-cnn. |
| **TensorShape not reused** | `cma-autograd` | Recomputes conv output dims inline instead of using `cma-cnn::TensorShape`. |
| **Hardcoded `std::f32::consts::PI`** | `cma-neural-network/src/network.rs` | Fragile if `Float` type changes from f32. |

---

## Verification Checklist

After each step:

- [ ] `cargo test --workspace` — all tests pass
- [ ] `cargo clippy --workspace -- -D warnings` — no warnings
- [ ] `cargo check --workspace --target wasm32-unknown-unknown` — WASM builds (after Step 2)
- [ ] `cargo fmt --all -- --check` — formatting clean
- [ ] Manual review of gradient check tests (after Step 1)
