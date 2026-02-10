//! # Sequential Container
//!
//! Autograd-aware sequential container that chains layers.
//!
//! ```rust,ignore
//! use cma_autograd::prelude::*;
//! use cma_autograd::sequential::Sequential;
//!
//! let model = Sequential::new()
//!     .add(Conv2DModule::new(1, 6, 5, 1, 0))
//!     .add(ReLULayer::new())
//!     .add(MaxPool2DLayer::new(2, 2))
//!     .add(Flatten::new())
//!     .add(Linear::new(150, 10));
//!
//! let output = model.forward(&input);
//! let loss = cross_entropy_loss(&output, &target);
//! loss.backward();
//! ```

use crate::module::Parameter;
use crate::tensor::Tensor;
use crate::Float;

// ═══════════════════════════════════════════════════════════════════════════
// Layer trait — unified interface for Sequential
// ═══════════════════════════════════════════════════════════════════════════

/// Unified trait for all layers (trainable and stateless).
///
/// This trait is the building block for the `Sequential` container.
/// Stateless layers simply return empty `parameters()`.
pub trait Layer {
    /// Forward pass.
    fn forward(&self, input: &Tensor) -> Tensor;

    /// Trainable parameters (empty for stateless layers).
    fn parameters(&self) -> Vec<&Parameter> {
        vec![]
    }

    /// Layer name for display.
    fn name(&self) -> &'static str;

    /// Switch to training mode (for Dropout, BatchNorm).
    fn set_training(&mut self, _training: bool) {}

    /// Downcast support for weight export.
    fn as_any(&self) -> &dyn std::any::Any;
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer implementations for trainable layers
// ═══════════════════════════════════════════════════════════════════════════

impl Layer for crate::module::Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::module::Module::forward(self, input)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        crate::module::Module::parameters(self)
    }

    fn name(&self) -> &'static str {
        "Linear"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::module::Conv2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::module::Module::forward(self, input)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        crate::module::Module::parameters(self)
    }

    fn name(&self) -> &'static str {
        "Conv2D"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::layers::BatchNorm2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::BatchNorm2D::forward(self, input)
    }

    fn parameters(&self) -> Vec<&Parameter> {
        crate::layers::BatchNorm2D::parameters(self)
    }

    fn name(&self) -> &'static str {
        "BatchNorm2D"
    }

    fn set_training(&mut self, training: bool) {
        if training {
            self.train();
        } else {
            self.eval();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer implementations for stateless layers
// ═══════════════════════════════════════════════════════════════════════════

impl Layer for crate::layers::ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::ReLU::forward(self, input)
    }

    fn name(&self) -> &'static str {
        "ReLU"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::layers::Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::Sigmoid::forward(self, input)
    }

    fn name(&self) -> &'static str {
        "Sigmoid"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::layers::Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::Tanh::forward(self, input)
    }

    fn name(&self) -> &'static str {
        "Tanh"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::layers::Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::Flatten::forward(self, input)
    }

    fn name(&self) -> &'static str {
        "Flatten"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::layers::MaxPool2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::MaxPool2D::forward(self, input)
    }

    fn name(&self) -> &'static str {
        "MaxPool2D"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::layers::AvgPool2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::AvgPool2D::forward(self, input)
    }

    fn name(&self) -> &'static str {
        "AvgPool2D"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::layers::GlobalAvgPool2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::GlobalAvgPool2D::forward(self, input)
    }

    fn name(&self) -> &'static str {
        "GlobalAvgPool2D"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::layers::Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::Softmax::forward(self, input)
    }

    fn name(&self) -> &'static str {
        "Softmax"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Layer for crate::layers::Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        crate::layers::Dropout::forward(self, input)
    }

    fn name(&self) -> &'static str {
        "Dropout"
    }

    fn set_training(&mut self, training: bool) {
        if training {
            self.train();
        } else {
            self.eval();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sequential container
// ═══════════════════════════════════════════════════════════════════════════

/// Sequential container — chains layers in order.
///
/// ```rust,ignore
/// let model = Sequential::new()
///     .add(Linear::new(784, 128))
///     .add(ReLULayer::new())
///     .add(Linear::new(128, 10));
///
/// // Forward
/// let output = model.forward(&input);
///
/// // Access parameters for optimizer
/// let params = model.parameters();
///
/// // Switch train/eval
/// model.train();
/// model.eval();
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    /// Create an empty sequential container.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer (builder pattern — consumes self).
    pub fn add<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Add a layer by reference (push pattern — borrows self).
    pub fn push<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    /// Forward pass through all layers sequentially.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Collect all trainable parameters from all layers.
    pub fn parameters(&self) -> Vec<&Parameter> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    /// Total number of trainable parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Zero all gradients across all layers.
    pub fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Switch all layers to training mode.
    pub fn train(&mut self) {
        for layer in &mut self.layers {
            layer.set_training(true);
        }
    }

    /// Switch all layers to eval mode.
    pub fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.set_training(false);
        }
    }

    /// Number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the container is empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Access layers for iteration (used by export module).
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    /// Print a summary of the model architecture.
    pub fn summary(&self) {
        println!("Sequential model: {} layers", self.layers.len());
        println!("{:-<50}", "");
        for (i, layer) in self.layers.iter().enumerate() {
            let n_params: usize = layer.parameters().iter().map(|p| p.numel()).sum();
            if n_params > 0 {
                println!("  [{:2}] {:20} ({} params)", i, layer.name(), n_params);
            } else {
                println!("  [{:2}] {}", i, layer.name());
            }
        }
        println!("{:-<50}", "");
        println!("Total parameters: {}", self.num_parameters());
    }

    /// Create a training builder (fluent API).
    ///
    /// ```rust,ignore
    /// let history = model.trainer(&mut optimizer)
    ///     .train_data(&train_inputs, &train_targets)
    ///     .validation_data(&val_inputs, &val_targets)
    ///     .loss_fn(cross_entropy_loss)
    ///     .epochs(10)
    ///     .batch_size(64)
    ///     .early_stopping(3)
    ///     .fit();
    /// ```
    pub fn trainer<'a, O: crate::optim::Optimizer>(
        &'a self,
        optimizer: &'a mut O,
    ) -> CnnTrainer<'a, O> {
        CnnTrainer::new(self, optimizer)
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CnnTrainer — fluent builder for autograd training loop
// ═══════════════════════════════════════════════════════════════════════════

/// Fluent training builder for autograd Sequential models.
///
/// Created via `model.trainer(&mut optimizer)`. Configure with chained methods,
/// then call `.fit()` to run training.
///
/// ```rust,ignore
/// let history = model.trainer(&mut optimizer)
///     .train_data(&train_inputs, &train_targets)
///     .validation_data(&val_inputs, &val_targets)
///     .loss_fn(cross_entropy_loss)
///     .epochs(10)
///     .batch_size(64)
///     .early_stopping(3)
///     .fit();
/// ```
pub struct CnnTrainer<'a, O: crate::optim::Optimizer> {
    model: &'a Sequential,
    optimizer: &'a mut O,
    train_inputs: Option<&'a [ndarray::ArrayD<Float>]>,
    train_targets: Option<&'a [ndarray::ArrayD<Float>]>,
    val_inputs: Option<&'a [ndarray::ArrayD<Float>]>,
    val_targets: Option<&'a [ndarray::ArrayD<Float>]>,
    loss_fn: fn(&Tensor, &Tensor) -> Tensor,
    epochs: usize,
    batch_size: usize,
    verbose: bool,
    early_stopping_patience: usize,
}

impl<'a, O: crate::optim::Optimizer> CnnTrainer<'a, O> {
    /// Create a new trainer (prefer `model.trainer(&mut optimizer)`).
    pub fn new(model: &'a Sequential, optimizer: &'a mut O) -> Self {
        Self {
            model,
            optimizer,
            train_inputs: None,
            train_targets: None,
            val_inputs: None,
            val_targets: None,
            loss_fn: crate::loss::cross_entropy_loss,
            epochs: 10,
            batch_size: 32,
            verbose: true,
            early_stopping_patience: 0,
        }
    }

    /// Set the training data (inputs and targets).
    pub fn train_data(
        mut self,
        inputs: &'a [ndarray::ArrayD<Float>],
        targets: &'a [ndarray::ArrayD<Float>],
    ) -> Self {
        self.train_inputs = Some(inputs);
        self.train_targets = Some(targets);
        self
    }

    /// Set the validation data (inputs and targets).
    pub fn validation_data(
        mut self,
        inputs: &'a [ndarray::ArrayD<Float>],
        targets: &'a [ndarray::ArrayD<Float>],
    ) -> Self {
        self.val_inputs = Some(inputs);
        self.val_targets = Some(targets);
        self
    }

    /// Set the loss function (default: `cross_entropy_loss`).
    pub fn loss_fn(mut self, f: fn(&Tensor, &Tensor) -> Tensor) -> Self {
        self.loss_fn = f;
        self
    }

    /// Set the number of training epochs (default: 10).
    pub fn epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }

    /// Set the mini-batch size (default: 32).
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Enable/disable verbose output (default: true).
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Set early stopping patience (0 = disabled, default: 0).
    ///
    /// Stops training if validation loss doesn't improve for `patience` epochs.
    pub fn early_stopping(mut self, patience: usize) -> Self {
        self.early_stopping_patience = patience;
        self
    }

    /// Run training and return epoch metrics history.
    pub fn fit(self) -> Vec<EpochMetrics> {
        let inputs = self
            .train_inputs
            .expect("CnnTrainer: train_data() must be called before fit()");
        let targets = self
            .train_targets
            .expect("CnnTrainer: train_data() must be called before fit()");

        let validation = match (self.val_inputs, self.val_targets) {
            (Some(vi), Some(vt)) => Some((vi, vt)),
            _ => None,
        };

        train(
            self.model,
            self.optimizer,
            inputs,
            targets,
            validation,
            self.loss_fn,
            TrainerConfig {
                epochs: self.epochs,
                batch_size: self.batch_size,
                verbose: self.verbose,
                early_stopping_patience: self.early_stopping_patience,
            },
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Trainer helper
// ═══════════════════════════════════════════════════════════════════════════

/// Metrics from a single epoch.
#[derive(Debug, Clone)]
pub struct EpochMetrics {
    /// Training loss (average over batches).
    pub train_loss: Float,
    /// Validation loss (if validation data provided).
    pub val_loss: Option<Float>,
    /// Training accuracy (if classification).
    pub train_accuracy: Option<Float>,
    /// Validation accuracy (if classification).
    pub val_accuracy: Option<Float>,
}

/// Configuration for the training loop (internal).
struct TrainerConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size for mini-batch training.
    pub batch_size: usize,
    /// Whether to print progress every epoch.
    pub verbose: bool,
    /// Early stopping patience (0 = disabled).
    pub early_stopping_patience: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            verbose: true,
            early_stopping_patience: 0,
        }
    }
}

/// Internal training loop used by `CnnTrainer::fit()`.
fn train<O: crate::optim::Optimizer>(
    model: &Sequential,
    optimizer: &mut O,
    inputs: &[ndarray::ArrayD<Float>],
    targets: &[ndarray::ArrayD<Float>],
    validation: Option<(&[ndarray::ArrayD<Float>], &[ndarray::ArrayD<Float>])>,
    loss_fn: fn(&Tensor, &Tensor) -> Tensor,
    config: TrainerConfig,
) -> Vec<EpochMetrics> {
    let n = inputs.len();
    assert_eq!(n, targets.len(), "inputs and targets must have same length");

    let mut history = Vec::with_capacity(config.epochs);
    let mut best_val_loss = Float::INFINITY;
    let mut patience_counter = 0usize;

    for epoch in 0..config.epochs {
        // ── Training phase ──
        let mut total_loss: Float = 0.0;
        let mut total_correct: usize = 0;
        let mut total_samples: usize = 0;

        // Create shuffled indices
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rand::rng());

        // Mini-batch iteration
        for batch_start in (0..n).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(n);
            let batch_indices = &indices[batch_start..batch_end];

            // Accumulate gradients over mini-batch
            let mut batch_loss: Float = 0.0;

            for &idx in batch_indices {
                let x = Tensor::new(inputs[idx].clone(), true);
                let t = Tensor::new(targets[idx].clone(), false);

                let pred = model.forward(&x);
                let loss = loss_fn(&pred, &t);

                batch_loss += loss.item();

                // Check classification accuracy
                let pred_data = pred.data();
                let target_data = &targets[idx];
                if pred_data.ndim() >= 1 && pred_data.len() > 1 {
                    let pred_class = pred_data
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let true_class = target_data
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    if pred_class == true_class {
                        total_correct += 1;
                    }
                }

                loss.backward();
                total_samples += 1;
            }

            optimizer.step();
            optimizer.zero_grad();

            total_loss += batch_loss;
        }

        let train_loss = total_loss / total_samples as Float;
        let train_accuracy = if total_samples > 0 {
            Some(total_correct as Float / total_samples as Float)
        } else {
            None
        };

        // ── Validation phase ──
        let (val_loss, val_accuracy) = if let Some((val_x, val_t)) = validation {
            crate::engine::no_grad(|| {
                let mut vloss: Float = 0.0;
                let mut vcorrect: usize = 0;

                for (x_data, t_data) in val_x.iter().zip(val_t.iter()) {
                    let x = Tensor::new(x_data.clone(), false);
                    let t = Tensor::new(t_data.clone(), false);
                    let pred = model.forward(&x);
                    let loss = loss_fn(&pred, &t);
                    vloss += loss.item();

                    // Classification accuracy
                    let pred_data = pred.data();
                    if pred_data.len() > 1 {
                        let pc = pred_data
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let tc = t_data
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        if pc == tc {
                            vcorrect += 1;
                        }
                    }
                }

                let vl = vloss / val_x.len() as Float;
                let va = vcorrect as Float / val_x.len() as Float;
                (Some(vl), Some(va))
            })
        } else {
            (None, None)
        };

        let metrics = EpochMetrics {
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
        };

        if config.verbose {
            print!(
                "Epoch {:3}/{}: train_loss={:.4}",
                epoch + 1,
                config.epochs,
                train_loss
            );
            if let Some(acc) = train_accuracy {
                print!(", train_acc={:.2}%", acc * 100.0);
            }
            if let Some(vl) = val_loss {
                print!(", val_loss={:.4}", vl);
            }
            if let Some(va) = val_accuracy {
                print!(", val_acc={:.2}%", va * 100.0);
            }
            println!();
        }

        history.push(metrics);

        // Early stopping
        if config.early_stopping_patience > 0 {
            if let Some(vl) = val_loss {
                if vl < best_val_loss - 1e-4 {
                    best_val_loss = vl;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= config.early_stopping_patience {
                        if config.verbose {
                            println!(
                                "Early stopping at epoch {} (patience={})",
                                epoch + 1,
                                config.early_stopping_patience
                            );
                        }
                        break;
                    }
                }
            }
        }
    }

    history
}
