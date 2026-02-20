//! Internal training logic for neural networks.
//!
//! This module contains the core training algorithms (backpropagation, gradient
//! accumulation) extracted from the Network struct. This separation allows:
//!
//! - Clean architecture: Network describes math, Trainer executes computation
//! - Future extensibility: Different compute backends (CPU multi-thread, GPU)
//! - Testability: Training logic can be tested independently
//!
//! This module is internal (`pub(crate)`) and not exposed in the public API.
//! Users interact with training through `Network::train()`, `Network::train_batch()`,
//! or the `TrainingBuilder` fluent interface.

use ndarray::{Array1, Array2, Axis};
use rand::rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::Float;
use crate::compute::{ComputeDevice, ComputeDeviceError};
use crate::network::{Activation, ForwardResult, LossFunction, Network};

/// Accumulated gradients for a training batch.
///
/// Contains the gradients for all layers, ready to be applied via the optimizer.
#[allow(dead_code)]
pub(crate) struct BatchGradients {
    /// Weight gradients for each layer (averaged over batch)
    pub weights: Vec<Array2<Float>>,
    /// Bias gradients for each layer (averaged over batch)
    pub biases: Vec<Array1<Float>>,
}

/// Internal trainer that executes the training logic.
///
/// This struct is created temporarily during training and released after.
/// It doesn't own the network, just borrows it mutably.
///
/// Pre-allocates gradient buffers to avoid repeated allocations during training.
pub(crate) struct Trainer<'a> {
    network: &'a mut Network,
    device: ComputeDevice,
    /// Pre-allocated gradient buffers for weights (reused across batches)
    accumulated_weights: Vec<Array2<Float>>,
    /// Pre-allocated gradient buffers for biases (reused across batches)
    accumulated_biases: Vec<Array1<Float>>,
}

impl<'a> Trainer<'a> {
    /// Gets a reference to the network.
    pub fn network(&self) -> &Network {
        self.network
    }

    /// Gets a mutable reference to the network.
    pub fn network_mut(&mut self) -> &mut Network {
        self.network
    }

    /// Creates a new trainer for the given network.
    #[allow(dead_code)]
    pub fn new(
        network: &'a mut Network,
        device: ComputeDevice,
    ) -> Result<Self, ComputeDeviceError> {
        device.validate()?;

        // Pre-allocate gradient buffers based on network architecture
        let accumulated_weights: Vec<Array2<Float>> = network
            .layers
            .iter()
            .map(|layer| Array2::zeros(layer.weights.dim()))
            .collect();

        let accumulated_biases: Vec<Array1<Float>> = network
            .layers
            .iter()
            .map(|layer| Array1::zeros(layer.biases.dim()))
            .collect();

        Ok(Self {
            network,
            device,
            accumulated_weights,
            accumulated_biases,
        })
    }

    /// Creates a new trainer with CPU device (infallible).
    pub fn cpu(network: &'a mut Network) -> Self {
        // Pre-allocate gradient buffers based on network architecture
        let accumulated_weights: Vec<Array2<Float>> = network
            .layers
            .iter()
            .map(|layer| Array2::zeros(layer.weights.dim()))
            .collect();

        let accumulated_biases: Vec<Array1<Float>> = network
            .layers
            .iter()
            .map(|layer| Array1::zeros(layer.biases.dim()))
            .collect();

        Self {
            network,
            device: ComputeDevice::Cpu,
            accumulated_weights,
            accumulated_biases,
        }
    }

    /// Trains on a single example.
    pub fn train_single(&mut self, input: &Array1<Float>, target: &Array1<Float>) {
        match self.device {
            ComputeDevice::Cpu => self.train_single_cpu(input, target),
            // For single example, parallel doesn't make sense, use CPU
            ComputeDevice::CpuParallel => self.train_single_cpu(input, target),
            ComputeDevice::Gpu => unreachable!("GPU validated at construction"),
        }
    }

    /// Trains on a batch of examples.
    pub fn train_batch(&mut self, inputs: &[Array1<Float>], targets: &[Array1<Float>]) {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Number of inputs must match number of targets"
        );
        assert!(!inputs.is_empty(), "Batch cannot be empty");

        match self.device {
            ComputeDevice::Cpu => self.train_batch_cpu(inputs, targets),
            #[cfg(feature = "parallel")]
            ComputeDevice::CpuParallel => self.train_batch_parallel(inputs, targets),
            #[cfg(not(feature = "parallel"))]
            ComputeDevice::CpuParallel => unreachable!("CpuParallel validated at construction"),
            ComputeDevice::Gpu => unreachable!("GPU validated at construction"),
        }
    }

    // =========================================================================
    // CPU Implementation
    // =========================================================================

    /// CPU implementation of single-example training.
    fn train_single_cpu(&mut self, input: &Array1<Float>, target: &Array1<Float>) {
        // Forward pass with full information
        let forward_result = self.forward_with_rng(input);
        let activations = &forward_result.activations;
        let pre_activations = &forward_result.pre_activations;
        let dropout_masks = &forward_result.dropout_masks;
        let final_output = activations.last().unwrap();

        // Compute deltas via backpropagation
        let deltas = self.compute_deltas(
            target,
            final_output,
            activations,
            pre_activations,
            dropout_masks,
        );

        // Apply gradients using optimizer
        self.apply_gradients_single(&deltas, activations);
    }

    /// CPU implementation of batch training.
    fn train_batch_cpu(&mut self, inputs: &[Array1<Float>], targets: &[Array1<Float>]) {
        let batch_size = inputs.len() as Float;

        // Reset accumulated gradients to zero (reuse pre-allocated buffers)
        for grad in self.accumulated_weights.iter_mut() {
            grad.fill(0.0);
        }
        for grad in self.accumulated_biases.iter_mut() {
            grad.fill(0.0);
        }

        // Accumulate gradients for each example
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let forward_result = self.forward_with_rng(input);
            let activations = &forward_result.activations;
            let pre_activations = &forward_result.pre_activations;
            let dropout_masks = &forward_result.dropout_masks;
            let final_output = activations.last().unwrap();

            // Compute deltas
            let deltas = self.compute_deltas(
                target,
                final_output,
                activations,
                pre_activations,
                dropout_masks,
            );

            // Accumulate gradients (using pre-allocated buffers)
            for (i, delta) in deltas.iter().enumerate() {
                let prev_activation = &activations[i];

                let weights_gradient = -delta
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&prev_activation.view().insert_axis(Axis(0)));
                let biases_gradient = -delta;

                self.accumulated_weights[i] = &self.accumulated_weights[i] + &weights_gradient;
                self.accumulated_biases[i] = &self.accumulated_biases[i] + &biases_gradient;
            }
        }

        // Average and apply gradients
        self.apply_gradients_batch(batch_size);
    }

    /// Parallel CPU implementation of batch training using Rayon.
    ///
    /// Parallelizes the forward pass and gradient computation across the batch,
    /// then reduces (sums) the gradients and applies them.
    #[cfg(feature = "parallel")]
    fn train_batch_parallel(&mut self, inputs: &[Array1<Float>], targets: &[Array1<Float>]) {
        let batch_size = inputs.len() as Float;
        let num_layers = self.network.layers.len();

        // Get layer dimensions for initializing gradient accumulators
        let weight_dims: Vec<_> = self
            .network
            .layers
            .iter()
            .map(|l| l.weights.dim())
            .collect();
        let bias_dims: Vec<_> = self.network.layers.iter().map(|l| l.biases.len()).collect();

        // Compute gradients in parallel for each sample
        // Note: We don't use dropout in parallel mode to avoid RNG synchronization issues
        let gradients: Vec<(Vec<Array2<Float>>, Vec<Array1<Float>>)> = inputs
            .par_iter()
            .zip(targets.par_iter())
            .map(|(input, target)| self.compute_sample_gradients(input, target))
            .collect();

        // Reset accumulated gradients to zero (reuse pre-allocated buffers)
        for grad in self.accumulated_weights.iter_mut() {
            grad.fill(0.0);
        }
        for grad in self.accumulated_biases.iter_mut() {
            grad.fill(0.0);
        }

        // Reduce: sum all gradients into pre-allocated buffers
        for (sample_weights, sample_biases) in gradients {
            for i in 0..num_layers {
                self.accumulated_weights[i] = &self.accumulated_weights[i] + &sample_weights[i];
                self.accumulated_biases[i] = &self.accumulated_biases[i] + &sample_biases[i];
            }
        }

        // Average and apply gradients
        self.apply_gradients_batch(batch_size);
    }

    /// Computes gradients for a single sample (thread-safe, no mutation).
    /// Used by the parallel implementation.
    #[cfg(feature = "parallel")]
    fn compute_sample_gradients(
        &self,
        input: &Array1<Float>,
        target: &Array1<Float>,
    ) -> (Vec<Array2<Float>>, Vec<Array1<Float>>) {
        // Forward pass without dropout (deterministic for parallel execution)
        let forward_result = self.network.forward_full_internal(input, &mut rand::rng());
        let activations = &forward_result.activations;
        let pre_activations = &forward_result.pre_activations;
        let dropout_masks = &forward_result.dropout_masks;
        let final_output = activations.last().unwrap();

        // Compute deltas
        let deltas = self.compute_deltas(
            target,
            final_output,
            activations,
            pre_activations,
            dropout_masks,
        );

        // Compute gradients for each layer
        let mut weights_gradients = Vec::with_capacity(self.network.layers.len());
        let mut biases_gradients = Vec::with_capacity(self.network.layers.len());

        for (i, delta) in deltas.iter().enumerate() {
            let prev_activation = &activations[i];

            let weights_gradient = -delta
                .view()
                .insert_axis(Axis(1))
                .dot(&prev_activation.view().insert_axis(Axis(0)));
            let biases_gradient = -delta.clone();

            weights_gradients.push(weights_gradient);
            biases_gradients.push(biases_gradient);
        }

        (weights_gradients, biases_gradients)
    }

    // =========================================================================
    // Shared computation helpers
    // =========================================================================

    /// Performs forward pass using stored RNG for reproducibility.
    fn forward_with_rng(&mut self, input: &Array1<Float>) -> ForwardResult {
        // Take ownership of stored RNG temporarily
        if let Some(mut stored_rng) = self.network.rng.take() {
            let result = self.network.forward_full_internal(input, &mut stored_rng);
            self.network.rng = Some(stored_rng);
            result
        } else {
            self.network.forward_full_internal(input, &mut rng())
        }
    }

    /// Computes deltas (error signals) for all layers via backpropagation.
    fn compute_deltas(
        &self,
        target: &Array1<Float>,
        final_output: &Array1<Float>,
        _activations: &[Array1<Float>],
        pre_activations: &[Array1<Float>],
        dropout_masks: &[Option<Array1<Float>>],
    ) -> Vec<Array1<Float>> {
        let output_layer_idx = self.network.layers.len() - 1;
        let output_activation = self.network.layers[output_layer_idx].activation;

        // Compute output layer delta (optimized for common activation/loss pairs)
        let output_delta = match (&output_activation, &self.network.loss_function) {
            // Sigmoid + BCE: derivative simplifies to (target - output)
            (Activation::Sigmoid, LossFunction::BinaryCrossEntropy) => target - final_output,
            // Softmax + CCE: derivative simplifies to (target - output)
            (Activation::Softmax, LossFunction::CategoricalCrossEntropy) => target - final_output,
            // MSE: derivative is (prediction - target), negate for gradient descent
            (_, LossFunction::MSE) => target - final_output,
            // General case
            _ => {
                let loss_gradient = self.network.loss_function.derivative(final_output, target);
                let activation_derivative = output_activation
                    .derivative_from_preactivation(&pre_activations[output_layer_idx]);
                -&loss_gradient * &activation_derivative
            }
        };

        let mut deltas = vec![output_delta];

        // Backpropagate through hidden layers
        for i in (0..self.network.layers.len() - 1).rev() {
            let current_delta = deltas.last().unwrap();
            let mut errors = self.network.layers[i + 1].weights.t().dot(current_delta);

            // Apply dropout mask to gradient
            if let Some(ref mask) = dropout_masks[i] {
                errors = &errors * mask;
            }

            // Use pre-activation for derivative (mathematically correct)
            let activation_derivative = self.network.layers[i]
                .activation
                .derivative_from_preactivation(&pre_activations[i]);
            let delta = &errors * &activation_derivative;
            deltas.push(delta);
        }

        // Reverse to match layer order
        deltas.reverse();
        deltas
    }

    /// Applies gradients from a single example.
    fn apply_gradients_single(&mut self, deltas: &[Array1<Float>], activations: &[Array1<Float>]) {
        for (i, delta) in deltas.iter().enumerate() {
            let prev_activation = &activations[i];

            // Compute gradients
            let mut weights_gradient = -delta
                .view()
                .insert_axis(Axis(1))
                .dot(&prev_activation.view().insert_axis(Axis(0)));
            let biases_gradient = -delta;

            // Add regularization gradient if needed
            if let Some(reg_grad) = self
                .network
                .regularization
                .gradient_opt(&self.network.layers[i].weights)
            {
                weights_gradient += &reg_grad;
            }

            // Update via optimizer
            self.network.optimizer_states_weights[i].step(
                &mut self.network.layers[i].weights,
                &weights_gradient,
                &self.network.optimizer,
            );

            self.network.optimizer_states_biases[i].step(
                &mut self.network.layers[i].biases,
                &biases_gradient,
                &self.network.optimizer,
            );
        }
    }

    /// Applies averaged gradients from a batch.
    fn apply_gradients_batch(&mut self, batch_size: Float) {
        for i in 0..self.network.layers.len() {
            // Average the gradients (modify in place)
            self.accumulated_weights[i].mapv_inplace(|g| g / batch_size);
            self.accumulated_biases[i].mapv_inplace(|g| g / batch_size);

            // Add regularization gradient if needed
            if let Some(reg_grad) = self
                .network
                .regularization
                .gradient_opt(&self.network.layers[i].weights)
            {
                self.accumulated_weights[i] = &self.accumulated_weights[i] + &reg_grad;
            }

            // Update via optimizer
            self.network.optimizer_states_weights[i].step(
                &mut self.network.layers[i].weights,
                &self.accumulated_weights[i],
                &self.network.optimizer,
            );

            self.network.optimizer_states_biases[i].step(
                &mut self.network.layers[i].biases,
                &self.accumulated_biases[i],
                &self.network.optimizer,
            );
        }
    }
}
