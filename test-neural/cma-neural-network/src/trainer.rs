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

use crate::compute::{ComputeDevice, ComputeDeviceError};
use crate::network::{Activation, ForwardResult, LossFunction, Network};

/// Accumulated gradients for a training batch.
///
/// Contains the gradients for all layers, ready to be applied via the optimizer.
#[allow(dead_code)]
pub(crate) struct BatchGradients {
    /// Weight gradients for each layer (averaged over batch)
    pub weights: Vec<Array2<f64>>,
    /// Bias gradients for each layer (averaged over batch)
    pub biases: Vec<Array1<f64>>,
}

/// Internal trainer that executes the training logic.
///
/// This struct is created temporarily during training and released after.
/// It doesn't own the network, just borrows it mutably.
pub(crate) struct Trainer<'a> {
    network: &'a mut Network,
    device: ComputeDevice,
}

impl<'a> Trainer<'a> {
    /// Creates a new trainer for the given network.
    #[allow(dead_code)]
    pub fn new(
        network: &'a mut Network,
        device: ComputeDevice,
    ) -> Result<Self, ComputeDeviceError> {
        device.validate()?;
        Ok(Self { network, device })
    }

    /// Creates a new trainer with CPU device (infallible).
    pub fn cpu(network: &'a mut Network) -> Self {
        Self {
            network,
            device: ComputeDevice::Cpu,
        }
    }

    /// Trains on a single example.
    pub fn train_single(&mut self, input: &Array1<f64>, target: &Array1<f64>) {
        match self.device {
            ComputeDevice::Cpu => self.train_single_cpu(input, target),
            ComputeDevice::Gpu => unreachable!("GPU validated at construction"),
        }
    }

    /// Trains on a batch of examples.
    pub fn train_batch(&mut self, inputs: &[Array1<f64>], targets: &[Array1<f64>]) {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Number of inputs must match number of targets"
        );
        assert!(!inputs.is_empty(), "Batch cannot be empty");

        match self.device {
            ComputeDevice::Cpu => self.train_batch_cpu(inputs, targets),
            ComputeDevice::Gpu => unreachable!("GPU validated at construction"),
        }
    }

    // =========================================================================
    // CPU Implementation
    // =========================================================================

    /// CPU implementation of single-example training.
    fn train_single_cpu(&mut self, input: &Array1<f64>, target: &Array1<f64>) {
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
    fn train_batch_cpu(&mut self, inputs: &[Array1<f64>], targets: &[Array1<f64>]) {
        let batch_size = inputs.len() as f64;

        // Initialize accumulated gradients
        let mut accumulated_weights: Vec<Array2<f64>> = self
            .network
            .layers
            .iter()
            .map(|layer| Array2::zeros(layer.weights.dim()))
            .collect();

        let mut accumulated_biases: Vec<Array1<f64>> = self
            .network
            .layers
            .iter()
            .map(|layer| Array1::zeros(layer.biases.dim()))
            .collect();

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

            // Accumulate gradients
            for (i, delta) in deltas.iter().enumerate() {
                let prev_activation = &activations[i];

                let weights_gradient = -delta
                    .view()
                    .insert_axis(Axis(1))
                    .dot(&prev_activation.view().insert_axis(Axis(0)));
                let biases_gradient = -delta;

                accumulated_weights[i] = &accumulated_weights[i] + &weights_gradient;
                accumulated_biases[i] = &accumulated_biases[i] + &biases_gradient;
            }
        }

        // Average and apply gradients
        self.apply_gradients_batch(&accumulated_weights, &accumulated_biases, batch_size);
    }

    // =========================================================================
    // Shared computation helpers
    // =========================================================================

    /// Performs forward pass using stored RNG for reproducibility.
    fn forward_with_rng(&mut self, input: &Array1<f64>) -> ForwardResult {
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
        target: &Array1<f64>,
        final_output: &Array1<f64>,
        _activations: &[Array1<f64>],
        pre_activations: &[Array1<f64>],
        dropout_masks: &[Option<Array1<f64>>],
    ) -> Vec<Array1<f64>> {
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
    fn apply_gradients_single(&mut self, deltas: &[Array1<f64>], activations: &[Array1<f64>]) {
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
    fn apply_gradients_batch(
        &mut self,
        accumulated_weights: &[Array2<f64>],
        accumulated_biases: &[Array1<f64>],
        batch_size: f64,
    ) {
        for i in 0..self.network.layers.len() {
            // Average the gradients
            let mut avg_weights_gradient = &accumulated_weights[i] / batch_size;
            let avg_biases_gradient = &accumulated_biases[i] / batch_size;

            // Add regularization gradient if needed
            if let Some(reg_grad) = self
                .network
                .regularization
                .gradient_opt(&self.network.layers[i].weights)
            {
                avg_weights_gradient += &reg_grad;
            }

            // Update via optimizer
            self.network.optimizer_states_weights[i].step(
                &mut self.network.layers[i].weights,
                &avg_weights_gradient,
                &self.network.optimizer,
            );

            self.network.optimizer_states_biases[i].step(
                &mut self.network.layers[i].biases,
                &avg_biases_gradient,
                &self.network.optimizer,
            );
        }
    }
}
