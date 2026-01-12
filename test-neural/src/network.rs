use ndarray::{Array1, Array2, Axis};
use rand::rng;
use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::optimizer::{OptimizerType, OptimizerState1D, OptimizerState2D};

/// Weight initialization methods for neural networks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WeightInit {
    /// Uniform distribution in [-1, 1] (simple, for shallow networks)
    Uniform,
    /// Xavier/Glorot initialization (for Tanh, Sigmoid, Softmax)
    Xavier,
    /// He initialization (for ReLU, LeakyReLU, ELU)
    He,
    /// LeCun initialization (for SELU)
    LeCun,
}

impl WeightInit {
    /// Initialize a weight matrix based on the initialization method.
    ///
    /// # Arguments
    /// - `rows`: Number of rows (output size)
    /// - `cols`: Number of columns (input size)
    /// - `rng`: Random number generator
    ///
    /// # Returns
    /// Initialized weight matrix
    fn initialize_weights(&self, rows: usize, cols: usize, rng: &mut impl Rng) -> Array2<f64> {
        match self {
            WeightInit::Uniform => {
                Array2::from_shape_fn((rows, cols), |_| rng.random_range(-1.0..1.0))
            },
            WeightInit::Xavier => {
                // Xavier: std = sqrt(2 / (input_size + output_size))
                let std = (2.0 / (rows + cols) as f64).sqrt();
                Array2::from_shape_fn((rows, cols), |_| {
                    // Box-Muller transform for Gaussian distribution
                    let u1: f64 = rng.random();
                    let u2: f64 = rng.random();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    z * std
                })
            },
            WeightInit::He => {
                // He: std = sqrt(2 / input_size)
                let std = (2.0 / cols as f64).sqrt();
                Array2::from_shape_fn((rows, cols), |_| {
                    // Box-Muller transform for Gaussian distribution
                    let u1: f64 = rng.random();
                    let u2: f64 = rng.random();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    z * std
                })
            },
            WeightInit::LeCun => {
                // LeCun: std = sqrt(1 / input_size)
                let std = (1.0 / cols as f64).sqrt();
                Array2::from_shape_fn((rows, cols), |_| {
                    // Box-Muller transform for Gaussian distribution
                    let u1: f64 = rng.random();
                    let u2: f64 = rng.random();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    z * std
                })
            },
        }
    }

    /// Get recommended initialization method for an activation function.
    pub fn for_activation(activation: Activation) -> Self {
        match activation {
            Activation::Sigmoid | Activation::Tanh | Activation::Softsign 
            | Activation::HardSigmoid | Activation::HardTanh | Activation::Softmax => {
                WeightInit::Xavier
            },
            Activation::ReLU | Activation::LeakyReLU | Activation::ELU 
            | Activation::GELU | Activation::Swish | Activation::Mish 
            | Activation::Softplus => {
                WeightInit::He
            },
            Activation::SELU => {
                WeightInit::LeCun
            },
            Activation::Linear => {
                WeightInit::Xavier
            },
        }
    }
}

/// Available activation functions for neural network layers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    ELU,
    SELU,
    Swish,
    GELU,
    Mish,
    Softplus,
    Softsign,
    HardSigmoid,
    HardTanh,
    Softmax,
    Linear,
}

/// Available loss functions for training.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error - for regression
    MSE,
    /// Mean Absolute Error - for robust regression
    MAE,
    /// Binary Cross-Entropy - for binary classification
    BinaryCrossEntropy,
    /// Categorical Cross-Entropy - for multi-class classification
    CategoricalCrossEntropy,
    /// Huber Loss - robust to outliers
    Huber,
}

impl LossFunction {
    /// Compute the loss value between predictions and targets.
    pub fn compute(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        match self {
            LossFunction::MSE => {
                let diff = predictions - targets;
                (&diff * &diff).sum() / predictions.len() as f64
            },
            LossFunction::MAE => {
                (predictions - targets).mapv(|x| x.abs()).sum() / predictions.len() as f64
            },
            LossFunction::BinaryCrossEntropy => {
                let epsilon = 1e-15;
                let mut sum = 0.0;
                for (p, t) in predictions.iter().zip(targets.iter()) {
                    let p_clamped = p.max(epsilon).min(1.0 - epsilon);
                    sum += -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln());
                }
                sum / predictions.len() as f64
            },
            LossFunction::CategoricalCrossEntropy => {
                let epsilon = 1e-15;
                let mut sum = 0.0;
                for (p, t) in predictions.iter().zip(targets.iter()) {
                    let p_clamped = p.max(epsilon);
                    sum += -t * p_clamped.ln();
                }
                sum
            },
            LossFunction::Huber => {
                let delta = 1.0;
                let diff = predictions - targets;
                let mut sum = 0.0;
                for &d in diff.iter() {
                    let abs_d = d.abs();
                    if abs_d <= delta {
                        sum += 0.5 * d * d;
                    } else {
                        sum += delta * (abs_d - 0.5 * delta);
                    }
                }
                sum / predictions.len() as f64
            },
        }
    }

    /// Compute the derivative (gradient) of the loss function.
    /// Returns the error signal to be backpropagated.
    pub fn derivative(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64> {
        match self {
            LossFunction::MSE => {
                // d/dx[(y - x)^2] = -2(y - x) = 2(x - y)
                // Simplified for gradient descent: (x - y)
                predictions - targets
            },
            LossFunction::MAE => {
                // d/dx[|y - x|] = sign(x - y)
                (predictions - targets).mapv(|x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 })
            },
            LossFunction::BinaryCrossEntropy => {
                // d/dx[-y*ln(x) - (1-y)*ln(1-x)] = -y/x + (1-y)/(1-x) = (x - y) / (x(1-x))
                // Simplified when used with sigmoid: (x - y)
                let epsilon = 1e-15;
                let mut result = Array1::zeros(predictions.len());
                for (i, (p, t)) in predictions.iter().zip(targets.iter()).enumerate() {
                    let p_clamped = p.max(epsilon).min(1.0 - epsilon);
                    result[i] = (p_clamped - t) / (p_clamped * (1.0 - p_clamped));
                }
                result
            },
            LossFunction::CategoricalCrossEntropy => {
                // d/dx[-y*ln(x)] = -y/x
                // Simplified when used with softmax: (x - y)
                let epsilon = 1e-15;
                let mut result = Array1::zeros(predictions.len());
                for (i, (p, t)) in predictions.iter().zip(targets.iter()).enumerate() {
                    let p_clamped = p.max(epsilon);
                    result[i] = -t / p_clamped;
                }
                result
            },
            LossFunction::Huber => {
                let delta = 1.0;
                let diff = predictions - targets;
                diff.mapv(|d| {
                    if d.abs() <= delta {
                        d
                    } else {
                        delta * d.signum()
                    }
                })
            },
        }
    }
}

impl Activation {
    /// Apply the activation function to an array.
    pub fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::Sigmoid => x.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::Tanh => x.mapv(|x| x.tanh()),
            Activation::ReLU => x.mapv(|x| x.max(0.0)),
            Activation::LeakyReLU => x.mapv(|x| if x > 0.0 { x } else { 0.01 * x }),
            Activation::ELU => x.mapv(|x| if x > 0.0 { x } else { 1.0 * (x.exp() - 1.0) }),
            Activation::SELU => {
                let lambda = 1.0507;
                let alpha = 1.6733;
                x.mapv(|x| lambda * if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            },
            Activation::Swish => x.mapv(|x| x / (1.0 + (-x).exp())),
            Activation::GELU => x.mapv(|x| {
                0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() 
                    * (x + 0.044715 * x.powi(3))).tanh())
            }),
            Activation::Mish => x.mapv(|x| x * ((1.0 + x.exp()).ln()).tanh()),
            Activation::Softplus => x.mapv(|x| (1.0 + x.exp()).ln()),
            Activation::Softsign => x.mapv(|x| x / (1.0 + x.abs())),
            Activation::HardSigmoid => x.mapv(|x| (0.2 * x + 0.5).max(0.0).min(1.0)),
            Activation::HardTanh => x.mapv(|x| x.max(-1.0).min(1.0)),
            Activation::Softmax => {
                // Numerical stability: subtract max before exp
                let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_x = x.mapv(|v| (v - max).exp());
                let sum = exp_x.sum();
                exp_x / sum
            },
            Activation::Linear => x.clone(),
        }
    }

    /// Compute the derivative of the activation function.
    /// For activations already applied (post-activation values).
    pub fn derivative(&self, x: &Array1<f64>) -> Array1<f64> {
        match self {
            Activation::Sigmoid => x * &(1.0 - x),
            Activation::Tanh => x.mapv(|x| 1.0 - x.powi(2)),
            Activation::ReLU => x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyReLU => x.mapv(|x| if x > 0.0 { 1.0 } else { 0.01 }),
            Activation::ELU => x.mapv(|x| if x > 0.0 { 1.0 } else { x + 1.0 }),
            Activation::SELU => {
                let lambda = 1.0507;
                x.mapv(|x| if x > 0.0 { lambda } else { lambda })
            },
            Activation::Swish => {
                let sigmoid = x.mapv(|x| 1.0 / (1.0 + (-x).exp()));
                let swish = x * &sigmoid;
                &swish + &sigmoid * &(1.0 - &swish)
            },
            Activation::GELU => {
                // Approximation simplifiée de la dérivée
                x.mapv(|x| {
                    let cdf = 0.5 * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() 
                        * (x + 0.044715 * x.powi(3))).tanh());
                    cdf + x * 0.5 * (1.0 - cdf.powi(2))
                })
            },
            Activation::Mish => {
                // Dérivée complexe de Mish (approximation)
                x.mapv(|x| {
                    let omega = 4.0 * (x + 1.0) + 4.0 * x.exp() + x.exp().powi(2) + x.exp() * (4.0 * x + 6.0);
                    let delta = 2.0 * x.exp() + x.exp().powi(2) + 2.0;
                    omega / delta.powi(2)
                })
            },
            Activation::Softplus => x.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            Activation::Softsign => x.mapv(|x| 1.0 / (1.0 + x.abs()).powi(2)),
            Activation::HardSigmoid => x.mapv(|x| {
                let val = 0.2 * x + 0.5;
                if val > 0.0 && val < 1.0 { 0.2 } else { 0.0 }
            }),
            Activation::HardTanh => x.mapv(|x| if x > -1.0 && x < 1.0 { 1.0 } else { 0.0 }),
            Activation::Softmax => {
                // Pour Softmax, la dérivée est plus complexe (Jacobienne)
                // Approximation simple: utiliser x * (1 - x) comme pour sigmoid
                x * &(1.0 - x)
            },
            Activation::Linear => Array1::ones(x.len()),
        }
    }
}

/// A layer in the neural network.
#[derive(Clone, Serialize, Deserialize)]
struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: Activation,
}

/// A feedforward neural network with configurable depth.
///
/// This network implements backpropagation for training and allows
/// customizable activation functions for each layer.
///
/// # Architecture
/// - Input layer (size defined by user)
/// - Multiple hidden layers with configurable activations
/// - Output layer with configurable activation
///
/// # Examples
/// ```
/// // Single hidden layer (simple network)
/// let net = Network::new(2, 5, 1, Activation::Tanh, Activation::Sigmoid, LossFunction::MSE, OptimizerType::adam(0.001));
///
/// // Multiple hidden layers (deep network)
/// let net = Network::new_deep(
///     2,
///     vec![10, 8, 5],
///     1,
///     vec![Activation::ReLU, Activation::ReLU, Activation::ReLU],
///     Activation::Sigmoid,
///     LossFunction::BinaryCrossEntropy,
///     OptimizerType::adam(0.001)
/// );
/// ```
#[derive(Serialize, Deserialize)]
pub struct Network {
    /// All layers (hidden + output)
    layers: Vec<Layer>,
    /// Input size for reference
    input_size: usize,
    /// Loss function for training
    loss_function: LossFunction,
    /// Optimizer type
    optimizer: OptimizerType,
    /// Optimizer states for weights
    optimizer_states_weights: Vec<OptimizerState2D>,
    /// Optimizer states for biases
    optimizer_states_biases: Vec<OptimizerState1D>,
}

impl Network {
    /// Creates a new neural network with one hidden layer.
    ///
    /// This is a convenience method for simple networks. For deep networks
    /// with multiple hidden layers, use `new_deep()`.
    ///
    /// Uses automatic weight initialization based on activation functions.
    ///
    /// # Arguments
    /// - `input_size`: Number of input neurons
    /// - `hidden_size`: Number of neurons in the hidden layer
    /// - `output_size`: Number of output neurons
    /// - `hidden_activation`: Activation function for hidden layer
    /// - `output_activation`: Activation function for output layer
    /// - `loss_function`: Loss function for training
    ///
    /// # Example
    /// ```
    /// // XOR with Tanh hidden, Sigmoid output, Binary Cross-Entropy loss
    /// let network = Network::new(
    ///     2, 5, 1, 
    ///     Activation::Tanh, 
    ///     Activation::Sigmoid,
    ///     LossFunction::BinaryCrossEntropy,
    ///     OptimizerType::adam(0.001)
    /// );
    /// ```
    pub fn new(
        input_size: usize, 
        hidden_size: usize, 
        output_size: usize,
        hidden_activation: Activation,
        output_activation: Activation,
        loss_function: LossFunction,
        optimizer: OptimizerType,
    ) -> Self {
        // Use recommended initialization for each activation
        let hidden_init = WeightInit::for_activation(hidden_activation);
        let output_init = WeightInit::for_activation(output_activation);
        
        Self::new_deep_with_init(
            input_size,
            vec![hidden_size],
            output_size,
            vec![hidden_activation],
            output_activation,
            loss_function,
            vec![hidden_init],
            output_init,
            optimizer,
        )
    }

    /// Creates a new deep neural network with multiple hidden layers.
    ///
    /// Uses automatic weight initialization based on activation functions.
    ///
    /// # Arguments
    /// - `input_size`: Number of input neurons
    /// - `hidden_sizes`: Vector of sizes for each hidden layer (e.g., [10, 8, 5] for 3 layers)
    /// - `output_size`: Number of output neurons
    /// - `hidden_activations`: Activation function for each hidden layer
    /// - `output_activation`: Activation function for output layer
    /// - `loss_function`: Loss function for training
    ///
    /// # Panics
    /// Panics if `hidden_sizes.len() != hidden_activations.len()`
    ///
    /// # Example
    /// ```
    /// // Deep network with 3 hidden layers
    /// let network = Network::new_deep(
    ///     2,                                      // 2 inputs
    ///     vec![10, 8, 5],                        // 3 hidden layers
    ///     1,                                      // 1 output
    ///     vec![Activation::ReLU, Activation::ReLU, Activation::ReLU],
    ///     Activation::Sigmoid,
    ///     LossFunction::BinaryCrossEntropy,
    ///     OptimizerType::adam(0.001)
    /// );
    /// ```
    pub fn new_deep(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        hidden_activations: Vec<Activation>,
        output_activation: Activation,
        loss_function: LossFunction,
        optimizer: OptimizerType,
    ) -> Self {
        // Use recommended initialization for each activation
        let hidden_inits: Vec<WeightInit> = hidden_activations.iter()
            .map(|&act| WeightInit::for_activation(act))
            .collect();
        let output_init = WeightInit::for_activation(output_activation);
        
        Self::new_deep_with_init(
            input_size,
            hidden_sizes,
            output_size,
            hidden_activations,
            output_activation,
            loss_function,
            hidden_inits,
            output_init,
            optimizer,
        )
    }

    /// Creates a new deep neural network with custom weight initialization.
    ///
    /// This method gives full control over weight initialization for each layer.
    ///
    /// # Arguments
    /// - `input_size`: Number of input neurons
    /// - `hidden_sizes`: Vector of sizes for each hidden layer
    /// - `output_size`: Number of output neurons
    /// - `hidden_activations`: Activation function for each hidden layer
    /// - `output_activation`: Activation function for output layer
    /// - `loss_function`: Loss function for training
    /// - `hidden_inits`: Weight initialization method for each hidden layer
    /// - `output_init`: Weight initialization method for output layer
    ///
    /// # Panics
    /// Panics if lengths don't match
    ///
    /// # Example
    /// ```
    /// // Deep network with custom initialization
    /// let network = Network::new_deep_with_init(
    ///     2,
    ///     vec![10, 8],
    ///     1,
    ///     vec![Activation::ReLU, Activation::ReLU],
    ///     Activation::Sigmoid,
    ///     LossFunction::BinaryCrossEntropy,
    ///     vec![WeightInit::He, WeightInit::He],
    ///     WeightInit::Xavier,
    ///     OptimizerType::adam(0.001)
    /// );
    /// ```
    pub fn new_deep_with_init(
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        hidden_activations: Vec<Activation>,
        output_activation: Activation,
        loss_function: LossFunction,
        hidden_inits: Vec<WeightInit>,
        output_init: WeightInit,
        optimizer: OptimizerType,
    ) -> Self {
        assert_eq!(
            hidden_sizes.len(),
            hidden_activations.len(),
            "Number of hidden layers must match number of activations"
        );
        assert_eq!(
            hidden_sizes.len(),
            hidden_inits.len(),
            "Number of hidden layers must match number of initializations"
        );

        let mut rng = rng();
        let mut layers = Vec::new();

        // Create hidden layers
        let mut prev_size = input_size;
        for (i, &size) in hidden_sizes.iter().enumerate() {
            let weights = hidden_inits[i].initialize_weights(size, prev_size, &mut rng);
            let biases = Array1::zeros(size);  // Biases initialized to 0
            
            layers.push(Layer {
                weights,
                biases,
                activation: hidden_activations[i],
            });
            
            prev_size = size;
        }

        // Create output layer
        let weights = output_init.initialize_weights(output_size, prev_size, &mut rng);
        let biases = Array1::zeros(output_size);
        
        layers.push(Layer {
            weights,
            biases,
            activation: output_activation,
        });

        // Initialize optimizer states for all layers
        let optimizer_states_weights: Vec<OptimizerState2D> = layers.iter()
            .map(|layer| {
                let shape = layer.weights.dim();
                OptimizerState2D::new(shape, &optimizer)
            })
            .collect();
        
        let optimizer_states_biases: Vec<OptimizerState1D> = layers.iter()
            .map(|layer| {
                let size = layer.biases.len();
                OptimizerState1D::new(size, &optimizer)
            })
            .collect();

        Network {
            layers,
            input_size,
            loss_function,
            optimizer,
            optimizer_states_weights,
            optimizer_states_biases,
        }
    }
}

impl Network {
    /// Performs a forward pass through the network.
    ///
    /// # Arguments
    /// - `input`: Input vector
    ///
    /// # Returns
    /// Vector of all layer activations (including input and final output).
    /// Index 0 is the input, last index is the final output.
    ///
    /// # Example
    /// ```
    /// let activations = network.forward(&array![0.0, 1.0]);
    /// let prediction = activations.last().unwrap();
    /// ```
    fn forward(&self, input: &Array1<f64>) -> Vec<Array1<f64>> {
        let mut activations = vec![input.clone()];
        
        // Forward pass through all layers
        for layer in &self.layers {
            let z = layer.weights.dot(activations.last().unwrap()) + &layer.biases;
            let a = layer.activation.apply(&z);
            activations.push(a);
        }
        
        activations
    }
}

impl Network {
    /// Trains the network on a single input-target pair using backpropagation.
    ///
    /// Updates weights and biases based on the error between prediction and target.
    /// Uses the configured loss function to compute gradients and the configured
    /// optimizer to update parameters.
    ///
    /// # Arguments
    /// - `input`: Input vector
    /// - `target`: Expected output vector
    ///
    /// Note: Learning rate is now part of the optimizer configuration.
    /// For backward compatibility, you can override the learning rate by modifying
    /// the optimizer before training.
    ///
    /// # Algorithm
    /// 1. Forward pass to get all activations
    /// 2. Calculate output layer error using loss function
    /// 3. Backpropagate error through all hidden layers
    /// 4. Update all weights and biases using the optimizer
    ///
    /// # Example
    /// ```
    /// // Learning rate is in the optimizer
    /// let mut network = Network::new(
    ///     2, 5, 1,
    ///     Activation::Tanh,
    ///     Activation::Sigmoid,
    ///     LossFunction::BinaryCrossEntropy,
    ///     OptimizerType::adam(0.001)
    /// );
    /// network.train(&array![0.0, 1.0], &array![1.0]);
    /// ```
    pub fn train(&mut self, input: &Array1<f64>, target: &Array1<f64>) {
        // Forward pass
        let activations = self.forward(input);
        let final_output = activations.last().unwrap();
        
        // Compute output layer delta
        let output_layer_idx = self.layers.len() - 1;
        let output_activation = self.layers[output_layer_idx].activation;
        
        let output_delta = match (&output_activation, &self.loss_function) {
            // Sigmoid + Binary Cross-Entropy: derivative simplifies
            (Activation::Sigmoid, LossFunction::BinaryCrossEntropy) => {
                target - final_output
            },
            // Softmax + Categorical Cross-Entropy: derivative simplifies
            (Activation::Softmax, LossFunction::CategoricalCrossEntropy) => {
                target - final_output
            },
            // MSE: derivative is (prediction - target), negate for gradient descent
            (_, LossFunction::MSE) => {
                target - final_output
            },
            // General case: use loss gradient and activation derivative
            _ => {
                let loss_gradient = self.loss_function.derivative(final_output, target);
                -&loss_gradient * &output_activation.derivative(final_output)
            }
        };
        
        // Backpropagate through all layers
        let mut deltas = vec![output_delta];
        
        // Go backwards through hidden layers
        for i in (0..self.layers.len() - 1).rev() {
            let current_delta = deltas.last().unwrap();
            let errors = self.layers[i + 1].weights.t().dot(current_delta);
            let delta = &errors * &self.layers[i].activation.derivative(&activations[i + 1]);
            deltas.push(delta);
        }
        
        // Reverse deltas to match layer order
        deltas.reverse();
        
        // Update weights and biases for all layers using optimizer
        for (i, delta) in deltas.iter().enumerate() {
            let prev_activation = &activations[i];
            
            // Compute gradients (negative because delta already has correct sign)
            let weights_gradient = -delta.view().insert_axis(Axis(1))
                .dot(&prev_activation.view().insert_axis(Axis(0)));
            let biases_gradient = -delta;
            
            // Update using optimizer
            self.optimizer_states_weights[i].step(
                &mut self.layers[i].weights,
                &weights_gradient,
                &self.optimizer,
            );
            
            self.optimizer_states_biases[i].step(
                &mut self.layers[i].biases,
                &biases_gradient,
                &self.optimizer,
            );
        }
    }

    /// Evaluates the network on given input-target pairs without updating weights.
    ///
    /// Returns the average loss over all samples.
    ///
    /// # Arguments
    /// - `inputs`: Vector of input arrays
    /// - `targets`: Vector of target arrays
    ///
    /// # Returns
    /// Average loss value
    ///
    /// # Example
    /// ```
    /// let loss = network.evaluate(&inputs, &targets);
    /// println!("Average loss: {:.4}", loss);
    /// ```
    pub fn evaluate(&self, inputs: &Vec<Array1<f64>>, targets: &Vec<Array1<f64>>) -> f64 {
        let mut total_loss = 0.0;
        
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let activations = self.forward(input);
            let prediction = activations.last().unwrap();
            total_loss += self.loss_function.compute(prediction, target);
        }
        
        total_loss / inputs.len() as f64
    }

    /// Makes a prediction for a single input without requiring a target.
    ///
    /// This is the main inference method - use it to get predictions after training.
    ///
    /// # Arguments
    /// - `input`: Input vector
    ///
    /// # Returns
    /// Output vector (network's prediction)
    ///
    /// # Example
    /// ```
    /// let prediction = network.predict(&array![0.0, 1.0]);
    /// println!("Prediction: {:.3}", prediction[0]);
    /// ```
    pub fn predict(&self, input: &Array1<f64>) -> Array1<f64> {
        let activations = self.forward(input);
        activations.last().unwrap().clone()
    }
}