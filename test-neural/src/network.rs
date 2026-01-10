use ndarray::{Array1, Array2, Axis};
use rand::thread_rng;
use rand::Rng;

/// Available activation functions for neural network layers.
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone, Copy)]
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

/// A simple feedforward neural network with one hidden layer.
///
/// This network implements backpropagation for training and allows
/// customizable activation functions for hidden and output layers.
///
/// # Architecture
/// - Input layer (size defined by user)
/// - Hidden layer with configurable activation
/// - Output layer with configurable activation
pub struct Network {
    /// Weights connecting input layer to hidden layer (hidden_size × input_size)
    weights1: Array2<f64>,
    /// Biases for the hidden layer
    biases1: Array1<f64>,

    /// Weights connecting hidden layer to output layer (output_size × hidden_size)
    weights2: Array2<f64>,
    /// Biases for the output layer
    biases2: Array1<f64>,

    /// Activation function for hidden layer
    hidden_activation: Activation,
    /// Activation function for output layer
    output_activation: Activation,
    /// Loss function for training
    loss_function: LossFunction,
}

impl Network {
    /// Creates a new neural network with random weights and biases.
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
    /// // XOR with ReLU hidden, Sigmoid output, Binary Cross-Entropy loss
    /// let network = Network::new(
    ///     2, 3, 1, 
    ///     Activation::ReLU, 
    ///     Activation::Sigmoid,
    ///     LossFunction::BinaryCrossEntropy
    /// );
    ///
    /// // Multi-class classification with GELU and Softmax
    /// let network = Network::new(
    ///     784, 128, 10, 
    ///     Activation::GELU, 
    ///     Activation::Softmax,
    ///     LossFunction::CategoricalCrossEntropy
    /// );
    /// ```
    pub fn new(
        input_size: usize, 
        hidden_size: usize, 
        output_size: usize,
        hidden_activation: Activation,
        output_activation: Activation,
        loss_function: LossFunction,
    ) -> Self {
        let mut rng = thread_rng();
 
        let weights1 = Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen_range(-1.0..1.0));
        let biases1 = Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-1.0..1.0));
        let weights2 = Array2::from_shape_fn((output_size, hidden_size), |_| rng.gen_range(-1.0..1.0));
        let biases2 = Array1::from_shape_fn(output_size, |_| rng.gen_range(-1.0..1.0));
 
        Network {
            weights1,
            biases1,
            weights2,
            biases2,
            hidden_activation,
            output_activation,
            loss_function,
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
    /// Tuple containing:
    /// - Hidden layer output (after activation)
    /// - Final layer input (before activation)
    /// - Final layer output (after activation - network prediction)
    ///
    /// # Example
    /// ```
    /// let (hidden, final_in, prediction) = network.forward(&array![0.0, 1.0]);
    /// ```
    pub fn forward(&self, input: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let hidden_input = self.weights1.dot(input) + &self.biases1;
        let hidden_output = self.hidden_activation.apply(&hidden_input);
        let final_input = self.weights2.dot(&hidden_output) + &self.biases2;
        let final_output = self.output_activation.apply(&final_input);
 
        (hidden_output, final_input, final_output)
    }
}

impl Network {
    /// Trains the network on a single input-target pair using backpropagation.
    ///
    /// Updates weights and biases based on the error between prediction and target.
    /// Uses the configured loss function to compute gradients.
    ///
    /// # Arguments
    /// - `input`: Input vector
    /// - `target`: Expected output vector
    /// - `learning_rate`: Controls how much to adjust weights (typically 0.001 - 0.1)
    ///
    /// # Algorithm
    /// 1. Forward pass to get prediction
    /// 2. Calculate output layer error using loss function
    /// 3. Backpropagate error to hidden layer
    /// 4. Update all weights and biases
    ///
    /// # Example
    /// ```
    /// network.train(&array![0.0, 1.0], &array![1.0], 0.1);
    /// ```
    pub fn train(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64) {
        let (hidden_output, _final_input, final_output) = self.forward(input);
 
        // For output layer: compute error signal
        // For specific combinations, the derivative simplifies nicely
        let output_delta = match (&self.output_activation, &self.loss_function) {
            // Sigmoid + Binary Cross-Entropy: derivative simplifies to -(target - prediction)
            (Activation::Sigmoid, LossFunction::BinaryCrossEntropy) => {
                target - &final_output
            },
            // Softmax + Categorical Cross-Entropy: derivative simplifies to -(target - prediction)
            (Activation::Softmax, LossFunction::CategoricalCrossEntropy) => {
                target - &final_output
            },
            // MSE: derivative is (prediction - target), negate for gradient descent
            (_, LossFunction::MSE) => {
                target - &final_output
            },
            // General case: use loss gradient and activation derivative
            _ => {
                let loss_gradient = self.loss_function.derivative(&final_output, target);
                -&loss_gradient * &self.output_activation.derivative(&final_output)
            }
        };
 
        let hidden_errors = self.weights2.t().dot(&output_delta);
        let hidden_delta = &hidden_errors * &self.hidden_activation.derivative(&hidden_output);
 
        let weights2_update = output_delta.view().insert_axis(Axis(1)).dot(&hidden_output.view().insert_axis(Axis(0))) * learning_rate;
        let biases2_update = &output_delta * learning_rate;
        
        let weights1_update = hidden_delta.view().insert_axis(Axis(1)).dot(&input.view().insert_axis(Axis(0))) * learning_rate;
        let biases1_update = &hidden_delta * learning_rate;
        
        self.weights2 = &self.weights2 + &weights2_update;
        self.biases2 = &self.biases2 + &biases2_update;
        self.weights1 = &self.weights1 + &weights1_update;
        self.biases1 = &self.biases1 + &biases1_update;
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
            let (_, _, prediction) = self.forward(input);
            total_loss += self.loss_function.compute(&prediction, target);
        }
        
        total_loss / inputs.len() as f64
    }
}