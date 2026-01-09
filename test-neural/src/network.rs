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
    Swish,
    GELU,
    Mish,
    Softplus,
    Linear,
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
            Activation::Swish => x.mapv(|x| x / (1.0 + (-x).exp())),
            Activation::GELU => x.mapv(|x| {
                0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() 
                    * (x + 0.044715 * x.powi(3))).tanh())
            }),
            Activation::Mish => x.mapv(|x| x * ((1.0 + x.exp()).ln()).tanh()),
            Activation::Softplus => x.mapv(|x| (1.0 + x.exp()).ln()),
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
    ///
    /// # Example
    /// ```
    /// // XOR with ReLU hidden, Sigmoid output
    /// let network = Network::new(2, 3, 1, Activation::ReLU, Activation::Sigmoid);
    ///
    /// // Multi-class classification with GELU and Softmax
    /// let network = Network::new(784, 128, 10, Activation::GELU, Activation::Sigmoid);
    /// ```
    pub fn new(
        input_size: usize, 
        hidden_size: usize, 
        output_size: usize,
        hidden_activation: Activation,
        output_activation: Activation,
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
    ///
    /// # Arguments
    /// - `input`: Input vector
    /// - `target`: Expected output vector
    /// - `learning_rate`: Controls how much to adjust weights (typically 0.001 - 0.1)
    ///
    /// # Algorithm
    /// 1. Forward pass to get prediction
    /// 2. Calculate output layer error and gradients
    /// 3. Backpropagate error to hidden layer
    /// 4. Update all weights and biases
    ///
    /// # Example
    /// ```
    /// network.train(&array![0.0, 1.0], &array![1.0], 0.1);
    /// ```
    pub fn train(&mut self, input: &Array1<f64>, target: &Array1<f64>, learning_rate: f64) {
        let (hidden_output, final_input, final_output) = self.forward(input);
 
        let output_errors = target - &final_output;
        let output_delta = &output_errors * &self.output_activation.derivative(&final_output);
 
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
}