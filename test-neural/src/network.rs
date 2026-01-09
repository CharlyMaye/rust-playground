use ndarray::{Array1, Array2, Axis};
use rand::thread_rng;
use rand::Rng;

/// A simple feedforward neural network with one hidden layer.
///
/// This network uses the sigmoid activation function and implements
/// backpropagation for training.
///
/// # Architecture
/// - Input layer (size defined by user)
/// - Hidden layer with sigmoid activation
/// - Output layer with sigmoid activation
pub struct Network {
    /// Weights connecting input layer to hidden layer (hidden_size × input_size)
    weights1: Array2<f64>,
    /// Biases for the hidden layer
    biases1: Array1<f64>,

    /// Weights connecting hidden layer to output layer (output_size × hidden_size)
    weights2: Array2<f64>,
    /// Biases for the output layer
    biases2: Array1<f64>,
}

impl Network {
    /// Creates a new neural network with random weights and biases.
    ///
    /// # Arguments
    /// - `input_size`: Number of input neurons
    /// - `hidden_size`: Number of neurons in the hidden layer
    /// - `output_size`: Number of output neurons
    ///
    /// # Example
    /// ```
    /// let network = Network::new(2, 3, 1); // XOR problem: 2 inputs, 3 hidden, 1 output
    /// ```
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
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
        }
    }
}

impl Network {
    /// Sigmoid activation function: 1 / (1 + e^-x)
    ///
    /// Maps input values to range [0, 1].
    fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
        x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }
 
    /// Derivative of the sigmoid function: σ(x) * (1 - σ(x))
    ///
    /// Used during backpropagation to calculate gradients.
    fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
        x * &(1.0 - x)
    }
 
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
        let hidden_output = Self::sigmoid(&hidden_input);
        let final_input = self.weights2.dot(&hidden_output) + &self.biases2;
        let final_output = Self::sigmoid(&final_input);
 
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
        let output_delta = &output_errors * &Self::sigmoid_derivative(&final_output);
 
        let hidden_errors = self.weights2.t().dot(&output_delta);
        let hidden_delta = &hidden_errors * &Self::sigmoid_derivative(&hidden_output);
 
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