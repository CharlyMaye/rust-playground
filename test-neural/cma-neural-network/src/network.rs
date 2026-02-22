use crate::{Dim, Float};
use crate::callbacks::Callback;
use crate::optimizer::{OptimizerState1D, OptimizerState2D, OptimizerType};
use ndarray::{Array1, Array2, Zip};
use rand::Rng;
use rand::SeedableRng;
use rand::rng;
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};

/// Regularization type to prevent overfitting.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RegularizationType {
    /// No regularization
    None,
    /// L1 regularization (Lasso) - encourages sparsity
    L1 { lambda: Float },
    /// L2 regularization (Ridge/Weight Decay) - penalizes large weights
    L2 { lambda: Float },
    /// Elastic Net - combines L1 and L2
    ElasticNet { l1_ratio: Float, lambda: Float },
}

impl RegularizationType {
    /// Creates L1 regularization with the specified lambda.
    pub fn l1(lambda: Float) -> Self {
        RegularizationType::L1 { lambda }
    }

    /// Creates L2 regularization with the specified lambda (typical: 0.0001 - 0.01).
    pub fn l2(lambda: Float) -> Self {
        RegularizationType::L2 { lambda }
    }

    /// Creates Elastic Net regularization.
    pub fn elastic_net(l1_ratio: Float, lambda: Float) -> Self {
        RegularizationType::ElasticNet { l1_ratio, lambda }
    }

    /// Computes the regularization penalty on weights.
    pub fn penalty(&self, weights: &Array2<Float>) -> Float {
        match self {
            RegularizationType::None => 0.0,
            // iter().map().sum(): single pass, no intermediate array (avoids mapv alloc)
            RegularizationType::L1 { lambda } => {
                lambda * weights.iter().map(|&w| w.abs()).sum::<Float>()
            }
            RegularizationType::L2 { lambda } => {
                0.5 * lambda * weights.iter().map(|&w| w * w).sum::<Float>()
            }
            RegularizationType::ElasticNet { l1_ratio, lambda } => {
                let l1_part = l1_ratio * weights.iter().map(|&w| w.abs()).sum::<Float>();
                let l2_part =
                    0.5 * (1.0 - l1_ratio) * weights.iter().map(|&w| w * w).sum::<Float>();
                lambda * (l1_part + l2_part)
            }
        }
    }

    /// Computes the regularization gradient to add to weight gradients.
    /// Returns None if no regularization (to avoid allocation).
    pub fn gradient_opt(&self, weights: &Array2<Float>) -> Option<Array2<Float>> {
        match self {
            RegularizationType::None => None,
            RegularizationType::L1 { lambda } => Some(weights.mapv(|w| lambda * w.signum())),
            RegularizationType::L2 { lambda } => Some(weights.mapv(|w| lambda * w)),
            RegularizationType::ElasticNet { l1_ratio, lambda } => {
                Some(weights.mapv(|w| lambda * (l1_ratio * w.signum() + (1.0 - l1_ratio) * w)))
            }
        }
    }

    /// Computes the regularization gradient to add to weight gradients.
    pub fn gradient(&self, weights: &Array2<Float>) -> Array2<Float> {
        self.gradient_opt(weights)
            .unwrap_or_else(|| Array2::zeros(weights.dim()))
    }
}

/// Dropout configuration for a layer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DropoutConfig {
    /// Probability of deactivating a neuron (0.0 = no dropout, 0.5 = 50% deactivated)
    pub rate: Float,
}

impl DropoutConfig {
    /// Creates a dropout configuration with the specified rate.
    pub fn new(rate: Float) -> Self {
        assert!(
            (0.0..1.0).contains(&rate),
            "Dropout rate must be in [0.0, 1.0)"
        );
        DropoutConfig { rate }
    }
}

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
    fn initialize_weights(&self, rows: usize, cols: usize, rng: &mut impl Rng) -> Array2<Float> {
        let std = match self {
            WeightInit::Uniform => {
                return Array2::from_shape_fn((rows, cols), |_| rng.random_range(-1.0..1.0));
            }
            // Xavier: std = sqrt(2 / (fan_in + fan_out))
            WeightInit::Xavier => (2.0 / (rows + cols) as Float).sqrt(),
            // He: std = sqrt(2 / fan_in)
            WeightInit::He => (2.0 / cols as Float).sqrt(),
            // LeCun: std = sqrt(1 / fan_in)
            WeightInit::LeCun => (1.0 / cols as Float).sqrt(),
        };
        let data = crate::init::randn_vec(rows * cols, std, rng);
        Array2::from_shape_vec((rows, cols), data).unwrap()
    }

    /// Get recommended initialization method for an activation function.
    pub fn for_activation(activation: Activation) -> Self {
        match activation {
            Activation::Sigmoid
            | Activation::Tanh
            | Activation::Softsign
            | Activation::HardSigmoid
            | Activation::HardTanh
            | Activation::Softmax => WeightInit::Xavier,
            Activation::ReLU
            | Activation::LeakyReLU
            | Activation::ELU
            | Activation::GELU
            | Activation::Swish
            | Activation::Mish
            | Activation::Softplus => WeightInit::He,
            Activation::SELU => WeightInit::LeCun,
            Activation::Linear => WeightInit::Xavier,
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
    pub fn compute(&self, predictions: &Array1<Float>, targets: &Array1<Float>) -> Float {
        match self {
            LossFunction::MSE => {
                let diff = predictions - targets;
                (&diff * &diff).sum() / predictions.len() as Float
            }
            LossFunction::MAE => {
                // Zip fold: avoids allocating (predictions - targets) + second mapv for abs
                Zip::from(predictions)
                    .and(targets)
                    .fold(0.0, |acc, &p, &t| acc + (p - t).abs())
                    / predictions.len() as Float
            }
            LossFunction::BinaryCrossEntropy => {
                let epsilon: Float = 1e-15;
                let n = predictions.len() as Float;
                Zip::from(predictions)
                    .and(targets)
                    .fold(0.0, |acc, &p, &t| {
                        let pc = p.max(epsilon).min(1.0 - epsilon);
                        acc - (t * pc.ln() + (1.0 - t) * (1.0 - pc).ln())
                    })
                    / n
            }
            LossFunction::CategoricalCrossEntropy => {
                let epsilon: Float = 1e-15;
                Zip::from(predictions)
                    .and(targets)
                    .fold(0.0, |acc, &p, &t| acc - t * p.max(epsilon).ln())
            }
            LossFunction::Huber => {
                let delta: Float = 1.0;
                let n = predictions.len() as Float;
                // Zip fold: avoids allocating `diff` then mapv then sum (3 passes → 1)
                Zip::from(predictions)
                    .and(targets)
                    .fold(0.0, |acc, &p, &t| {
                        let d = p - t;
                        acc + if d.abs() <= delta {
                            0.5 * d * d
                        } else {
                            delta * (d.abs() - 0.5 * delta)
                        }
                    })
                    / n
            }
        }
    }

    /// Compute the derivative (gradient) of the loss function.
    /// Returns the error signal to be backpropagated.
    pub fn derivative(
        &self,
        predictions: &Array1<Float>,
        targets: &Array1<Float>,
    ) -> Array1<Float> {
        match self {
            LossFunction::MSE => {
                // d/dx[(y - x)^2] = -2(y - x) = 2(x - y)
                // Simplified for gradient descent: (x - y)
                predictions - targets
            }
            LossFunction::MAE => {
                // d/dx[|y - x|] = sign(x - y)
                (predictions - targets).mapv(|x| {
                    if x > 0.0 {
                        1.0
                    } else if x < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
            }
            LossFunction::BinaryCrossEntropy => {
                // d/dx[-y*ln(x) - (1-y)*ln(1-x)] = (x - y) / (x(1-x))
                let epsilon: Float = 1e-15;
                // Fused Zip: clamp and derivative in one pass — avoids intermediate p_clamped array
                Zip::from(predictions)
                    .and(targets)
                    .map_collect(|&p, &t| {
                        let pc = p.max(epsilon).min(1.0 - epsilon);
                        (pc - t) / (pc * (1.0 - pc))
                    })
            }
            LossFunction::CategoricalCrossEntropy => {
                // d/dx[-y*ln(x)] = -y/x
                let epsilon: Float = 1e-15;
                Zip::from(predictions)
                    .and(targets)
                    .map_collect(|&p, &t| -t / p.max(epsilon))
            }
            LossFunction::Huber => {
                let delta = 1.0;
                // Fused Zip: avoids allocating `diff` then `mapv(...)` (2 allocs → 1)
                Zip::from(predictions)
                    .and(targets)
                    .map_collect(|&p, &t| {
                        let d = p - t;
                        if d.abs() <= delta { d } else { delta * d.signum() }
                    })
            }
        }
    }
}

impl Activation {
    /// Returns the name of the activation function as a string.
    pub fn name(&self) -> &'static str {
        match self {
            Activation::Sigmoid => "Sigmoid",
            Activation::Tanh => "Tanh",
            Activation::ReLU => "ReLU",
            Activation::LeakyReLU => "LeakyReLU",
            Activation::ELU => "ELU",
            Activation::SELU => "SELU",
            Activation::Swish => "Swish",
            Activation::GELU => "GELU",
            Activation::Mish => "Mish",
            Activation::Softplus => "Softplus",
            Activation::Softsign => "Softsign",
            Activation::HardSigmoid => "HardSigmoid",
            Activation::HardTanh => "HardTanh",
            Activation::Softmax => "Softmax",
            Activation::Linear => "Linear",
        }
    }

    /// Apply the activation function to a single scalar value.
    pub fn apply_scalar(&self, x: Float) -> Float {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            Activation::ELU => if x > 0.0 { x } else { x.exp() - 1.0 },
            Activation::SELU => {
                let lambda = 1.0507;
                let alpha = 1.6733;
                lambda * if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
            }
            Activation::Swish => x / (1.0 + (-x).exp()),
            Activation::GELU => {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }
            Activation::Mish => x * ((1.0 + x.exp()).ln()).tanh(),
            Activation::Softplus => (1.0 + x.exp()).ln(),
            Activation::Softsign => x / (1.0 + x.abs()),
            Activation::HardSigmoid => (0.2 * x + 0.5).clamp(0.0, 1.0),
            Activation::HardTanh => x.clamp(-1.0, 1.0),
            Activation::Softmax => x, // Softmax requires full context; identity for scalar
            Activation::Linear => x,
        }
    }

    /// Apply the activation function to an array.
    pub fn apply(&self, x: &Array1<Float>) -> Array1<Float> {
        match self {
            Activation::Softmax => {
                // Softmax with numerical stability
                let max = x.fold(Float::NEG_INFINITY, |a, &b| a.max(b));
                // mapv once + in-place /= — avoids the second allocation from `exp_x / sum`
                let mut exp_x = x.mapv(|v| (v - max).exp());
                exp_x /= exp_x.sum();
                exp_x
            }
            Activation::Linear => x.clone(),
            _ => x.mapv(|v| self.apply_scalar(v)),
        }
    }

    /// Compute the derivative of the activation function from POST-activation values.
    /// Use this for Sigmoid, Tanh, ReLU, LeakyReLU, HardSigmoid, HardTanh, Linear.
    /// For other activations, prefer `derivative_from_preactivation`.
    pub fn derivative(&self, a: &Array1<Float>) -> Array1<Float> {
        match self {
            // These can be computed from post-activation
            Activation::Sigmoid => a * &(1.0 - a),
            Activation::Tanh => a.mapv(|a| 1.0 - a.powi(2)),
            Activation::ReLU => a.mapv(|a| if a > 0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyReLU => a.mapv(|a| if a > 0.0 { 1.0 } else { 0.01 }),
            Activation::HardSigmoid => a.mapv(|a| {
                // a = clamp(0.2*z + 0.5, 0, 1), so if a is strictly between 0 and 1
                if a > 0.0 && a < 1.0 { 0.2 } else { 0.0 }
            }),
            Activation::HardTanh => a.mapv(|a| if a > -1.0 && a < 1.0 { 1.0 } else { 0.0 }),
            Activation::Linear => Array1::ones(a.len()),
            // For Softmax, the Jacobian is complex but when combined with CCE,
            // it simplifies to (output - target). This fallback is rarely used.
            Activation::Softmax => a * &(1.0 - a),
            // For the following, we need pre-activation z, so we approximate
            // by using the stored pre-activation when available
            _ => {
                // Fallback: these should use derivative_from_preactivation
                // Return ones as a safe fallback (caller should use correct method)
                Array1::ones(a.len())
            }
        }
    }

    /// Compute the derivative of the activation function from PRE-activation values (z).
    /// This is mathematically correct for all activation functions.
    pub fn derivative_from_preactivation(&self, z: &Array1<Float>) -> Array1<Float> {
        match self {
            Activation::Sigmoid => {
                // Single mapv: fuses σ(x) and σ(x)*(1-σ(x)) — was 3 allocs (sig, 1-sig, multiply)
                z.mapv(|x| { let s = 1.0 / (1.0 + (-x).exp()); s * (1.0 - s) })
            }
            Activation::Tanh => z.mapv(|x| 1.0 - x.tanh().powi(2)),
            Activation::ReLU => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }),
            Activation::LeakyReLU => z.mapv(|x| if x > 0.0 { 1.0 } else { 0.01 }),
            Activation::ELU => {
                let alpha = 1.0;
                z.mapv(|x| if x > 0.0 { 1.0 } else { alpha * x.exp() })
            }
            Activation::SELU => {
                let lambda = 1.0507;
                let alpha = 1.6733;
                z.mapv(|x| {
                    if x > 0.0 {
                        lambda
                    } else {
                        lambda * alpha * x.exp()
                    }
                })
            }
            Activation::Swish => {
                // Swish(z) = z * sigmoid(z)
                // Swish'(z) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
                //           = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
                z.mapv(|x| {
                    let sig = 1.0 / (1.0 + (-x).exp());
                    sig * (1.0 + x * (1.0 - sig))
                })
            }
            Activation::GELU => {
                // GELU(z) ≈ 0.5 * z * (1 + tanh(sqrt(2/π) * (z + 0.044715 * z³)))
                // Derivative: more complex, using the standard approximation
                let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
                z.mapv(|x| {
                    let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                    let tanh_inner = inner.tanh();
                    let sech2 = 1.0 - tanh_inner.powi(2);
                    let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x.powi(2));
                    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
                })
            }
            Activation::Mish => {
                // Mish(z) = z * tanh(softplus(z)) = z * tanh(ln(1 + e^z))
                // Mish'(z) = tanh(sp) + z * sech²(sp) * sigmoid(z)
                // where sp = softplus(z) = ln(1 + e^z)
                z.mapv(|x| {
                    let sp = (1.0 + x.exp()).ln(); // softplus
                    let tanh_sp = sp.tanh();
                    let sech2_sp = 1.0 - tanh_sp.powi(2);
                    let sigmoid = 1.0 / (1.0 + (-x).exp());
                    tanh_sp + x * sech2_sp * sigmoid
                })
            }
            Activation::Softplus => {
                // Softplus(z) = ln(1 + e^z)
                // Softplus'(z) = sigmoid(z)
                z.mapv(|x| 1.0 / (1.0 + (-x).exp()))
            }
            Activation::Softsign => {
                // Softsign(z) = z / (1 + |z|)
                // Softsign'(z) = 1 / (1 + |z|)²
                z.mapv(|x| 1.0 / (1.0 + x.abs()).powi(2))
            }
            Activation::HardSigmoid => {
                // HardSigmoid(z) = clamp(0.2*z + 0.5, 0, 1)
                z.mapv(|x| {
                    let val = 0.2 * x + 0.5;
                    if val > 0.0 && val < 1.0 { 0.2 } else { 0.0 }
                })
            }
            Activation::HardTanh => z.mapv(|x| if x > -1.0 && x < 1.0 { 1.0 } else { 0.0 }),
            Activation::Softmax => {
                // For Softmax, the full Jacobian is complex.
                // When used with CCE loss, the combined gradient simplifies.
                // This is handled specially in train() and train_batch().
                //
                // SAFETY: This code path should NEVER be reached in correct usage.
                // Softmax must be paired with CategoricalCrossEntropy, which bypasses
                // this derivative entirely (using the simplified target - output).
                unreachable!(
                    "Softmax derivative should not be called directly. \
                     Use Softmax + CategoricalCrossEntropy which simplifies to (output - target). \
                     If you see this error, check your loss function configuration."
                )
            }
            Activation::Linear => Array1::ones(z.len()),
        }
    }

    /// Returns true if this activation requires pre-activation (z) for correct derivative.
    pub fn needs_preactivation(&self) -> bool {
        matches!(
            self,
            Activation::ELU
                | Activation::SELU
                | Activation::Swish
                | Activation::GELU
                | Activation::Mish
                | Activation::Softplus
                | Activation::Softsign
        )
    }
}

/// A layer in the neural network.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Layer {
    pub(crate) weights: Array2<Float>,
    pub(crate) biases: Array1<Float>,
    pub(crate) activation: Activation,
    pub(crate) dropout: Option<DropoutConfig>,
}

/// Result of a forward pass, containing all information needed for backpropagation.
#[derive(Clone)]
pub(crate) struct ForwardResult {
    /// Pre-activation values (z) for each layer
    pub(crate) pre_activations: Vec<Array1<Float>>,
    /// Post-activation values (a) for each layer, including input at index 0
    pub(crate) activations: Vec<Array1<Float>>,
    /// Dropout masks for each layer (None if no dropout or eval mode)
    pub(crate) dropout_masks: Vec<Option<Array1<Float>>>,
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
/// # Reproducibility
/// Set a seed with `set_seed()` for reproducible training (dropout masks).
///
/// # Example
/// ```rust
/// use cma_neural_network::builder::NetworkBuilder;
/// use cma_neural_network::network::{Activation, LossFunction};
/// use cma_neural_network::optimizer::OptimizerType;
///
/// // Use the builder pattern to create networks
/// let network = NetworkBuilder::new(2, 1)
///     .hidden_layer(8, Activation::Tanh)
///     .output_activation(Activation::Sigmoid)
///     .loss(LossFunction::BinaryCrossEntropy)
///     .optimizer(OptimizerType::adam(0.001))
///     .build();
/// ```
#[derive(Serialize, Deserialize)]
pub struct Network {
    /// All layers (hidden + output)
    pub(crate) layers: Vec<Layer>,
    /// Input size for reference
    pub(crate) input_size: Dim,
    /// Loss function for training
    pub(crate) loss_function: LossFunction,
    /// Optimizer type
    pub(crate) optimizer: OptimizerType,
    /// Optimizer states for weights
    pub(crate) optimizer_states_weights: Vec<OptimizerState2D>,
    /// Optimizer states for biases
    pub(crate) optimizer_states_biases: Vec<OptimizerState1D>,
    /// Regularization type (L1, L2, None)
    pub(crate) regularization: RegularizationType,
    /// Training mode (true = apply dropout, false = inference mode)
    pub(crate) training_mode: bool,
    /// Optional seed for reproducibility (None = use system entropy)
    #[serde(skip)]
    pub(crate) rng_seed: Option<u64>,
    /// Cached RNG for performance (recreated if seed changes)
    #[serde(skip)]
    pub(crate) rng: Option<StdRng>,
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
    /// **Internal method**: Use `NetworkBuilder` for construction.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_deep_with_init(
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
            let biases = Array1::zeros(size); // Biases initialized to 0

            layers.push(Layer {
                weights,
                biases,
                activation: hidden_activations[i],
                dropout: None, // No dropout by default
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
            dropout: None, // No dropout on the output layer
        });

        // Initialize optimizer states for all layers
        let optimizer_states_weights: Vec<OptimizerState2D> = layers
            .iter()
            .map(|layer| {
                let shape = layer.weights.dim();
                OptimizerState2D::new(shape, &optimizer)
            })
            .collect();

        let optimizer_states_biases: Vec<OptimizerState1D> = layers
            .iter()
            .map(|layer| {
                let size = layer.biases.len();
                OptimizerState1D::new(size, &optimizer)
            })
            .collect();

        Network {
            layers,
            input_size: input_size as Dim,
            loss_function,
            optimizer,
            optimizer_states_weights,
            optimizer_states_biases,
            regularization: RegularizationType::None,
            training_mode: true,
            rng_seed: None,
            rng: None,
        }
    }

    /// Sets a seed for reproducible training.
    ///
    /// When a seed is set, dropout masks will be deterministic,
    /// making training reproducible across runs.
    ///
    /// # Example
    /// ```rust
    /// use cma_neural_network::builder::NetworkBuilder;
    /// use cma_neural_network::network::Activation;
    ///
    /// let mut network = NetworkBuilder::new(2, 1)
    ///     .hidden_layer(8, Activation::ReLU)
    ///     .build();
    ///
    /// network.set_seed(42);  // Reproducible training
    /// ```
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_seed = Some(seed);
        self.rng = Some(StdRng::seed_from_u64(seed));
    }

    /// Clears the seed, using system entropy for randomness.
    pub fn clear_seed(&mut self) {
        self.rng_seed = None;
        self.rng = None;
    }

    /// Returns the current seed if set.
    pub fn seed(&self) -> Option<u64> {
        self.rng_seed
    }

    /// Switches to training mode (enables dropout).
    pub fn train_mode(&mut self) {
        self.training_mode = true;
    }

    /// Switches to evaluation/inference mode (disables dropout).
    pub fn eval_mode(&mut self) {
        self.training_mode = false;
    }
}

impl Network {
    /// Forward pass for evaluation (no dropout, no RNG needed).
    /// Always runs in "eval mode" regardless of training_mode flag.
    fn forward_eval(&self, input: &Array1<Float>) -> Vec<Array1<Float>> {
        let mut activations = vec![input.clone()];

        for layer in &self.layers {
            let z = layer.weights.dot(activations.last().unwrap()) + &layer.biases;
            let a = layer.activation.apply(&z);
            // No dropout applied in eval mode
            activations.push(a);
        }

        activations
    }

    /// Internal forward pass implementation.
    ///
    /// This is exposed to the trainer module for backpropagation.
    pub(crate) fn forward_full_internal(
        &self,
        input: &Array1<Float>,
        rng: &mut impl Rng,
    ) -> ForwardResult {
        let mut activations = vec![input.clone()];
        let mut pre_activations = Vec::with_capacity(self.layers.len());
        let mut dropout_masks = Vec::with_capacity(self.layers.len());

        // Forward pass through all layers
        for layer in &self.layers {
            let z = layer.weights.dot(activations.last().unwrap()) + &layer.biases;
            let mut a = layer.activation.apply(&z);

            // Store pre-activation
            pre_activations.push(z);

            // Apply dropout if in training mode
            let mask = if self.training_mode
                && let Some(dropout_config) = layer.dropout
            {
                let keep_prob = 1.0 - dropout_config.rate;
                // Create dropout mask with inverted scaling
                let mask: Array1<Float> = Array1::from_shape_fn(a.len(), |_| {
                    if rng.random::<Float>() < keep_prob {
                        1.0 / keep_prob // Inverted dropout (scaling during training)
                    } else {
                        0.0
                    }
                });
                a = &a * &mask;
                Some(mask)
            } else {
                None
            };
            dropout_masks.push(mask);

            activations.push(a);
        }

        ForwardResult {
            pre_activations,
            activations,
            dropout_masks,
        }
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
    /// # Algorithm
    /// 1. Forward pass to get all activations and pre-activations
    /// 2. Calculate output layer error using loss function
    /// 3. Backpropagate error through all hidden layers (with dropout mask)
    /// 4. Update all weights and biases using the optimizer
    pub fn train(&mut self, input: &Array1<Float>, target: &Array1<Float>) {
        // Delegate to the Trainer (CPU by default)
        let mut trainer = crate::trainer::Trainer::cpu(self);
        trainer.train_single(input, target);
    }

    /// Train the network on a batch of examples (mini-batch training).
    ///
    /// This method is more efficient than calling `train()` multiple times because:
    /// - Gradients are accumulated over the entire batch
    /// - Optimizer updates are applied once per batch instead of once per example
    /// - Provides more stable gradient estimates (reduced variance)
    /// - Better utilization of vectorized operations
    ///
    /// # Arguments
    /// - `inputs`: Vector of input arrays (batch of inputs)
    /// - `targets`: Vector of target arrays (batch of targets)
    ///
    /// # Panics
    /// Panics if inputs.len() != targets.len() or if batch is empty
    ///
    /// # Example
    /// ```rust
    /// use cma_neural_network::builder::NetworkBuilder;
    /// use cma_neural_network::network::Activation;
    /// use ndarray::array;
    ///
    /// let mut network = NetworkBuilder::new(2, 1)
    ///     .hidden_layer(5, Activation::Tanh)
    ///     .build();
    ///
    /// let inputs = vec![array![0.0, 0.0], array![0.0, 1.0], array![1.0, 0.0]];
    /// let targets = vec![array![0.0], array![1.0], array![1.0]];
    ///
    /// network.train_batch(&inputs, &targets);
    /// ```
    pub fn train_batch(&mut self, inputs: &[Array1<Float>], targets: &[Array1<Float>]) {
        // Delegate to the Trainer (CPU by default)
        let mut trainer = crate::trainer::Trainer::cpu(self);
        trainer.train_batch(inputs, targets);
    }

    /// Evaluates the network on given input-target pairs without updating weights.
    ///
    /// Returns the average loss over all samples, including regularization penalty.
    ///
    /// # Arguments
    /// - `inputs`: Vector of input arrays
    /// - `targets`: Vector of target arrays
    ///
    /// # Returns
    /// Average loss value
    pub fn evaluate(&self, inputs: &[Array1<Float>], targets: &[Array1<Float>]) -> Float {
        let mut total_loss = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            // Always use eval mode for evaluation (no dropout)
            let activations = self.forward_eval(input);
            let prediction = activations.last().unwrap();
            total_loss += self.loss_function.compute(prediction, target);
        }

        let base_loss = total_loss / inputs.len() as Float;

        // Add regularization penalty
        let reg_penalty: Float = self
            .layers
            .iter()
            .map(|layer| self.regularization.penalty(&layer.weights))
            .sum();

        base_loss + reg_penalty / inputs.len() as Float
    }

    /// Makes a prediction for a single input.
    ///
    /// This is the main inference method - use it to get predictions after training.
    /// Always runs without dropout, regardless of training_mode.
    ///
    /// # Arguments
    /// - `input`: Input vector
    ///
    /// # Returns
    /// Output vector (network's prediction)
    pub fn predict(&self, input: &Array1<Float>) -> Array1<Float> {
        // Always use eval mode for predictions (no dropout)
        let activations = self.forward_eval(input);
        activations.last().unwrap().clone()
    }

    /// Trains the network with support for callbacks and optional learning rate scheduler.
    ///
    /// **Internal method**: Use `network.trainer().fit()` instead.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fit(
        &mut self,
        train_dataset: &mut crate::dataset::Dataset,
        val_dataset: Option<&crate::dataset::Dataset>,
        epochs: usize,
        batch_size: usize,
        device: crate::compute::ComputeDevice,
        mut scheduler: Option<&mut crate::callbacks::LearningRateScheduler>,
        callbacks: &mut Vec<Box<dyn crate::callbacks::Callback>>,
        eval_every: usize,
    ) -> Vec<(Float, Option<Float>)> {
        // Initialize the scheduler if it exists
        if let Some(sched) = scheduler.as_mut() {
            sched.current_lr = match &self.optimizer {
                crate::optimizer::OptimizerType::SGD { learning_rate } => *learning_rate,
                crate::optimizer::OptimizerType::Momentum { learning_rate, .. } => *learning_rate,
                crate::optimizer::OptimizerType::RMSprop { learning_rate, .. } => *learning_rate,
                crate::optimizer::OptimizerType::Adam { learning_rate, .. } => *learning_rate,
                crate::optimizer::OptimizerType::AdamW { learning_rate, .. } => *learning_rate,
            };
            sched.on_train_begin(self);
        }

        // Call on_train_begin
        for callback in callbacks.iter_mut() {
            callback.on_train_begin(self);
        }

        let mut history = Vec::new();

        // Create trainer ONCE for all epochs (performance!)
        let mut trainer = crate::trainer::Trainer::new(self, device)
            .expect("Device should be validated before calling fit()");

        for epoch in 0..epochs {
            // Call on_epoch_begin
            if let Some(sched) = scheduler.as_mut() {
                sched.on_epoch_begin(epoch, trainer.network_mut());
            }
            for callback in callbacks.iter_mut() {
                callback.on_epoch_begin(epoch, trainer.network_mut());
            }

            // Shuffle dataset in-place (no clone!)
            train_dataset.shuffle();

            // Train on batches using iterator (no allocations!)
            for (batch_inputs, batch_targets) in train_dataset.batches(batch_size) {
                trainer.train_batch(batch_inputs, batch_targets);
            }

            // Evaluate losses only every eval_every epochs (or last epoch)
            let should_evaluate = (epoch + 1) % eval_every == 0 || epoch + 1 == epochs;

            let (train_loss, val_loss) = if should_evaluate {
                let train_loss = trainer
                    .network()
                    .evaluate(train_dataset.inputs(), train_dataset.targets());
                let val_loss =
                    val_dataset.map(|val| trainer.network().evaluate(val.inputs(), val.targets()));
                (train_loss, val_loss)
            } else {
                // Skip evaluation, use dummy values
                (0.0, None)
            };

            history.push((train_loss, val_loss));

            // Call scheduler on_epoch_end and update
            if let Some(sched) = scheduler.as_mut() {
                sched.on_epoch_end(epoch, trainer.network_mut(), train_loss, val_loss);
                sched.update_optimizer_lr(&mut trainer.network_mut().optimizer);
            }

            // Call on_epoch_end
            let mut should_continue = true;
            for callback in callbacks.iter_mut() {
                if !callback.on_epoch_end(epoch, trainer.network_mut(), train_loss, val_loss) {
                    should_continue = false;
                    break;
                }
            }

            if !should_continue {
                break;
            }
        }

        // Call on_train_end
        if let Some(sched) = scheduler.as_mut() {
            sched.on_train_end(trainer.network_mut());
        }
        for callback in callbacks.iter_mut() {
            callback.on_train_end(trainer.network_mut());
        }

        history
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Public API for introspection (useful for visualization)
    // ═══════════════════════════════════════════════════════════════════════

    /// Returns a string representation of the network architecture.
    /// Example: "2 → [8, 4] → 1"
    pub fn architecture_string(&self) -> String {
        let hidden_sizes: Vec<String> = self
            .layers
            .iter()
            .take(self.layers.len().saturating_sub(1))
            .map(|l| l.weights.nrows().to_string())
            .collect();

        let output_size = self.layers.last().map(|l| l.weights.nrows()).unwrap_or(0);

        if hidden_sizes.is_empty() {
            format!("{} → {}", self.input_size, output_size)
        } else {
            format!(
                "{} → [{}] → {}",
                self.input_size,
                hidden_sizes.join(", "),
                output_size
            )
        }
    }

    /// Returns the number of layers (hidden + output).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the input size of the network.
    pub fn input_size(&self) -> usize {
        self.input_size as usize
    }

    /// Returns the output size of the network.
    pub fn output_size(&self) -> usize {
        self.layers.last().map(|l| l.weights.nrows()).unwrap_or(0)
    }

    /// Returns information about each layer for visualization.
    /// Each tuple contains: (weights, biases, activation_name)
    pub fn get_layers_info(&self) -> Vec<(&Array2<Float>, &Array1<Float>, &str)> {
        self.layers
            .iter()
            .map(|l| (&l.weights, &l.biases, l.activation.name()))
            .collect()
    }

    /// Performs a forward pass and returns all intermediate activations.
    /// Returns: Vec of (pre_activation, post_activation, activation_name) for each layer.
    pub fn get_all_activations(
        &self,
        input: &Array1<Float>,
    ) -> Vec<(Array1<Float>, Array1<Float>, String)> {
        let mut current = input.clone();
        let mut results = Vec::new();

        for layer in &self.layers {
            let pre = layer.weights.dot(&current) + &layer.biases;
            let post = layer.activation.apply(&pre);
            results.push((pre, post.clone(), layer.activation.name().to_string()));
            current = post;
        }

        results
    }
}
