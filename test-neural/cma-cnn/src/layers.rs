//! # Couches CNN
//!
//! Implémentation des couches pour réseaux de neurones convolutifs:
//! - Conv2D: Convolution 2D
//! - MaxPool2D / AvgPool2D: Pooling spatial
//! - BatchNorm2D: Normalisation par batch
//! - Flatten: Conversion 4D → 2D
//!
//! ## Références
//!
//! - LeCun et al. (1998): Convolutions et pooling
//! - Ioffe & Szegedy (2015): Batch Normalization
//! - He et al. (2015): Initialisation pour ReLU

use ndarray::{Array1, Array4};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::ops::{avgpool2d, conv2d_naive, global_avgpool2d, maxpool2d};
use crate::tensor::{Tensor4D, TensorShape};

/// Type de couche
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    GlobalAvgPool2D,
    BatchNorm2D,
    Dropout2D,
    Flatten,
    Activation,
}

/// Trait commun à toutes les couches
pub trait Layer: Send + Sync {
    /// Propagation avant
    fn forward(&self, input: &Tensor4D) -> Tensor4D;

    /// Type de la couche
    fn layer_type(&self) -> LayerType;

    /// Nombre de paramètres entraînables
    fn num_parameters(&self) -> usize;

    /// Shape de sortie étant donné un shape d'entrée
    fn output_shape(&self, input_shape: TensorShape) -> TensorShape;

    /// Description pour debug
    fn summary(&self) -> String;
}

// ═══════════════════════════════════════════════════════════════════════════
// Conv2D - Convolution 2D
// ═══════════════════════════════════════════════════════════════════════════

/// Couche de convolution 2D
///
/// # Architecture (LeCun et al., 1998)
///
/// Applique `out_channels` filtres de taille `kernel_size × kernel_size`
/// sur l'entrée avec `in_channels` canaux.
///
/// # Exemple
///
/// ```rust,ignore
/// // 1 channel input → 32 filters, kernel 3x3
/// let conv = Conv2D::new(1, 32, 3, 1, 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2D {
    /// Nombre de canaux d'entrée
    pub in_channels: usize,
    /// Nombre de filtres (canaux de sortie)
    pub out_channels: usize,
    /// Taille du kernel (carré)
    pub kernel_size: usize,
    /// Stride (pas de déplacement)
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Poids [out_channels, in_channels, kernel_h, kernel_w]
    pub weights: Array4<f64>,
    /// Biais [out_channels]
    pub bias: Array1<f64>,
    /// Utiliser le biais?
    pub use_bias: bool,
}

impl Conv2D {
    /// Crée une nouvelle couche Conv2D avec initialisation He
    ///
    /// # Arguments
    /// * `in_channels` - Nombre de canaux d'entrée (1 pour grayscale, 3 pour RGB)
    /// * `out_channels` - Nombre de filtres
    /// * `kernel_size` - Taille du kernel (ex: 3 pour 3x3)
    /// * `stride` - Pas de déplacement (1 = chaque pixel)
    /// * `padding` - Zéros autour de l'image
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let mut rng = rand::rng();

        // Initialisation He (pour ReLU)
        // Variance = 2 / (fan_in)
        // fan_in = in_channels * kernel_size * kernel_size
        let fan_in = in_channels * kernel_size * kernel_size;
        let std = (2.0 / fan_in as f64).sqrt();

        // Génère les poids avec distribution normale
        let weights_vec: Vec<f64> = (0..out_channels * in_channels * kernel_size * kernel_size)
            .map(|_| {
                // Box-Muller transform
                let u1: f64 = rng.random();
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * std
            })
            .collect();

        let weights = Array4::from_shape_vec(
            (out_channels, in_channels, kernel_size, kernel_size),
            weights_vec,
        )
        .unwrap();

        // Biais initialisé à zéro
        let bias = Array1::zeros(out_channels);

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
            use_bias: true,
        }
    }

    /// Crée Conv2D sans biais (utile avant BatchNorm)
    pub fn without_bias(mut self) -> Self {
        self.use_bias = false;
        self
    }

    /// Crée Conv2D avec padding "same" (conserve la taille)
    pub fn same_padding(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        let padding = kernel_size / 2;
        Self::new(in_channels, out_channels, kernel_size, 1, padding)
    }
}

impl Layer for Conv2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let bias = if self.use_bias {
            Some(&self.bias)
        } else {
            None
        };
        conv2d_naive(input, &self.weights, bias, self.stride, self.padding)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Conv2D
    }

    fn num_parameters(&self) -> usize {
        let weights = self.out_channels * self.in_channels * self.kernel_size * self.kernel_size;
        let bias = if self.use_bias { self.out_channels } else { 0 };
        weights + bias
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape.after_conv(
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
        )
    }

    fn summary(&self) -> String {
        format!(
            "Conv2D({} → {}, {}x{}, stride={}, pad={})",
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MaxPool2D - Max Pooling
// ═══════════════════════════════════════════════════════════════════════════

/// Couche de Max Pooling 2D
///
/// Réduit la dimension spatiale en prenant le maximum sur chaque fenêtre.
/// Introduit l'invariance aux petites translations.
///
/// # Exemple
///
/// ```rust,ignore
/// // Pool 2x2 avec stride 2 → divise la résolution par 2
/// let pool = MaxPool2D::new(2, 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxPool2D {
    pub pool_size: usize,
    pub stride: usize,
}

impl MaxPool2D {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self { pool_size, stride }
    }

    /// Pool 2x2 stride 2 (le plus commun)
    pub fn default_2x2() -> Self {
        Self::new(2, 2)
    }
}

impl Layer for MaxPool2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let (output, _indices) = maxpool2d(input, self.pool_size, self.stride);
        output
    }

    fn layer_type(&self) -> LayerType {
        LayerType::MaxPool2D
    }

    fn num_parameters(&self) -> usize {
        0 // Pas de paramètres entraînables
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape.after_pool(self.pool_size, self.stride)
    }

    fn summary(&self) -> String {
        format!(
            "MaxPool2D({}x{}, stride={})",
            self.pool_size, self.pool_size, self.stride
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AvgPool2D - Average Pooling
// ═══════════════════════════════════════════════════════════════════════════

/// Couche de Average Pooling 2D
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvgPool2D {
    pub pool_size: usize,
    pub stride: usize,
}

impl AvgPool2D {
    pub fn new(pool_size: usize, stride: usize) -> Self {
        Self { pool_size, stride }
    }
}

impl Layer for AvgPool2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        avgpool2d(input, self.pool_size, self.stride)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::AvgPool2D
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape.after_pool(self.pool_size, self.stride)
    }

    fn summary(&self) -> String {
        format!(
            "AvgPool2D({}x{}, stride={})",
            self.pool_size, self.pool_size, self.stride
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GlobalAvgPool2D - Global Average Pooling
// ═══════════════════════════════════════════════════════════════════════════

/// Global Average Pooling 2D
///
/// Réduit [batch, channels, H, W] → [batch, channels, 1, 1]
/// Utilisé dans les architectures modernes (ResNet, EfficientNet) à la place
/// des couches fully-connected.
///
/// # Avantages
/// - Pas de paramètres entraînables
/// - Régularisation implicite
/// - Invariance à la taille d'entrée
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalAvgPool2D;

impl GlobalAvgPool2D {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GlobalAvgPool2D {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for GlobalAvgPool2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        global_avgpool2d(input)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::GlobalAvgPool2D
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape.after_global_pool()
    }

    fn summary(&self) -> String {
        "GlobalAvgPool2D".to_string()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BatchNorm2D - Batch Normalization
// ═══════════════════════════════════════════════════════════════════════════

/// Batch Normalization 2D (Ioffe & Szegedy, 2015)
///
/// Normalise les activations par batch pour stabiliser l'entraînement.
///
/// # Formule
/// ```text
/// y = γ * (x - μ) / √(σ² + ε) + β
/// ```
///
/// # Avantages
/// - Permet des learning rates plus élevés
/// - Réduit la dépendance à l'initialisation
/// - Régularisation implicite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNorm2D {
    pub num_features: usize,
    /// Paramètres appris: scale (γ)
    pub gamma: Array1<f64>,
    /// Paramètres appris: shift (β)
    pub beta: Array1<f64>,
    /// Moyenne courante (pour inference)
    pub running_mean: Array1<f64>,
    /// Variance courante (pour inference)
    pub running_var: Array1<f64>,
    /// Momentum pour running stats
    pub momentum: f64,
    /// Epsilon pour stabilité numérique
    pub eps: f64,
    /// Mode training (true) ou eval (false)
    pub training: bool,
}

impl BatchNorm2D {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            gamma: Array1::ones(num_features),
            beta: Array1::zeros(num_features),
            running_mean: Array1::zeros(num_features),
            running_var: Array1::ones(num_features),
            momentum: 0.1,
            eps: 1e-5,
            training: true,
        }
    }

    pub fn eval_mode(&mut self) {
        self.training = false;
    }

    pub fn train_mode(&mut self) {
        self.training = true;
    }
}

impl Layer for BatchNorm2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        let shape = input.shape();
        let data = input.data();

        let mut output =
            ndarray::Array4::zeros((shape.batch, shape.channels, shape.height, shape.width));

        for c in 0..shape.channels {
            let (mean, var) = if self.training {
                // Calcule mean et var sur le batch
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let n = (shape.batch * shape.height * shape.width) as f64;

                for b in 0..shape.batch {
                    for h in 0..shape.height {
                        for w in 0..shape.width {
                            let val = data[[b, c, h, w]];
                            sum += val;
                            sum_sq += val * val;
                        }
                    }
                }

                let mean = sum / n;
                let var = sum_sq / n - mean * mean;
                (mean, var)
            } else {
                (self.running_mean[c], self.running_var[c])
            };

            let std = (var + self.eps).sqrt();

            for b in 0..shape.batch {
                for h in 0..shape.height {
                    for w in 0..shape.width {
                        let val = data[[b, c, h, w]];
                        let normalized = (val - mean) / std;
                        output[[b, c, h, w]] = self.gamma[c] * normalized + self.beta[c];
                    }
                }
            }
        }

        Tensor4D::from_array(output)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::BatchNorm2D
    }

    fn num_parameters(&self) -> usize {
        // gamma et beta sont appris
        self.num_features * 2
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape // BatchNorm ne change pas la shape
    }

    fn summary(&self) -> String {
        format!("BatchNorm2D({})", self.num_features)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dropout2D - Spatial Dropout
// ═══════════════════════════════════════════════════════════════════════════

/// Dropout 2D (Spatial Dropout)
///
/// Désactive des canaux entiers plutôt que des pixels individuels.
/// Plus efficace pour les CNN car les pixels adjacents sont corrélés.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dropout2D {
    pub rate: f64,
    pub training: bool,
}

impl Dropout2D {
    pub fn new(rate: f64) -> Self {
        Self {
            rate,
            training: true,
        }
    }

    pub fn eval_mode(&mut self) {
        self.training = false;
    }

    pub fn train_mode(&mut self) {
        self.training = true;
    }
}

impl Layer for Dropout2D {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        if !self.training || self.rate == 0.0 {
            return input.clone();
        }

        let mut rng = rand::rng();
        let shape = input.shape();
        let data = input.data();
        let scale = 1.0 / (1.0 - self.rate);

        let mut output = data.clone();

        // Dropout par canal (spatial dropout)
        for b in 0..shape.batch {
            for c in 0..shape.channels {
                let drop: bool = rng.random::<f64>() < self.rate;
                if drop {
                    for h in 0..shape.height {
                        for w in 0..shape.width {
                            output[[b, c, h, w]] = 0.0;
                        }
                    }
                } else {
                    for h in 0..shape.height {
                        for w in 0..shape.width {
                            output[[b, c, h, w]] *= scale;
                        }
                    }
                }
            }
        }

        Tensor4D::from_array(output)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Dropout2D
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape
    }

    fn summary(&self) -> String {
        format!("Dropout2D(p={})", self.rate)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Flatten - 4D → 2D
// ═══════════════════════════════════════════════════════════════════════════

/// Flatten: Convertit [batch, C, H, W] → vecteur pour couches Dense
///
/// Utilisé pour connecter les couches CNN aux couches fully-connected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flatten {
    /// Shape d'entrée (pour unflatten au backward)
    input_shape: Option<TensorShape>,
}

impl Flatten {
    pub fn new() -> Self {
        Self { input_shape: None }
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Flatten {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        // Note: Flatten retourne techniquement un Array2, pas Tensor4D
        // Mais pour l'interface unifiée, on garde Tensor4D avec H=1, W=flat_size
        let shape = input.shape();
        let flat = input.flatten();

        // Convertit Array2 [batch, flat] → Tensor4D [batch, 1, 1, flat]
        let flat_size = shape.channels * shape.height * shape.width;
        let mut data = ndarray::Array4::zeros((shape.batch, 1, 1, flat_size));

        for b in 0..shape.batch {
            for i in 0..flat_size {
                data[[b, 0, 0, i]] = flat[[b, i]];
            }
        }

        Tensor4D::from_array(data)
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Flatten
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        let flat_size = input_shape.channels * input_shape.height * input_shape.width;
        TensorShape::new(input_shape.batch, 1, 1, flat_size)
    }

    fn summary(&self) -> String {
        "Flatten".to_string()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Activation Layer Wrapper
// ═══════════════════════════════════════════════════════════════════════════

/// Couche d'activation (wrapper autour de cma_neural_network::Activation)
///
/// Réutilise les activations de cma-neural-network en les appliquant
/// élément par élément sur les Tensor4D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationLayer {
    pub activation: cma_neural_network::Activation,
}

impl ActivationLayer {
    /// Crée une couche avec l'activation spécifiée
    pub fn new(activation: cma_neural_network::Activation) -> Self {
        Self { activation }
    }

    pub fn relu() -> Self {
        Self::new(cma_neural_network::Activation::ReLU)
    }

    pub fn leaky_relu() -> Self {
        Self::new(cma_neural_network::Activation::LeakyReLU)
    }

    pub fn sigmoid() -> Self {
        Self::new(cma_neural_network::Activation::Sigmoid)
    }

    pub fn tanh() -> Self {
        Self::new(cma_neural_network::Activation::Tanh)
    }

    pub fn gelu() -> Self {
        Self::new(cma_neural_network::Activation::GELU)
    }

    pub fn mish() -> Self {
        Self::new(cma_neural_network::Activation::Mish)
    }

    pub fn swish() -> Self {
        Self::new(cma_neural_network::Activation::Swish)
    }

    pub fn softmax() -> Self {
        Self::new(cma_neural_network::Activation::Softmax)
    }

    pub fn elu() -> Self {
        Self::new(cma_neural_network::Activation::ELU)
    }

    pub fn selu() -> Self {
        Self::new(cma_neural_network::Activation::SELU)
    }
}

impl Layer for ActivationLayer {
    fn forward(&self, input: &Tensor4D) -> Tensor4D {
        // Applique l'activation élément par élément
        // On utilise la même logique que cma_neural_network::Activation::apply
        // mais sur un scalaire au lieu d'un Array1
        input.map(|x| apply_activation_scalar(self.activation, x))
    }

    fn layer_type(&self) -> LayerType {
        LayerType::Activation
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        input_shape
    }

    fn summary(&self) -> String {
        self.activation.name().to_string()
    }
}

/// Applique une activation sur un scalaire
/// Réplique la logique de cma_neural_network::Activation::apply pour un seul élément
fn apply_activation_scalar(activation: cma_neural_network::Activation, x: f64) -> f64 {
    use cma_neural_network::Activation;
    match activation {
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Tanh => x.tanh(),
        Activation::ReLU => x.max(0.0),
        Activation::LeakyReLU => {
            if x > 0.0 {
                x
            } else {
                0.01 * x
            }
        }
        Activation::ELU => {
            if x > 0.0 {
                x
            } else {
                (x.exp() - 1.0)
            }
        }
        Activation::SELU => {
            let lambda = 1.0507;
            let alpha = 1.6733;
            lambda * if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
        }
        Activation::Swish => x / (1.0 + (-x).exp()),
        Activation::GELU => {
            0.5 * x
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        }
        Activation::Mish => x * ((1.0 + x.exp()).ln()).tanh(),
        Activation::Softplus => (1.0 + x.exp()).ln(),
        Activation::Softsign => x / (1.0 + x.abs()),
        Activation::HardSigmoid => (0.2 * x + 0.5).clamp(0.0, 1.0),
        Activation::HardTanh => x.clamp(-1.0, 1.0),
        Activation::Softmax => x, // Softmax nécessite le contexte complet, identité ici
        Activation::Linear => x,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_creation() {
        let conv = Conv2D::new(1, 32, 3, 1, 1);
        assert_eq!(conv.in_channels, 1);
        assert_eq!(conv.out_channels, 32);
        assert_eq!(conv.kernel_size, 3);
        // He init: variance ≈ 2/9
        assert!(conv.weights.iter().all(|&w| w.abs() < 3.0));
    }

    #[test]
    fn test_conv2d_output_shape() {
        let conv = Conv2D::new(1, 32, 5, 1, 0);
        let input_shape = TensorShape::new(1, 1, 28, 28);
        let output_shape = conv.output_shape(input_shape);

        // 28 - 5 + 1 = 24
        assert_eq!(output_shape.height, 24);
        assert_eq!(output_shape.width, 24);
        assert_eq!(output_shape.channels, 32);
    }

    #[test]
    fn test_conv2d_same_padding() {
        let conv = Conv2D::same_padding(1, 32, 3);
        let input_shape = TensorShape::new(1, 1, 28, 28);
        let output_shape = conv.output_shape(input_shape);

        // Same padding préserve la taille
        assert_eq!(output_shape.height, 28);
        assert_eq!(output_shape.width, 28);
    }

    #[test]
    fn test_maxpool2d() {
        let pool = MaxPool2D::default_2x2();
        let input_shape = TensorShape::new(1, 32, 24, 24);
        let output_shape = pool.output_shape(input_shape);

        // 24 / 2 = 12
        assert_eq!(output_shape.height, 12);
        assert_eq!(output_shape.width, 12);
        assert_eq!(output_shape.channels, 32);
    }

    #[test]
    fn test_batchnorm2d() {
        let bn = BatchNorm2D::new(32);
        assert_eq!(bn.num_parameters(), 64); // gamma + beta

        let input_shape = TensorShape::new(4, 32, 14, 14);
        assert_eq!(bn.output_shape(input_shape), input_shape);
    }

    #[test]
    fn test_flatten_output() {
        let flatten = Flatten::new();
        let input_shape = TensorShape::new(4, 64, 7, 7);
        let output_shape = flatten.output_shape(input_shape);

        assert_eq!(output_shape.batch, 4);
        assert_eq!(output_shape.width, 64 * 7 * 7);
    }

    #[test]
    fn test_activation_layer() {
        let relu = ActivationLayer::relu();
        let input = Tensor4D::from_array(ndarray::Array4::from_elem((1, 1, 2, 2), -1.0));
        let output = relu.forward(&input);

        assert!(output.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_layer_parameters_count() {
        // Conv2D: 32 * 1 * 5 * 5 + 32 = 832
        let conv = Conv2D::new(1, 32, 5, 1, 0);
        assert_eq!(conv.num_parameters(), 832);

        // Sans biais: 32 * 1 * 5 * 5 = 800
        let conv_no_bias = Conv2D::new(1, 32, 5, 1, 0).without_bias();
        assert_eq!(conv_no_bias.num_parameters(), 800);

        // Pool n'a pas de paramètres
        let pool = MaxPool2D::new(2, 2);
        assert_eq!(pool.num_parameters(), 0);
    }
}
