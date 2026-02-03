//! # LeNet-5 (LeCun et al., 1998)
//!
//! Première architecture CNN à succès commercial, utilisée pour la reconnaissance
//! de chiffres manuscrits sur les chèques bancaires.
//!
//! ## Paper Original
//!
//! **"Gradient-Based Learning Applied to Document Recognition"**
//! Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
//! Proceedings of the IEEE, 1998
//! http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
//!
//! ## Architecture Originale (32x32 input)
//!
//! ```text
//! Input (32x32x1)
//!     │
//! ┌───▼───────────────────┐
//! │ C1: Conv 5x5, 6 maps  │ → 28x28x6
//! │ S2: AvgPool 2x2       │ → 14x14x6
//! └───┬───────────────────┘
//!     │
//! ┌───▼───────────────────┐
//! │ C3: Conv 5x5, 16 maps │ → 10x10x16
//! │ S4: AvgPool 2x2       │ → 5x5x16
//! └───┬───────────────────┘
//!     │
//! ┌───▼───────────────────┐
//! │ C5: Conv 5x5, 120     │ → 1x1x120
//! │ F6: FC 120 → 84       │
//! │ Output: FC 84 → 10    │
//! └───────────────────────┘
//! ```
//!
//! ## Adaptation MNIST (28x28 input)
//!
//! Pour MNIST, on utilise un padding initial ou on adapte les tailles.
//! Cette implémentation supporte les deux modes.
//!
//! ## Innovations Clés (1998)
//!
//! 1. **Partage de poids** (weight sharing): même filtre appliqué partout
//! 2. **Sous-échantillonnage** (subsampling): réduit la dimension spatiale
//! 3. **Architecture profonde**: 7 couches (révolutionnaire pour l'époque)
//! 4. **Backpropagation bout-en-bout**: entraînement différentiable
//!
//! ## Exemple
//!
//! ```rust,ignore
//! use cma_models::lenet::{LeNet5, LeNet5Config};
//!
//! // Version standard pour MNIST
//! let model = LeNet5::new(10);
//!
//! // Version personnalisée
//! let model = LeNet5::with_config(LeNet5Config {
//!     num_classes: 10,
//!     input_size: 28,
//!     use_batch_norm: true,  // Modernisation
//!     activation: "relu",     // ReLU au lieu de tanh
//! });
//! ```

use serde::{Deserialize, Serialize};

use cma_cnn::{
    ActivationLayer, AvgPool2D, BatchNorm2D, Conv2D, Flatten, MaxPool2D, Sequential, Tensor4D,
    TensorShape,
};

/// Configuration de LeNet-5
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeNet5Config {
    /// Nombre de classes en sortie (10 pour MNIST)
    pub num_classes: usize,
    /// Taille de l'image d'entrée (28 pour MNIST, 32 pour original)
    pub input_size: usize,
    /// Utiliser BatchNorm (modernisation, pas dans le paper original)
    pub use_batch_norm: bool,
    /// Activation: "tanh" (original) ou "relu" (moderne)
    pub activation: String,
    /// Nombre de canaux d'entrée (1 pour grayscale)
    pub in_channels: usize,
}

impl Default for LeNet5Config {
    fn default() -> Self {
        Self {
            num_classes: 10,
            input_size: 28,
            use_batch_norm: false,
            activation: "tanh".to_string(), // Fidèle au paper
            in_channels: 1,
        }
    }
}

impl LeNet5Config {
    /// Config pour MNIST (28x28)
    pub fn mnist() -> Self {
        Self::default()
    }

    /// Config originale du paper (32x32)
    pub fn original() -> Self {
        Self {
            num_classes: 10,
            input_size: 32,
            use_batch_norm: false,
            activation: "tanh".to_string(),
            in_channels: 1,
        }
    }

    /// Config moderne avec ReLU et BatchNorm
    pub fn modern() -> Self {
        Self {
            num_classes: 10,
            input_size: 28,
            use_batch_norm: true,
            activation: "relu".to_string(),
            in_channels: 1,
        }
    }
}

/// LeNet-5: Architecture CNN Historique (1998)
///
/// # Architecture
///
/// | Couche | Type | Output Shape | Params |
/// |--------|------|--------------|--------|
/// | Input | - | 1×28×28 | 0 |
/// | C1 | Conv 5×5, 6 | 6×24×24 | 156 |
/// | S2 | AvgPool 2×2 | 6×12×12 | 0 |
/// | C3 | Conv 5×5, 16 | 16×8×8 | 2,416 |
/// | S4 | AvgPool 2×2 | 16×4×4 | 0 |
/// | C5 | Conv 4×4, 120 | 120×1×1 | 30,840 |
/// | **Total** | | | **~33k** |
///
/// Note: Le paper original utilisait une table de connexion partielle pour C3,
/// cette implémentation utilise des connexions complètes (standard moderne).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeNet5 {
    /// Couches convolutionnelles
    pub conv_layers: Sequential,
    /// Configuration
    pub config: LeNet5Config,
}

impl LeNet5 {
    /// Crée LeNet-5 avec configuration par défaut pour MNIST
    ///
    /// # Arguments
    /// * `num_classes` - Nombre de classes (10 pour MNIST)
    ///
    /// # Exemple
    /// ```rust,ignore
    /// let model = LeNet5::new(10);
    /// assert_eq!(model.num_parameters(), 33_456);
    /// ```
    pub fn new(num_classes: usize) -> Self {
        let mut config = LeNet5Config::mnist();
        config.num_classes = num_classes;
        Self::with_config(config)
    }

    /// Crée LeNet-5 avec configuration personnalisée
    pub fn with_config(config: LeNet5Config) -> Self {
        let activation = match config.activation.as_str() {
            "relu" => ActivationLayer::relu(),
            "sigmoid" => ActivationLayer::sigmoid(),
            _ => ActivationLayer::tanh(), // Default: tanh (original)
        };

        // Calcul de la taille du kernel C5 pour avoir output 1x1
        // Pour input 28x28: après C1(5x5) → 24, S2(2x2) → 12, C3(5x5) → 8, S4(2x2) → 4
        // Donc C5 kernel = 4 pour 28x28
        // Pour input 32x32: après C1 → 28, S2 → 14, C3 → 10, S4 → 5
        // Donc C5 kernel = 5 pour 32x32
        let c5_kernel = if config.input_size == 32 { 5 } else { 4 };

        let mut conv_layers = Sequential::named("LeNet-5");

        // C1: Convolution Layer (6 feature maps, 5x5 kernel)
        conv_layers = conv_layers.add_conv2d(Conv2D::new(config.in_channels, 6, 5, 1, 0));
        if config.use_batch_norm {
            conv_layers = conv_layers.add_batchnorm(BatchNorm2D::new(6));
        }
        conv_layers = conv_layers.add_activation(activation.clone());

        // S2: Subsampling (Average Pooling 2x2)
        // Note: Le paper original utilisait une forme de pooling avec poids appris
        conv_layers = conv_layers.add_avgpool(AvgPool2D::new(2, 2));

        // C3: Convolution Layer (16 feature maps, 5x5 kernel)
        conv_layers = conv_layers.add_conv2d(Conv2D::new(6, 16, 5, 1, 0));
        if config.use_batch_norm {
            conv_layers = conv_layers.add_batchnorm(BatchNorm2D::new(16));
        }
        conv_layers = conv_layers.add_activation(activation.clone());

        // S4: Subsampling (Average Pooling 2x2)
        conv_layers = conv_layers.add_avgpool(AvgPool2D::new(2, 2));

        // C5: Convolution Layer (120 feature maps)
        // Dans le paper, C5 est en fait une couche fully-connected déguisée
        conv_layers = conv_layers.add_conv2d(Conv2D::new(16, 120, c5_kernel, 1, 0));
        if config.use_batch_norm {
            conv_layers = conv_layers.add_batchnorm(BatchNorm2D::new(120));
        }
        conv_layers = conv_layers.add_activation(activation);

        // Flatten pour connexion aux couches FC
        conv_layers = conv_layers.add_flatten();

        Self {
            conv_layers,
            config,
        }
    }

    /// Propagation avant (couches conv uniquement)
    ///
    /// Retourne les features aplaties prêtes pour les couches Dense.
    /// Utilisez un Network de cma-neural-network pour les couches FC.
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        self.conv_layers.forward(input)
    }

    /// Nombre total de paramètres (couches conv)
    pub fn num_parameters(&self) -> usize {
        self.conv_layers.num_parameters()
    }

    /// Affiche le résumé du modèle
    pub fn summary(&self) {
        let input_shape = TensorShape::new(
            1,
            self.config.in_channels,
            self.config.input_size,
            self.config.input_size,
        );
        self.conv_layers.summary(input_shape);
    }

    /// Shape de sortie (features aplaties)
    pub fn output_size(&self) -> usize {
        let input_shape = TensorShape::new(
            1,
            self.config.in_channels,
            self.config.input_size,
            self.config.input_size,
        );
        let output = self.conv_layers.output_shape(input_shape);
        output.width // Après flatten
    }
}

/// Crée le classifieur FC pour LeNet-5
///
/// # Architecture FC (paper original)
/// - F6: 120 → 84 (tanh)
/// - Output: 84 → num_classes
///
/// # Exemple
/// ```rust,ignore
/// use cma_neural_network::NetworkBuilder;
/// use cma_models::lenet::{LeNet5, create_lenet5_classifier};
///
/// let cnn = LeNet5::new(10);
/// let classifier = create_lenet5_classifier(cnn.output_size(), 10);
/// ```
pub fn create_lenet5_classifier(input_size: usize, num_classes: usize) -> String {
    // Retourne la configuration recommandée pour le NetworkBuilder
    format!(
        r#"// Classifieur FC pour LeNet-5
// Input: {} features (sortie du CNN)
// Output: {} classes

use cma_neural_network::{{NetworkBuilder, Activation, LossFunction, OptimizerType}};

let classifier = NetworkBuilder::new({}, {})
    .hidden_layer(84, Activation::Tanh)  // F6 du paper
    .output_activation(Activation::Softmax)
    .loss(LossFunction::CategoricalCrossEntropy)
    .optimizer(OptimizerType::adam(0.001))
    .build();"#,
        input_size, num_classes, input_size, num_classes
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lenet5_creation() {
        let model = LeNet5::new(10);
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_lenet5_mnist_shape() {
        let model = LeNet5::new(10);
        let input = Tensor4D::zeros(TensorShape::new(1, 1, 28, 28));
        let output = model.forward(&input);

        // Après C1(5x5) → 24, S2 → 12, C3(5x5) → 8, S4 → 4, C5(4x4) → 1
        // Flatten: 120 * 1 * 1 = 120
        assert_eq!(output.shape().width, 120);
    }

    #[test]
    fn test_lenet5_original_shape() {
        let model = LeNet5::with_config(LeNet5Config::original());
        let input = Tensor4D::zeros(TensorShape::new(1, 1, 32, 32));
        let output = model.forward(&input);

        // Pour 32x32: C1 → 28, S2 → 14, C3 → 10, S4 → 5, C5(5x5) → 1
        assert_eq!(output.shape().width, 120);
    }

    #[test]
    fn test_lenet5_modern() {
        let model = LeNet5::with_config(LeNet5Config::modern());
        // Avec BatchNorm, plus de paramètres
        assert!(model.num_parameters() > LeNet5::new(10).num_parameters());
    }

    #[test]
    fn test_lenet5_batch() {
        let model = LeNet5::new(10);
        let batch = Tensor4D::random(TensorShape::new(32, 1, 28, 28));
        let output = model.forward(&batch);

        assert_eq!(output.shape().batch, 32);
        assert_eq!(output.shape().width, 120);
    }
}
