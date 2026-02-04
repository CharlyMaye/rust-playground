//! # Sequential Container
//!
//! Container pour empiler des couches séquentiellement.
//!
//! ## Exemple
//!
//! ```rust,ignore
//! use cma_cnn::{Sequential, Conv2D, MaxPool2D, Flatten, ActivationLayer};
//!
//! let model = Sequential::new()
//!     .add_conv2d(Conv2D::new(1, 32, 5, 1, 0))
//!     .add_activation(ActivationLayer::relu())
//!     .add_pool(MaxPool2D::new(2, 2))
//!     .add_flatten(Flatten::new());
//! ```

use serde::{Deserialize, Serialize};

use crate::layers::{
    ActivationLayer, AvgPool2D, BatchNorm2D, Conv2D, Dropout2D, Flatten, GlobalAvgPool2D, Layer,
    MaxPool2D,
};
use crate::tensor::{Tensor4D, TensorShape};

/// Couche boxée pour stockage hétérogène
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoxedLayer {
    Conv2D(Conv2D),
    MaxPool2D(MaxPool2D),
    AvgPool2D(AvgPool2D),
    GlobalAvgPool2D(GlobalAvgPool2D),
    BatchNorm2D(BatchNorm2D),
    Dropout2D(Dropout2D),
    Flatten(Flatten),
    Activation(ActivationLayer),
}

impl BoxedLayer {
    fn as_layer(&self) -> &dyn Layer {
        match self {
            BoxedLayer::Conv2D(l) => l,
            BoxedLayer::MaxPool2D(l) => l,
            BoxedLayer::AvgPool2D(l) => l,
            BoxedLayer::GlobalAvgPool2D(l) => l,
            BoxedLayer::BatchNorm2D(l) => l,
            BoxedLayer::Dropout2D(l) => l,
            BoxedLayer::Flatten(l) => l,
            BoxedLayer::Activation(l) => l,
        }
    }
}

/// Container séquentiel pour empiler des couches
///
/// # Architecture
///
/// Les couches sont exécutées dans l'ordre d'ajout:
/// ```text
/// Input → Layer1 → Layer2 → ... → LayerN → Output
/// ```
///
/// # Exemple (LeNet-5 style)
///
/// ```rust,ignore
/// let model = Sequential::new()
///     // Block 1
///     .add_conv2d(Conv2D::new(1, 6, 5, 1, 0))    // 28x28 → 24x24
///     .add_activation(ActivationLayer::relu())
///     .add_pool(MaxPool2D::new(2, 2))            // 24x24 → 12x12
///     // Block 2
///     .add_conv2d(Conv2D::new(6, 16, 5, 1, 0))   // 12x12 → 8x8
///     .add_activation(ActivationLayer::relu())
///     .add_pool(MaxPool2D::new(2, 2))            // 8x8 → 4x4
///     // Flatten
///     .add_flatten(Flatten::new());              // 16*4*4 = 256
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequential {
    layers: Vec<BoxedLayer>,
    /// Nom du modèle (pour debug/logging)
    name: String,
}

impl Sequential {
    /// Crée un nouveau Sequential vide
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            name: "Sequential".to_string(),
        }
    }

    /// Crée un Sequential avec un nom personnalisé
    pub fn named(name: &str) -> Self {
        Self {
            layers: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Ajoute une couche Conv2D
    pub fn add_conv2d(mut self, layer: Conv2D) -> Self {
        self.layers.push(BoxedLayer::Conv2D(layer));
        self
    }

    /// Ajoute une couche MaxPool2D
    pub fn add_maxpool(mut self, layer: MaxPool2D) -> Self {
        self.layers.push(BoxedLayer::MaxPool2D(layer));
        self
    }

    /// Ajoute une couche AvgPool2D
    pub fn add_avgpool(mut self, layer: AvgPool2D) -> Self {
        self.layers.push(BoxedLayer::AvgPool2D(layer));
        self
    }

    /// Ajoute une couche GlobalAvgPool2D
    pub fn add_global_avgpool(mut self) -> Self {
        self.layers
            .push(BoxedLayer::GlobalAvgPool2D(GlobalAvgPool2D::new()));
        self
    }

    /// Ajoute une couche BatchNorm2D
    pub fn add_batchnorm(mut self, layer: BatchNorm2D) -> Self {
        self.layers.push(BoxedLayer::BatchNorm2D(layer));
        self
    }

    /// Ajoute une couche Dropout2D
    pub fn add_dropout(mut self, layer: Dropout2D) -> Self {
        self.layers.push(BoxedLayer::Dropout2D(layer));
        self
    }

    /// Ajoute une couche Flatten
    pub fn add_flatten(mut self) -> Self {
        self.layers.push(BoxedLayer::Flatten(Flatten::new()));
        self
    }

    /// Ajoute une couche d'activation
    pub fn add_activation(mut self, layer: ActivationLayer) -> Self {
        self.layers.push(BoxedLayer::Activation(layer));
        self
    }

    /// Raccourci: Conv2D + ReLU
    pub fn add_conv_relu(
        self,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        self.add_conv2d(Conv2D::new(in_ch, out_ch, kernel, stride, padding))
            .add_activation(ActivationLayer::relu())
    }

    /// Raccourci: Conv2D + BatchNorm + ReLU
    pub fn add_conv_bn_relu(
        self,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        self.add_conv2d(Conv2D::new(in_ch, out_ch, kernel, stride, padding).without_bias())
            .add_batchnorm(BatchNorm2D::new(out_ch))
            .add_activation(ActivationLayer::relu())
    }

    /// Propagation avant (version avec référence, clone nécessaire)
    pub fn forward(&self, input: &Tensor4D) -> Tensor4D {
        self.forward_owned(input.clone())
    }

    /// Propagation avant optimisée (prend ownership, évite le clone)
    pub fn forward_owned(&self, input: Tensor4D) -> Tensor4D {
        let mut x = input;
        for layer in &self.layers {
            x = layer.as_layer().forward(&x);
        }
        x
    }

    /// Nombre total de paramètres
    pub fn num_parameters(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.as_layer().num_parameters())
            .sum()
    }

    /// Nombre de couches
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Calcule la shape de sortie pour une shape d'entrée donnée
    pub fn output_shape(&self, input_shape: TensorShape) -> TensorShape {
        let mut shape = input_shape;
        for layer in &self.layers {
            shape = layer.as_layer().output_shape(shape);
        }
        shape
    }

    /// Affiche un résumé du modèle (style Keras)
    pub fn summary(&self, input_shape: TensorShape) {
        println!("Model: {}", self.name);
        println!("{}", "=".repeat(70));
        println!(
            "{:<30} {:>20} {:>15}",
            "Layer (type)", "Output Shape", "Param #"
        );
        println!("{}", "=".repeat(70));

        let mut shape = input_shape;
        let mut total_params = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            let l = layer.as_layer();
            shape = l.output_shape(shape);
            let params = l.num_parameters();
            total_params += params;

            println!(
                "{:<30} {:>20} {:>15}",
                format!("{} ({})", l.summary(), i),
                format!("{}", shape),
                params
            );
        }

        println!("{}", "=".repeat(70));
        println!("Total params: {}", total_params);
        println!("Trainable params: {}", total_params);
        println!("Non-trainable params: 0");
        println!("{}", "=".repeat(70));
    }

    /// Passe en mode évaluation (désactive dropout, utilise running stats pour BatchNorm)
    pub fn eval_mode(&mut self) {
        for layer in &mut self.layers {
            match layer {
                BoxedLayer::BatchNorm2D(bn) => bn.eval_mode(),
                BoxedLayer::Dropout2D(d) => d.eval_mode(),
                _ => {}
            }
        }
    }

    /// Passe en mode entraînement
    pub fn train_mode(&mut self) {
        for layer in &mut self.layers {
            match layer {
                BoxedLayer::BatchNorm2D(bn) => bn.train_mode(),
                BoxedLayer::Dropout2D(d) => d.train_mode(),
                _ => {}
            }
        }
    }

    /// Accès aux couches
    pub fn layers(&self) -> &[BoxedLayer] {
        &self.layers
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_creation() {
        let model = Sequential::new()
            .add_conv2d(Conv2D::new(1, 32, 3, 1, 1))
            .add_activation(ActivationLayer::relu())
            .add_maxpool(MaxPool2D::new(2, 2));

        assert_eq!(model.num_layers(), 3);
    }

    #[test]
    fn test_sequential_output_shape() {
        let model = Sequential::new()
            .add_conv2d(Conv2D::new(1, 32, 5, 1, 0)) // 28→24
            .add_maxpool(MaxPool2D::new(2, 2)) // 24→12
            .add_conv2d(Conv2D::new(32, 64, 5, 1, 0)) // 12→8
            .add_maxpool(MaxPool2D::new(2, 2)) // 8→4
            .add_flatten(); // 64*4*4=1024

        let input_shape = TensorShape::new(1, 1, 28, 28);
        let output_shape = model.output_shape(input_shape);

        assert_eq!(output_shape.width, 64 * 4 * 4);
    }

    #[test]
    fn test_sequential_forward() {
        let model = Sequential::new()
            .add_conv2d(Conv2D::new(1, 8, 3, 1, 1))
            .add_activation(ActivationLayer::relu())
            .add_maxpool(MaxPool2D::new(2, 2));

        let input = Tensor4D::random(TensorShape::new(2, 1, 28, 28));
        let output = model.forward(&input);

        assert_eq!(output.shape().batch, 2);
        assert_eq!(output.shape().channels, 8);
        assert_eq!(output.shape().height, 14);
        assert_eq!(output.shape().width, 14);
    }

    #[test]
    fn test_sequential_params_count() {
        let model = Sequential::new()
            .add_conv2d(Conv2D::new(1, 6, 5, 1, 0)) // 1*6*5*5 + 6 = 156
            .add_conv2d(Conv2D::new(6, 16, 5, 1, 0)); // 6*16*5*5 + 16 = 2416

        assert_eq!(model.num_parameters(), 156 + 2416);
    }

    #[test]
    fn test_conv_bn_relu_shortcut() {
        let model = Sequential::new().add_conv_bn_relu(1, 32, 3, 1, 1);

        assert_eq!(model.num_layers(), 3); // Conv + BN + ReLU

        // Conv sans bias + BN (gamma + beta)
        // 32*1*3*3 = 288 + 64 = 352
        assert_eq!(model.num_parameters(), 288 + 64);
    }
}
