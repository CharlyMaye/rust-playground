//! # CMA-Models: Architectures Historiques de Deep Learning
//!
//! Réimplémentations fidèles des architectures CNN qui ont marqué l'histoire du Deep Learning.
//!
//! ## Écosystème
//!
//! ```text
//! cma-models ← VOUS ÊTES ICI (Architectures prêtes à l'emploi)
//!     │
//!     └── cma-cnn (Couches: Conv2D, MaxPool2D, BatchNorm2D)
//!             │
//!             └── cma-neural-network (Base: Dense, Activations, Optimiseurs)
//! ```
//!
//! ## Architectures Disponibles
//!
//! | Année | Architecture | Paper | Usage |
//! |-------|--------------|-------|-------|
//! | 1998  | LeNet-5      | LeCun et al. | MNIST, petites images |
//! | 2012  | AlexNet      | Krizhevsky et al. | ImageNet, révolution DL |
//! | 2014  | VGG          | Simonyan & Zisserman | Features extraction |
//! | 2015  | ResNet       | He et al. | Réseaux très profonds |
//! | 2019  | EfficientNet | Tan & Le | État de l'art efficace |
//!
//! ## Exemple Rapide
//!
//! ```rust,ignore
//! use cma_models::lenet::LeNet5;
//!
//! // Crée LeNet-5 pour MNIST (10 classes)
//! let model = LeNet5::new(10);
//! println!("Params: {}", model.num_parameters());
//!
//! // Forward pass
//! let input = Tensor4D::random(TensorShape::new(1, 1, 28, 28));
//! let features = model.forward(&input);
//! ```
//!
//! ## Références Académiques
//!
//! - LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition"
//! - Krizhevsky et al. (2012): "ImageNet Classification with Deep CNNs"
//! - Simonyan & Zisserman (2014): "Very Deep Convolutional Networks"
//! - He et al. (2015): "Deep Residual Learning for Image Recognition"
//! - Tan & Le (2019): "EfficientNet: Rethinking Model Scaling"

pub mod lenet;
pub mod alexnet;
pub mod vgg;
pub mod resnet;
pub mod efficientnet;

// Re-exports
pub use lenet::{LeNet5, LeNet5Config};
pub use alexnet::{AlexNet, AlexNetConfig};
pub use vgg::{VGG16, VGG19, VGGConfig};
pub use resnet::{ResNet18, ResNet34, ResNet50, ResNetConfig, ResidualBlock};
pub use efficientnet::{EfficientNetB0, EfficientNetConfig, MBConvBlock};

// Re-exports from cma-cnn
pub use cma_cnn::{
    Tensor4D, TensorShape,
    Sequential, Conv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D,
    BatchNorm2D, Dropout2D, Flatten, ActivationLayer,
    Layer,
};
