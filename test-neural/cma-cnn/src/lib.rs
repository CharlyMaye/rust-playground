//! # CMA-CNN: Convolutional Neural Network Layers
//!
//! Extension de `cma-neural-network` avec des couches convolutionnelles pour le traitement d'images.
//!
//! ## Architecture de l'Écosystème
//!
//! ```text
//! cma-models (Architectures prêtes: LeNet-5, ResNet, etc.)
//!     │
//!     └── cma-cnn (Ce crate: Conv2D, MaxPool2D, BatchNorm2D) ← VOUS ÊTES ICI
//!             │
//!             └── cma-neural-network (Base: Dense, Activations, Optimiseurs)
//! ```
//!
//! ## Fonctionnalités
//!
//! - **Conv2D**: Convolution 2D avec support padding/stride
//! - **MaxPool2D / AvgPool2D**: Pooling spatial
//! - **BatchNorm2D**: Normalisation par batch
//! - **Flatten**: Conversion tenseur 4D → vecteur 1D
//! - **Sequential**: Container pour empiler des couches
//!
//! ## Exemple Rapide
//!
//! ```rust,ignore
//! use cma_cnn::{Conv2D, MaxPool2D, Flatten, Sequential};
//! use cma_cnn::Activation;
//!
//! let model = Sequential::new()
//!     .add(Conv2D::new(1, 32, 3, 1, 1))  // 1 channel → 32 filters, 3x3 kernel
//!     .add(Activation::ReLU)
//!     .add(MaxPool2D::new(2, 2))
//!     .add(Flatten::new())
//!     .add(Dense::new(128, 10));
//! ```
//!
//! ## Références
//!
//! - LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition"
//! - Krizhevsky et al. (2012): "ImageNet Classification with Deep CNNs" (AlexNet)
//! - He et al. (2015): "Deep Residual Learning for Image Recognition" (ResNet)

pub mod layers;
pub mod ops;
pub mod sequential;
pub mod tensor;

// Re-exports
pub use layers::{
    ActivationLayer, AvgPool2D, BatchNorm2D, Conv2D, Dropout2D, Flatten, GlobalAvgPool2D, Layer,
    LayerType, MaxPool2D,
};
pub use ops::{Padding, col2im, conv2d_im2col, im2col, im2col_single};
pub use sequential::Sequential;
pub use tensor::{Tensor4D, TensorShape};

// Re-export from cma-neural-network for convenience
pub use cma_neural_network::{Activation, LossFunction, Network, NetworkBuilder, OptimizerType};
