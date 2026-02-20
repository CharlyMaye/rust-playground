//! # CMA-Models: Historic Deep Learning Architectures
//!
//! Faithful reimplementations of the CNN architectures that shaped the history of Deep Learning.
//!
//! ## Ecosystem
//!
//! ```text
//! cma-models ← YOU ARE HERE (Ready-to-use architectures)
//!     │
//!     └── cma-cnn (Layers: Conv2D, MaxPool2D, BatchNorm2D)
//!             │
//!             └── cma-neural-network (Base: Dense, Activations, Optimizers)
//! ```
//!
//! ## Available Architectures
//!
//! | Year | Architecture | Paper | Usage |
//! |-------|--------------|-------|-------|
//! | 1998  | LeNet-5      | LeCun et al. | MNIST, small images |
//! | 2012  | AlexNet      | Krizhevsky et al. | ImageNet, DL revolution |
//! | 2014  | VGG          | Simonyan & Zisserman | Feature extraction |
//! | 2015  | ResNet       | He et al. | Very deep networks |
//! | 2019  | EfficientNet | Tan & Le | Efficient state of the art |
//!
//! ## Quick Example
//!
//! ```rust,ignore
//! use cma_models::lenet::LeNet5;
//!
//! // Create LeNet-5 for MNIST (10 classes)
//! let model = LeNet5::new(10);
//! println!("Params: {}", model.num_parameters());
//!
//! // Forward pass
//! let input = Tensor4D::random(TensorShape::new(1, 1, 28, 28));
//! let features = model.forward(&input);
//! ```
//!
//! ## Academic References
//!
//! - LeCun et al. (1998): "Gradient-Based Learning Applied to Document Recognition"
//! - Krizhevsky et al. (2012): "ImageNet Classification with Deep CNNs"
//! - Simonyan & Zisserman (2014): "Very Deep Convolutional Networks"
//! - He et al. (2015): "Deep Residual Learning for Image Recognition"
//! - Tan & Le (2019): "EfficientNet: Rethinking Model Scaling"

pub mod alexnet;
pub mod efficientnet;
pub mod lenet;
pub mod resnet;
pub mod vgg;

// Re-exports
pub use alexnet::{AlexNet, AlexNetConfig};
pub use efficientnet::{EfficientNetB0, EfficientNetConfig, MBConvBlock};
pub use lenet::{LeNet5, LeNet5Config};
pub use resnet::{
    ResNet, ResNet18, ResNet34, ResNet50, ResNetBuilder, ResNetConfig, ResidualBlock,
};
pub use vgg::{VGG16, VGG19, VGGConfig};

// Re-exports from cma-cnn
pub use cma_cnn::{
    ActivationLayer, AvgPool2D, BatchNorm2D, Conv2D, Dropout2D, Flatten, GlobalAvgPool2D, Layer,
    MaxPool2D, Sequential, Tensor4D, TensorShape,
};
