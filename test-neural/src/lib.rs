//! # Test Neural Network
//!
//! A flexible feedforward neural network library in Rust.
//!
//! ## Features
//!
//! - **Multiple activation functions**: Sigmoid, Tanh, ReLU, LeakyReLU, ELU, SELU, Swish, GELU, Mish, Softmax, etc.
//! - **Multiple loss functions**: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
//! - **Deep architectures**: Support for multiple hidden layers
//! - **Weight initialization**: Xavier, He, LeCun (automatic selection based on activation)
//! - **Advanced optimizers**: SGD, Momentum, RMSprop, Adam, AdamW
//! - **Regularization**: L1, L2, Elastic Net, Dropout
//! - **Model serialization**: JSON (human-readable) and Binary (compact)
//! - **Evaluation metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC/AUC
//! - **Training utilities**: Mini-batch training, callbacks, early stopping, learning rate scheduling
//!
//! ## Quick Start
//!
//! ```rust
//! use test_neural::builder::NetworkBuilder;
//! use test_neural::network::{Activation, LossFunction};
//! use test_neural::optimizer::OptimizerType;
//! use ndarray::array;
//!
//! // Create a network using the builder pattern
//! let mut network = NetworkBuilder::new(2, 1)
//!     .hidden_layer(8, Activation::Tanh)
//!     .output_activation(Activation::Sigmoid)
//!     .loss(LossFunction::BinaryCrossEntropy)
//!     .optimizer(OptimizerType::adam(0.001))
//!     .build();
//!
//! // Train on a single example
//! let input = array![0.0, 1.0];
//! let target = array![1.0];
//! network.train(&input, &target);
//!
//! // Make predictions
//! let prediction = network.predict(&input);
//! ```
//!
//! ## Mini-batch Training with Callbacks
//!
//! ```rust,ignore
//! use test_neural::builder::{NetworkBuilder, NetworkTrainer};
//! use test_neural::callbacks::{EarlyStopping, ProgressBar};
//! use test_neural::dataset::Dataset;
//!
//! let mut network = NetworkBuilder::new(2, 1)
//!     .hidden_layer(8, Activation::Tanh)
//!     .build();
//!
//! let history = network.trainer()
//!     .train_data(&train_dataset)
//!     .validation_data(&val_dataset)
//!     .epochs(100)
//!     .batch_size(32)
//!     .callback(Box::new(EarlyStopping::new(10, 0.0001)))
//!     .fit();
//! ```

pub mod network;
pub mod optimizer;
pub mod dataset;
pub mod callbacks;
pub mod io;
pub mod metrics;
pub mod builder;
