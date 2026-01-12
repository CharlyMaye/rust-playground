//! Test Neural Network - A simple neural network library in Rust
//!
//! This library provides a flexible feedforward neural network implementation
//! with support for:
//! - Multiple activation functions
//! - Multiple loss functions  
//! - Deep architectures (multiple hidden layers)
//! - Xavier/He weight initialization
//! - Advanced optimizers (SGD, Momentum, RMSprop, Adam, AdamW)
//! - Model serialization (JSON and binary)
//! - Evaluation metrics (accuracy, precision, recall, F1, confusion matrix, ROC/AUC)
//!
//! # Quick Start
//!
//! ```rust
//! use test_neural::network::{Network, Activation, LossFunction};
//! use test_neural::optimizer::OptimizerType;
//! use ndarray::array;
//!
//! // Create a network for XOR problem with Adam optimizer
//! let mut network = Network::new(
//!     2, 5, 1,
//!     Activation::Tanh,
//!     Activation::Sigmoid,
//!     LossFunction::BinaryCrossEntropy,
//!     OptimizerType::adam(0.001)
//! );
//!
//! // Train it (learning rate is now in the optimizer)
//! let input = array![0.0, 1.0];
//! let target = array![1.0];
//! network.train(&input, &target);
//!
//! // Make predictions
//! let prediction = network.predict(&input);
//! ```

pub mod network;
pub mod optimizer;
pub mod dataset;
pub mod io;
pub mod metrics;
