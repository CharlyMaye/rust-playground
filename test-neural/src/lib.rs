//! Test Neural Network - A simple neural network library in Rust
//!
//! This library provides a flexible feedforward neural network implementation
//! with support for:
//! - Multiple activation functions
//! - Multiple loss functions  
//! - Deep architectures (multiple hidden layers)
//! - Xavier/He weight initialization
//! - Model serialization (JSON and binary)
//!
//! # Quick Start
//!
//! ```rust
//! use test_neural::network::{Network, Activation, LossFunction};
//! use ndarray::array;
//!
//! // Create a network for XOR problem
//! let mut network = Network::new(
//!     2, 5, 1,
//!     Activation::Tanh,
//!     Activation::Sigmoid,
//!     LossFunction::BinaryCrossEntropy
//! );
//!
//! // Train it
//! let input = array![0.0, 1.0];
//! let target = array![1.0];
//! network.train(&input, &target, 0.1);
//!
//! // Make predictions
//! let prediction = network.predict(&input);
//! ```

pub mod network;
pub mod io;
