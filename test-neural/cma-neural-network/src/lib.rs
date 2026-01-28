//! # CMA Neural Network
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
//! use cma_neural_network::builder::NetworkBuilder;
//! use cma_neural_network::network::{Activation, LossFunction};
//! use cma_neural_network::optimizer::OptimizerType;
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
//! use cma_neural_network::builder::{NetworkBuilder, NetworkTrainer};
//! use cma_neural_network::callbacks::{EarlyStopping, ProgressBar};
//! use cma_neural_network::dataset::Dataset;
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

pub mod builder;
pub mod callbacks;
pub mod compute;
pub mod dataset;
pub mod io;
pub mod metrics;
pub mod network;
pub mod optimizer;

// Internal modules (not exposed in public API)
pub(crate) mod trainer;

// Re-exports for convenience
pub use builder::{NetworkBuilder, NetworkTrainer, TrainingBuilder};
pub use callbacks::{
    Callback, DeltaMode, EarlyStopping, LRSchedule, LearningRateScheduler, ModelCheckpoint,
    ProgressBar,
};
pub use compute::{ComputeDevice, ComputeDeviceError};
pub use dataset::Dataset;
pub use io::{IoError, load_binary, load_json, save_binary, save_json};
pub use metrics::{BinaryMetrics, accuracy, binary_metrics};
pub use network::{
    Activation, DropoutConfig, LossFunction, Network, RegularizationType, WeightInit,
};
pub use optimizer::OptimizerType;
