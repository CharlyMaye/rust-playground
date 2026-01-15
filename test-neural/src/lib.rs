//! # Test Neural Network
//!
//! This crate re-exports `cma_neural_network` for backward compatibility.
//!
//! All functionality is provided by the `cma-neural-network` library crate.
//!
//! ## Quick Start
//!
//! ```rust
//! use test_neural::prelude::*;
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

// Re-export the entire cma_neural_network crate
pub use cma_neural_network::*;

/// Prelude module for convenient imports
pub mod prelude {
    pub use cma_neural_network::{
        // Core types
        Network, Activation, LossFunction, WeightInit, RegularizationType, DropoutConfig,
        // Optimizer
        OptimizerType,
        // Dataset
        Dataset,
        // Builder
        NetworkBuilder, NetworkTrainer, TrainingBuilder,
        // Callbacks
        Callback, EarlyStopping, DeltaMode, LearningRateScheduler, LRSchedule, ModelCheckpoint, ProgressBar,
        // Metrics
        accuracy, binary_metrics, BinaryMetrics,
        // I/O
        save_json, load_json, save_binary, load_binary, IoError,
    };
}
