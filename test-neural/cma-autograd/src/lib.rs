//! # CMA-Autograd: Automatic Differentiation Engine
//!
//! Automatic differentiation system for neural networks in Rust.
//!
//! ## Architecture
//!
//! ```text
//! cma-models (Architectures: LeNet, ResNet, VGG)
//!     │
//!     ├── cma-cnn (Conv2D, MaxPool2D, BatchNorm2D)
//!     │       │
//!     │       └── cma-neural-network (Dense, Activations, Optimizers)
//!     │
//!     └── cma-autograd (THIS CRATE: Autograd, Tensor, Backward) ← YOU ARE HERE
//!             │
//!             └── cma-neural-network (Float type, base)
//! ```
//!
//! ## Key Concepts
//!
//! - **Tensor**: Data structure with gradient tracking
//! - **GradFn**: Backward functions attached to operations
//! - **Engine**: Backpropagation engine (topological sort)
//! - **Layer/TrainableLayer**: Trait hierarchy for layers
//! - **Parameter**: Wrapper for trainable tensors
//!
//! ## Quick Example
//!
//! ```rust,ignore
//! use cma_autograd::prelude::*;
//!
//! // Create tensors with gradient tracking
//! let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], true);
//! let w = Tensor::from_vec(vec![0.5, -0.3, 0.8], &[3], true);
//!
//! // Operations build the computation graph automatically
//! let y = &x * &w;
//! let loss = y.sum();
//!
//! // Backward propagates the gradients
//! loss.backward();
//!
//! // Gradients are accessible
//! println!("dL/dw = {:?}", w.grad());
//! ```

pub mod engine;
pub mod grad_fn;
pub mod layers;
pub mod loss;
pub mod module;
pub mod ops;
pub mod optim;
pub mod sequential;
pub mod tensor;
pub mod cnn_ops;
pub mod builder;

#[cfg(feature = "export")]
pub mod export;

// Re-export Float from cma-neural-network
pub use cma_neural_network::Float;

/// Prelude — import everything you need with `use cma_autograd::prelude::*`
pub mod prelude {
    pub use crate::builder::{CnnBuilder, ConvPoolBlock};
    pub use crate::engine::{no_grad, NoGradGuard};
    pub use crate::layers::{
        AvgPool2D, BatchNorm2D, Dropout, Flatten, GlobalAvgPool2D,
        MaxPool2D as MaxPool2DLayer, ReLU as ReLULayer,
        Sigmoid as SigmoidLayer, Softmax, Tanh as TanhLayer,
    };
    pub use crate::loss::{binary_cross_entropy_loss, cross_entropy_loss, mse_loss};
    pub use crate::module::{Conv2D as Conv2DModule, Linear, Module, Parameter, TrainableLayer};
    pub use crate::ops;
    pub use crate::optim::{Adam, Optimizer, SGD};
    pub use crate::sequential::{CnnTrainer, EpochMetrics, Sequential};
    pub use crate::tensor::Tensor;
    pub use crate::Float;
}
