# 🏗️ Builder Pattern Guide

The Builder Pattern is the recommended API for building and training neural networks with `test-neural`. It provides a fluent, intuitive interface that replaces multiple traditional construction methods.

## Table of Contents

- [Why the Builder Pattern?](#why-the-builder-pattern)
- [NetworkBuilder](#networkbuilder)
- [TrainingBuilder](#trainingbuilder)
- [Complete Examples](#complete-examples)
- [Comparison with Traditional API](#comparison-with-traditional-api)

---

## Why the Builder Pattern?

### Problems Solved

❌ **Before** - Method proliferation:
- `Network::new()` - simple network (1 hidden layer)
- `Network::new_deep()` - deep network (auto init)
- `Network::new_deep_with_init()` - deep network (manual init)
- `Network::fit()` - training with callbacks
- `Network::fit_with_scheduler()` - training with scheduler
- Manual management of `Vec<Box<dyn Callback>>`
- Confusion about which method to use

✅ **After** - One unified way:
- `NetworkBuilder` - intuitive construction via chaining
- `.trainer()` - unified training
- No need to manage Vec manually
- Self-documenting API

---

## NetworkBuilder

### Simple Construction

```rust
use test_neural::builder::NetworkBuilder;
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;

let network = NetworkBuilder::new(2, 1)  // input_size, output_size
    .hidden_layer(8, Activation::Tanh)   // 1 hidden layer
    .build();
```

### Deep Network

```rust
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(16, Activation::ReLU)
    .hidden_layer(8, Activation::ReLU)
    .hidden_layer(4, Activation::Tanh)
    .build();
```

### Complete Configuration

```rust
let network = NetworkBuilder::new(input_size, output_size)
    // Hidden layers
    .hidden_layer(64, Activation::ReLU)
    .hidden_layer(32, Activation::ReLU)
    .hidden_layer(16, Activation::Tanh)
    
    // Output
    .output_activation(Activation::Sigmoid)  // default: Sigmoid
    .loss(LossFunction::BinaryCrossEntropy)  // default: BCE
    
    // Optimizer
    .optimizer(OptimizerType::adam(0.001))   // default: Adam(0.001)
    
    // Regularization
    .dropout(0.3)           // applied to hidden layers
    .l2(0.01)               // L2 regularization
    
    // Optional initialization
    .weight_init(WeightInit::He)  // otherwise: auto based on activation
    
    .build();
```

### Regularization Options

```rust
// L1 (Lasso) - encourages sparsity
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(8, Activation::Tanh)
    .l1(0.001)
    .build();

// L2 (Ridge) - penalizes large weights
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(8, Activation::Tanh)
    .l2(0.01)
    .build();

// Elastic Net - combines L1 and L2
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(8, Activation::Tanh)
    .elastic_net(0.5, 0.01)  // l1_ratio=0.5, lambda=0.01
    .build();

// Dropout + L2 (recommended)
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(16, Activation::ReLU)
    .hidden_layer(8, Activation::ReLU)
    .dropout(0.3)
    .l2(0.001)
    .build();
```

### Default Values

If you don't specify certain options, the defaults are:
- `output_activation`: `Activation::Sigmoid`
- `loss`: `LossFunction::BinaryCrossEntropy`
- `optimizer`: `OptimizerType::adam(0.001)`
- `weight_init`: Auto-detection based on activation
- `dropout`: None
- `regularization`: None

---

## TrainingBuilder

### Simple Training

```rust
use test_neural::builder::NetworkTrainer;  // Trait for .trainer()

let history = network.trainer()
    .train_data(&train_dataset)
    .epochs(100)
    .fit();
```

### With Validation

```rust
let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(100)
    .batch_size(32)
    .fit();
```

### With Callbacks

```rust
use test_neural::callbacks::{EarlyStopping, ModelCheckpoint, ProgressBar};

let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(200)
    .batch_size(32)
    .callback(Box::new(EarlyStopping::new(10, 0.0001)))
    .callback(Box::new(ModelCheckpoint::new("best_model.json", true)))
    .callback(Box::new(ProgressBar::new(200)))
    .fit();
```

### With Learning Rate Scheduler

```rust
use test_neural::callbacks::{LearningRateScheduler, LRSchedule};

// StepLR: reduces LR every N epochs
let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(100)
    .batch_size(32)
    .scheduler(LearningRateScheduler::new(
        LRSchedule::StepLR { 
            step_size: 30,  // every 30 epochs
            gamma: 0.1      // multiply by 0.1
        }
    ))
    .fit();

// ReduceOnPlateau: reduces LR when loss plateaus (recommended!)
let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(100)
    .batch_size(32)
    .scheduler(LearningRateScheduler::new(
        LRSchedule::ReduceOnPlateau { 
            patience: 10,          // wait 10 epochs
            factor: 0.5,           // divide by 2
            min_delta: 0.0001     // minimum improvement
        }
    ))
    .fit();

// ExponentialLR: exponential decay
let history = network.trainer()
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    .epochs(100)
    .batch_size(32)
    .scheduler(LearningRateScheduler::new(
        LRSchedule::ExponentialLR { gamma: 0.95 }
    ))
    .fit();
```

### Complete Configuration (all combined)

```rust
let history = network.trainer()
    // Data
    .train_data(&train_dataset)
    .validation_data(&val_dataset)
    
    // Hyperparameters
    .epochs(200)
    .batch_size(32)
    
    // Learning rate scheduling
    .scheduler(LearningRateScheduler::new(
        LRSchedule::ReduceOnPlateau { 
            patience: 10, 
            factor: 0.5, 
            min_delta: 0.0001 
        }
    ))
    
    // Callbacks (in execution order)
    .callback(Box::new(ProgressBar::new(200)))
    .callback(Box::new(ModelCheckpoint::new("best_model.json", true)))
    .callback(Box::new(EarlyStopping::new(20, 0.00001)))
    
    .fit();

// history contains (train_loss, val_loss) for each epoch
println!("Final loss: {:.6}", history.last().unwrap().1.unwrap());
```

---

## Complete Examples

### Example 1: Binary Classification (XOR)

```rust
use test_neural::builder::{NetworkBuilder, NetworkTrainer};
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::dataset::Dataset;
use test_neural::callbacks::{EarlyStopping, ModelCheckpoint};
use ndarray::array;

fn main() {
    // XOR data
    let inputs = vec![
        array![0.0, 0.0],
        array![0.0, 1.0],
        array![1.0, 0.0],
        array![1.0, 1.0],
    ];
    
    let targets = vec![
        array![0.0],
        array![1.0],
        array![1.0],
        array![0.0],
    ];
    
    let dataset = Dataset::new(inputs.clone(), targets.clone());
    let (train, val) = dataset.split(0.75);
    
    // Build the network
    let mut network = NetworkBuilder::new(2, 1)
        .hidden_layer(8, Activation::Tanh)
        .output_activation(Activation::Sigmoid)
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerType::adam(0.01))
        .build();
    
    // Training
    let history = network.trainer()
        .train_data(&train)
        .validation_data(&val)
        .epochs(1000)
        .batch_size(2)
        .callback(Box::new(EarlyStopping::new(50, 0.0001)))
        .callback(Box::new(ModelCheckpoint::new("best_xor.json", true)))
        .fit();
    
    // Predictions
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let prediction = network.predict(input);
        println!("[{:.0}, {:.0}] → {:.3} (expected {:.0})", 
            input[0], input[1], prediction[0], target[0]);
    }
}
```

### Example 2: Deep Network with Regularization

```rust
use test_neural::builder::{NetworkBuilder, NetworkTrainer};
use test_neural::network::{Activation, LossFunction};
use test_neural::optimizer::OptimizerType;
use test_neural::callbacks::{LearningRateScheduler, LRSchedule, ProgressBar};

fn main() {
    let mut network = NetworkBuilder::new(784, 10)  // MNIST-like
        .hidden_layer(128, Activation::ReLU)
        .hidden_layer(64, Activation::ReLU)
        .hidden_layer(32, Activation::ReLU)
        .output_activation(Activation::Softmax)
        .loss(LossFunction::CategoricalCrossEntropy)
        .optimizer(OptimizerType::adam(0.001))
        .dropout(0.3)
        .l2(0.0001)
        .build();
    
    let history = network.trainer()
        .train_data(&train_dataset)
        .validation_data(&val_dataset)
        .epochs(100)
        .batch_size(64)
        .scheduler(LearningRateScheduler::new(
            LRSchedule::ReduceOnPlateau { 
                patience: 5, 
                factor: 0.5, 
                min_delta: 0.001 
            }
        ))
        .callback(Box::new(ProgressBar::new(100)))
        .fit();
    
    println!("Training completed: {} epochs", history.len());
}
```

---

## Comparison with Traditional API

### Construction

**Traditional**:
```rust
// Simple
let network = Network::new(
    2, 8, 1,
    Activation::Tanh,
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.01)
);

// Deep
let network = Network::new_deep(
    2,
    vec![16, 8, 4],           // Vec<usize>
    1,
    vec![Activation::ReLU, Activation::ReLU, Activation::Tanh],  // Vec<Activation>
    Activation::Sigmoid,
    LossFunction::BinaryCrossEntropy,
    OptimizerType::adam(0.001)
);
```

**Builder**:
```rust
// Simple
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(8, Activation::Tanh)
    .build();

// Deep
let network = NetworkBuilder::new(2, 1)
    .hidden_layer(16, Activation::ReLU)
    .hidden_layer(8, Activation::ReLU)
    .hidden_layer(4, Activation::Tanh)
    .build();
```

### Training

**Traditional**:
```rust
// Without scheduler
let mut callbacks: Vec<Box<dyn Callback>> = vec![
    Box::new(EarlyStopping::new(10, 0.0001)),
];
let history = network.fit(&train, Some(&val), 100, 32, &mut callbacks);

// With scheduler
let mut scheduler = LearningRateScheduler::new(...);
let mut callbacks: Vec<Box<dyn Callback>> = vec![...];
let history = network.fit_with_scheduler(
    &train, Some(&val), 100, 32, &mut scheduler, &mut callbacks
);
```

**Builder**:
```rust
// All unified
let history = network.trainer()
    .train_data(&train)
    .validation_data(&val)
    .epochs(100)
    .batch_size(32)
    .callback(Box::new(EarlyStopping::new(10, 0.0001)))
    .scheduler(LearningRateScheduler::new(...))
    .fit();
```

---

## Conclusion

The Builder Pattern offers:

✅ **Intuitive API** - Self-documenting code  
✅ **Fewer errors** - No more Vec to manage  
✅ **Flexibility** - Combine any options  
✅ **Unification** - One way to do things  
✅ **Extensibility** - Easy to add new options  
✅ **Backward compatible** - Old API still available

🚀 **Start here**: `cargo run --example getting_started`
