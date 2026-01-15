fn main() {
    println!("Test Neural Network Library");
    println!("============================");
    println!();
    println!("This is a library for building neural networks in Rust.");
    println!();
    println!("Features:");
    println!("  • Builder Pattern: Fluent API for easy construction");
    println!("  • 15 activation functions (Sigmoid, ReLU, Tanh, GELU, Mish, etc.)");
    println!("  • 5 loss functions (MSE, MAE, BCE, CCE, Huber)");
    println!("  • 5 optimizers (SGD, Momentum, RMSprop, Adam, AdamW)");
    println!("  • Regularization (Dropout, L1, L2, Elastic Net)");
    println!("  • Mini-batch training with Dataset management");
    println!("  • Callbacks (EarlyStopping, ModelCheckpoint, LR Scheduler)");
    println!("  • Deep architectures with multiple hidden layers");
    println!("  • Xavier/He/LeCun weight initialization");
    println!("  • Model serialization (JSON and binary)");
    println!("  • Evaluation metrics (accuracy, precision, recall, F1, ROC/AUC)");
    println!();
    println!("To see examples, run:");
    println!("  cargo run --example getting_started   - ⭐ Complete guide (START HERE!)");
    println!("  cargo run --example serialization     - Save/load models (JSON & binary)");
    println!("  cargo run --example minibatch_demo    - Mini-batch training (2x faster!)");
    println!("  cargo run --example metrics_demo      - Evaluation metrics (accuracy, F1, ROC)");
    println!();
    println!("For more information, see the readme.md file.");
}