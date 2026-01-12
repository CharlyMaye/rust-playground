fn main() {
    println!("Test Neural Network Library");
    println!("============================");
    println!();
    println!("This is a library for building neural networks in Rust.");
    println!();
    println!("Features:");
    println!("  • 15 activation functions (Sigmoid, ReLU, Tanh, GELU, Mish, etc.)");
    println!("  • 5 loss functions (MSE, MAE, BCE, CCE, Huber)");
    println!("  • 5 optimizers (SGD, Momentum, RMSprop, Adam, AdamW)");
    println!("  • Regularization (Dropout, L1, L2, Elastic Net)");
    println!("  • Deep architectures with multiple hidden layers");
    println!("  • Xavier/He/LeCun weight initialization");
    println!("  • Model serialization (JSON and binary)");
    println!("  • Evaluation metrics (accuracy, precision, recall, F1, ROC/AUC)");
    println!();
    println!("To see examples, run:");
    println!("  cargo run --example xor_tests            - Test all loss functions and deep networks");
    println!("  cargo run --example serialization        - Demo save/load functionality");
    println!("  cargo run --example metrics_demo         - Demo evaluation metrics (accuracy, F1, etc.)");
    println!("  cargo run --example optimizer_comparison - Compare optimizers (SGD, Adam, RMSprop, etc.)");
    println!("  cargo run --example regularization_demo  - Demo regularization (Dropout, L1, L2)");
    println!();
    println!("For more information, see the readme.md file.");
}