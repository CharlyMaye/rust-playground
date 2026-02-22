//! WebAssembly AlexNet-Mini CNN for MNIST
//!
//! Uses trained CNN weights exported from cma_autograd via cma_cnn Sequential.
//! Model format: CnnModelWithMetadata (CNN feature extractor + FC classifier).

neural_wasm_shared::define_cnn_mnist_network! {
    struct_name = MnistAlexNetNetwork,
    model_file = "alexnet_model.bin",
    model_info_name = "AlexNet-Mini MNIST Classifier",
    description = "AlexNet-Mini CNN trained end-to-end with autograd (Krizhevsky et al., 2012)",
    arch_name = "AlexNet-Mini",
    output_features = 2304,
}
