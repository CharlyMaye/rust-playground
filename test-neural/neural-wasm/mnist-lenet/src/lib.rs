//! WebAssembly LeNet-5 CNN for MNIST
//!
//! Uses trained CNN weights exported from cma_autograd via cma_cnn Sequential.
//! Model format: CnnModelWithMetadata (CNN feature extractor + FC classifier).

neural_wasm_shared::define_cnn_mnist_network! {
    struct_name = MnistLeNetNetwork,
    model_file = "lenet_model.bin",
    model_info_name = "LeNet-5 MNIST Classifier",
    description = "LeNet-5 CNN trained end-to-end with autograd (LeCun et al., 1998)",
    arch_name = "LeNet-5",
    output_features = 120,
}
