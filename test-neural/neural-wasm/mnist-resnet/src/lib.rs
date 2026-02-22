//! WebAssembly ResNet-style CNN for MNIST
//!
//! Uses trained CNN weights exported from cma_autograd via cma_cnn Sequential.
//! Model format: CnnModelWithMetadata (CNN feature extractor + FC classifier).

neural_wasm_shared::define_cnn_mnist_network! {
    struct_name = MnistResNetNetwork,
    model_file = "resnet_model.bin",
    model_info_name = "ResNet-MNIST Classifier",
    description = "Deep CNN trained end-to-end with autograd (ResNet-style)",
    arch_name = "ResNet-MNIST",
    output_features = 64,
}
