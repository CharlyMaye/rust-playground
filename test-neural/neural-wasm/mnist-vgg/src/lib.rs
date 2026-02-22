//! WebAssembly VGG-Tiny CNN for MNIST
//!
//! Uses trained CNN weights exported from cma_autograd via cma_cnn Sequential.
//! Model format: CnnModelWithMetadata (CNN feature extractor + FC classifier).

neural_wasm_shared::define_cnn_mnist_network! {
    struct_name = MnistVGGNetwork,
    model_file = "vgg_model.bin",
    model_info_name = "VGG-Tiny MNIST Classifier",
    description = "VGG-Tiny CNN trained end-to-end with autograd (Simonyan & Zisserman, 2014)",
    arch_name = "VGG-Tiny",
    output_features = 3136,
}
