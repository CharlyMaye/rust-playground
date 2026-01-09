mod network;
use network::{Network, Activation};
use ndarray::array;
//https://evolveasdev.com/blogs/tutorial/building-a-neural-network-from-scratch-in-rust
fn main() {
    // XOR problem with ReLU hidden layer and Sigmoid output
    let mut network = Network::new(
        2,
        3,
        1,
        Activation::ReLU, Activation::Sigmoid
    );
 
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
 
    let learning_rate = 0.1;
    let epochs = 250_000;
 
    println!("Training XOR with ReLU (hidden) + Sigmoid (output)...\n");
 
    for _ in 0..epochs {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(input, target, learning_rate);
        }
    }
 
    println!("Results:");
    for input in inputs.iter() {
        let (_, _, output) = network.forward(input);
        println!("{:?} -> {:.3}", input, output[0]);
    }
}