mod network;
use network::Network;
use ndarray::array;
//https://evolveasdev.com/blogs/tutorial/building-a-neural-network-from-scratch-in-rust
fn main() {
    let mut network = Network::new(2, 3, 1);
 
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
    let epochs = 100_000;
 
    for _ in 0..epochs {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            network.train(input, target, learning_rate);
        }
    }
 
    for input in inputs.iter() {
        let (_, _, output) = network.forward(input);
        println!("{:?} -> {:?}", input, output);
    }
}