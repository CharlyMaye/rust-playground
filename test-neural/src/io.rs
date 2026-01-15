//! Input/Output utilities for neural networks.
//!
//! This module provides serialization and deserialization functions
//! for saving and loading trained neural networks to/from disk.
//!
//! Supports multiple formats:
//! - **JSON**: Human-readable, good for inspection and debugging
//! - **Binary (Bincode)**: Compact and fast, recommended for production
//!
//! # Examples
//!
//! ```rust
//! use test_neural::network::{Network, Activation, LossFunction};
//! use test_neural::io;
//!
//! // Train a network
//! let mut network = Network::new(2, 5, 1, 
//!     Activation::Tanh, 
//!     Activation::Sigmoid,
//!     LossFunction::BinaryCrossEntropy);
//!
//! // ... training code ...
//!
//! // Save to JSON
//! io::save_json(&network, "model.json")?;
//!
//! // Save to binary
//! io::save_binary(&network, "model.bin")?;
//!
//! // Load from JSON
//! let loaded = io::load_json("model.json")?;
//!
//! // Load from binary
//! let loaded = io::load_binary("model.bin")?;
//! ```

use crate::network::Network;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

/// Error type for I/O operations.
#[derive(Debug)]
pub enum IoError {
    /// File system error
    Io(std::io::Error),
    /// JSON serialization/deserialization error
    Json(serde_json::Error),
    /// Binary serialization/deserialization error
    Bincode(bincode::Error),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            IoError::Io(e) => write!(f, "I/O error: {}", e),
            IoError::Json(e) => write!(f, "JSON error: {}", e),
            IoError::Bincode(e) => write!(f, "Binary encoding error: {}", e),
        }
    }
}

impl std::error::Error for IoError {}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        IoError::Io(e)
    }
}

impl From<serde_json::Error> for IoError {
    fn from(e: serde_json::Error) -> Self {
        IoError::Json(e)
    }
}

impl From<bincode::Error> for IoError {
    fn from(e: bincode::Error) -> Self {
        IoError::Bincode(e)
    }
}

pub type Result<T> = std::result::Result<T, IoError>;

/// Save a neural network to a JSON file.
///
/// JSON format is human-readable and useful for:
/// - Debugging and inspection
/// - Version control (readable diffs)
/// - Manual editing if needed
///
/// # Arguments
/// - `network`: The network to save
/// - `path`: Path to the output file
///
/// # Returns
/// - `Ok(())` on success
/// - `Err(IoError)` on failure
///
/// # Example
/// ```rust
/// io::save_json(&network, "model.json")?;
/// ```
pub fn save_json<P: AsRef<Path>>(network: &Network, path: P) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, network)?;
    Ok(())
}

/// Load a neural network from a JSON file.
///
/// # Arguments
/// - `path`: Path to the JSON file
///
/// # Returns
/// - `Ok(Network)` on success
/// - `Err(IoError)` on failure
///
/// # Example
/// ```rust
/// let network = io::load_json("model.json")?;
/// ```
pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Network> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let network = serde_json::from_reader(reader)?;
    Ok(network)
}

/// Save a neural network to a binary file (Bincode format).
///
/// Binary format is:
/// - **Compact**: Much smaller than JSON
/// - **Fast**: Faster serialization/deserialization
/// - **Recommended for production**
///
/// # Arguments
/// - `network`: The network to save
/// - `path`: Path to the output file
///
/// # Returns
/// - `Ok(())` on success
/// - `Err(IoError)` on failure
///
/// # Example
/// ```rust
/// io::save_binary(&network, "model.bin")?;
/// ```
pub fn save_binary<P: AsRef<Path>>(network: &Network, path: P) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let encoded = bincode::serialize(network)?;
    writer.write_all(&encoded)?;
    writer.flush()?;
    Ok(())
}

/// Load a neural network from a binary file (Bincode format).
///
/// # Arguments
/// - `path`: Path to the binary file
///
/// # Returns
/// - `Ok(Network)` on success
/// - `Err(IoError)` on failure
///
/// # Example
/// ```rust
/// let network = io::load_binary("model.bin")?;
/// ```
pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Network> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let network = bincode::deserialize_from(reader)?;
    Ok(network)
}

/// Get the size of a network when serialized.
///
/// Useful for comparing format efficiency or checking model size.
///
/// # Arguments
/// - `network`: The network to measure
///
/// # Returns
/// Tuple of (JSON size in bytes, Binary size in bytes)
///
/// # Example
/// ```rust
/// let (json_size, bin_size) = io::get_serialized_size(&network);
/// println!("JSON: {} bytes, Binary: {} bytes", json_size, bin_size);
/// println!("Compression ratio: {:.2}x", json_size as f64 / bin_size as f64);
/// ```
pub fn get_serialized_size(network: &Network) -> (usize, usize) {
    let json_size = serde_json::to_string(network)
        .map(|s| s.len())
        .unwrap_or(0);
    
    let bin_size = bincode::serialize(network)
        .map(|b| b.len())
        .unwrap_or(0);
    
    (json_size, bin_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{Activation, LossFunction};
    use crate::optimizer::OptimizerType;
    use ndarray::array;
    use std::fs;

    #[test]
    fn test_save_load_json() {
        use crate::builder::NetworkBuilder;
        let network = NetworkBuilder::new(2, 1)
            .hidden_layer(3, Activation::Tanh)
            .output_activation(Activation::Sigmoid)
            .loss(LossFunction::MSE)
            .optimizer(OptimizerType::sgd(0.1))
            .build();

        let path = "test_model.json";
        
        // Save
        save_json(&network, path).expect("Failed to save JSON");
        
        // Load
        let loaded = load_json(path).expect("Failed to load JSON");
        
        // Verify it loads successfully (can't access private fields)
        // Basic smoke test: predict should work
        let input = array![0.5, 0.5];
        let _ = loaded.predict(&input);
        
        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_save_load_binary() {
        use crate::builder::NetworkBuilder;
        let network = NetworkBuilder::new(2, 1)
            .hidden_layer(3, Activation::Tanh)
            .output_activation(Activation::Sigmoid)
            .loss(LossFunction::MSE)
            .optimizer(OptimizerType::adam(0.001))
            .build();

        let path = "test_model.bin";
        
        // Save
        save_binary(&network, path).expect("Failed to save binary");
        
        // Load
        let loaded = load_binary(path).expect("Failed to load binary");
        
        // Verify it loads successfully
        let input = array![0.5, 0.5];
        let _ = loaded.predict(&input);
        
        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_serialized_size() {
        use crate::builder::NetworkBuilder;
        let network = NetworkBuilder::new(2, 1)
            .hidden_layer(5, Activation::ReLU)
            .output_activation(Activation::Sigmoid)
            .loss(LossFunction::BinaryCrossEntropy)
            .optimizer(OptimizerType::sgd(0.1))
            .build();

        let (json_size, bin_size) = get_serialized_size(&network);
        
        assert!(json_size > 0, "JSON size should be positive");
        assert!(bin_size > 0, "Binary size should be positive");
        assert!(bin_size < json_size, "Binary should be more compact than JSON");
    }
}
