//! Dataset management for neural network training.
//!
//! This module provides structures for organizing training data, creating
//! mini-batches, shuffling data, and splitting into train/validation/test sets.
//!
//! # Example
//!
//! ```rust
//! use test_neural::dataset::Dataset;
//! use ndarray::array;
//!
//! let inputs = vec![array![0.0, 0.0], array![0.0, 1.0]];
//! let targets = vec![array![0.0], array![1.0]];
//! let mut dataset = Dataset::new(inputs, targets);
//!
//! dataset.shuffle();
//! for (batch_inputs, batch_targets) in dataset.batches(32) {
//!     println!("Batch size: {}", batch_inputs.len());
//! }
//! ```

use ndarray::Array1;
use rand::{rng, Rng};

/// A dataset containing input-target pairs for neural network training.
///
/// Manages pairs of (input, target) vectors and provides utilities for:
/// - Creating mini-batches for training
/// - Shuffling data to prevent learning order dependencies
/// - Splitting into train/validation/test sets
/// - Iterating over batches
#[derive(Debug, Clone)]
pub struct Dataset {
    inputs: Vec<Array1<f64>>,
    targets: Vec<Array1<f64>>,
}

impl Dataset {
    /// Creates a new dataset from input and target vectors.
    ///
    /// # Arguments
    /// * `inputs` - Vector of input arrays
    /// * `targets` - Vector of target arrays (must have same length as inputs)
    ///
    /// # Panics
    /// Panics if `inputs.len() != targets.len()`
    ///
    /// # Example
    /// ```rust
    /// use test_neural::dataset::Dataset;
    /// use ndarray::array;
    ///
    /// let inputs = vec![array![0.0, 0.0], array![0.0, 1.0]];
    /// let targets = vec![array![0.0], array![1.0]];
    /// let dataset = Dataset::new(inputs, targets);
    /// assert_eq!(dataset.len(), 2);
    /// ```
    pub fn new(inputs: Vec<Array1<f64>>, targets: Vec<Array1<f64>>) -> Self {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Number of inputs must match number of targets"
        );
        Dataset { inputs, targets }
    }
    
    /// Returns the number of examples in the dataset.
    pub fn len(&self) -> usize {
        self.inputs.len()
    }
    
    /// Returns true if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
    
    /// Returns a reference to the input vectors.
    pub fn inputs(&self) -> &Vec<Array1<f64>> {
        &self.inputs
    }
    
    /// Returns a reference to the target vectors.
    pub fn targets(&self) -> &Vec<Array1<f64>> {
        &self.targets
    }
    
    /// Shuffles the dataset randomly in-place.
    ///
    /// This is important to prevent the network from learning the order of examples.
    /// Should be called before each training epoch.
    ///
    /// Uses the Fisher-Yates algorithm for O(n) time complexity with O(1) additional memory.
    ///
    /// # Example
    /// ```rust
    /// use test_neural::dataset::Dataset;
    /// use ndarray::array;
    ///
    /// let inputs = vec![array![1.0], array![2.0], array![3.0]];
    /// let targets = vec![array![1.0], array![2.0], array![3.0]];
    /// let mut dataset = Dataset::new(inputs, targets);
    /// dataset.shuffle();
    /// assert_eq!(dataset.len(), 3);
    /// ```
    pub fn shuffle(&mut self) {
        let mut rng = rng();
        let n = self.len();
        
        // Fisher-Yates shuffle in-place (O(n), pas de clone!)
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            self.inputs.swap(i, j);
            self.targets.swap(i, j);
        }
    }
    
    /// Splits the dataset into training and test sets.
    ///
    /// # Arguments
    /// * `train_ratio` - Proportion of data for training (e.g., 0.8 for 80% train, 20% test)
    ///
    /// # Returns
    /// A tuple `(train_dataset, test_dataset)`
    ///
    /// # Panics
    /// Panics if `train_ratio` is not between 0 and 1 (exclusive)
    ///
    /// # Example
    /// ```rust
    /// use test_neural::dataset::Dataset;
    /// use ndarray::array;
    ///
    /// let inputs = vec![array![1.0], array![2.0], array![3.0], array![4.0], array![5.0]];
    /// let targets = vec![array![1.0], array![2.0], array![3.0], array![4.0], array![5.0]];
    /// let dataset = Dataset::new(inputs, targets);
    /// let (train, test) = dataset.split(0.8);
    /// assert_eq!(train.len(), 4);
    /// assert_eq!(test.len(), 1);
    /// ```
    pub fn split(self, train_ratio: f64) -> (Dataset, Dataset) {
        assert!(train_ratio > 0.0 && train_ratio < 1.0, "train_ratio must be between 0 and 1");
        
        let split_idx = (self.len() as f64 * train_ratio) as usize;
        
        let (train_inputs, test_inputs) = self.inputs.split_at(split_idx);
        let (train_targets, test_targets) = self.targets.split_at(split_idx);
        
        let train = Dataset::new(train_inputs.to_vec(), train_targets.to_vec());
        let test = Dataset::new(test_inputs.to_vec(), test_targets.to_vec());
        
        (train, test)
    }
    
    /// Splits the dataset into training, validation, and test sets.
    ///
    /// # Arguments
    /// * `train_ratio` - Proportion for training (e.g., 0.7)
    /// * `val_ratio` - Proportion for validation (e.g., 0.15)
    /// * The remainder goes to test (e.g., 0.15)
    ///
    /// # Returns
    /// A tuple `(train_dataset, val_dataset, test_dataset)`
    ///
    /// # Panics
    /// Panics if `train_ratio + val_ratio >= 1.0`
    ///
    /// # Example
    /// ```rust
    /// use test_neural::dataset::Dataset;
    /// use ndarray::array;
    ///
    /// let inputs: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
    /// let targets: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
    /// let dataset = Dataset::new(inputs, targets);
    /// let (train, val, test) = dataset.split_three(0.7, 0.15);
    /// assert_eq!(train.len(), 7);
    /// ```
    pub fn split_three(self, train_ratio: f64, val_ratio: f64) -> (Dataset, Dataset, Dataset) {
        assert!(
            train_ratio + val_ratio < 1.0,
            "train_ratio + val_ratio must be less than 1.0"
        );
        
        let train_idx = (self.len() as f64 * train_ratio) as usize;
        let val_idx = train_idx + (self.len() as f64 * val_ratio) as usize;
        
        let train_inputs = self.inputs[..train_idx].to_vec();
        let train_targets = self.targets[..train_idx].to_vec();
        
        let val_inputs = self.inputs[train_idx..val_idx].to_vec();
        let val_targets = self.targets[train_idx..val_idx].to_vec();
        
        let test_inputs = self.inputs[val_idx..].to_vec();
        let test_targets = self.targets[val_idx..].to_vec();
        
        (
            Dataset::new(train_inputs, train_targets),
            Dataset::new(val_inputs, val_targets),
            Dataset::new(test_inputs, test_targets),
        )
    }
    
    /// Creates an iterator over mini-batches of the dataset.
    ///
    /// # Arguments
    /// * `batch_size` - Number of examples per batch (e.g., 32, 64, 128)
    ///
    /// # Returns
    /// An iterator that yields `(batch_inputs, batch_targets)` tuples.
    /// The last batch may be smaller if the dataset size is not divisible by batch_size.
    ///
    /// # Example
    /// ```rust
    /// use test_neural::dataset::Dataset;
    /// use ndarray::array;
    ///
    /// let inputs: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
    /// let targets: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
    /// let dataset = Dataset::new(inputs, targets);
    ///
    /// let batches: Vec<_> = dataset.batches(3).collect();
    /// assert_eq!(batches.len(), 4);
    /// ```
    pub fn batches(&self, batch_size: usize) -> DatasetBatchIterator<'_> {
        DatasetBatchIterator {
            dataset: self,
            batch_size,
            current_idx: 0,
        }
    }
}

/// Iterator for traversing a dataset in mini-batches.
pub struct DatasetBatchIterator<'a> {
    dataset: &'a Dataset,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> Iterator for DatasetBatchIterator<'a> {
    type Item = (Vec<Array1<f64>>, Vec<Array1<f64>>);
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }
        
        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.len());
        
        let batch_inputs = self.dataset.inputs[self.current_idx..end_idx].to_vec();
        let batch_targets = self.dataset.targets[self.current_idx..end_idx].to_vec();
        
        self.current_idx = end_idx;
        
        Some((batch_inputs, batch_targets))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_dataset_creation() {
        let inputs = vec![array![0.0, 0.0], array![0.0, 1.0]];
        let targets = vec![array![0.0], array![1.0]];
        let dataset = Dataset::new(inputs, targets);
        
        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_dataset_split() {
        let inputs = vec![
            array![0.0], array![1.0], array![2.0], array![3.0], array![4.0],
        ];
        let targets = vec![
            array![0.0], array![1.0], array![2.0], array![3.0], array![4.0],
        ];
        let dataset = Dataset::new(inputs, targets);
        
        let (train, test) = dataset.split(0.8);
        
        assert_eq!(train.len(), 4);
        assert_eq!(test.len(), 1);
    }

    #[test]
    fn test_dataset_split_three() {
        let inputs: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
        let targets: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
        let dataset = Dataset::new(inputs, targets);
        
        let (train, val, test) = dataset.split_three(0.7, 0.15);
        
        assert_eq!(train.len(), 7);
        assert_eq!(val.len(), 1);
        assert_eq!(test.len(), 2);
    }

    #[test]
    fn test_dataset_batches() {
        let inputs: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
        let targets: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
        let dataset = Dataset::new(inputs, targets);
        
        let batches: Vec<_> = dataset.batches(3).collect();
        
        assert_eq!(batches.len(), 4);  // 10 éléments avec batch_size=3 → 4 batches
        assert_eq!(batches[0].0.len(), 3);
        assert_eq!(batches[1].0.len(), 3);
        assert_eq!(batches[2].0.len(), 3);
        assert_eq!(batches[3].0.len(), 1);  // Dernier batch plus petit
    }

    #[test]
    fn test_dataset_shuffle() {
        let inputs: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
        let targets: Vec<_> = (0..10).map(|i| array![i as f64]).collect();
        let mut dataset = Dataset::new(inputs.clone(), targets);
        
        dataset.shuffle();
        
        // Vérifier que le dataset a toujours la même taille
        assert_eq!(dataset.len(), 10);
        
        // Vérifier que les données sont différentes de l'ordre original (très probable)
        let all_same = dataset.inputs().iter()
            .zip(inputs.iter())
            .all(|(a, b)| a[0] == b[0]);
        
        // Il y a une très faible probabilité que shuffle ne change rien
        // mais avec 10 éléments, c'est presque impossible
        assert!(!all_same || dataset.len() <= 1);
    }
}
