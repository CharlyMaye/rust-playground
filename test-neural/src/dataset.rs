//! Module pour gérer les datasets et l'entraînement par mini-batch
//!
//! Ce module fournit des structures pour organiser les données d'entraînement,
//! créer des batches, shuffle les données, et split train/val/test.

use ndarray::Array1;
use rand::seq::SliceRandom;
use rand::rng;

/// Dataset pour l'entraînement de réseaux de neurones
/// 
/// Gère les paires (input, target) et permet de :
/// - Créer des batches pour l'entraînement
/// - Shuffle les données
/// - Split train/validation/test
/// - Itérer sur les batches
#[derive(Debug, Clone)]
pub struct Dataset {
    inputs: Vec<Array1<f64>>,
    targets: Vec<Array1<f64>>,
}

impl Dataset {
    /// Crée un nouveau dataset à partir d'inputs et targets
    /// 
    /// # Panics
    /// Panic si inputs.len() != targets.len()
    /// 
    /// # Example
    /// ```
    /// use ndarray::array;
    /// 
    /// let inputs = vec![array![0.0, 0.0], array![0.0, 1.0]];
    /// let targets = vec![array![0.0], array![1.0]];
    /// let dataset = Dataset::new(inputs, targets);
    /// ```
    pub fn new(inputs: Vec<Array1<f64>>, targets: Vec<Array1<f64>>) -> Self {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Number of inputs must match number of targets"
        );
        Dataset { inputs, targets }
    }
    
    /// Retourne le nombre d'exemples dans le dataset
    pub fn len(&self) -> usize {
        self.inputs.len()
    }
    
    /// Vérifie si le dataset est vide
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty()
    }
    
    /// Retourne une référence aux inputs
    pub fn inputs(&self) -> &Vec<Array1<f64>> {
        &self.inputs
    }
    
    /// Retourne une référence aux targets
    pub fn targets(&self) -> &Vec<Array1<f64>> {
        &self.targets
    }
    
    /// Shuffle le dataset (mélange aléatoirement l'ordre des exemples)
    /// 
    /// Important pour éviter que le réseau n'apprenne l'ordre des exemples.
    /// À appeler avant chaque epoch.
    /// 
    /// # Example
    /// ```
    /// let mut dataset = Dataset::new(inputs, targets);
    /// dataset.shuffle();  // Mélange aléatoirement
    /// ```
    pub fn shuffle(&mut self) {
        let mut rng = rng();
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.shuffle(&mut rng);
        
        let inputs: Vec<_> = indices.iter().map(|&i| self.inputs[i].clone()).collect();
        let targets: Vec<_> = indices.iter().map(|&i| self.targets[i].clone()).collect();
        
        self.inputs = inputs;
        self.targets = targets;
    }
    
    /// Split le dataset en train et test
    /// 
    /// # Arguments
    /// - `train_ratio`: Proportion pour l'entraînement (ex: 0.8 = 80% train, 20% test)
    /// 
    /// # Returns
    /// Tuple (train_dataset, test_dataset)
    /// 
    /// # Example
    /// ```
    /// let dataset = Dataset::new(inputs, targets);
    /// let (train, test) = dataset.split(0.8);  // 80% train, 20% test
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
    
    /// Split le dataset en train, validation et test
    /// 
    /// # Arguments
    /// - `train_ratio`: Proportion pour l'entraînement (ex: 0.7)
    /// - `val_ratio`: Proportion pour la validation (ex: 0.15)
    /// - Le reste sera pour le test (ex: 0.15)
    /// 
    /// # Returns
    /// Tuple (train_dataset, val_dataset, test_dataset)
    /// 
    /// # Example
    /// ```
    /// let dataset = Dataset::new(inputs, targets);
    /// let (train, val, test) = dataset.split_three(0.7, 0.15);  // 70/15/15
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
    
    /// Crée un itérateur sur les batches
    /// 
    /// # Arguments
    /// - `batch_size`: Taille des batches (ex: 32, 64, 128)
    /// 
    /// # Returns
    /// Iterator qui produit des tuples (batch_inputs, batch_targets)
    /// 
    /// # Example
    /// ```
    /// let dataset = Dataset::new(inputs, targets);
    /// for (batch_inputs, batch_targets) in dataset.batches(32) {
    ///     network.train_batch(&batch_inputs, &batch_targets);
    /// }
    /// ```
    pub fn batches(&self, batch_size: usize) -> DatasetBatchIterator<'_> {
        DatasetBatchIterator {
            dataset: self,
            batch_size,
            current_idx: 0,
        }
    }
}

/// Iterator pour parcourir un dataset par batches
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
