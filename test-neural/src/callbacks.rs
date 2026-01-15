//! Training callbacks for neural networks.
//!
//! Callbacks allow injecting custom code at different points during training:
//! after each epoch, after each batch, etc.
//!
//! Available callbacks:
//! - **EarlyStopping**: Stops training if loss stops improving
//! - **ModelCheckpoint**: Automatically saves the best model
//! - **LearningRateScheduler**: Dynamically adjusts the learning rate
//! - **ProgressBar**: Displays real-time training progress

use crate::network::Network;
use crate::optimizer::OptimizerType;
use std::path::PathBuf;

/// Base trait for all callbacks.
///
/// Callbacks are invoked at different points during training
/// and can modify behavior or collect statistics.
pub trait Callback {
    /// Called at the beginning of training.
    fn on_train_begin(&mut self, _network: &Network) {}
    
    /// Called at the end of training.
    fn on_train_end(&mut self, _network: &Network) {}
    
    /// Called at the beginning of each epoch.
    ///
    /// # Arguments
    /// - `epoch`: Epoch number (0-indexed)
    /// - `network`: Reference to the network
    fn on_epoch_begin(&mut self, _epoch: usize, _network: &Network) {}
    
    /// Called at the end of each epoch.
    ///
    /// # Arguments
    /// - `epoch`: Epoch number (0-indexed)
    /// - `network`: Reference to the network
    /// - `train_loss`: Training loss
    /// - `val_loss`: Validation loss (None if no validation)
    ///
    /// # Returns
    /// `true` to continue training, `false` to stop
    fn on_epoch_end(&mut self, _epoch: usize, _network: &Network, _train_loss: f64, _val_loss: Option<f64>) -> bool {
        true
    }
}

/// EarlyStopping - Stops training if loss stops improving.
///
/// Monitors validation loss and stops training after `patience` epochs
/// without improvement. Helps prevent overfitting.
///
/// # Example
/// ```rust
/// use test_neural::callbacks::{EarlyStopping, DeltaMode};
///
/// // Absolute comparison (default): improvement if loss decreases by at least min_delta
/// let early_stop = EarlyStopping::new(10, 0.0001);
///
/// // Relative comparison: improvement if loss decreases by at least min_delta * best_loss
/// let early_stop_relative = EarlyStopping::new(10, 0.01)  // 1% improvement required
///     .mode(DeltaMode::Relative);
/// ```

/// Comparison mode for determining improvement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeltaMode {
    /// Absolute comparison: improvement if `loss < best_loss - min_delta`
    Absolute,
    /// Relative comparison: improvement if `loss < best_loss * (1 - min_delta)`
    /// Use min_delta as a percentage (e.g., 0.01 for 1%)
    Relative,
}

impl Default for DeltaMode {
    fn default() -> Self {
        DeltaMode::Absolute
    }
}

#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Number of epochs to wait without improvement before stopping
    patience: usize,

    /// Minimum improvement required to count as improvement
    min_delta: f64,

    /// Comparison mode (absolute or relative)
    delta_mode: DeltaMode,

    /// Best loss observed
    best_loss: f64,

    /// Number of epochs without improvement
    wait: usize,

    /// Indicates if training should stop
    stopped: bool,

    /// Epoch where the best model was found
    best_epoch: usize,
}

impl EarlyStopping {
    /// Creates a new EarlyStopping callback.
    ///
    /// # Arguments
    /// - `patience`: Number of epochs to wait without improvement
    /// - `min_delta`: Minimum required improvement
    ///   - In Absolute mode (default): absolute value (e.g., 0.0001)
    ///   - In Relative mode: percentage (e.g., 0.01 for 1%)
    pub fn new(patience: usize, min_delta: f64) -> Self {
        EarlyStopping {
            patience,
            min_delta,
            delta_mode: DeltaMode::default(),
            best_loss: f64::INFINITY,
            wait: 0,
            stopped: false,
            best_epoch: 0,
        }
    }

    /// Sets the comparison mode (builder pattern).
    ///
    /// # Arguments
    /// - `mode`: `DeltaMode::Absolute` or `DeltaMode::Relative`
    ///
    /// # Example
    /// ```rust
    /// use test_neural::callbacks::{EarlyStopping, DeltaMode};
    ///
    /// // Require 1% relative improvement
    /// let early_stop = EarlyStopping::new(10, 0.01)
    ///     .mode(DeltaMode::Relative);
    /// ```
    pub fn mode(mut self, mode: DeltaMode) -> Self {
        self.delta_mode = mode;
        self
    }

    /// Returns whether training was stopped.
    pub fn stopped(&self) -> bool {
        self.stopped
    }
    
    /// Returns the epoch of the best model.
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }
    
    /// Returns the best loss.
    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }
}

impl Callback for EarlyStopping {
    fn on_train_begin(&mut self, _network: &Network) {
        self.best_loss = f64::INFINITY;
        self.wait = 0;
        self.stopped = false;
        self.best_epoch = 0;
    }
    
    fn on_epoch_end(&mut self, epoch: usize, _network: &Network, _train_loss: f64, val_loss: Option<f64>) -> bool {
        if let Some(loss) = val_loss {
            // Check improvement based on delta mode
            let improved = match self.delta_mode {
                DeltaMode::Absolute => loss < self.best_loss - self.min_delta,
                DeltaMode::Relative => {
                    if self.best_loss.is_infinite() {
                        true  // First epoch is always an improvement
                    } else {
                        loss < self.best_loss * (1.0 - self.min_delta)
                    }
                }
            };

            if improved {
                self.best_loss = loss;
                self.best_epoch = epoch;
                self.wait = 0;
            } else {
                self.wait += 1;
                if self.wait >= self.patience {
                    self.stopped = true;
                    println!("\n⚠️ EarlyStopping: Stopped at epoch {} (best epoch: {}, loss: {:.6})",
                             epoch, self.best_epoch, self.best_loss);
                    return false;  // Stop training
                }
            }
        }
        true  // Continue
    }
}

/// ModelCheckpoint - Automatically saves the best model.
///
/// Saves the model when validation loss improves.
/// Allows recovering the best model even if training overfits afterwards.
///
/// # Example
/// ```rust
/// use test_neural::callbacks::ModelCheckpoint;
///
/// let checkpoint = ModelCheckpoint::new("best_model.json", true);
/// ```
#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    /// Path where to save the model
    filepath: PathBuf,
    
    /// Save only if improvement
    save_best_only: bool,
    
    /// Best loss observed
    best_loss: f64,
    
    /// Save format (true = JSON, false = Binary)
    use_json: bool,
}

impl ModelCheckpoint {
    /// Creates a new ModelCheckpoint callback.
    ///
    /// # Arguments
    /// - `filepath`: File path (e.g., "best_model.json" or "best_model.bin")
    /// - `save_best_only`: If true, save only when loss improves
    pub fn new(filepath: &str, save_best_only: bool) -> Self {
        let path = PathBuf::from(filepath);
        let use_json = path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext == "json")
            .unwrap_or(true);
        
        ModelCheckpoint {
            filepath: path,
            save_best_only,
            best_loss: f64::INFINITY,
            use_json,
        }
    }
    
    /// Returns the best loss.
    pub fn best_loss(&self) -> f64 {
        self.best_loss
    }
}

impl Callback for ModelCheckpoint {
    fn on_train_begin(&mut self, _network: &Network) {
        self.best_loss = f64::INFINITY;
    }
    
    fn on_epoch_end(&mut self, epoch: usize, network: &Network, _train_loss: f64, val_loss: Option<f64>) -> bool {
        if let Some(loss) = val_loss {
            let should_save = if self.save_best_only {
                loss < self.best_loss
            } else {
                true
            };
            
            if should_save {
                self.best_loss = loss;
                
                let result = if self.use_json {
                    crate::io::save_json(network, self.filepath.to_str().unwrap())
                } else {
                    crate::io::save_binary(network, self.filepath.to_str().unwrap())
                };
                
                match result {
                    Ok(_) => println!("💾 ModelCheckpoint: Modèle sauvegardé à l'epoch {} (loss: {:.6})", epoch, loss),
                    Err(e) => eprintln!("❌ Erreur sauvegarde: {}", e),
                }
            }
        }
        true  // Continue toujours
    }
}

/// Learning rate adjustment strategy.
#[derive(Debug, Clone)]
pub enum LRSchedule {
    /// Reduces LR by a factor at fixed epochs.
    /// Example: StepLR { step_size: 10, gamma: 0.5 } divides LR by 2 every 10 epochs
    StepLR { step_size: usize, gamma: f64 },
    
    /// Reduces LR when loss plateaus.
    /// Example: ReduceOnPlateau { patience: 5, factor: 0.5 } divides LR by 2 after 5 epochs without improvement
    ReduceOnPlateau { patience: usize, factor: f64, min_delta: f64 },
    
    /// Exponential decay of LR.
    /// Example: ExponentialLR { gamma: 0.95 } multiplies LR by 0.95 each epoch
    ExponentialLR { gamma: f64 },
}

/// LearningRateScheduler - Dynamically adjusts the learning rate.
///
/// Several strategies available:
/// - **StepLR**: Reduces LR at regular intervals
/// - **ReduceOnPlateau**: Reduces LR when loss stagnates
/// - **ExponentialLR**: Exponential decay
///
/// # Example
/// ```rust
/// use test_neural::callbacks::{LearningRateScheduler, LRSchedule};
///
/// let scheduler = LearningRateScheduler::new(
///     LRSchedule::ReduceOnPlateau {
///         patience: 5,
///         factor: 0.5,
///         min_delta: 0.0001
///     }
/// );
/// ```
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    schedule: LRSchedule,
    
    // State for ReduceOnPlateau
    best_loss: f64,
    wait: usize,
    
    // Current LR (pub to allow access from fit())
    pub current_lr: f64,
}

impl LearningRateScheduler {
    /// Creates a new learning rate scheduler.
    pub fn new(schedule: LRSchedule) -> Self {
        LearningRateScheduler {
            schedule,
            best_loss: f64::INFINITY,
            wait: 0,
            current_lr: 0.0,
        }
    }
    
    /// Returns the current learning rate.
    pub fn current_lr(&self) -> f64 {
        self.current_lr
    }
    
    /// Updates the optimizer's learning rate.
    pub fn update_optimizer_lr(&mut self, optimizer: &mut OptimizerType) {
        match optimizer {
            OptimizerType::SGD { learning_rate } => *learning_rate = self.current_lr,
            OptimizerType::Momentum { learning_rate, .. } => *learning_rate = self.current_lr,
            OptimizerType::RMSprop { learning_rate, .. } => *learning_rate = self.current_lr,
            OptimizerType::Adam { learning_rate, .. } => *learning_rate = self.current_lr,
            OptimizerType::AdamW { learning_rate, .. } => *learning_rate = self.current_lr,
        }
    }
}

impl Callback for LearningRateScheduler {
    fn on_train_begin(&mut self, _network: &Network) {
        // Note: Le LR sera géré par la méthode fit() qui a accès à l'optimizer
        self.best_loss = f64::INFINITY;
        self.wait = 0;
    }
    
    fn on_epoch_end(&mut self, epoch: usize, _network: &Network, _train_loss: f64, val_loss: Option<f64>) -> bool {
        // Note: On ne peut pas modifier network ici car c'est une référence immutable
        // Le LR sera mis à jour dans la méthode d'entraînement
        
        match &self.schedule {
            LRSchedule::StepLR { step_size, gamma } => {
                if (epoch + 1).is_multiple_of(*step_size) {
                    let new_lr = self.current_lr * gamma;
                    println!("📉 LR Scheduler: Epoch {} - Réduction LR {:.6} → {:.6}", 
                             epoch, self.current_lr, new_lr);
                    self.current_lr = new_lr;
                }
            },
            
            LRSchedule::ReduceOnPlateau { patience, factor, min_delta } => {
                if let Some(loss) = val_loss {
                    if loss < self.best_loss - min_delta {
                        self.best_loss = loss;
                        self.wait = 0;
                    } else {
                        self.wait += 1;
                        if self.wait >= *patience {
                            let new_lr = self.current_lr * factor;
                            println!("\n📉 LR Scheduler: Epoch {} - Plateau detected, reducing LR {:.6} → {:.6}", 
                                     epoch, self.current_lr, new_lr);
                            self.current_lr = new_lr;
                            self.wait = 0;
                        }
                    }
                }
            },
            
            LRSchedule::ExponentialLR { gamma } => {
                let new_lr = self.current_lr * gamma;
                if epoch > 0 && epoch.is_multiple_of(10) {
                    println!("📉 LR Scheduler: Epoch {} - LR = {:.6}", epoch, new_lr);
                }
                self.current_lr = new_lr;
            },
        }
        
        true  // Continue toujours
    }
}

/// ProgressBar - Displays training progress in real-time.
///
/// Shows in real-time:
/// - Epoch progression
/// - Training and validation loss
/// - Elapsed time and estimated remaining time
///
/// # Example
/// ```rust
/// use test_neural::callbacks::ProgressBar;
///
/// let progress = ProgressBar::new(100);  // 100 epochs
/// ```
#[derive(Debug, Clone)]
pub struct ProgressBar {
    total_epochs: usize,
    start_time: Option<std::time::Instant>,
    verbose: bool,
}

impl ProgressBar {
    /// Creates a new progress bar.
    ///
    /// # Arguments
    /// - `total_epochs`: Total number of epochs
    pub fn new(total_epochs: usize) -> Self {
        ProgressBar {
            total_epochs,
            start_time: None,
            verbose: true,
        }
    }
    
    /// Enables/disables verbose mode.
    pub fn set_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl Callback for ProgressBar {
    fn on_train_begin(&mut self, _network: &Network) {
        self.start_time = Some(std::time::Instant::now());
        if self.verbose {
            println!("🚀 Début de l'entraînement ({} epochs)", self.total_epochs);
        }
    }
    
    fn on_train_end(&mut self, _network: &Network) {
        if let Some(start) = self.start_time {
            let duration = start.elapsed();
            if self.verbose {
                println!("✅ Entraînement terminé en {:.2}s", duration.as_secs_f64());
            }
        }
    }
    
    fn on_epoch_end(&mut self, epoch: usize, _network: &Network, train_loss: f64, val_loss: Option<f64>) -> bool {
        if self.verbose {
            let progress = (epoch + 1) as f64 / self.total_epochs as f64 * 100.0;
            
            if let (Some(start), Some(val)) = (self.start_time, val_loss) {
                let elapsed = start.elapsed().as_secs_f64();
                let eta = elapsed / (epoch + 1) as f64 * (self.total_epochs - epoch - 1) as f64;
                
                print!("\rEpoch {}/{} [{:.1}%] - train_loss: {:.6} - val_loss: {:.6} - ETA: {:.0}s   ",
                       epoch + 1, self.total_epochs, progress, train_loss, val, eta);
            } else {
                print!("\rEpoch {}/{} [{:.1}%] - train_loss: {:.6}   ",
                       epoch + 1, self.total_epochs, progress, train_loss);
            }
            
            use std::io::Write;
            std::io::stdout().flush().ok();
            
            // Nouvelle ligne tous les 10 epochs ou à la fin
            if (epoch + 1).is_multiple_of(10) || epoch + 1 == self.total_epochs {
                println!();
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{Activation, LossFunction};
    use crate::optimizer::OptimizerType;

    #[test]
    fn test_early_stopping_triggers() {
        use crate::builder::NetworkBuilder;
        let mut early_stop = EarlyStopping::new(3, 0.001);
        let network = NetworkBuilder::new(2, 1)
            .hidden_layer(5, Activation::Sigmoid)
            .output_activation(Activation::Sigmoid)
            .loss(LossFunction::MSE)
            .optimizer(OptimizerType::sgd(0.1))
            .build();
        
        early_stop.on_train_begin(&network);
        
        // Pas d'amélioration pendant 3 epochs
        assert!(early_stop.on_epoch_end(0, &network, 1.0, Some(1.0)));
        assert!(early_stop.on_epoch_end(1, &network, 0.9, Some(1.0)));
        assert!(early_stop.on_epoch_end(2, &network, 0.8, Some(1.0)));
        assert!(!early_stop.on_epoch_end(3, &network, 0.7, Some(1.0)));  // Arrête ici
        
        assert!(early_stop.stopped());
    }

    #[test]
    fn test_early_stopping_improvement() {
        use crate::builder::NetworkBuilder;
        let mut early_stop = EarlyStopping::new(3, 0.001);
        let network = NetworkBuilder::new(2, 1)
            .hidden_layer(5, Activation::Sigmoid)
            .output_activation(Activation::Sigmoid)
            .loss(LossFunction::MSE)
            .optimizer(OptimizerType::sgd(0.1))
            .build();
        
        early_stop.on_train_begin(&network);
        
        // Amélioration continue
        assert!(early_stop.on_epoch_end(0, &network, 1.0, Some(1.0)));
        assert!(early_stop.on_epoch_end(1, &network, 0.9, Some(0.5)));  // Amélioration
        assert!(early_stop.on_epoch_end(2, &network, 0.8, Some(0.3)));  // Amélioration
        
        assert!(!early_stop.stopped());
        assert_eq!(early_stop.best_epoch(), 2);
    }

    #[test]
    fn test_lr_scheduler_step() {
        use crate::builder::NetworkBuilder;
        let mut scheduler = LearningRateScheduler::new(
            LRSchedule::StepLR { step_size: 2, gamma: 0.5 }
        );
        
        let network = NetworkBuilder::new(2, 1)
            .hidden_layer(5, Activation::Sigmoid)
            .output_activation(Activation::Sigmoid)
            .loss(LossFunction::MSE)
            .optimizer(OptimizerType::sgd(1.0))
            .build();
        
        // Initialise manuellement le LR (normalement fait par fit())
        scheduler.current_lr = 1.0;
        scheduler.on_train_begin(&network);
        assert_eq!(scheduler.current_lr(), 1.0);
        
        scheduler.on_epoch_end(0, &network, 1.0, Some(1.0));
        assert_eq!(scheduler.current_lr(), 1.0);  // Pas encore
        
        scheduler.on_epoch_end(1, &network, 0.9, Some(0.9));
        assert_eq!(scheduler.current_lr(), 0.5);  // Réduit à epoch 2
    }
}
