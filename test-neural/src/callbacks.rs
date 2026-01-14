/// Module pour les callbacks d'entraînement
/// 
/// Les callbacks permettent d'injecter du code personnalisé à différents moments
/// de l'entraînement : après chaque epoch, après chaque batch, etc.
/// 
/// Callbacks disponibles :
/// - **EarlyStopping** : Arrête l'entraînement si la loss ne s'améliore plus
/// - **ModelCheckpoint** : Sauvegarde automatique du meilleur modèle
/// - **LearningRateScheduler** : Ajuste dynamiquement le learning rate
/// - **ProgressBar** : Affiche la progression en temps réel

use crate::network::Network;
use crate::optimizer::OptimizerType;
use std::path::PathBuf;

/// Trait de base pour tous les callbacks
/// 
/// Les callbacks sont appelés à différents moments de l'entraînement
/// et peuvent modifier le comportement ou collecter des statistiques.
pub trait Callback {
    /// Appelé au début de l'entraînement
    fn on_train_begin(&mut self, _network: &Network) {}
    
    /// Appelé à la fin de l'entraînement
    fn on_train_end(&mut self, _network: &Network) {}
    
    /// Appelé au début de chaque epoch
    /// 
    /// # Arguments
    /// - `epoch`: Numéro de l'epoch (0-indexed)
    /// - `network`: Référence au réseau
    fn on_epoch_begin(&mut self, _epoch: usize, _network: &Network) {}
    
    /// Appelé à la fin de chaque epoch
    /// 
    /// # Arguments
    /// - `epoch`: Numéro de l'epoch (0-indexed)
    /// - `network`: Référence au réseau
    /// - `train_loss`: Loss d'entraînement
    /// - `val_loss`: Loss de validation (None si pas de validation)
    /// 
    /// # Returns
    /// `true` pour continuer l'entraînement, `false` pour arrêter
    fn on_epoch_end(&mut self, _epoch: usize, _network: &Network, _train_loss: f64, _val_loss: Option<f64>) -> bool {
        true  // Continue par défaut
    }
}

/// EarlyStopping - Arrête l'entraînement si la loss ne s'améliore plus
/// 
/// Surveille la validation loss et arrête l'entraînement après `patience`
/// epochs sans amélioration. Évite l'overfitting.
/// 
/// # Example
/// ```
/// use test_neural::callbacks::EarlyStopping;
/// 
/// let early_stop = EarlyStopping::new(10, 0.0001);  // patience=10, min_delta=0.0001
/// ```
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Nombre d'epochs à attendre sans amélioration avant d'arrêter
    patience: usize,
    
    /// Amélioration minimale requise pour compter comme amélioration
    min_delta: f64,
    
    /// Meilleure loss observée
    best_loss: f64,
    
    /// Nombre d'epochs sans amélioration
    wait: usize,
    
    /// Indique si l'entraînement doit s'arrêter
    stopped: bool,
    
    /// Epoch où le meilleur modèle a été trouvé
    best_epoch: usize,
}

impl EarlyStopping {
    /// Crée un nouveau callback EarlyStopping
    /// 
    /// # Arguments
    /// - `patience`: Nombre d'epochs à attendre sans amélioration
    /// - `min_delta`: Amélioration minimale requise (ex: 0.0001)
    pub fn new(patience: usize, min_delta: f64) -> Self {
        EarlyStopping {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            wait: 0,
            stopped: false,
            best_epoch: 0,
        }
    }
    
    /// Vérifie si l'entraînement a été arrêté
    pub fn stopped(&self) -> bool {
        self.stopped
    }
    
    /// Retourne l'epoch du meilleur modèle
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }
    
    /// Retourne la meilleure loss
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
            // Amélioration si loss diminue de plus de min_delta
            if loss < self.best_loss - self.min_delta {
                self.best_loss = loss;
                self.best_epoch = epoch;
                self.wait = 0;
            } else {
                self.wait += 1;
                if self.wait >= self.patience {
                    self.stopped = true;
                    println!("⚠️ EarlyStopping: Arrêt à l'epoch {} (meilleur epoch: {}, loss: {:.6})", 
                             epoch, self.best_epoch, self.best_loss);
                    return false;  // Arrête l'entraînement
                }
            }
        }
        true  // Continue
    }
}

/// ModelCheckpoint - Sauvegarde automatique du meilleur modèle
/// 
/// Sauvegarde le modèle quand la validation loss s'améliore.
/// Permet de récupérer le meilleur modèle même si l'entraînement overfitte ensuite.
/// 
/// # Example
/// ```
/// use test_neural::callbacks::ModelCheckpoint;
/// 
/// let checkpoint = ModelCheckpoint::new("best_model.json", true);
/// ```
#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    /// Chemin où sauvegarder le modèle
    filepath: PathBuf,
    
    /// Sauvegarder uniquement si amélioration
    save_best_only: bool,
    
    /// Meilleure loss observée
    best_loss: f64,
    
    /// Format de sauvegarde (true = JSON, false = Binary)
    use_json: bool,
}

impl ModelCheckpoint {
    /// Crée un nouveau callback ModelCheckpoint
    /// 
    /// # Arguments
    /// - `filepath`: Chemin du fichier (ex: "best_model.json" ou "best_model.bin")
    /// - `save_best_only`: Si true, sauvegarde uniquement quand loss s'améliore
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
    
    /// Retourne la meilleure loss
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

/// Stratégie d'ajustement du learning rate
#[derive(Debug, Clone)]
pub enum LRSchedule {
    /// Réduit LR par un facteur à des epochs fixes
    /// Ex: StepLR { step_size: 10, gamma: 0.5 } divise LR par 2 tous les 10 epochs
    StepLR { step_size: usize, gamma: f64 },
    
    /// Réduit LR quand la loss plateau
    /// Ex: ReduceOnPlateau { patience: 5, factor: 0.5 } divise LR par 2 après 5 epochs sans amélioration
    ReduceOnPlateau { patience: usize, factor: f64, min_delta: f64 },
    
    /// Décroissance exponentielle du LR
    /// Ex: ExponentialLR { gamma: 0.95 } multiplie LR par 0.95 chaque epoch
    ExponentialLR { gamma: f64 },
}

/// LearningRateScheduler - Ajuste dynamiquement le learning rate
/// 
/// Plusieurs stratégies disponibles :
/// - **StepLR** : Réduit LR à intervalles réguliers
/// - **ReduceOnPlateau** : Réduit LR quand la loss stagne
/// - **ExponentialLR** : Décroissance exponentielle
/// 
/// # Example
/// ```
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
    
    // État pour ReduceOnPlateau
    best_loss: f64,
    wait: usize,
    
    // LR actuel (pub pour permettre l'accès depuis fit())
    pub current_lr: f64,
}

impl LearningRateScheduler {
    /// Crée un nouveau scheduler de learning rate
    pub fn new(schedule: LRSchedule) -> Self {
        LearningRateScheduler {
            schedule,
            best_loss: f64::INFINITY,
            wait: 0,
            current_lr: 0.0,
        }
    }
    
    /// Retourne le learning rate actuel
    pub fn current_lr(&self) -> f64 {
        self.current_lr
    }
    
    /// Met à jour le learning rate de l'optimizer
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
                if (epoch + 1) % step_size == 0 {
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
                            println!("📉 LR Scheduler: Epoch {} - Plateau détecté, réduction LR {:.6} → {:.6}", 
                                     epoch, self.current_lr, new_lr);
                            self.current_lr = new_lr;
                            self.wait = 0;
                        }
                    }
                }
            },
            
            LRSchedule::ExponentialLR { gamma } => {
                let new_lr = self.current_lr * gamma;
                if epoch > 0 && epoch % 10 == 0 {
                    println!("📉 LR Scheduler: Epoch {} - LR = {:.6}", epoch, new_lr);
                }
                self.current_lr = new_lr;
            },
        }
        
        true  // Continue toujours
    }
}

/// ProgressBar - Affiche la progression de l'entraînement
/// 
/// Affiche en temps réel :
/// - Progression des epochs
/// - Loss d'entraînement et validation
/// - Temps écoulé et temps estimé restant
/// 
/// # Example
/// ```
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
    /// Crée une nouvelle barre de progression
    /// 
    /// # Arguments
    /// - `total_epochs`: Nombre total d'epochs
    pub fn new(total_epochs: usize) -> Self {
        ProgressBar {
            total_epochs,
            start_time: None,
            verbose: true,
        }
    }
    
    /// Active/désactive le mode verbose
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
            if (epoch + 1) % 10 == 0 || epoch + 1 == self.total_epochs {
                println!();
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{Network, Activation, LossFunction};
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
