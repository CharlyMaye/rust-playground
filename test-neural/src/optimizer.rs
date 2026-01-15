//! Module d'optimiseurs pour l'entraînement des réseaux de neurones
//!
//! Ce module fournit différents algorithmes d'optimisation pour mettre à jour
//! les poids du réseau pendant l'entraînement.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Type d'optimiseur disponible
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Descente de gradient stochastique (SGD) - Simple et rapide
    SGD { learning_rate: f64 },
    
    /// SGD avec momentum - Accélère dans les bonnes directions
    Momentum { 
        learning_rate: f64, 
        beta: f64  // Typiquement 0.9
    },
    
    /// RMSprop - Adapte le learning rate par paramètre
    RMSprop { 
        learning_rate: f64, 
        beta: f64,      // Typiquement 0.9
        epsilon: f64    // Typiquement 1e-8
    },
    
    /// Adam - Adaptive Moment Estimation (standard moderne)
    Adam { 
        learning_rate: f64,
        beta1: f64,     // Typiquement 0.9 (momentum)
        beta2: f64,     // Typiquement 0.999 (variance)
        epsilon: f64    // Typiquement 1e-8
    },
    
    /// AdamW - Adam avec Weight Decay découplé
    AdamW { 
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64  // Typiquement 0.01
    },
}

impl OptimizerType {
    /// Crée un optimiseur SGD avec le learning rate spécifié
    pub fn sgd(learning_rate: f64) -> Self {
        OptimizerType::SGD { learning_rate }
    }
    
    /// Crée un optimiseur Momentum avec paramètres par défaut
    pub fn momentum(learning_rate: f64) -> Self {
        OptimizerType::Momentum { 
            learning_rate, 
            beta: 0.9 
        }
    }
    
    /// Crée un optimiseur RMSprop avec paramètres par défaut
    pub fn rmsprop(learning_rate: f64) -> Self {
        OptimizerType::RMSprop { 
            learning_rate, 
            beta: 0.9, 
            epsilon: 1e-8 
        }
    }
    
    /// Crée un optimiseur Adam avec paramètres par défaut (recommandé)
    pub fn adam(learning_rate: f64) -> Self {
        OptimizerType::Adam { 
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8
        }
    }
    
    /// Crée un optimiseur AdamW avec paramètres par défaut
    pub fn adamw(learning_rate: f64, weight_decay: f64) -> Self {
        OptimizerType::AdamW { 
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay
        }
    }
}

/// État de l'optimiseur pour une matrice de poids
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState2D {
    /// Premier moment (momentum) - utilisé par Momentum, Adam, AdamW
    pub m: Option<Array2<f64>>,
    
    /// Second moment (variance) - utilisé par RMSprop, Adam, AdamW
    pub v: Option<Array2<f64>>,
    
    /// Nombre d'itérations (pour bias correction dans Adam)
    pub t: usize,
}

impl OptimizerState2D {
    /// Crée un nouvel état pour une matrice de taille donnée
    pub fn new(shape: (usize, usize), optimizer: &OptimizerType) -> Self {
        let needs_m = matches!(optimizer, 
            OptimizerType::Momentum { .. } | 
            OptimizerType::Adam { .. } | 
            OptimizerType::AdamW { .. }
        );
        
        let needs_v = matches!(optimizer,
            OptimizerType::RMSprop { .. } |
            OptimizerType::Adam { .. } | 
            OptimizerType::AdamW { .. }
        );
        
        OptimizerState2D {
            m: if needs_m { Some(Array2::zeros(shape)) } else { None },
            v: if needs_v { Some(Array2::zeros(shape)) } else { None },
            t: 0,
        }
    }
    
    /// Met à jour les poids avec le gradient calculé
    pub fn step(
        &mut self,
        weights: &mut Array2<f64>,
        gradient: &Array2<f64>,
        optimizer: &OptimizerType,
    ) {
        self.t += 1;
        
        match optimizer {
            OptimizerType::SGD { learning_rate } => {
                // Simple gradient descent: w = w - lr * grad
                *weights -= &(gradient * *learning_rate);
            }
            
            OptimizerType::Momentum { learning_rate, beta } => {
                // Momentum: m = beta * m + grad
                //           w = w - lr * m
                let m = self.m.as_mut().expect("Momentum state not initialized");
                *m = &(m.clone() * *beta) + gradient;
                *weights -= &(&*m * *learning_rate);
            }
            
            OptimizerType::RMSprop { learning_rate, beta, epsilon } => {
                // RMSprop: v = beta * v + (1 - beta) * grad^2
                //          w = w - lr * grad / (sqrt(v) + epsilon)
                let v = self.v.as_mut().expect("RMSprop state not initialized");
                *v = &(v.clone() * *beta) + &(gradient.mapv(|g| g * g) * (1.0 - beta));
                
                let update = gradient / &(v.mapv(|x| x.sqrt()) + *epsilon);
                *weights -= &(update * *learning_rate);
            }
            
            OptimizerType::Adam { learning_rate, beta1, beta2, epsilon } => {
                // Adam: m = beta1 * m + (1 - beta1) * grad
                //       v = beta2 * v + (1 - beta2) * grad^2
                //       m_hat = m / (1 - beta1^t)
                //       v_hat = v / (1 - beta2^t)
                //       w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
                
                let m = self.m.as_mut().expect("Adam m state not initialized");
                let v = self.v.as_mut().expect("Adam v state not initialized");
                
                // Update biased first moment estimate
                *m = &(m.clone() * *beta1) + &(gradient * (1.0 - beta1));
                
                // Update biased second moment estimate
                *v = &(v.clone() * *beta2) + &(gradient.mapv(|g: f64| g * g) * (1.0 - beta2));
                
                // Compute bias-corrected moments
                let m_hat = &*m / (1.0 - beta1.powi(self.t as i32));
                let v_hat = &*v / (1.0 - beta2.powi(self.t as i32));
                
                // Update weights
                let update = &m_hat / &(v_hat.mapv(|x: f64| x.sqrt()) + *epsilon);
                *weights -= &(update * *learning_rate);
            }
            
            OptimizerType::AdamW { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                // AdamW: Same as Adam but with decoupled weight decay
                //        w = w * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
                
                let m = self.m.as_mut().expect("AdamW m state not initialized");
                let v = self.v.as_mut().expect("AdamW v state not initialized");
                
                // Update biased moments
                *m = &(m.clone() * *beta1) + &(gradient * (1.0 - beta1));
                *v = &(v.clone() * *beta2) + &(gradient.mapv(|g: f64| g * g) * (1.0 - beta2));
                
                // Bias correction
                let m_hat = &*m / (1.0 - beta1.powi(self.t as i32));
                let v_hat = &*v / (1.0 - beta2.powi(self.t as i32));
                
                // Weight decay (decoupled)
                *weights *= 1.0 - learning_rate * weight_decay;
                
                // Adam update
                let update = &m_hat / &(v_hat.mapv(|x: f64| x.sqrt()) + *epsilon);
                *weights -= &(update * *learning_rate);
            }
        }
    }
}

/// État de l'optimiseur pour un vecteur de biais
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState1D {
    /// Premier moment (momentum)
    pub m: Option<Array1<f64>>,
    
    /// Second moment (variance)
    pub v: Option<Array1<f64>>,
    
    /// Nombre d'itérations
    pub t: usize,
}

impl OptimizerState1D {
    /// Crée un nouvel état pour un vecteur de taille donnée
    pub fn new(size: usize, optimizer: &OptimizerType) -> Self {
        let needs_m = matches!(optimizer, 
            OptimizerType::Momentum { .. } | 
            OptimizerType::Adam { .. } | 
            OptimizerType::AdamW { .. }
        );
        
        let needs_v = matches!(optimizer,
            OptimizerType::RMSprop { .. } |
            OptimizerType::Adam { .. } | 
            OptimizerType::AdamW { .. }
        );
        
        OptimizerState1D {
            m: if needs_m { Some(Array1::zeros(size)) } else { None },
            v: if needs_v { Some(Array1::zeros(size)) } else { None },
            t: 0,
        }
    }
    
    /// Met à jour les biais avec le gradient calculé
    pub fn step(
        &mut self,
        biases: &mut Array1<f64>,
        gradient: &Array1<f64>,
        optimizer: &OptimizerType,
    ) {
        self.t += 1;
        
        match optimizer {
            OptimizerType::SGD { learning_rate } => {
                *biases -= &(gradient * *learning_rate);
            }
            
            OptimizerType::Momentum { learning_rate, beta } => {
                let m = self.m.as_mut().expect("Momentum state not initialized");
                *m = &(m.clone() * *beta) + gradient;
                *biases -= &(&*m * *learning_rate);
            }
            
            OptimizerType::RMSprop { learning_rate, beta, epsilon } => {
                let v = self.v.as_mut().expect("RMSprop state not initialized");
                *v = &(v.clone() * *beta) + &(gradient.mapv(|g: f64| g * g) * (1.0 - beta));
                
                let update = gradient / &(v.mapv(|x: f64| x.sqrt()) + *epsilon);
                *biases -= &(update * *learning_rate);
            }
            
            OptimizerType::Adam { learning_rate, beta1, beta2, epsilon } => {
                let m = self.m.as_mut().expect("Adam m state not initialized");
                let v = self.v.as_mut().expect("Adam v state not initialized");
                
                *m = &(m.clone() * *beta1) + &(gradient * (1.0 - beta1));
                *v = &(v.clone() * *beta2) + &(gradient.mapv(|g: f64| g * g) * (1.0 - beta2));
                
                let m_hat = &*m / (1.0 - beta1.powi(self.t as i32));
                let v_hat = &*v / (1.0 - beta2.powi(self.t as i32));
                
                let update = &m_hat / &(v_hat.mapv(|x: f64| x.sqrt()) + *epsilon);
                *biases -= &(update * *learning_rate);
            }
            
            OptimizerType::AdamW { learning_rate, beta1, beta2, epsilon, weight_decay } => {
                let m = self.m.as_mut().expect("AdamW m state not initialized");
                let v = self.v.as_mut().expect("AdamW v state not initialized");
                
                *m = &(m.clone() * *beta1) + &(gradient * (1.0 - beta1));
                *v = &(v.clone() * *beta2) + &(gradient.mapv(|g: f64| g * g) * (1.0 - beta2));
                
                let m_hat = &*m / (1.0 - beta1.powi(self.t as i32));
                let v_hat = &*v / (1.0 - beta2.powi(self.t as i32));
                
                // Weight decay pour biais (généralement pas appliqué, mais pour cohérence)
                *biases *= 1.0 - learning_rate * weight_decay;
                
                let update = &m_hat / &(v_hat.mapv(|x: f64| x.sqrt()) + *epsilon);
                *biases -= &(update * *learning_rate);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_update() {
        let mut weights = Array2::from_elem((2, 2), 1.0);
        let gradient = Array2::from_elem((2, 2), 0.1);
        let mut state = OptimizerState2D::new((2, 2), &OptimizerType::sgd(0.1));
        
        state.step(&mut weights, &gradient, &OptimizerType::sgd(0.1));
        
        // w = 1.0 - 0.1 * 0.1 = 0.99
        assert!((weights[[0, 0]] - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_adam_initialization() {
        let optimizer = OptimizerType::adam(0.001);
        let state = OptimizerState2D::new((3, 3), &optimizer);
        
        assert!(state.m.is_some());
        assert!(state.v.is_some());
        assert_eq!(state.t, 0);
    }

    #[test]
    fn test_optimizer_constructors() {
        let sgd = OptimizerType::sgd(0.1);
        assert!(matches!(sgd, OptimizerType::SGD { .. }));
        
        let adam = OptimizerType::adam(0.001);
        assert!(matches!(adam, OptimizerType::Adam { .. }));
        
        let rmsprop = OptimizerType::rmsprop(0.01);
        assert!(matches!(rmsprop, OptimizerType::RMSprop { .. }));
    }
}
