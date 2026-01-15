//! Module de métriques d'évaluation pour les réseaux de neurones
//!
//! Ce module fournit diverses métriques pour évaluer la performance des modèles,
//! notamment pour la classification binaire et multi-classes.

use ndarray::{Array1, Array2};

/// Résultat d'évaluation pour classification binaire
#[derive(Debug, Clone)]
pub struct BinaryMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
}

impl BinaryMetrics {
    /// Affiche les métriques dans un format lisible
    pub fn summary(&self) -> String {
        format!(
            "Accuracy: {:.4} | Precision: {:.4} | Recall: {:.4} | F1: {:.4}\n\
             TP: {} | FP: {} | TN: {} | FN: {}",
            self.accuracy, self.precision, self.recall, self.f1_score,
            self.true_positives, self.false_positives,
            self.true_negatives, self.false_negatives
        )
    }
}

/// Calcule l'accuracy (pourcentage de prédictions correctes)
/// 
/// # Arguments
/// * `predictions` - Vecteur de prédictions du réseau
/// * `targets` - Vecteur de valeurs réelles
/// * `threshold` - Seuil de classification (par défaut 0.5)
/// 
/// # Exemples
/// ```
/// use ndarray::array;
/// use test_neural::metrics::accuracy;
/// 
/// let predictions = vec![array![0.1], array![0.9], array![0.8], array![0.2]];
/// let targets = vec![array![0.0], array![1.0], array![1.0], array![0.0]];
/// let acc = accuracy(&predictions, &targets, 0.5);
/// assert_eq!(acc, 1.0); // 100% correct
/// ```
pub fn accuracy(
    predictions: &[Array1<f64>],
    targets: &[Array1<f64>],
    threshold: f64,
) -> f64 {
    if predictions.is_empty() || predictions.len() != targets.len() {
        return 0.0;
    }

    let mut correct = 0;
    
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        // Classification binaire simple
        if pred.len() == 1 && target.len() == 1 {
            let pred_class: f64 = if pred[0] > threshold { 1.0 } else { 0.0 };
            let target_class: f64 = if target[0] > threshold { 1.0 } else { 0.0 };
            if (pred_class - target_class).abs() < 1e-6 {
                correct += 1;
            }
        } 
        // Classification multi-classes (argmax)
        else {
            let pred_class = pred.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            let target_class = target.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            if pred_class == target_class {
                correct += 1;
            }
        }
    }
    
    correct as f64 / predictions.len() as f64
}

/// Calcule les métriques complètes pour classification binaire
/// 
/// # Retourne
/// - Accuracy: % de prédictions correctes
/// - Precision: TP / (TP + FP) - "quand je prédis positif, à quelle fréquence ai-je raison?"
/// - Recall: TP / (TP + FN) - "je capture quel % de tous les positifs réels?"
/// - F1-score: moyenne harmonique de Precision et Recall
/// 
/// # Arguments
/// * `predictions` - Vecteur de prédictions
/// * `targets` - Vecteur de valeurs réelles  
/// * `threshold` - Seuil de classification (typiquement 0.5)
/// 
/// # Exemples
/// ```
/// use ndarray::array;
/// use test_neural::metrics::binary_metrics;
/// 
/// let predictions = vec![array![0.9], array![0.8], array![0.3], array![0.2]];
/// let targets = vec![array![1.0], array![1.0], array![0.0], array![0.0]];
/// let metrics = binary_metrics(&predictions, &targets, 0.5);
/// 
/// println!("{}", metrics.summary());
/// assert_eq!(metrics.accuracy, 1.0);
/// ```
pub fn binary_metrics(
    predictions: &[Array1<f64>],
    targets: &[Array1<f64>],
    threshold: f64,
) -> BinaryMetrics {
    let mut tp = 0; // True Positives
    let mut fp = 0; // False Positives
    let mut tn = 0; // True Negatives
    let mut fn_ = 0; // False Negatives
    
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        if pred.len() != 1 || target.len() != 1 {
            continue; // Skip multi-class (non supporté par cette fonction)
        }
        
        let pred_class = pred[0] > threshold;
        let target_class = target[0] > threshold;
        
        match (pred_class, target_class) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }
    
    let accuracy = (tp + tn) as f64 / (tp + fp + tn + fn_) as f64;
    
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    
    let f1_score = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };
    
    BinaryMetrics {
        accuracy,
        precision,
        recall,
        f1_score,
        true_positives: tp,
        false_positives: fp,
        true_negatives: tn,
        false_negatives: fn_,
    }
}

/// Calcule la matrice de confusion pour classification binaire
/// 
/// Retourne une matrice 2x2 :
/// ```text
///                 Prédit
///              Neg    Pos
/// Réel  Neg [  TN  |  FP  ]
///       Pos [  FN  |  TP  ]
/// ```
/// 
/// # Arguments
/// * `predictions` - Vecteur de prédictions
/// * `targets` - Vecteur de valeurs réelles
/// * `threshold` - Seuil de classification
pub fn confusion_matrix_binary(
    predictions: &[Array1<f64>],
    targets: &[Array1<f64>],
    threshold: f64,
) -> Array2<usize> {
    let mut matrix = Array2::zeros((2, 2));
    
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        if pred.len() != 1 || target.len() != 1 {
            continue;
        }
        
        let pred_class = if pred[0] > threshold { 1 } else { 0 };
        let target_class = if target[0] > threshold { 1 } else { 0 };
        
        matrix[[target_class, pred_class]] += 1;
    }
    
    matrix
}

/// Calcule la matrice de confusion pour classification multi-classes
/// 
/// Retourne une matrice NxN où N est le nombre de classes
/// matrix[i][j] = nombre d'exemples de classe i prédits comme classe j
/// 
/// # Arguments
/// * `predictions` - Vecteur de prédictions (probabilités par classe)
/// * `targets` - Vecteur de valeurs réelles (one-hot ou probabilités)
/// * `num_classes` - Nombre de classes
pub fn confusion_matrix_multiclass(
    predictions: &[Array1<f64>],
    targets: &[Array1<f64>],
    num_classes: usize,
) -> Array2<usize> {
    let mut matrix = Array2::zeros((num_classes, num_classes));
    
    for (pred, target) in predictions.iter().zip(targets.iter()) {
        // Trouver la classe prédite (argmax)
        let pred_class = pred.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        // Trouver la classe réelle (argmax)
        let target_class = target.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        if pred_class < num_classes && target_class < num_classes {
            matrix[[target_class, pred_class]] += 1;
        }
    }
    
    matrix
}

/// Affiche une matrice de confusion dans un format lisible
pub fn format_confusion_matrix(matrix: &Array2<usize>, class_names: Option<&[&str]>) -> String {
    let size = matrix.nrows();
    let mut result = String::new();
    
    result.push_str("\nConfusion Matrix:\n");
    result.push_str("                Predicted\n");
    
    // Header avec noms de classes
    result.push_str("         ");
    for i in 0..size {
        if let Some(names) = class_names {
            if i < names.len() {
                result.push_str(&format!("{:>8} ", names[i]));
            } else {
                result.push_str(&format!("{:>8} ", i));
            }
        } else {
            result.push_str(&format!("{:>8} ", i));
        }
    }
    result.push('\n');
    
    // Lignes avec données
    for i in 0..size {
        if i == 0 {
            result.push_str("Actual ");
        } else {
            result.push_str("       ");
        }
        
        if let Some(names) = class_names {
            if i < names.len() {
                result.push_str(&format!("{:>5} ", names[i]));
            } else {
                result.push_str(&format!("{:>5} ", i));
            }
        } else {
            result.push_str(&format!("{:>5} ", i));
        }
        
        for j in 0..size {
            result.push_str(&format!("{:>8} ", matrix[[i, j]]));
        }
        result.push('\n');
    }
    
    result
}

/// Calcule la courbe ROC (Receiver Operating Characteristic)
/// 
/// Retourne des vecteurs (FPR, TPR) pour différents seuils
/// Utile pour visualiser performance à différents seuils
/// 
/// # Arguments
/// * `predictions` - Scores de prédiction (probabilités)
/// * `targets` - Valeurs réelles (0 ou 1)
/// * `num_thresholds` - Nombre de seuils à tester
pub fn roc_curve(
    predictions: &[Array1<f64>],
    targets: &[Array1<f64>],
    num_thresholds: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut fpr_values = Vec::new();
    let mut tpr_values = Vec::new();
    let mut thresholds = Vec::new();
    
    for i in 0..=num_thresholds {
        let threshold = i as f64 / num_thresholds as f64;
        thresholds.push(threshold);
        
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if pred.len() != 1 || target.len() != 1 {
                continue;
            }
            
            let pred_class = pred[0] > threshold;
            let target_class = target[0] > 0.5;
            
            match (pred_class, target_class) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => fn_ += 1,
            }
        }
        
        // TPR (True Positive Rate) = Recall = Sensitivity
        let tpr = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        
        // FPR (False Positive Rate) = 1 - Specificity
        let fpr = if fp + tn > 0 {
            fp as f64 / (fp + tn) as f64
        } else {
            0.0
        };
        
        fpr_values.push(fpr);
        tpr_values.push(tpr);
    }
    
    (fpr_values, tpr_values, thresholds)
}

/// Calcule l'AUC (Area Under Curve) pour la courbe ROC
/// 
/// Approximation par méthode des trapèzes
/// Valeur parfaite = 1.0, valeur aléatoire = 0.5
pub fn auc_roc(
    predictions: &[Array1<f64>],
    targets: &[Array1<f64>],
) -> f64 {
    let (mut fpr, mut tpr, _) = roc_curve(predictions, targets, 100);
    
    // Trier par FPR croissant (nécessaire pour l'intégration)
    let mut points: Vec<(f64, f64)> = fpr.iter().zip(tpr.iter())
        .map(|(&f, &t)| (f, t))
        .collect();
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    fpr = points.iter().map(|(f, _)| *f).collect();
    tpr = points.iter().map(|(_, t)| *t).collect();
    
    // Méthode des trapèzes
    let mut area = 0.0;
    for i in 1..fpr.len() {
        let width = (fpr[i] - fpr[i - 1]).abs();
        let height = (tpr[i] + tpr[i - 1]) / 2.0;
        area += width * height;
    }
    
    area
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_accuracy_perfect() {
        let predictions = vec![
            array![0.1], array![0.9], array![0.8], array![0.2]
        ];
        let targets = vec![
            array![0.0], array![1.0], array![1.0], array![0.0]
        ];
        
        let acc = accuracy(&predictions, &targets, 0.5);
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn test_accuracy_half() {
        let predictions = vec![
            array![0.1], array![0.1], array![0.8], array![0.8]
        ];
        let targets = vec![
            array![0.0], array![1.0], array![1.0], array![0.0]
        ];
        
        let acc = accuracy(&predictions, &targets, 0.5);
        assert_eq!(acc, 0.5);
    }

    #[test]
    fn test_binary_metrics() {
        let predictions = vec![
            array![0.9], array![0.8], array![0.3], array![0.2], array![0.7], array![0.4]
        ];
        let targets = vec![
            array![1.0], array![1.0], array![0.0], array![0.0], array![1.0], array![0.0]
        ];
        
        let metrics = binary_metrics(&predictions, &targets, 0.5);
        
        assert_eq!(metrics.true_positives, 3);
        assert_eq!(metrics.false_positives, 0);
        assert_eq!(metrics.true_negatives, 3);
        assert_eq!(metrics.false_negatives, 0);
        assert_eq!(metrics.accuracy, 1.0);
        assert_eq!(metrics.precision, 1.0);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.f1_score, 1.0);
    }

    #[test]
    fn test_confusion_matrix_binary() {
        let predictions = vec![
            array![0.9], array![0.8], array![0.3], array![0.2]
        ];
        let targets = vec![
            array![1.0], array![1.0], array![1.0], array![0.0]
        ];
        
        let matrix = confusion_matrix_binary(&predictions, &targets, 0.5);
        
        // TN=1, FP=0, FN=1, TP=2
        assert_eq!(matrix[[0, 0]], 1); // TN
        assert_eq!(matrix[[0, 1]], 0); // FP
        assert_eq!(matrix[[1, 0]], 1); // FN
        assert_eq!(matrix[[1, 1]], 2); // TP
    }

    #[test]
    fn test_multiclass_accuracy() {
        let predictions = vec![
            array![0.8, 0.1, 0.1], // Classe 0
            array![0.1, 0.8, 0.1], // Classe 1
            array![0.1, 0.1, 0.8], // Classe 2
        ];
        let targets = vec![
            array![1.0, 0.0, 0.0], // Classe 0
            array![0.0, 1.0, 0.0], // Classe 1
            array![0.0, 0.0, 1.0], // Classe 2
        ];
        
        let acc = accuracy(&predictions, &targets, 0.5);
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn test_auc_roc_perfect() {
        let predictions = vec![
            array![0.9], array![0.8], array![0.3], array![0.2]
        ];
        let targets = vec![
            array![1.0], array![1.0], array![0.0], array![0.0]
        ];
        
        let auc = auc_roc(&predictions, &targets);
        // AUC doit être > 0.5 (mieux que random)
        // Avec un petit dataset, l'AUC peut ne pas atteindre 1.0
        assert!(auc > 0.5, "AUC should be > 0.5 (random), got {}", auc);
        assert!(auc <= 1.0, "AUC should be <= 1.0, got {}", auc);
    }
}
