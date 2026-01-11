# Guide des Métriques d'Évaluation - Approches et Approfondissements

Ce document détaille les différentes approches pour évaluer un réseau de neurones et comment approfondir chaque métrique.

## 📊 Vue d'Ensemble

### Métriques Implémentées

| Métrique | Usage | Binaire | Multi-Classe | Complexité |
|----------|-------|---------|--------------|------------|
| **Accuracy** | Pourcentage correct | ✅ | ✅ | Simple |
| **Precision** | Vrais positifs / Prédits positifs | ✅ | ✅ | Moyen |
| **Recall** | Vrais positifs / Réels positifs | ✅ | ✅ | Moyen |
| **F1-Score** | Moyenne harmonique P/R | ✅ | ✅ | Moyen |
| **Confusion Matrix** | Vue détaillée erreurs | ✅ | ✅ | Simple |
| **ROC Curve** | Performance à tous seuils | ✅ | 🔶 | Avancé |
| **AUC** | Aire sous courbe ROC | ✅ | 🔶 | Avancé |

---

## 1. Accuracy (Exactitude)

### Définition
```
Accuracy = (Prédictions Correctes) / (Total Prédictions)
         = (TP + TN) / (TP + TN + FP + FN)
```

### Quand l'utiliser
- ✅ **Dataset équilibré** (50% classe A, 50% classe B)
- ✅ **Première métrique** à regarder (simple et intuitive)
- ✅ **Validation rapide** pendant l'entraînement

### Quand NE PAS l'utiliser
- ❌ **Dataset déséquilibré** (99% classe A, 1% classe B)
  - Exemple: Détection de fraude (fraudes rares)
  - Un modèle qui prédit toujours "pas de fraude" aura 99% accuracy mais est inutile
- ❌ **Coûts asymétriques** (faux négatif ≠ faux positif)
  - Exemple: Diagnostic médical (manquer un cancer est pire qu'un faux positif)

### Implémentation Actuelle
```rust
pub fn accuracy(predictions: &[Array1<f64>], targets: &[Array1<f64>], threshold: f64) -> f64
```
- Supporte binaire (seuil) et multi-classes (argmax)
- Simple et rapide
- Pas de dépendances externes

### Approfondissements Possibles

#### 1.1 Balanced Accuracy
Pour datasets déséquilibrés :
```rust
pub fn balanced_accuracy(predictions, targets, threshold) -> f64 {
    // Moyenne du recall par classe
    // = (Sensitivity + Specificity) / 2
    let metrics = binary_metrics(predictions, targets, threshold);
    let sensitivity = metrics.recall;
    let specificity = metrics.true_negatives as f64 / 
                      (metrics.true_negatives + metrics.false_positives) as f64;
    (sensitivity + specificity) / 2.0
}
```
**Usage:** Détection d'anomalies, datasets médicaux

#### 1.2 Top-K Accuracy
Pour classification multi-classes :
```rust
pub fn top_k_accuracy(predictions: &[Array1<f64>], targets: &[Array1<f64>], k: usize) -> f64 {
    // Correct si la vraie classe est dans les k prédictions les plus probables
    // Utilisé dans ImageNet (top-5 accuracy)
}
```
**Usage:** ImageNet, classification sur beaucoup de classes (1000+)

#### 1.3 Per-Class Accuracy
```rust
pub fn per_class_accuracy(predictions, targets) -> Vec<f64> {
    // Accuracy séparée pour chaque classe
    // Identifie les classes problématiques
}
```
**Usage:** Débogage, analyse de performance par classe

---

## 2. Precision, Recall, F1-Score

### Définitions

**Precision (Précision):**
```
Precision = TP / (TP + FP)
= "Quand je prédis positif, à quelle fréquence ai-je raison?"
```

**Recall (Rappel / Sensibilité):**
```
Recall = TP / (TP + FN)
= "Je capture quel % de tous les positifs réels?"
```

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
= Moyenne harmonique de Precision et Recall
```

### Trade-off Precision vs Recall

| Seuil | Precision | Recall | Usage |
|-------|-----------|--------|-------|
| **Élevé (0.9)** | ⬆️ Haute | ⬇️ Basse | Éviter faux positifs (spam filter) |
| **Moyen (0.5)** | ➡️ Équilibré | ➡️ Équilibré | Général |
| **Bas (0.1)** | ⬇️ Basse | ⬆️ Haute | Capturer tous les positifs (diagnostic médical) |

### Cas d'Usage

| Contexte | Priorité | Raison |
|----------|----------|--------|
| **Spam Filter** | 🔴 Precision | Ne pas bloquer vrais emails |
| **Détection Cancer** | 🔴 Recall | Ne pas manquer de malades |
| **Recommandations** | 🔴 Precision | Montrer contenu pertinent |
| **Moteur Recherche** | 🟡 F1 (équilibré) | Pertinence et couverture |

### Implémentation Actuelle
```rust
pub fn binary_metrics(predictions, targets, threshold) -> BinaryMetrics {
    // Retourne struct avec accuracy, precision, recall, f1_score, TP/FP/TN/FN
}
```

### Approfondissements Possibles

#### 2.1 Precision-Recall Curve
```rust
pub fn precision_recall_curve(predictions, targets, num_thresholds) 
    -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // (precision_values, recall_values, thresholds)
    // Visualise trade-off precision/recall
    // Utile pour choisir le bon seuil
}
```

#### 2.2 Average Precision (AP)
```rust
pub fn average_precision(predictions, targets) -> f64 {
    // Aire sous la courbe Precision-Recall
    // Métrique standard pour Object Detection
    // Utilisé dans COCO dataset, PASCAL VOC
}
```

#### 2.3 F-Beta Score
```rust
pub fn f_beta_score(precision: f64, recall: f64, beta: f64) -> f64 {
    // F_β = (1 + β²) × (P × R) / (β² × P + R)
    // β = 0.5: Favorise Precision
    // β = 1.0: F1 (équilibré)
    // β = 2.0: Favorise Recall
}
```
**Usage:** Ajuster l'importance de P vs R selon le contexte

#### 2.4 Macro/Micro/Weighted Averages (Multi-Classe)
```rust
pub enum AverageMethod {
    Macro,    // Moyenne simple des métriques par classe
    Micro,    // Calculer sur TP/FP/FN globaux
    Weighted, // Moyenne pondérée par nombre d'exemples
}

pub fn precision_multiclass(predictions, targets, method: AverageMethod) -> f64
```

**Exemple:**
```
Classes: A (100 ex), B (10 ex)
Precision_A = 0.9, Precision_B = 0.5

Macro:    (0.9 + 0.5) / 2 = 0.70  // Traite classes également
Weighted: (0.9×100 + 0.5×10) / 110 = 0.86  // Pondère par fréquence
```

---

## 3. Confusion Matrix

### Définition

**Binaire (2x2):**
```
                Prédit
             Neg    Pos
Réel  Neg [  TN  |  FP  ]
      Pos [  FN  |  TP  ]
```

**Multi-Classe (NxN):**
```
matrix[i][j] = nombre d'exemples de classe i prédits comme classe j
```

### Interprétation

| Métrique | Formule | Signification |
|----------|---------|---------------|
| **True Positive Rate** | TP / (TP + FN) | = Recall = Sensitivity |
| **False Positive Rate** | FP / (FP + TN) | Taux fausses alarmes |
| **True Negative Rate** | TN / (TN + FP) | = Specificity |
| **False Negative Rate** | FN / (FN + TP) | Taux manqués |

### Implémentation Actuelle
```rust
pub fn confusion_matrix_binary(predictions, targets, threshold) -> Array2<usize>
pub fn confusion_matrix_multiclass(predictions, targets, num_classes) -> Array2<usize>
pub fn format_confusion_matrix(matrix, class_names) -> String
```

### Approfondissements Possibles

#### 3.1 Normalized Confusion Matrix
```rust
pub fn confusion_matrix_normalized(predictions, targets, num_classes, 
                                   normalize: NormalizeMethod) -> Array2<f64> {
    enum NormalizeMethod {
        True,   // Normaliser par ligne (somme = 1 par vraie classe)
        Pred,   // Normaliser par colonne (somme = 1 par prédiction)
        All,    // Normaliser par total (toute matrice somme = 1)
    }
}
```
**Usage:** Visualisation, comparaison entre datasets de tailles différentes

#### 3.2 Métriques Dérivées de la Matrice
```rust
pub struct ConfusionMetrics {
    pub sensitivity: f64,     // = Recall = TPR
    pub specificity: f64,     // = TNR
    pub positive_likelihood_ratio: f64,  // TPR / FPR
    pub negative_likelihood_ratio: f64,  // FNR / TNR
    pub diagnostic_odds_ratio: f64,      // PLR / NLR
}
```

#### 3.3 Cohen's Kappa
```rust
pub fn cohens_kappa(confusion_matrix: &Array2<usize>) -> f64 {
    // Mesure accord au-delà du hasard
    // κ = (p_o - p_e) / (1 - p_e)
    // 1.0 = accord parfait, 0 = accord aléatoire
}
```
**Usage:** Inter-rater reliability, annoter qualité

---

## 4. ROC Curve & AUC

### ROC Curve (Receiver Operating Characteristic)

**Définition:**
- Graphique: FPR (x-axis) vs TPR (y-axis) à différents seuils
- Montre trade-off entre sensibilité et spécificité

**Implémentation Actuelle:**
```rust
pub fn roc_curve(predictions, targets, num_thresholds) 
    -> (Vec<f64>, Vec<f64>, Vec<f64>)  // (FPR, TPR, thresholds)
```

### AUC (Area Under Curve)

**Définition:**
- Aire sous la courbe ROC
- **1.0** = Prédictions parfaites (tous les positifs avant tous les négatifs)
- **0.5** = Performance aléatoire (ligne diagonale)
- **< 0.5** = Pire que random (modèle inversé!)

**Interprétation:**
```
AUC = Probabilité qu'un exemple positif aléatoire 
      ait un score plus élevé qu'un exemple négatif aléatoire
```

**Implémentation Actuelle:**
```rust
pub fn auc_roc(predictions, targets) -> f64
```

### Avantages ROC/AUC

✅ **Indépendant du seuil** - Évalue performance globale
✅ **Résistant déséquilibre** - Contrairement à accuracy
✅ **Standard industrie** - Benchmarking, publications

### Limites

❌ **Datasets très déséquilibrés** - Préférer Precision-Recall
❌ **Multi-classes** - Nécessite One-vs-Rest ou One-vs-One
❌ **Petit dataset** - Courbe instable (peu de points)

### Approfondissements Possibles

#### 4.1 Partial AUC
```rust
pub fn partial_auc(predictions, targets, fpr_range: (f64, f64)) -> f64 {
    // AUC dans une région spécifique de FPR
    // Utile si on s'intéresse à un taux FPR spécifique
    // Ex: Partial AUC entre FPR 0.0-0.1 pour high-precision tasks
}
```

#### 4.2 Multi-Class ROC
```rust
pub fn roc_auc_multiclass(predictions, targets, num_classes, 
                          method: MultiClassMethod) -> f64 {
    enum MultiClassMethod {
        OneVsRest,  // N courbes ROC (classe i vs reste)
        OneVsOne,   // N×(N-1)/2 courbes (toutes paires)
    }
}
```

#### 4.3 Bootstrap Confidence Intervals
```rust
pub fn auc_confidence_interval(predictions, targets, 
                                num_bootstraps: usize, 
                                confidence: f64) -> (f64, f64, f64) {
    // (lower_bound, auc, upper_bound)
    // Donne incertitude sur l'AUC
    // Utile pour petits datasets
}
```

---

## 5. Métriques Avancées (Non Implémentées)

### 5.1 Log Loss (Cross-Entropy Loss)
```rust
pub fn log_loss(predictions: &[Array1<f64>], targets: &[Array1<f64>]) -> f64 {
    // Pénalise fortement prédictions confiantes mais fausses
    // Standard pour compétitions (Kaggle)
    // Meilleur que accuracy car prend en compte probabilités
}
```

### 5.2 Brier Score
```rust
pub fn brier_score(predictions: &[Array1<f64>], targets: &[Array1<f64>]) -> f64 {
    // = Mean Squared Error entre probabilités prédites et réelles
    // Plus sensible aux prédictions extrêmes que log loss
}
```

### 5.3 Calibration Curve
```rust
pub fn calibration_curve(predictions, targets, num_bins: usize) 
    -> (Vec<f64>, Vec<f64>) {
    // (mean_predicted_probability, fraction_of_positives)
    // Vérifie si probabilités prédites sont calibrées
    // Ex: Parmi toutes les prédictions à 70%, 70% doivent être positives
}
```

### 5.4 Matthews Correlation Coefficient (MCC)
```rust
pub fn matthews_correlation_coefficient(confusion_matrix: &Array2<usize>) -> f64 {
    // MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    // Range: [-1, 1]
    // Prend en compte toutes les 4 valeurs (TP/TN/FP/FN)
    // Meilleur que accuracy pour datasets déséquilibrés
}
```

---

## 6. Métriques Spécialisées

### 6.1 Object Detection
```rust
pub struct ObjectDetectionMetrics {
    pub map_50: f64,        // mAP @ IoU=0.5
    pub map_50_95: f64,     // mAP @ IoU=0.5:0.05:0.95
    pub ar_1: f64,          // Average Recall @ 1 detection
    pub ar_10: f64,         // Average Recall @ 10 detections
}
```

### 6.2 Segmentation
```rust
pub fn iou(pred_mask: &Array2<bool>, true_mask: &Array2<bool>) -> f64 {
    // Intersection over Union
    // = Area(intersection) / Area(union)
}

pub fn dice_coefficient(pred_mask: &Array2<bool>, true_mask: &Array2<bool>) -> f64 {
    // = 2 × |A ∩ B| / (|A| + |B|)
    // Équivalent à F1 pour pixels
}
```

### 6.3 Ranking Metrics
```rust
pub fn mean_average_precision(predictions: &[Vec<f64>], targets: &[Vec<usize>]) -> f64 {
    // Pour systèmes de recommandation
}

pub fn ndcg(predictions: &[f64], relevances: &[f64], k: usize) -> f64 {
    // Normalized Discounted Cumulative Gain
    // Pour ranking et moteurs de recherche
}
```

---

## 7. Choix de Métrique par Domaine

### Machine Learning Général

| Problème | Métrique Principale | Métriques Secondaires |
|----------|---------------------|----------------------|
| **Classification Binaire Équilibrée** | Accuracy | Precision, Recall, F1 |
| **Classification Binaire Déséquilibrée** | F1, AUPRC | Recall, MCC |
| **Classification Multi-Classe** | Accuracy (macro) | Confusion Matrix, per-class F1 |
| **Ranking / Recommandation** | MAP, NDCG | Precision@K, Recall@K |
| **Probabilités Calibrées** | Log Loss, Brier Score | Calibration Curve |

### Domaines Spécifiques

| Domaine | Métrique | Pourquoi |
|---------|----------|----------|
| **Détection Fraude** | Recall, AUPRC | Ne pas manquer fraudes (coûteux) |
| **Spam Filter** | Precision | Ne pas bloquer vrais emails |
| **Diagnostic Médical** | Sensitivity (Recall) | Ne pas manquer malades |
| **Vision par Ordinateur** | mAP, IoU | Standard pour benchmarks |
| **NLP (Sentiment Analysis)** | Accuracy, F1 (macro) | Classes peuvent être déséquilibrées |
| **Search Engine** | NDCG, MAP | Ordre des résultats important |

---

## 8. Bonnes Pratiques

### 8.1 Toujours Rapporter Plusieurs Métriques
```rust
pub struct EvaluationReport {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub auc: f64,
    pub confusion_matrix: Array2<usize>,
}
```
**Pourquoi:** Une seule métrique peut être trompeuse

### 8.2 Utiliser Cross-Validation
```rust
pub fn cross_validate_metrics(
    model_builder: impl Fn() -> Network,
    dataset: &Dataset,
    k_folds: usize
) -> Vec<EvaluationReport> {
    // Retourne métriques pour chaque fold
    // Donne incertitude sur performance
}
```

### 8.3 Stratification pour Datasets Déséquilibrés
```rust
pub fn stratified_split(dataset: &Dataset, test_ratio: f64) 
    -> (Dataset, Dataset) {
    // Maintient proportions de classes dans train/test
    // Critique pour datasets déséquilibrés
}
```

### 8.4 Threshold Tuning
```rust
pub fn find_optimal_threshold(
    predictions: &[Array1<f64>],
    targets: &[Array1<f64>],
    metric: MetricType
) -> f64 {
    enum MetricType {
        MaxF1,          // Maximiser F1
        MaxAccuracy,    // Maximiser Accuracy
        TargetRecall(f64),  // Atteindre recall minimum
        TargetPrecision(f64), // Atteindre precision minimum
    }
}
```

---

## 9. Visualisation des Métriques

### 9.1 Plots à Implémenter
```rust
// Nécessite intégration avec plotters ou similar
pub fn plot_roc_curve(fpr: &[f64], tpr: &[f64]) -> Result<(), Error>
pub fn plot_precision_recall_curve(precision: &[f64], recall: &[f64]) -> Result<(), Error>
pub fn plot_confusion_matrix_heatmap(matrix: &Array2<usize>) -> Result<(), Error>
pub fn plot_learning_curve(train_metrics: &[f64], val_metrics: &[f64]) -> Result<(), Error>
```

### 9.2 Export pour Outils Externes
```rust
pub fn export_metrics_csv(metrics: &EvaluationReport, path: &str) -> Result<(), Error>
pub fn export_metrics_json(metrics: &EvaluationReport, path: &str) -> Result<(), Error>
```

---

## 10. Roadmap Métriques

### Phase 1: ✅ Complétée
- [x] Accuracy (binaire + multi-classes)
- [x] Precision, Recall, F1
- [x] Confusion Matrix
- [x] ROC Curve & AUC

### Phase 2: Recommandé Prochainement
- [ ] Log Loss / Cross-Entropy
- [ ] Precision-Recall Curve & Average Precision
- [ ] Matthews Correlation Coefficient (MCC)
- [ ] Per-Class Metrics (multi-classe)

### Phase 3: Avancé
- [ ] Calibration Curve & Brier Score
- [ ] Bootstrap Confidence Intervals
- [ ] Multi-Class ROC (One-vs-Rest, One-vs-One)
- [ ] Threshold Optimization

### Phase 4: Spécialisé
- [ ] Object Detection (mAP, IoU)
- [ ] Ranking (NDCG, MAP)
- [ ] Regression (MAE, MSE, R²)
- [ ] Time Series (MAPE, SMAPE)

---

## Références

1. **Scikit-learn Metrics** - https://scikit-learn.org/stable/modules/model_evaluation.html
2. **ROC Analysis** - Fawcett, T. (2006). "An introduction to ROC analysis"
3. **Precision-Recall vs ROC** - Davis & Goadrich (2006)
4. **Calibration** - Guo et al. (2017). "On Calibration of Modern Neural Networks"
5. **Multi-Class Metrics** - Sokolova & Lapalme (2009). "A systematic analysis of performance measures"
