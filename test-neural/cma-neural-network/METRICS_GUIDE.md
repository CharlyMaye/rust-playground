# Evaluation Metrics Guide - Approaches and Deep Dives

This document details the different approaches to evaluating neural networks and how to leverage each metric.

## 📊 Overview

### Implemented Metrics

| Metric | Usage | Binary | Multi-Class | Complexity |
|--------|-------|--------|-------------|------------|
| **Accuracy** | Percentage correct | ✅ | ✅ | Simple |
| **Precision** | True positives / Predicted positives | ✅ | ✅ | Medium |
| **Recall** | True positives / Actual positives | ✅ | ✅ | Medium |
| **F1-Score** | Harmonic mean P/R | ✅ | ✅ | Medium |
| **Confusion Matrix** | Detailed error view | ✅ | ✅ | Simple |
| **ROC Curve** | Performance at all thresholds | ✅ | 🔶 | Advanced |
| **AUC** | Area under ROC curve | ✅ | 🔶 | Advanced |

---

## 1. Accuracy

### Definition
```
Accuracy = (Correct Predictions) / (Total Predictions)
         = (TP + TN) / (TP + TN + FP + FN)
```

### When to Use
- ✅ **Balanced dataset** (50% class A, 50% class B)
- ✅ **First metric** to look at (simple and intuitive)
- ✅ **Quick validation** during training

### When NOT to Use
- ❌ **Imbalanced dataset** (99% class A, 1% class B)
  - Example: Fraud detection (fraud is rare)
  - A model that always predicts "no fraud" will have 99% accuracy but is useless
- ❌ **Asymmetric costs** (false negative ≠ false positive)
  - Example: Medical diagnosis (missing cancer is worse than false positive)

### Current Implementation
```rust
pub fn accuracy(predictions: &[Array1<f64>], targets: &[Array1<f64>], threshold: f64) -> f64
```
- Supports binary (threshold) and multi-class (argmax)
- Simple and fast
- No external dependencies

### Possible Extensions

#### 1.1 Balanced Accuracy
For imbalanced datasets:
```rust
pub fn balanced_accuracy(predictions, targets, threshold) -> f64 {
    // Average recall per class
    // = (Sensitivity + Specificity) / 2
    let metrics = binary_metrics(predictions, targets, threshold);
    let sensitivity = metrics.recall;
    let specificity = metrics.true_negatives as f64 / 
                      (metrics.true_negatives + metrics.false_positives) as f64;
    (sensitivity + specificity) / 2.0
}
```
**Use case:** Anomaly detection, medical datasets

#### 1.2 Top-K Accuracy
For multi-class classification:
```rust
pub fn top_k_accuracy(predictions: &[Array1<f64>], targets: &[Array1<f64>], k: usize) -> f64 {
    // Correct if true class is in top k most probable predictions
    // Used in ImageNet (top-5 accuracy)
}
```
**Use case:** ImageNet, classification with many classes (1000+)

#### 1.3 Per-Class Accuracy
```rust
pub fn per_class_accuracy(predictions, targets) -> Vec<f64> {
    // Separate accuracy for each class
    // Identifies problematic classes
}
```
**Use case:** Debugging, per-class performance analysis

---

## 2. Precision, Recall, F1-Score

### Definitions

**Precision:**
```
Precision = TP / (TP + FP)
= "When I predict positive, how often am I right?"
```

**Recall (Sensitivity):**
```
Recall = TP / (TP + FN)
= "What % of all actual positives do I capture?"
```

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
= Harmonic mean of Precision and Recall
```

### Precision vs Recall Trade-off

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| **High (0.9)** | ⬆️ High | ⬇️ Low | Avoid false positives (spam filter) |
| **Medium (0.5)** | ➡️ Balanced | ➡️ Balanced | General |
| **Low (0.1)** | ⬇️ Low | ⬆️ High | Capture all positives (medical diagnosis) |

### Use Cases

| Context | Priority | Reason |
|---------|----------|--------|
| **Spam Filter** | 🔴 Precision | Don't block real emails |
| **Cancer Detection** | 🔴 Recall | Don't miss sick patients |
| **Recommendations** | 🔴 Precision | Show relevant content |
| **Search Engine** | 🟡 F1 (balanced) | Relevance and coverage |

### Current Implementation
```rust
pub fn binary_metrics(predictions, targets, threshold) -> BinaryMetrics {
    // Returns struct with accuracy, precision, recall, f1_score, TP/FP/TN/FN
}
```

### Possible Extensions

#### 2.1 Precision-Recall Curve
```rust
pub fn precision_recall_curve(predictions, targets, num_thresholds) 
    -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // (precision_values, recall_values, thresholds)
    // Visualizes precision/recall trade-off
    // Useful for choosing the right threshold
}
```

#### 2.2 Average Precision (AP)
```rust
pub fn average_precision(predictions, targets) -> f64 {
    // Area under the Precision-Recall curve
    // Standard metric for Object Detection
    // Used in COCO dataset, PASCAL VOC
}
```

#### 2.3 F-Beta Score
```rust
pub fn f_beta_score(precision: f64, recall: f64, beta: f64) -> f64 {
    // F_β = (1 + β²) × (P × R) / (β² × P + R)
    // β = 0.5: Favors Precision
    // β = 1.0: F1 (balanced)
    // β = 2.0: Favors Recall
}
```
**Use case:** Adjust importance of P vs R based on context

#### 2.4 Macro/Micro/Weighted Averages (Multi-Class)
```rust
pub enum AverageMethod {
    Macro,    // Simple average of per-class metrics
    Micro,    // Calculate on global TP/FP/FN
    Weighted, // Weighted average by number of examples
}

pub fn precision_multiclass(predictions, targets, method: AverageMethod) -> f64
```

**Example:**
```
Classes: A (100 ex), B (10 ex)
Precision_A = 0.9, Precision_B = 0.5

Macro:    (0.9 + 0.5) / 2 = 0.70  // Treats classes equally
Weighted: (0.9×100 + 0.5×10) / 110 = 0.86  // Weighted by frequency
```

---

## 3. Confusion Matrix

### Definition

**Binary (2x2):**
```
                Predicted
             Neg    Pos
Actual  Neg [  TN  |  FP  ]
        Pos [  FN  |  TP  ]
```

**Multi-Class (NxN):**
```
matrix[i][j] = number of examples of class i predicted as class j
```

### Interpretation

| Metric | Formula | Meaning |
|--------|---------|---------|
| **True Positive Rate** | TP / (TP + FN) | = Recall = Sensitivity |
| **False Positive Rate** | FP / (FP + TN) | False alarm rate |
| **True Negative Rate** | TN / (TN + FP) | = Specificity |
| **False Negative Rate** | FN / (FN + TP) | Miss rate |

### Current Implementation
```rust
pub fn confusion_matrix_binary(predictions, targets, threshold) -> Array2<usize>
pub fn confusion_matrix_multiclass(predictions, targets, num_classes) -> Array2<usize>
pub fn format_confusion_matrix(matrix, class_names) -> String
```

### Possible Extensions

#### 3.1 Normalized Confusion Matrix
```rust
pub fn confusion_matrix_normalized(predictions, targets, num_classes, 
                                   normalize: NormalizeMethod) -> Array2<f64> {
    enum NormalizeMethod {
        True,   // Normalize by row (sum = 1 per true class)
        Pred,   // Normalize by column (sum = 1 per prediction)
        All,    // Normalize by total (entire matrix sums to 1)
    }
}
```
**Use case:** Visualization, comparison between datasets of different sizes

#### 3.2 Derived Metrics from Matrix
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
    // Measures agreement beyond chance
    // κ = (p_o - p_e) / (1 - p_e)
    // 1.0 = perfect agreement, 0 = random agreement
}
```
**Use case:** Inter-rater reliability, annotation quality

---

## 4. ROC Curve & AUC

### ROC Curve (Receiver Operating Characteristic)

**Definition:**
- Graph: FPR (x-axis) vs TPR (y-axis) at different thresholds
- Shows trade-off between sensitivity and specificity

**Current Implementation:**
```rust
pub fn roc_curve(predictions, targets, num_thresholds) 
    -> (Vec<f64>, Vec<f64>, Vec<f64>)  // (FPR, TPR, thresholds)
```

### AUC (Area Under Curve)

**Definition:**
- Area under the ROC curve
- **1.0** = Perfect predictions (all positives before all negatives)
- **0.5** = Random performance (diagonal line)
- **< 0.5** = Worse than random (inverted model!)

**Interpretation:**
```
AUC = Probability that a random positive example 
      has a higher score than a random negative example
```

**Current Implementation:**
```rust
pub fn auc_roc(predictions, targets) -> f64
```

### Advantages of ROC/AUC

✅ **Threshold independent** - Evaluates global performance
✅ **Resistant to imbalance** - Unlike accuracy
✅ **Industry standard** - Benchmarking, publications

### Limitations

❌ **Very imbalanced datasets** - Prefer Precision-Recall
❌ **Multi-class** - Requires One-vs-Rest or One-vs-One
❌ **Small dataset** - Unstable curve (few points)

### Possible Extensions

#### 4.1 Partial AUC
```rust
pub fn partial_auc(predictions, targets, fpr_range: (f64, f64)) -> f64 {
    // AUC in a specific region of FPR
    // Useful if interested in specific FPR rate
    // Ex: Partial AUC between FPR 0.0-0.1 for high-precision tasks
}
```

#### 4.2 Multi-Class ROC
```rust
pub fn roc_auc_multiclass(predictions, targets, num_classes, 
                          method: MultiClassMethod) -> f64 {
    enum MultiClassMethod {
        OneVsRest,  // N ROC curves (class i vs rest)
        OneVsOne,   // N×(N-1)/2 curves (all pairs)
    }
}
```

#### 4.3 Bootstrap Confidence Intervals
```rust
pub fn auc_confidence_interval(predictions, targets, 
                                num_bootstraps: usize, 
                                confidence: f64) -> (f64, f64, f64) {
    // (lower_bound, auc, upper_bound)
    // Gives uncertainty on AUC
    // Useful for small datasets
}
```

---

## 5. Advanced Metrics (Not Implemented)

### 5.1 Log Loss (Cross-Entropy Loss)
```rust
pub fn log_loss(predictions: &[Array1<f64>], targets: &[Array1<f64>]) -> f64 {
    // Heavily penalizes confident but wrong predictions
    // Standard for competitions (Kaggle)
    // Better than accuracy because it accounts for probabilities
}
```

### 5.2 Brier Score
```rust
pub fn brier_score(predictions: &[Array1<f64>], targets: &[Array1<f64>]) -> f64 {
    // = Mean Squared Error between predicted and actual probabilities
    // More sensitive to extreme predictions than log loss
}
```

### 5.3 Calibration Curve
```rust
pub fn calibration_curve(predictions, targets, num_bins: usize) 
    -> (Vec<f64>, Vec<f64>) {
    // (mean_predicted_probability, fraction_of_positives)
    // Checks if predicted probabilities are calibrated
    // Ex: Among all predictions at 70%, 70% should be positive
}
```

### 5.4 Matthews Correlation Coefficient (MCC)
```rust
pub fn matthews_correlation_coefficient(confusion_matrix: &Array2<usize>) -> f64 {
    // MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    // Range: [-1, 1]
    // Takes into account all 4 values (TP/TN/FP/FN)
    // Better than accuracy for imbalanced datasets
}
```

---

## 6. Specialized Metrics

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
    // Equivalent to F1 for pixels
}
```

### 6.3 Ranking Metrics
```rust
pub fn mean_average_precision(predictions: &[Vec<f64>], targets: &[Vec<usize>]) -> f64 {
    // For recommendation systems
}

pub fn ndcg(predictions: &[f64], relevances: &[f64], k: usize) -> f64 {
    // Normalized Discounted Cumulative Gain
    // For ranking and search engines
}
```

---

## 7. Metric Selection by Domain

### General Machine Learning

| Problem | Primary Metric | Secondary Metrics |
|---------|----------------|-------------------|
| **Balanced Binary Classification** | Accuracy | Precision, Recall, F1 |
| **Imbalanced Binary Classification** | F1, AUPRC | Recall, MCC |
| **Multi-Class Classification** | Accuracy (macro) | Confusion Matrix, per-class F1 |
| **Ranking / Recommendation** | MAP, NDCG | Precision@K, Recall@K |
| **Calibrated Probabilities** | Log Loss, Brier Score | Calibration Curve |

### Specific Domains

| Domain | Metric | Why |
|--------|--------|-----|
| **Fraud Detection** | Recall, AUPRC | Don't miss fraud (costly) |
| **Spam Filter** | Precision | Don't block real emails |
| **Medical Diagnosis** | Sensitivity (Recall) | Don't miss sick patients |
| **Computer Vision** | mAP, IoU | Standard for benchmarks |
| **NLP (Sentiment Analysis)** | Accuracy, F1 (macro) | Classes may be imbalanced |
| **Search Engine** | NDCG, MAP | Result order matters |

---

## 8. Best Practices

### 8.1 Always Report Multiple Metrics
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
**Why:** A single metric can be misleading

### 8.2 Use Cross-Validation
```rust
pub fn cross_validate_metrics(
    model_builder: impl Fn() -> Network,
    dataset: &Dataset,
    k_folds: usize
) -> Vec<EvaluationReport> {
    // Returns metrics for each fold
    // Gives uncertainty on performance
}
```

### 8.3 Stratification for Imbalanced Datasets
```rust
pub fn stratified_split(dataset: &Dataset, test_ratio: f64) 
    -> (Dataset, Dataset) {
    // Maintains class proportions in train/test
    // Critical for imbalanced datasets
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
        MaxF1,          // Maximize F1
        MaxAccuracy,    // Maximize Accuracy
        TargetRecall(f64),  // Achieve minimum recall
        TargetPrecision(f64), // Achieve minimum precision
    }
}
```

---

## 9. Metric Visualization

### 9.1 Plots to Implement
```rust
// Requires integration with plotters or similar
pub fn plot_roc_curve(fpr: &[f64], tpr: &[f64]) -> Result<(), Error>
pub fn plot_precision_recall_curve(precision: &[f64], recall: &[f64]) -> Result<(), Error>
pub fn plot_confusion_matrix_heatmap(matrix: &Array2<usize>) -> Result<(), Error>
pub fn plot_learning_curve(train_metrics: &[f64], val_metrics: &[f64]) -> Result<(), Error>
```

### 9.2 Export for External Tools
```rust
pub fn export_metrics_csv(metrics: &EvaluationReport, path: &str) -> Result<(), Error>
pub fn export_metrics_json(metrics: &EvaluationReport, path: &str) -> Result<(), Error>
```

---

## 10. Metrics Roadmap

### Phase 1: ✅ Completed
- [x] Accuracy (binary + multi-class)
- [x] Precision, Recall, F1
- [x] Confusion Matrix
- [x] ROC Curve & AUC

### Phase 2: Recommended Next
- [ ] Log Loss / Cross-Entropy
- [ ] Precision-Recall Curve & Average Precision
- [ ] Matthews Correlation Coefficient (MCC)
- [ ] Per-Class Metrics (multi-class)

### Phase 3: Advanced
- [ ] Calibration Curve & Brier Score
- [ ] Bootstrap Confidence Intervals
- [ ] Multi-Class ROC (One-vs-Rest, One-vs-One)
- [ ] Threshold Optimization

### Phase 4: Specialized
- [ ] Object Detection (mAP, IoU)
- [ ] Ranking (NDCG, MAP)
- [ ] Regression (MAE, MSE, R²)
- [ ] Time Series (MAPE, SMAPE)

---

## References

1. **Scikit-learn Metrics** - https://scikit-learn.org/stable/modules/model_evaluation.html
2. **ROC Analysis** - Fawcett, T. (2006). "An introduction to ROC analysis"
3. **Precision-Recall vs ROC** - Davis & Goadrich (2006)
4. **Calibration** - Guo et al. (2017). "On Calibration of Modern Neural Networks"
5. **Multi-Class Metrics** - Sokolova & Lapalme (2009). "A systematic analysis of performance measures"
