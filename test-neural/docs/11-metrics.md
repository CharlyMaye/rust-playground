# 11 — Metrics & Evaluation

> **Navigation** ← [10 — Training Loop](10-training-loop.md) | [README →](README.md)

---

## Level 1 — Concepts

### Why metrics beyond loss?

The training loss is an optimization target; it may not directly measure what you care about. For example:
- A model might achieve low cross-entropy loss but confuse the two most important classes.
- Accuracy can be misleading when classes are imbalanced (99% accuracy on a dataset that is 99% class A is trivial).

Evaluation metrics provide **task-specific measures** of model quality.

### The confusion matrix

For binary classification, every prediction falls into one of four categories:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

From these four numbers, all other metrics are derived.

### The key metrics

| Metric | "I want..." | Formula |
|--------|-------------|---------|
| **Accuracy** | Overall correctness | $(TP + TN) / \text{total}$ |
| **Precision** | When I predict positive, am I right? | $TP / (TP + FP)$ |
| **Recall** | Do I catch all positives? | $TP / (TP + FN)$ |
| **F1 Score** | Balance precision and recall | $2 \cdot P \cdot R / (P + R)$ |

### ROC and AUC

The **ROC curve** (Receiver Operating Characteristic) shows, across all possible classification thresholds, the trade-off between True Positive Rate (Recall) and False Positive Rate. A perfect model has AUC = 1.0; a random model has AUC = 0.5.

AUC is threshold-independent — it measures the quality of the **ranking** produced by the model, not the quality of any specific threshold. It is the standard metric when comparing classifiers on imbalanced datasets.

---

## Level 2 — Mathematics

### Binary classification threshold

For a model producing scalar output $\hat{p} \in (0, 1)$ (Sigmoid), classification is:

$$\hat{y} = \begin{cases} 1 & \hat{p} \geq \theta \\ 0 & \hat{p} < \theta \end{cases}$$

with default threshold $\theta = 0.5$. All binary metrics below are computed at this threshold.

For multi-class (Softmax output), the predicted class is $\hat{k} = \arg\max_k \hat{y}_k$.

---

### Accuracy

$$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]$$

**Binary case**: $= (TP + TN) / N$.

**Pitfall**: on an imbalanced dataset with 95% negative samples, a trivial classifier that always predicts negative achieves 95% accuracy with zero useful signal.

---

### Precision and Recall

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN} \quad (\text{also called sensitivity or True Positive Rate})$$

**Precision** answers: of all samples I labeled positive, how many are truly positive? High precision means few false alarms.

**Recall** answers: of all truly positive samples, how many did I label positive? High recall means few misses.

**Trade-off**: increasing the threshold $\theta$ increases precision and decreases recall. Decreasing $\theta$ does the opposite. The F1 score summarizes both.

---

### F1 Score

The F1 score is the **harmonic mean** of precision and recall:

$$F_1 = 2 \cdot \frac{P \cdot R}{P + R} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

The harmonic mean is lower than the arithmetic mean and is dominated by the smaller of the two values — it punishes large imbalances between precision and recall.

**Generalization — $F_\beta$ score**: when recall is $\beta$ times more important than precision:

$$F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}$$

- $\beta = 1$: equal weight → $F_1$
- $\beta = 2$: recall twice as important → $F_2$ (used in medical diagnostics where missing a positive is costly)
- $\beta = 0.5$: precision twice as important → $F_{0.5}$

---

### Confusion Matrix (Multi-class)

For a $K$-class problem, the confusion matrix $C \in \mathbb{Z}_{\geq 0}^{K \times K}$ is defined:

$$C_{kj} = \text{number of samples with true label } k \text{ predicted as label } j$$

The diagonal $C_{kk}$ counts correct predictions for class $k$. Off-diagonal entries are errors. The matrix visualizes which classes are most confused with each other — invaluable for understanding model failure modes.

Accuracy from the confusion matrix: $\text{Acc} = \text{trace}(C) / \sum_{kj} C_{kj}$.

---

### ROC Curve

The ROC curve plots the **True Positive Rate (Recall)** on the y-axis against the **False Positive Rate (FPR)** on the x-axis, as the classification threshold $\theta$ varies from 1 to 0:

$$\text{TPR}(\theta) = \frac{TP(\theta)}{TP(\theta) + FN(\theta)}, \quad \text{FPR}(\theta) = \frac{FP(\theta)}{FP(\theta) + TN(\theta)}$$

The library computes this curve at $N = 100$ uniformly-spaced thresholds in $[0, 1]$, returning vectors `(fpr, tpr, thresholds)`.

**Curve interpretation**:
- $(0, 0)$: threshold = 1, nothing predicted positive.
- $(1, 1)$: threshold = 0, everything predicted positive.
- The point $(0, 1)$ would be a perfect classifier.
- The diagonal from $(0,0)$ to $(1,1)$ represents a random classifier.

---

### AUC — Area Under the ROC Curve

$$\text{AUC} = \int_0^1 \text{TPR}(t) \, d(\text{FPR}(t))$$

The library computes this numerically using the **trapezoidal rule**:

$$\text{AUC} \approx \sum_{i=1}^{N-1} \frac{(\text{TPR}_i + \text{TPR}_{i+1})}{2} \cdot |\text{FPR}_{i+1} - \text{FPR}_i|$$

**Statistical interpretation (Wilcoxon-Mann-Whitney statistic)**:

$$\text{AUC} = P(\hat{p}_{\text{pos}} > \hat{p}_{\text{neg}})$$

where $\hat{p}_{\text{pos}}$ is the model's score for a randomly drawn positive sample and $\hat{p}_{\text{neg}}$ for a randomly drawn negative sample. AUC is the probability that the model ranks a positive sample above a negative one — a pure measure of **ranking quality**.

**AUC is threshold-independent**: it does not require choosing a threshold $\theta$. This makes it suitable for:
- Comparing models without committing to a threshold.
- Class-imbalanced problems where accuracy is misleading.

**Reference**: Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters*, 27(8), 861–874.

---

### Implementation notes

All metric functions in `metrics.rs` operate on raw `f32` slices and return either `Float` or composite structs. The `BinaryMetrics` struct collects all binary metrics in one pass:

```
accuracy, precision, recall, f1
true_positives, false_positives, true_negatives, false_negatives
```

The `format_confusion_matrix()` function produces a human-readable ASCII table for debugging multi-class models. The `roc_curve()` function returns three parallel vectors that can be plotted directly.

---

### Summary

| Function | Input | Output |
|----------|-------|--------|
| `accuracy(preds, targets)` | Probability vectors | `Float` |
| `binary_metrics(preds, targets)` | Sigmoid outputs, binary targets | `BinaryMetrics` struct |
| `confusion_matrix_binary(preds, targets)` | Same | `Array2<usize>` (2×2) |
| `confusion_matrix_multiclass(preds, targets, K)` | Softmax outputs, class indices | `Array2<usize>` (K×K) |
| `format_confusion_matrix(matrix)` | Confusion matrix | Pretty-printed `String` |
| `roc_curve(scores, labels, n_thresholds)` | Scalar scores, binary labels | `(Vec<Float>, Vec<Float>, Vec<Float>)` |
| `auc_roc(scores, labels)` | Same | `Float` |
