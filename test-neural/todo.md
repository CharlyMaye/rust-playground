# TODO - Améliorations du Réseau de Neurones

## ✅ Complété

- [x] Implémentation des fonctions d'activation configurables
- [x] Documentation détaillée de toutes les activations
- [x] Enum `Activation` avec 15 fonctions

## 🔄 À Étudier : Loss Functions (Fonctions de Perte)

### Concept de Base

La **loss function** (fonction de perte/coût) mesure **à quel point le réseau se trompe** dans ses prédictions.

**Objectif** : Minimiser l'erreur entre la prédiction et la valeur réelle.

```
Loss = Différence(Prédiction, Valeur_Réelle)
```

Plus la loss est **petite** → meilleure prédiction  
Plus la loss est **grande** → pire prédiction

---

### Actuellement dans le Code

```rust
let output_errors = target - &final_output;  // Erreur brute
let output_delta = &output_errors * &self.output_activation.derivative(&final_output);
```

**Loss actuelle :** MSE (Mean Squared Error) implicite

$$\text{MSE} = \frac{1}{n}\sum(y_{réel} - y_{prédit})^2$$

---

### Principales Loss Functions

#### 1. MSE (Mean Squared Error)
**Formule :** $\text{MSE} = \frac{1}{n}\sum(y - \hat{y})^2$

**Usage :** Régression (prédire des valeurs continues)

**Avantages :**
- ✅ Pénalise fortement les grandes erreurs
- ✅ Différentiable partout

**Inconvénients :**
- ❌ Pas optimal pour classification
- ❌ Gradient qui disparaît avec Sigmoid

**Exemple :**
```
Prédiction: 2.5, Réel: 3.0
Loss = (3.0 - 2.5)² = 0.25
```

---

#### 2. MAE (Mean Absolute Error)
**Formule :** $\text{MAE} = \frac{1}{n}\sum|y - \hat{y}|$

**Usage :** Régression (moins sensible aux outliers)

**Avantages :**
- ✅ Robuste aux outliers
- ✅ Interprétation intuitive

**Exemple :**
```
Prédiction: 2.5, Réel: 3.0
Loss = |3.0 - 2.5| = 0.5
```

---

#### 3. Binary Cross-Entropy (Log Loss)
**Formule :** $\text{BCE} = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$

**Usage :** Classification binaire (avec Sigmoid)

**Avantages :**
- ✅ Interprétation probabiliste
- ✅ Gradient plus stable que MSE pour classification
- ✅ Convergence plus rapide

**Inconvénients :**
- ❌ Nécessite prédictions dans [0, 1]
- ❌ Instable si prédiction = 0 ou 1 (log(0))

**Exemple :**
```
Prédiction: 0.9, Réel: 1 (chien)
Loss = -[1×log(0.9) + 0×log(0.1)] = 0.105

Prédiction: 0.1, Réel: 1 (chien)
Loss = -[1×log(0.1) + 0×log(0.9)] = 2.303  // Grosse erreur!
```

**Implémentation Rust :**
```rust
fn binary_cross_entropy(prediction: f64, target: f64) -> f64 {
    let epsilon = 1e-15; // Éviter log(0)
    let p = prediction.max(epsilon).min(1.0 - epsilon);
    -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
}
```

---

#### 4. Categorical Cross-Entropy
**Formule :** $\text{CCE} = -\sum y_i \log(\hat{y}_i)$

**Usage :** Classification multi-classes (avec Softmax)

**Avantages :**
- ✅ Standard pour multi-classes
- ✅ Interprétation probabiliste claire

**Exemple :**
```
Classes: [Chat, Chien, Oiseau]
Réel:    [1,    0,     0]      // C'est un chat
Prédit:  [0.7,  0.2,   0.1]
Loss = -(1×log(0.7) + 0×log(0.2) + 0×log(0.1)) = 0.357
```

**Implémentation Rust :**
```rust
fn categorical_cross_entropy(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let epsilon = 1e-15;
    -targets.iter()
        .zip(predictions.iter())
        .map(|(t, p)| t * (p.max(epsilon)).ln())
        .sum::<f64>()
}
```

---

#### 5. Huber Loss
**Formule :** 
$$\text{Huber} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{si } |y - \hat{y}| \leq \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{sinon} \end{cases}$$

**Usage :** Régression robuste aux outliers

**Avantages :**
- ✅ Combine MSE (petites erreurs) et MAE (grandes erreurs)
- ✅ Moins sensible aux outliers que MSE

---

### Relation avec le Training

**Cycle d'apprentissage :**

```
1. Forward pass → Prédiction
2. Calcul de la Loss → Mesurer l'erreur
3. Backpropagation → Calculer les gradients
4. Update des poids → Réduire la Loss
```

---

### Guide de Sélection des Loss Functions

| **Tâche** | **Activation Sortie** | **Loss Function Recommandée** |
|-----------|----------------------|-------------------------------|
| Régression | Linear | **MSE** (défaut) |
| Régression robuste | Linear | **MAE** ou **Huber** |
| Classification binaire | Sigmoid | **Binary Cross-Entropy** |
| Classification multi-classes | Softmax | **Categorical Cross-Entropy** |
| Détection d'objets | Variable | IoU Loss, Focal Loss |
| Segmentation | Softmax | Dice Loss, Focal Loss |

---

### Pourquoi Changer de Loss pour XOR ?

**Problème actuel :** MSE + Sigmoid pour classification binaire

**MSE pour classification :**
- ❌ Gradient qui disparaît quand proche de 0 ou 1
- ❌ Pas d'interprétation probabiliste
- ❌ Convergence plus lente

**Binary Cross-Entropy pour classification :**
- ✅ Gradient plus stable
- ✅ Converge plus vite
- ✅ Interprétation comme probabilité

---

### Visualisation de la Convergence

```
Haute Loss ━━━━━━━━━┓
                    ┃    Début
                    ┃      ↓
                    ┃      •
                    ┃     ╱
                    ┃    ╱
                    ┃   ╱     Training
                    ┃  ╱      ↓
                    ┃ ╱
Basse Loss ━━━━━━━━━┃╱________• Convergence
                    └────────────────→
                         Epochs
```

**Objectif du training** : Descendre cette courbe le plus vite possible.

---

## 📝 Prochaines Étapes

### À Implémenter

- [ ] Ajouter un enum `LossFunction` similaire à `Activation`
- [ ] Implémenter Binary Cross-Entropy
- [ ] Implémenter Categorical Cross-Entropy
- [ ] Implémenter MAE
- [ ] Implémenter Huber Loss
- [ ] Permettre de choisir la loss dans `Network::new()`
- [ ] Modifier `train()` pour utiliser la loss choisie
- [ ] Ajouter une méthode `evaluate()` pour calculer la loss sans update

### Structure Proposée

```rust
pub enum LossFunction {
    MSE,
    MAE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    Huber,
}

impl LossFunction {
    pub fn compute(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
        // Calcul de la loss
    }
    
    pub fn derivative(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Array1<f64> {
        // Gradient pour backprop
    }
}

pub struct Network {
    // ... existing fields
    loss_function: LossFunction,
}
```

### Tests à Effectuer

- [ ] Comparer MSE vs BCE sur XOR
- [ ] Mesurer la vitesse de convergence
- [ ] Tester sur problèmes multi-classes
- [ ] Valider les gradients numériquement

---

## 📚 Ressources

- **Cross-Entropy :** [Understanding Cross-Entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)
- **Loss Functions :** [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- **Comparaison :** [When to use which loss?](https://machinelearningmastery.com/loss-functions-for-neural-networks/)

---

## 🔗 Liens Internes

- Voir [readme.md](readme.md) pour la documentation des activations
- Voir [src/network.rs](src/network.rs) pour l'implémentation actuelle
