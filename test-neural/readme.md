# Réseau de Neurones en Rust

## Concepts Clés

### 1. Architecture (Couches/Neurones)

La **structure** de ton réseau : nombre de couches et nombre de neurones par couche.

- Dans ton code : `Network::new(2, 3, 1)` = 2 entrées → 3 neurones cachés → 1 sortie
- Plus de neurones/couches = plus de capacité d'apprentissage, mais risque de **surapprentissage**

### 2. Fonctions d'Activation (sigmoid, ReLU, tanh...)

Fonction qui **transforme** la sortie d'un neurone.

**Actuellement utilisée : Sigmoid**
- Formule : `1 / (1 + e^-x)`
- Sortie : entre `[0, 1]`

**Alternatives :**
- **ReLU** : `max(0, x)` → plus rapide, standard moderne
- **tanh** : `tanh(x)` → sortie entre `[-1, 1]`
- **Leaky ReLU**, **ELU**, etc.

---

## Fonctions d'Activation Détaillées

### Sigmoid (Logistic)
**Formule :** $f(x) = \frac{1}{1 + e^{-x}}$

**Dérivée :** $f'(x) = f(x) \cdot (1 - f(x))$

**Propriétés :**
- Sortie : `[0, 1]`
- Lisse et différentiable partout
- Interprétable comme une probabilité

**Avantages :**
- ✅ Sortie normalisée (bonne pour la couche de sortie en classification binaire)
- ✅ Gradient bien défini

**Inconvénients :**
- ❌ **Problème du gradient qui disparaît** (vanishing gradient) pour grandes/petites valeurs
- ❌ Sortie non centrée sur zéro
- ❌ Coûteux en calcul (`exp()`)

**Implémentation Rust :**
```rust
fn sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp()))
}

fn sigmoid_derivative(x: &Array1<f64>) -> Array1<f64> {
    x * &(1.0 - x)
}
```

---

### ReLU (Rectified Linear Unit)
**Formule :** $f(x) = \max(0, x)$

**Dérivée :** $f'(x) = \begin{cases} 1 & \text{si } x > 0 \\ 0 & \text{si } x \leq 0 \end{cases}$

**Propriétés :**
- Sortie : `[0, +∞)`
- Linéaire pour valeurs positives, zéro sinon
- **Standard moderne pour les couches cachées**

**Avantages :**
- ✅ **Très rapide** (simple comparaison et multiplication)
- ✅ Pas de gradient qui disparaît pour valeurs positives
- ✅ Favorise la sparsité (certains neurones s'éteignent)
- ✅ Convergence plus rapide que sigmoid/tanh

**Inconvénients :**
- ❌ **Problème des neurones morts** : si gradient = 0, le neurone ne s'active plus jamais
- ❌ Sortie non centrée sur zéro
- ❌ Non différentiable en x = 0 (en pratique, on prend 0 ou 1)

**Implémentation Rust :**
```rust
fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x.max(0.0))
}

fn relu_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
}
```

---

### Leaky ReLU
**Formule :** $f(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha x & \text{si } x \leq 0 \end{cases}$ (typiquement $\alpha = 0.01$)

**Dérivée :** $f'(x) = \begin{cases} 1 & \text{si } x > 0 \\ \alpha & \text{si } x \leq 0 \end{cases}$

**Propriétés :**
- Sortie : `(-∞, +∞)`
- Petite pente pour valeurs négatives

**Avantages :**
- ✅ Résout le problème des neurones morts de ReLU
- ✅ Rapide comme ReLU
- ✅ Garde un gradient pour valeurs négatives

**Inconvénients :**
- ❌ Résultats incohérents selon les tâches
- ❌ Nécessite un hyperparamètre (alpha)

**Implémentation Rust :**
```rust
fn leaky_relu(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { x } else { alpha * x })
}

fn leaky_relu_derivative(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { 1.0 } else { alpha })
}
```

---

### Tanh (Tangente Hyperbolique)
**Formule :** $f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

**Dérivée :** $f'(x) = 1 - f(x)^2$

**Propriétés :**
- Sortie : `[-1, 1]`
- **Centrée sur zéro** (contrairement à sigmoid)
- Version "étendue" de sigmoid

**Avantages :**
- ✅ Sortie centrée → convergence plus rapide que sigmoid
- ✅ Gradient plus fort que sigmoid
- ✅ Bon pour les couches cachées

**Inconvénients :**
- ❌ Problème du gradient qui disparaît (moins que sigmoid)
- ❌ Coûteux en calcul

**Implémentation Rust :**
```rust
fn tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x.tanh())
}

fn tanh_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| 1.0 - x.powi(2))
}
```

---

### ELU (Exponential Linear Unit)
**Formule :** $f(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha(e^x - 1) & \text{si } x \leq 0 \end{cases}$ (typiquement $\alpha = 1.0$)

**Dérivée :** $f'(x) = \begin{cases} 1 & \text{si } x > 0 \\ f(x) + \alpha & \text{si } x \leq 0 \end{cases}$

**Propriétés :**
- Sortie : `(-α, +∞)`
- Lisse partout

**Avantages :**
- ✅ Moyenne des activations proche de zéro
- ✅ Pas de neurones morts
- ✅ Gradient non-nul partout

**Inconvénients :**
- ❌ Coûteux (`exp()`)
- ❌ Légèrement plus lent que ReLU

**Implémentation Rust :**
```rust
fn elu(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
}

fn elu_derivative(x: &Array1<f64>, alpha: f64) -> Array1<f64> {
    x.mapv(|x| if x > 0.0 { 1.0 } else { alpha * x.exp() })
}
```

---

### Softmax (pour classification multi-classes)
**Formule :** $f(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$

**Propriétés :**
- Sortie : `[0, 1]` pour chaque neurone, somme = 1
- Convertit logits en probabilités
- **Uniquement pour la couche de sortie**

**Avantages :**
- ✅ Interprétation probabiliste claire
- ✅ Standard pour classification multi-classes

**Implémentation Rust :**
```rust
fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|x| (x - max).exp());
    let sum = exp_x.sum();
    exp_x / sum
}
```

---

## Fonctions d'Activation Avancées

### PReLU (Parametric ReLU)
**Formule :** $f(x) = \begin{cases} x & \text{si } x > 0 \\ \alpha x & \text{si } x \leq 0 \end{cases}$ où $\alpha$ est **appris** pendant l'entraînement

**Dérivée :** $f'(x) = \begin{cases} 1 & \text{si } x > 0 \\ \alpha & \text{si } x \leq 0 \end{cases}$

**Avantages :**
- ✅ Alpha adaptatif par neurone
- ✅ Plus flexible que Leaky ReLU

**Inconvénients :**
- ❌ Plus de paramètres à entraîner
- ❌ Risque de surapprentissage

**Implémentation Rust :**
```rust
fn prelu(x: &Array1<f64>, alpha: &Array1<f64>) -> Array1<f64> {
    x.iter().zip(alpha.iter())
        .map(|(&x, &a)| if x > 0.0 { x } else { a * x })
        .collect()
}
```

---

### GELU (Gaussian Error Linear Unit)
**Formule :** $f(x) = x \cdot \Phi(x)$ où $\Phi$ est la fonction de distribution cumulative gaussienne

**Approximation :** $f(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$

**Propriétés :**
- Lisse et non-monotone
- **Utilisé dans BERT, GPT**

**Avantages :**
- ✅ Performance SOTA sur transformers
- ✅ Lisse partout
- ✅ Probabilistiquement motivé

**Inconvénients :**
- ❌ Coûteux en calcul

**Implémentation Rust :**
```rust
fn gelu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| {
        0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() 
            * (x + 0.044715 * x.powi(3))).tanh())
    })
}
```

---

### Swish / SiLU (Sigmoid Linear Unit)
**Formule :** $f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$

**Dérivée :** $f'(x) = f(x) + \sigma(x)(1 - f(x))$

**Propriétés :**
- Lisse, non-monotone
- **Découvert par Google via recherche automatique**

**Avantages :**
- ✅ Meilleure performance que ReLU sur certaines tâches
- ✅ Lisse partout

**Inconvénients :**
- ❌ Plus coûteux que ReLU

**Implémentation Rust :**
```rust
fn swish(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x / (1.0 + (-x).exp()))
}

fn swish_derivative(x: &Array1<f64>) -> Array1<f64> {
    let sigmoid = x.mapv(|x| 1.0 / (1.0 + (-x).exp()));
    let swish = x * &sigmoid;
    &swish + &sigmoid * &(1.0 - &swish)
}
```

---

### Mish
**Formule :** $f(x) = x \cdot \tanh(\ln(1 + e^x)) = x \cdot \tanh(\text{softplus}(x))$

**Propriétés :**
- Lisse, non-monotone
- **Alternatives récente à Swish**

**Avantages :**
- ✅ Meilleure régularisation que ReLU/Swish
- ✅ Gradient non-nul pour valeurs négatives

**Inconvénients :**
- ❌ Très coûteux en calcul

**Implémentation Rust :**
```rust
fn mish(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x * ((1.0 + x.exp()).ln()).tanh())
}
```

---

### SELU (Scaled ELU)
**Formule :** $f(x) = \lambda \begin{cases} x & \text{si } x > 0 \\ \alpha(e^x - 1) & \text{si } x \leq 0 \end{cases}$

**Constantes :** $\lambda \approx 1.0507$, $\alpha \approx 1.6733$

**Propriétés :**
- Auto-normalisant (préserve moyenne=0, variance=1)
- **Conçu pour FeedForward Networks**

**Avantages :**
- ✅ Pas besoin de Batch Normalization
- ✅ Convergence plus rapide

**Inconvénients :**
- ❌ Sensible à l'initialisation (utiliser LeCun)
- ❌ Fonctionne mal avec Dropout

**Implémentation Rust :**
```rust
fn selu(x: &Array1<f64>) -> Array1<f64> {
    let lambda = 1.0507;
    let alpha = 1.6733;
    x.mapv(|x| {
        lambda * if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
    })
}
```

---

### Softplus
**Formule :** $f(x) = \ln(1 + e^x)$

**Dérivée :** $f'(x) = \frac{1}{1 + e^{-x}} = \sigma(x)$ (sigmoid!)

**Propriétés :**
- Version lisse de ReLU
- Toujours positif

**Avantages :**
- ✅ Différentiable partout
- ✅ Pas de neurones morts

**Inconvénients :**
- ❌ Coûteux (`exp`, `log`)
- ❌ Gradient qui disparaît pour grandes valeurs négatives

**Implémentation Rust :**
```rust
fn softplus(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| (1.0 + x.exp()).ln())
}

fn softplus_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| 1.0 / (1.0 + (-x).exp())) // sigmoid
}
```

---

### Softsign
**Formule :** $f(x) = \frac{x}{1 + |x|}$

**Dérivée :** $f'(x) = \frac{1}{(1 + |x|)^2}$

**Propriétés :**
- Sortie : `(-1, 1)`
- Alternative à tanh

**Avantages :**
- ✅ Plus rapide que tanh (pas d'exponentielle)
- ✅ Gradient décroît plus lentement

**Inconvénients :**
- ❌ Rarement utilisé en pratique

**Implémentation Rust :**
```rust
fn softsign(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x / (1.0 + x.abs()))
}

fn softsign_derivative(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| 1.0 / (1.0 + x.abs()).powi(2))
}
```

---

### Hard Sigmoid
**Formule :** $f(x) = \max(0, \min(1, 0.2x + 0.5))$

**Propriétés :**
- Approximation linéaire par morceaux de sigmoid
- Très rapide

**Avantages :**
- ✅ Calcul extrêmement rapide (pas d'exponentielle)
- ✅ Utile pour les appareils embarqués

**Implémentation Rust :**
```rust
fn hard_sigmoid(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| (0.2 * x + 0.5).max(0.0).min(1.0))
}
```

---

### Hard Tanh
**Formule :** $f(x) = \max(-1, \min(1, x))$

**Propriétés :**
- Approximation linéaire par morceaux de tanh
- Sortie : `[-1, 1]`

**Avantages :**
- ✅ Très rapide

**Implémentation Rust :**
```rust
fn hard_tanh(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|x| x.max(-1.0).min(1.0))
}
```

---

## Tableau Comparatif Complet

| **Fonction** | **Plage** | **Vitesse** | **Usage Principal** | **Depuis** |
|--------------|-----------|-------------|---------------------|------------|
| Sigmoid | [0, 1] | Lent | Sortie binaire | Classique |
| Tanh | [-1, 1] | Lent | Couches cachées | Classique |
| ReLU | [0, ∞) | **Très rapide** | Couches cachées (défaut) | 2010 |
| Leaky ReLU | (-∞, ∞) | **Très rapide** | Fix neurones morts | 2013 |
| PReLU | (-∞, ∞) | Rapide | Amélioration LeakyReLU | 2015 |
| ELU | (-α, ∞) | Moyen | Réseaux profonds | 2015 |
| SELU | (-λα, ∞) | Moyen | FeedForward (sans BN) | 2017 |
| Swish/SiLU | (-∞, ∞) | Moyen | Alternative ReLU | 2017 |
| GELU | (-∞, ∞) | Lent | **Transformers (GPT, BERT)** | 2016 |
| Mish | (-∞, ∞) | Lent | Vision profonde | 2019 |
| Softmax | [0, 1] (somme=1) | Moyen | Sortie multi-classe | Classique |
| Softplus | (0, ∞) | Lent | ReLU lisse | Classique |
| Hard Sigmoid | [0, 1] | **Très rapide** | Embarqué | Mobile |
| Hard Tanh | [-1, 1] | **Très rapide** | Embarqué | Mobile |

---

## Guide de Sélection

### Par Cas d'Usage

| **Cas d'Usage** | **Fonction Recommandée** | **Raison** |
|-----------------|--------------------------|------------|
| **Couches cachées (défaut 2024)** | **ReLU** | Rapide, efficace, standard industriel |
| Couches cachées (si neurones morts) | **Leaky ReLU** ou **ELU** | Gradient toujours actif |
| Couches cachées (réseaux profonds) | **SELU** ou **ELU** | Auto-normalisation, évite gradient qui disparaît |
| Couches cachées (recherche de performance) | **Swish** ou **Mish** | Performance SOTA sur certaines tâches |
| **Transformers / NLP (GPT, BERT)** | **GELU** | Standard pour attention mechanisms |
| **Vision par ordinateur (CNN)** | **ReLU** ou **Mish** | Rapide pour CNN, Mish pour profonds |
| Réseaux récurrents (RNN, LSTM) | **Tanh** | Standard historique pour gates |
| **Sortie classification binaire** | **Sigmoid** | Sortie [0,1] = probabilité |
| **Sortie classification multi-classes** | **Softmax** | Distribution de probabilités (somme=1) |
| **Sortie régression** | **Linéaire** (aucune) | Valeurs continues illimitées |
| Sortie régression (valeurs positives) | **Softplus** ou **ReLU** | Force sortie ≥ 0 |
| **Appareils embarqués / Mobile** | **Hard Sigmoid** / **Hard Tanh** | Pas d'exponentielle, ultra-rapide |
| Recherche / Expérimentation | **PReLU** | Alpha adaptatif par neurone |

### Par Priorité

#### 🏆 **Si tu veux la meilleure performance (sans contrainte)** :
1. **Couches cachées** : GELU, Swish, Mish
2. **Sortie** : Softmax (multi-classe), Sigmoid (binaire)

#### ⚡ **Si tu veux la rapidité (contrainte temps réel)** :
1. **Couches cachées** : ReLU, Leaky ReLU
2. **Embarqué** : Hard Sigmoid, Hard Tanh

#### 🎯 **Si tu veux la stabilité (réseaux très profonds)** :
1. **Couches cachées** : SELU (avec initialisation LeCun), ELU
2. **Éviter** : Sigmoid, Tanh (gradient qui disparaît)

#### 🔧 **Si tu débutes / prototype rapide** :
1. **Défaut recommandé** : ReLU partout sauf sortie
2. **Sortie** : Sigmoid (binaire), Softmax (multi-classe)

### Par Type de Réseau

| **Architecture** | **Couches Cachées** | **Sortie** |
|------------------|---------------------|------------|
| **Feedforward simple** | ReLU | Sigmoid / Softmax |
| **Feedforward profond** | SELU, ELU | Sigmoid / Softmax |
| **CNN (Computer Vision)** | ReLU, Mish | Softmax |
| **RNN / LSTM** | Tanh | Sigmoid / Softmax |
| **Transformer** | GELU | Softmax |
| **GAN (Générateur)** | ReLU, Leaky ReLU | Tanh |
| **GAN (Discriminateur)** | Leaky ReLU | Sigmoid |
| **Autoencoder** | ReLU | Sigmoid (binaire), Linéaire (continu) |
| **Reinforcement Learning** | ReLU, ELU | Linéaire, Softmax |

### Arbre de Décision

```
Quelle est ta couche ?
├─ Couche de SORTIE
│  ├─ Classification binaire ? → Sigmoid
│  ├─ Classification multi-classes ? → Softmax
│  ├─ Régression (valeurs continues) ? → Linéaire (aucune activation)
│  └─ Régression (valeurs positives) ? → Softplus / ReLU
│
└─ Couche CACHÉE
   ├─ Contrainte de VITESSE ?
   │  ├─ Ultra-rapide (embarqué) ? → Hard Sigmoid / Hard Tanh
   │  └─ Rapide → ReLU, Leaky ReLU
   │
   ├─ Type de RÉSEAU ?
   │  ├─ Transformer / NLP ? → GELU
   │  ├─ CNN profond ? → Mish
   │  ├─ RNN / LSTM ? → Tanh
   │  └─ Feedforward ? → Voir ci-dessous
   │
   ├─ Profondeur du RÉSEAU ?
   │  ├─ Peu de couches (< 5) ? → ReLU
   │  ├─ Profond (> 10 couches) ? → SELU, ELU
   │  └─ Très profond (> 50) ? → SELU avec LeCun init
   │
   ├─ Problème de NEURONES MORTS (gradient = 0) ?
   │  ├─ Oui → Leaky ReLU, PReLU, ELU
   │  └─ Non → ReLU
   │
   └─ Recherche de PERFORMANCE maximale ?
      ├─ Oui (GPU puissant) → Swish, Mish, GELU
      └─ Non → ReLU (défaut)
```

### Recommandations par Année

| **Époque** | **Standard** | **Contexte** |
|------------|--------------|--------------|
| 1990-2010 | Sigmoid, Tanh | Réseaux peu profonds |
| 2010-2015 | ReLU | Révolution deep learning |
| 2015-2017 | Leaky ReLU, ELU, PReLU | Amélioration ReLU |
| 2017-2019 | Swish, SELU | Auto-recherche Google |
| 2019-2024 | **GELU** (transformers), **Mish** (vision) | SOTA actuel |
| 2024+ | **GELU** (défaut NLP), **ReLU** (défaut vision) | Standard industriel |

### Combinaisons Éprouvées

**Classification d'images (CNN) :**
```rust
// Couches conv : ReLU ou Mish
// Couches fully-connected : ReLU
// Sortie : Softmax
```

**Modèle de langage (Transformer) :**
```rust
// Attention + FFN : GELU
// Sortie : Softmax
```

**Réseau profond (> 20 couches) :**
```rust
// Toutes couches cachées : SELU
// Initialisation : LeCun normal
// PAS de Batch Normalization
// Sortie : Sigmoid / Softmax
```

**Prototype rapide :**
```rust
// Couches cachées : ReLU
// Sortie : Sigmoid (binaire) ou Softmax (multi-classe)
```

### 3. Learning Rate (Taux d'apprentissage)

**Vitesse d'apprentissage** : à quel point modifier les poids à chaque étape.

- Actuellement : `0.1`
- **Trop petit** → apprentissage lent
- **Trop grand** → instabilité, ne converge pas
- **Typique** : `0.001` à `0.1`

---

## Fonctions de Perte (Loss Functions)

### Concept de Base

La **loss function** (fonction de perte/coût) mesure **à quel point le réseau se trompe** dans ses prédictions.

**Objectif :** Minimiser l'erreur entre la prédiction et la valeur réelle.

```
Loss = Différence(Prédiction, Valeur_Réelle)
```

Plus la loss est **petite** → meilleure prédiction  
Plus la loss est **grande** → pire prédiction

### Cycle d'apprentissage

```
1. Forward pass → Prédiction
2. Calcul de la Loss → Mesurer l'erreur
3. Backpropagation → Calculer les gradients
4. Update des poids → Réduire la Loss
```

---

### 1. MSE (Mean Squared Error)

**Formule :** $\text{MSE} = \frac{1}{n}\sum(y - \hat{y})^2$

**Usage :** Régression (prédire des valeurs continues)

**Avantages :**
- ✅ Pénalise fortement les grandes erreurs
- ✅ Différentiable partout
- ✅ Interprétation intuitive

**Inconvénients :**
- ❌ Pas optimal pour classification
- ❌ Gradient qui disparaît avec Sigmoid

**Exemple :**
```
Prédiction: 2.5, Réel: 3.0
Loss = (3.0 - 2.5)² = 0.25
```

**Implémentation Rust :**
```rust
fn mse(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let diff = predictions - targets;
    (&diff * &diff).sum() / predictions.len() as f64
}
```

---

### 2. MAE (Mean Absolute Error)

**Formule :** $\text{MAE} = \frac{1}{n}\sum|y - \hat{y}|$

**Usage :** Régression (moins sensible aux outliers)

**Avantages :**
- ✅ Robuste aux outliers
- ✅ Interprétation intuitive
- ✅ Toutes les erreurs traitées linéairement

**Inconvénients :**
- ❌ Gradients constants (convergence plus lente)
- ❌ Non différentiable en zéro

**Exemple :**
```
Prédiction: 2.5, Réel: 3.0
Loss = |3.0 - 2.5| = 0.5
```

**Implémentation Rust :**
```rust
fn mae(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    (predictions - targets).mapv(|x| x.abs()).sum() / predictions.len() as f64
}
```

---

### 3. Binary Cross-Entropy (Log Loss)

**Formule :** $\text{BCE} = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$

**Usage :** Classification binaire (avec Sigmoid)

**Avantages :**
- ✅ Interprétation probabiliste
- ✅ Gradient plus stable que MSE pour classification
- ✅ Convergence plus rapide
- ✅ Standard pour classification binaire

**Inconvénients :**
- ❌ Nécessite prédictions dans [0, 1]
- ❌ Instable si prédiction = 0 ou 1 (log(0))

**Exemple :**
```
Prédiction: 0.9, Réel: 1 (classe positive)
Loss = -[1×log(0.9) + 0×log(0.1)] = 0.105  // Bonne prédiction

Prédiction: 0.1, Réel: 1 (classe positive)
Loss = -[1×log(0.1) + 0×log(0.9)] = 2.303  // Grosse erreur!
```

**Implémentation Rust :**
```rust
fn binary_cross_entropy(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let epsilon = 1e-15; // Éviter log(0)
    let mut sum = 0.0;
    for (p, t) in predictions.iter().zip(targets.iter()) {
        let p_clamped = p.max(epsilon).min(1.0 - epsilon);
        sum += -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln());
    }
    sum / predictions.len() as f64
}
```

---

### 4. Categorical Cross-Entropy

**Formule :** $\text{CCE} = -\sum y_i \log(\hat{y}_i)$

**Usage :** Classification multi-classes (avec Softmax)

**Avantages :**
- ✅ Standard pour multi-classes
- ✅ Interprétation probabiliste claire
- ✅ Gradient bien adapté avec Softmax

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

### 5. Huber Loss

**Formule :** 
$$\text{Huber} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{si } |y - \hat{y}| \leq \delta \\ \delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{sinon} \end{cases}$$

**Usage :** Régression robuste aux outliers

**Avantages :**
- ✅ Combine MSE (petites erreurs) et MAE (grandes erreurs)
- ✅ Moins sensible aux outliers que MSE
- ✅ Différentiable partout

**Paramètre :** $\delta$ (typiquement 1.0) = seuil entre comportement MSE et MAE

**Implémentation Rust :**
```rust
fn huber_loss(predictions: &Array1<f64>, targets: &Array1<f64>, delta: f64) -> f64 {
    let diff = predictions - targets;
    let mut sum = 0.0;
    for &d in diff.iter() {
        let abs_d = d.abs();
        if abs_d <= delta {
            sum += 0.5 * d * d;  // MSE pour petites erreurs
        } else {
            sum += delta * (abs_d - 0.5 * delta);  // MAE pour grandes erreurs
        }
    }
    sum / predictions.len() as f64
}
```

---

### Guide de Sélection des Loss Functions

| **Tâche** | **Activation Sortie** | **Loss Function Recommandée** | **Pourquoi** |
|-----------|----------------------|-------------------------------|--------------|
| Régression | Linear | **MSE** | Standard, pénalise grandes erreurs |
| Régression robuste | Linear | **MAE** ou **Huber** | Résiste aux outliers |
| Classification binaire | Sigmoid | **Binary Cross-Entropy** | Interprétation probabiliste |
| Classification multi-classes | Softmax | **Categorical Cross-Entropy** | Standard multi-classes |
| Détection d'objets | Variable | IoU Loss, Focal Loss | Adapté aux boîtes |
| Segmentation | Softmax | Dice Loss, Focal Loss | Adapté aux pixels |

---

### Comparaison MSE vs Binary Cross-Entropy (XOR)

**Problème :** Classification binaire avec Sigmoid

#### MSE pour classification
- ❌ Gradient qui disparaît quand proche de 0 ou 1
- ❌ Pas d'interprétation probabiliste
- ❌ Convergence plus lente

#### Binary Cross-Entropy pour classification
- ✅ Gradient plus stable
- ✅ Converge plus vite
- ✅ Interprétation comme probabilité
- ✅ Meilleur choix pour XOR

**Résultats typiques (50k epochs, lr=0.5) :**
```
MSE:  Final loss: 0.0000 ✓
BCE:  Final loss: 0.0000 ✓
MAE:  Final loss: 0.2500 (nécessite lr=0.2, epochs=150k)
```

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

**Objectif du training :** Descendre cette courbe le plus vite possible en ajustant les poids.

---

## Documentation Recommandée

1. **[3Blue1Brown - Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk)** (YouTube) : visualisations excellentes
2. **[The Rust ML Book](https://rust-ml.github.io/book/)** : apprentissage automatique en Rust
3. **[ndarray docs](https://docs.rs/ndarray/latest/ndarray/)** : documentation de la bibliothèque
4. **Neural Networks from Scratch** (livre) : explications mathématiques détaillées
5. **[ML Cheatsheet - Loss Functions](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)** : référence complète

## Expérimentation

### Architecture
```rust
Network::new(2, 5, 1)   // 5 neurones cachés
Network::new(2, 10, 1)  // 10 neurones cachés
```

### Learning Rate
```rust
let learning_rate = 0.01;  // Plus lent
let learning_rate = 0.5;   // Plus rapide
let learning_rate = 1.0;   // Très rapide (attention à la stabilité)
```

### Fonction d'Activation et Loss
```rust
// Classification binaire (XOR)
Network::new(2, 5, 1, 
    Activation::Tanh,           // Couche cachée
    Activation::Sigmoid,        // Sortie
    LossFunction::BinaryCrossEntropy)

// Régression
Network::new(4, 10, 1,
    Activation::ReLU,           // Couche cachée
    Activation::Linear,         // Sortie
    LossFunction::MSE)

// Multi-classes
Network::new(784, 128, 10,
    Activation::GELU,           // Couche cachée
    Activation::Softmax,        // Sortie
    LossFunction::CategoricalCrossEntropy)
```
