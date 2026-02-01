# 🔍 Analyse Performance et Mémoire - Rapport Complet v2

**Date**: 2026-01-30  
**Auteur**: Agent IA  
**Scope**: cma-neural-network + neural-wasm

---

## 📁 **cma-neural-network**

### 🔴 **Problèmes Critiques**

#### 1. `network.rs` L40-L46 - Allocations inutiles dans `RegularizationType::penalty()`

```rust
RegularizationType::L1 { lambda } => lambda * weights.mapv(|w| w.abs()).sum(),
RegularizationType::L2 { lambda } => 0.5 * lambda * weights.mapv(|w| w * w).sum(),
```

**Problème**: `mapv()` crée une nouvelle matrice temporaire juste pour calculer une somme.

**Solution**:
```rust
RegularizationType::L1 { lambda } => lambda * weights.iter().map(|w| w.abs()).sum::<f64>(),
RegularizationType::L2 { lambda } => 0.5 * lambda * weights.iter().map(|w| w * w).sum::<f64>(),
```

---

#### 2. `network.rs` L54-L59 - `gradient_opt()` alloue systématiquement

```rust
RegularizationType::L1 { lambda } => Some(weights.mapv(|w| lambda * w.signum())),
```

**Problème**: Chaque appel alloue une nouvelle matrice de même taille que les poids.

**Solution**: Utiliser une version in-place qui ajoute directement au gradient accumulé :
```rust
pub fn add_gradient_to(&self, weights: &Array2<f64>, target: &mut Array2<f64>) {
    match self {
        RegularizationType::L1 { lambda } => {
            ndarray::Zip::from(target).and(weights).for_each(|t, &w| {
                *t += lambda * w.signum();
            });
        }
        // ...
    }
}
```

---

#### 3. `network.rs` L354-L361 - Activation `Softmax` : double allocation

```rust
Activation::Softmax => {
    let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x = x.mapv(|v| (v - max).exp());  // Allocation 1
    let sum = exp_x.sum();
    exp_x / sum  // Allocation 2 (nouvelle matrice)
}
```

**Solution**: Modifier in-place ou utiliser `mapv_into` :
```rust
Activation::Softmax => {
    let max = x.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mut result = x.mapv(|v| (v - max).exp());
    let sum = result.sum();
    result.mapv_inplace(|v| v / sum);
    result
}
```

---

#### 4. `trainer.rs` L206-L209 - Allocation répétée dans la boucle de training

```rust
self.accumulated_weights[i] = &self.accumulated_weights[i] + &weights_gradient;
self.accumulated_biases[i] = &self.accumulated_biases[i] + &biases_gradient;
```

**Problème**: Crée une nouvelle matrice à chaque itération au lieu de modifier en place.

**Solution**:
```rust
self.accumulated_weights[i] += &weights_gradient;
self.accumulated_biases[i] += &biases_gradient;
```

---

#### 5. `trainer.rs` L252-L255 - Mode parallèle : allocation inutile

```rust
let gradients: Vec<(Vec<Array2<f64>>, Vec<Array1<f64>>)> = inputs
    .par_iter()
    .zip(targets.par_iter())
    .map(|(input, target)| self.compute_sample_gradients(input, target))
    .collect();  // Alloue TOUT en mémoire
```

**Problème**: Collecte tous les gradients avant de les réduire → pic mémoire = `batch_size × num_layers × weights_size`.

**Solution**: Utiliser `reduce` de Rayon pour accumuler directement :
```rust
let gradients = inputs.par_iter().zip(targets.par_iter())
    .map(|(input, target)| self.compute_sample_gradients(input, target))
    .reduce(
        || (vec![Array2::zeros(dims)...], vec![Array1::zeros(dims)...]),
        |(mut acc_w, mut acc_b), (w, b)| {
            for i in 0..num_layers {
                acc_w[i] += &w[i];
                acc_b[i] += &b[i];
            }
            (acc_w, acc_b)
        }
    );
```

---

### 🟡 **Problèmes Modérés**

#### 6. `dataset.rs` L119-L128 - `shuffle_with_indices()` clone tout

```rust
let mut temp_inputs: Vec<Array1<f64>> = indices
    .iter()
    .map(|&idx| self.inputs[idx].clone())  // Clone chaque Array1!
    .collect();
```

**Problème**: Clone tous les arrays au lieu de permuter les pointeurs.

**Solution**: Utiliser une permutation in-place avec un algorithme de cycle :
```rust
pub fn shuffle_with_indices(&mut self, indices: &[usize]) {
    // In-place permutation using cycle following
    let mut visited = vec![false; self.len()];
    for start in 0..self.len() {
        if visited[start] || indices[start] == start { continue; }
        let mut current = start;
        while !visited[current] {
            visited[current] = true;
            let next = indices[current];
            if next != start {
                self.inputs.swap(start, next);
                self.targets.swap(start, next);
            }
            current = next;
        }
    }
}
```

---

#### 7. `network.rs` L362-L379 - `derivative()` fallback retourne `Array1::ones()`

```rust
_ => {
    // Fallback: these should use derivative_from_preactivation
    Array1::ones(a.len())  // Allocation inutile
}
```

**Problème**: Alloue un vecteur juste pour un fallback qui ne devrait pas être atteint.

**Solution**: Utiliser `panic!()` ou restructurer pour éviter ce cas.

---

#### 8. `network.rs` L888-L895 - `evaluate()` prend `&Vec<>` au lieu de `&[]`

```rust
pub fn evaluate(&self, inputs: &Vec<Array1<f64>>, targets: &Vec<Array1<f64>>) -> f64 {
```

**Problème**: Force l'utilisation d'un `Vec` alors qu'une slice suffirait.

**Solution**:
```rust
pub fn evaluate(&self, inputs: &[Array1<f64>], targets: &[Array1<f64>]) -> f64 {
```

---

#### 9. `metrics.rs` L65-L75 - Double passage sur les données

```rust
let pred_class = pred.iter()
    .enumerate()
    .max_by(...)
    .map(|(idx, _)| idx)
    .unwrap_or(0);

let target_class = target.iter()  // Second pass
    .enumerate()
    .max_by(...)
```

**Problème mineur**: Parcourt deux fois les données. OK pour de petits vecteurs, mais pourrait être fusionné.

---

### 🟢 **Points Positifs** (Bonnes pratiques déjà en place)

1. **Pre-allocation dans `Trainer`** : Les buffers de gradient sont pré-alloués et réutilisés
2. **Fisher-Yates shuffle** : Algorithme O(n) en place
3. **Batch iterator** : Retourne des slices, pas des copies
4. **ndarray::Zip** dans les optimizers : Opérations fusionnées efficaces

---

## 📁 **neural-wasm**

### 🔴 **Problèmes Critiques**

#### 10. `mnist/src/lib.rs` L77-L93 - API MNIST incorrecte

```rust
pub fn predict(&self, x1: f64, x2: f64) -> String {
    let input = array![x1, x2];  // MNIST a 784 features, pas 2!
```

**Problème**: L'API MNIST utilise le code XOR copié-collé. MNIST nécessite 784 pixels, pas 2 valeurs.

**Solution**: Corriger l'API pour accepter un `Float64Array` de 784 éléments.

---

#### 11. `mnist/src/lib.rs` - Model embarqué trop gros

```rust
const MODEL_BIN: &[u8] = include_bytes!("mnist_model.bin");
```

**Problème**: Le modèle MNIST (784→128→64→10) représente ~100K+ poids. Cela peut faire un fichier WASM de plusieurs Mo.

**Recommandation**: 
- Quantifier les poids (f32 au lieu de f64)
- Ou charger le modèle dynamiquement via fetch

---

#### 12. `shared/src/lib.rs` L85-L90 - `softmax()` alloue 3 fois

```rust
pub fn softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = values.iter().map(|&v| (v - max).exp()).collect();  // Alloc 1
    let sum: f64 = exp_values.iter().sum();
    exp_values.iter().map(|&v| v / sum).collect()  // Alloc 2
}
```

**Solution**:
```rust
pub fn softmax(values: &[f64]) -> Vec<f64> {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut result: Vec<f64> = values.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = result.iter().sum();
    result.iter_mut().for_each(|v| *v /= sum);
    result
}
```

---

### 🟡 **Problèmes Modérés**

#### 13. `train_mnist.rs` L39-L40 - Clone inutile lors du chargement

```rust
let inputs: Vec<Array1<f64>> = mnist_data.iter().map(|(i, _)| i.clone()).collect();
let targets: Vec<Array1<f64>> = mnist_data.iter().map(|(_, t)| t.clone()).collect();
```

**Solution**: Destructurer et consommer directement :
```rust
let (inputs, targets): (Vec<_>, Vec<_>) = mnist_data.into_iter().unzip();
```

---

#### 14. `iris/src/lib.rs` & `xor/src/lib.rs` - Sérialisation JSON répétée

```rust
serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
```

**Problème**: Chaque appel de `predict()`, `get_weights()`, etc. sérialise en JSON.

**Recommandation**: Pour les appels fréquents (predict), retourner des types simples :
```rust
#[wasm_bindgen]
pub fn predict_class(&self, x1: f64, x2: f64) -> u8 {  // Pas de JSON!
    // ...
}
```

---

## 📊 **Résumé des Priorités**

| Priorité | Fichier | Problème | Impact Mémoire | Impact Perf |
|----------|---------|----------|----------------|-------------|
| 🔴 P0 | trainer.rs | Accumulation += | Haut | Haut |
| 🔴 P0 | trainer.rs | Collect parallel | Très Haut | Moyen |
| 🔴 P0 | mnist/lib.rs | API incorrecte | - | Bloquant |
| 🟠 P1 | network.rs | Softmax double alloc | Moyen | Moyen |
| 🟠 P1 | network.rs | RegularizationType allocs | Moyen | Moyen |
| 🟡 P2 | dataset.rs | shuffle_with_indices clone | Moyen | Faible |
| 🟡 P2 | shared/lib.rs | softmax 3 allocs | Faible | Faible |
| 🟢 P3 | Divers | &Vec → &[] | Faible | Négligeable |

---

## ✅ TODO List - Corrections à effectuer

### 🔴 Priorité 0 - Critique

- [ ] **1. trainer.rs - Accumulation in-place**
  - Fichier: `cma-neural-network/src/trainer.rs`
  - Lignes: ~206-209
  - Remplacer `self.accumulated_weights[i] = &self.accumulated_weights[i] + &weights_gradient` par `self.accumulated_weights[i] += &weights_gradient`
  - Idem pour `accumulated_biases`

- [ ] **2. trainer.rs - Parallel reduce au lieu de collect**
  - Fichier: `cma-neural-network/src/trainer.rs`
  - Fonction: `train_batch_parallel()`
  - Remplacer `.collect()` par `.reduce()` pour éviter le pic mémoire

- [ ] **3. mnist/lib.rs - Corriger l'API MNIST**
  - Fichier: `neural-wasm/mnist/src/lib.rs`
  - Réécrire `predict()` pour accepter 784 pixels via `js_sys::Float64Array`
  - Supprimer les méthodes XOR copiées-collées (test_all avec XOR logic, etc.)

### 🟠 Priorité 1 - Important

- [ ] **4. network.rs - Optimiser Softmax**
  - Fichier: `cma-neural-network/src/network.rs`
  - Fonction: `Activation::apply()` cas `Softmax`
  - Utiliser `mapv_inplace` pour éviter la double allocation

- [ ] **5. network.rs - Optimiser RegularizationType::penalty()**
  - Fichier: `cma-neural-network/src/network.rs`
  - Lignes: ~40-46
  - Remplacer `mapv().sum()` par `iter().map().sum()`

- [ ] **6. network.rs - Ajouter méthode in-place pour gradient régularisation**
  - Fichier: `cma-neural-network/src/network.rs`
  - Ajouter `add_gradient_to()` à `RegularizationType`
  - Modifier `trainer.rs` pour utiliser cette nouvelle méthode

### 🟡 Priorité 2 - Modéré

- [ ] **7. dataset.rs - Optimiser shuffle_with_indices()**
  - Fichier: `cma-neural-network/src/dataset.rs`
  - Implémenter une permutation in-place (cycle following)

- [ ] **8. shared/lib.rs - Optimiser softmax()**
  - Fichier: `neural-wasm/shared/src/lib.rs`
  - Réduire de 3 allocations à 1

- [ ] **9. train_mnist.rs - Éviter les clones**
  - Fichier: `neural-wasm/mnist/src/train_mnist.rs`
  - Utiliser `into_iter().unzip()` au lieu de `iter().map().clone().collect()`

- [ ] **10. train_iris.rs - Éviter les clones**
  - Fichier: `neural-wasm/iris/src/train_iris.rs`
  - Même correction que pour MNIST

### 🟢 Priorité 3 - Mineur

- [ ] **11. network.rs - Changer signature evaluate()**
  - Fichier: `cma-neural-network/src/network.rs`
  - Changer `&Vec<Array1<f64>>` en `&[Array1<f64>]`

- [ ] **12. network.rs - Supprimer fallback derivative()**
  - Fichier: `cma-neural-network/src/network.rs`
  - Remplacer `Array1::ones()` par `unreachable!()` ou restructurer

- [ ] **13. xor/lib.rs & iris/lib.rs - Ajouter API sans JSON**
  - Fichiers: `neural-wasm/xor/src/lib.rs`, `neural-wasm/iris/src/lib.rs`
  - Ajouter `predict_class()` retournant un type simple pour les appels fréquents

### 📋 Bonus - Améliorations futures

- [ ] **14. Quantification f32 pour WASM**
  - Créer une version du réseau avec poids f32 pour réduire la taille WASM
  
- [ ] **15. Chargement dynamique du modèle MNIST**
  - Charger le modèle via fetch au lieu de include_bytes! pour MNIST

- [ ] **16. Benchmark suite**
  - Créer des benchmarks avec `criterion` pour mesurer l'impact des optimisations

---

## 📝 Notes pour l'agent

1. **Ordre d'exécution**: Suivre les priorités (P0 → P1 → P2 → P3)
2. **Tests**: Après chaque modification, exécuter `cargo test` dans le répertoire concerné
3. **Build WASM**: Après modifications dans neural-wasm, exécuter `./build.sh` dans chaque sous-répertoire
4. **Vérification**: Utiliser `cargo clippy` pour vérifier qu'il n'y a pas de nouvelles warnings
5. **Commits**: Faire un commit après chaque groupe de priorité terminé
