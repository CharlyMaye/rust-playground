# Analyse Performance - Neural Network Library

**Date**: 2026-01-29  
**Contexte**: Training MNIST (49k samples, 784→128→64→10) prend ~291s pour 20 epochs

---

## 📋 MÉTHODOLOGIE D'ANALYSE

### Étapes suivies:
1. ✅ Lecture complète du code source (`trainer.rs`, `network.rs`, `dataset.rs`, `optimizer.rs`)
2. ✅ Identification des patterns d'allocation mémoire (`grep` sur `clone()`, `to_vec()`, `zeros()`)
3. ✅ Analyse de la boucle d'entraînement principale (`fit()` dans `network.rs`)
4. ✅ Analyse du code de batch training (`train_batch_cpu()` dans `trainer.rs`)
5. ⏳ Mesures de performance à faire (profiling)
6. ⏳ Benchmarks comparatifs à faire

---

## 🔍 OBSERVATIONS FACTUELLES

### 1. Boucle d'entraînement principale

**Fichier**: `network.rs:968-1020`

```rust
let mut train_data = train_dataset.clone();  // LINE 968 ← CLONE COMPLET

let mut trainer = crate::trainer::Trainer::new(self, device)
    .expect("Device should be validated before calling fit()");

for epoch in 0..epochs {
    // ... callbacks ...
    
    train_data.shuffle();  // LINE 985
    
    for (batch_inputs, batch_targets) in train_data.batches(batch_size) {  // LINE 987
        trainer.train_batch(&batch_inputs, &batch_targets);
    }
    
    // Calcul des losses sur TOUT le dataset
    let train_loss = trainer.network()
        .evaluate(train_dataset.inputs(), train_dataset.targets());  // LINE 993
    let val_loss = val_dataset.map(|val| 
        trainer.network().evaluate(val.inputs(), val.targets()));    // LINE 995
    
    history.push((train_loss, val_loss));
}
```

**Observations**:
- **Ligne 968**: Clone du dataset complet (49k × 784 f64 = ~306 MB)
- **Ligne 985**: Shuffle in-place utilisant Fisher-Yates (bon algorithme)
- **Ligne 987**: Création de batches via itérateur
- **Lignes 993-995**: Évaluation complète sur train+val à CHAQUE epoch

**Quantification pour MNIST (20 epochs)**:
- Clone dataset: 1x (au début du fit)
- Shuffles: 20x (une fois par epoch)
- Évaluations train: 20 × 49k forward passes
- Évaluations val: 20 × 21k forward passes
- **Total évaluations**: 1.4M forward passes juste pour le monitoring

---

### 2. Création de batches

**Fichier**: `dataset.rs:250-253`

```rust
fn next(&mut self) -> Option<Self::Item> {
    // ...
    let batch_inputs = self.dataset.inputs[self.current_idx..end_idx].to_vec();   // COPIE
    let batch_targets = self.dataset.targets[self.current_idx..end_idx].to_vec(); // COPIE
    
    self.current_idx = end_idx;
    Some((batch_inputs, batch_targets))
}
```

**Observation**: 
- Type de retour: `(Vec<Array1<f64>>, Vec<Array1<f64>>)`
- Chaque batch copie les slices en nouveaux Vec

**Quantification**:
- Batch size = 1536
- Nombre de batches par epoch = ceil(49000/1536) = 32 batches
- Copies par epoch: 32 × 1536 × 2 = 98,304 Array1 copiés
- Copies pour 20 epochs: 1,966,080 Array1 copiés

---

### 3. Accumulation de gradients dans batch training

**Fichier**: `trainer.rs:130-145`

```rust
fn train_batch_cpu(&mut self, inputs: &[Array1<f64>], targets: &[Array1<f64>]) {
    let batch_size = inputs.len() as f64;

    // Allocation de buffers pour chaque batch
    let mut accumulated_weights: Vec<Array2<f64>> = self
        .network
        .layers
        .iter()
        .map(|layer| Array2::zeros(layer.weights.dim()))  // ALLOCATION
        .collect();

    let mut accumulated_biases: Vec<Array1<f64>> = self
        .network
        .layers
        .iter()
        .map(|layer| Array1::zeros(layer.biases.dim()))   // ALLOCATION
        .collect();

    // Boucle sur chaque exemple du batch
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let forward_result = self.forward_with_rng(input);
        // ... backprop ...
        
        // Accumulation (+= pour chaque sample)
        accumulated_weights[i] = &accumulated_weights[i] + &weights_gradient;  // LINE 175
        accumulated_biases[i] = &accumulated_biases[i] + &biases_gradient;    // LINE 176
    }

    // Apply gradients
    self.apply_gradients_batch(&accumulated_weights, &accumulated_biases, batch_size);
}
```

**Observations**:
- Allocation de buffers de gradients: **POUR CHAQUE BATCH**
- Architecture MNIST: 3 layers
  - Layer 1: 784×128 = 100,352 f64
  - Layer 2: 128×64 = 8,192 f64  
  - Layer 3: 64×10 = 640 f64
  - **Total: 109,184 f64 par batch**

**Quantification pour 20 epochs**:
- Nombre de batches total: 32 × 20 = 640 batches
- Allocations: 640 × 109,184 = **69,877,760 f64 alloués** (560 MB)

**Note importante**: Les lignes 175-176 créent aussi de nouvelles Array2 via l'opération `+`

---

### 4. Évaluation du loss

**Fichier**: `network.rs:900-922`

```rust
pub fn evaluate(&self, inputs: &Vec<Array1<f64>>, targets: &Vec<Array1<f64>>) -> f64 {
    let mut total_loss = 0.0;

    for (input, target) in inputs.iter().zip(targets.iter()) {
        // Forward pass pour CHAQUE sample
        let activations = self.forward_eval(input);
        let prediction = activations.last().unwrap();
        total_loss += self.loss_function.compute(prediction, target);
    }

    let base_loss = total_loss / inputs.len() as f64;

    // Add regularization penalty
    let reg_penalty: f64 = self
        .layers
        .iter()
        .map(|layer| self.regularization.penalty(&layer.weights))
        .sum();

    base_loss + reg_penalty / inputs.len() as f64
}
```

**Observation**: 
- Itère sur TOUS les inputs individuellement
- Forward pass pour chaque sample (pas de vectorisation batch)
- Pas de cache ou de sampling

**Pour 20 epochs MNIST**:
- Train eval: 20 × 49,000 = 980,000 forward passes
- Val eval: 20 × 21,000 = 420,000 forward passes
- **Total: 1.4M forward passes juste pour monitoring**

---

### 5. Forward pass

**Fichier**: `network.rs:749-760`

```rust
fn forward_eval(&self, input: &Array1<f64>) -> Vec<Array1<f64>> {
    let mut activations = vec![input.clone()];  // CLONE de l'input

    for layer in &self.layers {
        let pre_activation = layer.weights.dot(activations.last().unwrap()) + &layer.biases;
        let post_activation = layer.activation.apply(&pre_activation);
        activations.push(post_activation);
    }

    activations
}
```

**Observation**:
- Clone l'input à chaque forward pass
- Alloue un Vec de taille `num_layers + 1`
- Pour MNIST: 4 Array1 alloués par forward (input + 3 layers)

---

## 📊 CALCULS QUANTIFIÉS

### Configuration MNIST testée:
- Samples d'entraînement: 49,000
- Samples de validation: 21,000
- Architecture: 784 → 128 → 64 → 10
- Epochs: 20
- Batch size: 1536
- Batches par epoch: ceil(49000/1536) = 32

### Allocations estimées (20 epochs):

| Opération | Par batch | Total (20 epochs) |
|-----------|-----------|-------------------|
| **Gradient buffers** | 109,184 f64 | 69.8M f64 (560 MB) |
| **Batch copies** | 1536 × 2 Array1 | 1.97M Array1 copiés |
| **Forward eval (monitoring)** | - | 1.4M forward passes |
| **Dataset clone** | 306 MB | 1x (au début) |

---

## ❓ QUESTIONS SANS RÉPONSE (À MESURER)

1. **Temps réel passé dans chaque phase**:
   - Forward pass
   - Backward pass
   - Optimizer step
   - Monitoring (eval)
   - Allocation/désallocation mémoire

2. **Impact des allocations de gradients**:
   - Est-ce vraiment un bottleneck ? (à mesurer avec profiler)
   - L'allocateur Rust est-il efficace ici ?

3. **Impact du monitoring constant**:
   - Que se passe-t-il si on évalue seulement tous les 10 epochs ?
   - Quel est le coût réel des 1.4M forward passes d'évaluation ?

4. **Comparaison vectorisation**:
   - Évaluation batch vs sample-by-sample
   - Quel gain réel avec batch forward pass ?

5. **Impact des clones de Array1**:
   - `to_vec()` dans batches iterator
   - `clone()` dans forward_eval
   - Coût réel vs coût théorique ?

---

## 🎯 HYPOTHÈSES À TESTER

### Hypothèse H1: "Les allocations de gradients sont le bottleneck principal"
**Test**: Pré-allouer les buffers de gradients sur le Trainer, réutiliser entre batches  
**Mesure**: Comparer le temps avant/après  
**Prédiction**: Si vrai → speedup de 20-40%

### Hypothèse H2: "L'évaluation constante ralentit significativement"
**Test**: Évaluer seulement tous les 10 epochs au lieu de chaque epoch  
**Mesure**: Comparer le temps total  
**Prédiction**: Si vrai → speedup proportionnel (~5-10% car 1.4M/total forward passes)

### Hypothèse H3: "Le clone du dataset au début de fit() est négligeable"
**Test**: Utiliser des indices de shuffle au lieu de clone+shuffle  
**Mesure**: Temps de clone vs temps total  
**Prédiction**: Probablement <1% du temps total (306MB clone one-time vs répété work)

### Hypothèse H4: "Les copies de batches dans l'itérateur coûtent cher"
**Test**: Modifier l'itérateur pour retourner des slices `&[Array1<f64>]`  
**Mesure**: Temps avant/après + mesure d'allocation  
**Prédiction**: Si vrai → speedup de 5-15%

### Hypothèse H5: "La vectorisation batch du forward/backward pourrait aider"
**Test**: Implémenter forward_batch qui traite tout un batch en une matrice operation  
**Mesure**: Comparer forward_eval loop vs forward_batch matrix ops  
**Prédiction**: Si vrai → speedup de 2-3x sur la partie forward/backward

---

## 🔬 TESTS À EFFECTUER

### Test 1: Profiling avec cargo-flamegraph
```bash
cargo flamegraph --bin train_mnist -- --release
```
→ Identifier visuellement où le temps est passé

### Test 2: Benchmark avec/sans monitoring
Modifier `fit()` pour accepter un paramètre `eval_frequency`  
Comparer eval chaque epoch vs tous les 10 epochs

### Test 3: Allocation profiling avec heaptrack
```bash
heaptrack target/release/train_mnist
```
→ Mesurer les allocations réelles et leur impact

### Test 4: Benchmark Array operations
Créer un micro-benchmark pour mesurer:
- Coût de `Array2::zeros()`
- Coût de `clone()` sur Array1
- Coût de `&array1 + &array2` vs `array1 += &array2`

---

## 📝 NEXT STEPS

1. **Instrumenter le code** avec des timers pour mesurer chaque phase
2. **Profiler** avec flamegraph pour identifier les vrais hotspots
3. **Tester les hypothèses** une par une avec benchmarks reproductibles
4. **Documenter les résultats** avec des chiffres réels, pas théoriques

---

## ⚠️ ATTENTION: HYPOTHÈSES vs FAITS

**FAITS** (confirmés par lecture de code):
- ✅ Gradient buffers alloués à chaque batch
- ✅ Dataset cloné au début de fit()
- ✅ Évaluation complète train+val à chaque epoch
- ✅ Batches copiés par l'itérateur
- ✅ Forward eval sample-by-sample (pas vectorisé)

**HYPOTHÈSES** (non mesurées):
- ❓ L'allocation est le bottleneck principal
- ❓ L'évaluation prend X% du temps total
- ❓ La vectorisation donnerait un speedup de Y%
- ❓ Le parallélisme Rayon est contre-productif pour MNIST

**CONCLUSION**: Il faut MESURER, pas spéculer !
