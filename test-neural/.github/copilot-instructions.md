# Instructions pour GitHub Copilot

Ce projet est une bibliothèque de réseaux de neurones en Rust, conçue pour la performance et l'efficacité mémoire.

## 🎯 Principes Fondamentaux

### 0. English Only

- **All code in English** : variable names, function names, type names, struct fields
- **All comments in English** : `//`, `///`, `//!`, doc comments, TODOs
- **All strings in English** : error messages, log messages, println output
- **No French anywhere in source code**

### 1. Clean Code

- **Noms explicites** : Utiliser des noms de variables, fonctions et types descriptifs
- **Fonctions courtes** : Une fonction = une responsabilité
- **Pas de code mort** : Supprimer le code non utilisé
- **DRY** : Ne pas répéter le code, factoriser
- **Documentation** : Documenter les fonctions publiques avec `///`
- **Tests** : Chaque fonctionnalité doit avoir des tests unitaires

### 2. Builder Pattern Obligatoire

- **Toujours privilégier le pattern Builder** pour la construction d'objets complexes (modèles, trainers, configs)
- **API fluide** : chaîner les méthodes `.method()` plutôt qu'exposer des fonctions avec beaucoup de paramètres
- **Pas de fonctions à 4+ arguments** : si une fonction a trop de paramètres, créer un builder
- **Pas de backward compatibility** : quand un builder remplace une ancienne API, supprimer l'ancienne API entièrement (ne pas garder les deux)

### 3. Refactoring Complet

- **Jamais de demi-refacto** : une refactorisation doit être menée à terme — tous les call sites, tests, exports et documentation doivent être mis à jour
- **Supprimer le code mort** : après refacto, supprimer les anciennes fonctions, structs et exports devenus inutiles. Ne pas les garder "au cas où"
- **Pas de `pub` inutile** : si un symbole n'est plus utilisé en dehors du module, retirer `pub` ou le supprimer
- **Mettre à jour les tests** : les tests doivent utiliser la nouvelle API, pas l'ancienne
- **Ne jamais simplifier en supprimant des fonctionnalités** : une refacto améliore le design sans réduire les capacités

### 2. Performance First

- **Éviter les allocations** : Préférer les slices aux Vec quand possible
- **Réutiliser les buffers** : Pré-allouer et réutiliser plutôt que créer/détruire
- **Éviter les clones** : Utiliser des références, prendre ownership quand approprié
- **Vectorisation** : Utiliser les opérations ndarray plutôt que des boucles scalaires
- **SIMD-friendly** : Structurer les données pour les opérations vectorielles
- **Parallélisation** : Utiliser Rayon avec `#[cfg(feature = "parallel")]` pour les opérations batch

### 3. Gestion Mémoire Optimisée

- **Type Float** : Utiliser `Float` (alias de f32) au lieu de f64 pour réduire la mémoire de 50%
- **Pas de Box/Rc inutiles** : Préférer les types concrets sur la stack
- **Éviter les String** : Utiliser `&str` quand possible
- **Itérateurs** : Préférer les itérateurs aux collections intermédiaires
- **In-place** : Modifier en place plutôt que créer de nouvelles structures

## 📐 Conventions de Code

### Types Numériques

```rust
// ✅ Correct : utiliser Float
use crate::Float;
let value: Float = 0.5;
let array: Array1<Float> = Array1::zeros(10);

// ❌ Éviter : f64 en dur
let value: f64 = 0.5;  // NON
```

### Allocations

```rust
// ✅ Correct : réutiliser les buffers
fn process(&mut self, data: &[Float]) {
    self.buffer.fill(0.0);
    // ... utiliser self.buffer
}

// ❌ Éviter : allocations répétées
fn process(&self, data: &[Float]) -> Vec<Float> {
    let buffer = vec![0.0; data.len()];  // Allocation à chaque appel
    // ...
}
```

### Itérateurs vs Boucles

```rust
// ✅ Correct : itérateurs et opérations vectorisées
let sum: Float = data.iter().sum();
let result = array.mapv(|x| x.max(0.0));

// ❌ Éviter : boucles scalaires
let mut sum = 0.0;
for i in 0..data.len() {
    sum += data[i];
}
```

### Parallélisation

```rust
// ✅ Correct : parallélisation conditionnelle
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "parallel")]
fn process_batch(inputs: &[Input]) -> Vec<Output> {
    inputs.par_iter().map(process_one).collect()
}

#[cfg(not(feature = "parallel"))]
fn process_batch(inputs: &[Input]) -> Vec<Output> {
    inputs.iter().map(process_one).collect()
}
```

## 🏗️ Architecture

```
cma-neural-network/     # Core : Dense, Activations, Optimiseurs
    └── Float           # Type alias (f32 par défaut)
    
cma-cnn/                # Extensions CNN : Conv2D, Pool, BatchNorm
    └── utilise Float de cma-neural-network
    
cma-models/             # Architectures : LeNet, ResNet, VGG
    └── utilise cma-cnn + cma-neural-network
```

## ⚠️ Compatibilité WASM

- **Rayon** : NON compatible WASM → utiliser `#[cfg(feature = "parallel")]`
- **std::time** : Limité en WASM → utiliser des alternatives
- **Threads** : Non disponibles en WASM

## 🔧 Features

| Feature | Description | Compatible WASM |
|---------|-------------|-----------------|
| `default` | Mode standard | ✅ |
| `parallel` | Multi-thread Rayon | ❌ |

## 📝 Checklist avant commit

- [ ] Tous les tests passent (`cargo test`)
- [ ] Pas de warnings (`cargo check`)
- [ ] Code formaté (`cargo fmt`)
- [ ] Clippy satisfait (`cargo clippy`)
- [ ] Documentation à jour
- [ ] Pas d'allocations inutiles ajoutées
- [ ] Utilise `Float` et non `f64`
