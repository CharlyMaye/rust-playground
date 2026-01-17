# Neural WASM API Documentation

Cette documentation décrit l'API commune aux modules XOR et Iris.

## Format de Retour

Toutes les méthodes retournent des **String contenant du JSON** pour compatibilité WASM.

## Méthodes Communes

### `predict()`
Fait une prédiction sur des entrées données.

**XOR:**
```rust
pub fn predict(&self, x1: f64, x2: f64) -> String
```

**Iris:**
```rust
pub fn predict(&self, sepal_length: f64, sepal_width: f64, petal_length: f64, petal_width: f64) -> String
```

**Retour JSON:**
```json
{
  "prediction": 1,              // XOR: 0 ou 1
  "raw": 0.95,                  // XOR: sortie brute (0-1)
  "confidence": 90.5,           // Confiance en POURCENTAGE (0-100)
  "probabilities": [0.05, 0.95] // Probabilités pour chaque classe
}
```

**Iris retour JSON:**
```json
{
  "class": "Iris-setosa",       // Nom de la classe
  "class_idx": 0,               // Index de la classe
  "probabilities": [0.95, 0.03, 0.02],
  "confidence": 95.0            // Confiance en POURCENTAGE (0-100)
}
```

### `get_probabilities()`
Retourne les probabilités pour chaque classe.

**XOR:**
```rust
pub fn get_probabilities(&self, x1: f64, x2: f64) -> String
```

**Iris:**
```rust
pub fn get_probabilities(&self, sepal_length: f64, sepal_width: f64, petal_length: f64, petal_width: f64) -> String
```

**Retour:** Array JSON de probabilités (valeurs entre 0-1)
```json
[0.05, 0.95]  // XOR
[0.95, 0.03, 0.02]  // Iris
```

### `get_class_names()`
Retourne les noms des classes.

```rust
pub fn get_class_names(&self) -> String
```

**Retour:**
```json
["0", "1"]  // XOR
["Iris-setosa", "Iris-versicolor", "Iris-virginica"]  // Iris
```

### `test_all()`
Teste toutes les combinaisons d'entrées (XOR) ou toutes les fleurs du dataset (Iris).

```rust
pub fn test_all(&self) -> String
```

**Retour:** Array JSON de résultats
```json
[
  {
    "inputs": [0, 0],
    "expected": 0,
    "prediction": 0,
    "raw": 0.05,
    "confidence": 90.0,
    "correct": true
  },
  ...
]
```

### `model_info()`
Retourne les informations sur le modèle.

```rust
pub fn model_info(&self) -> String
```

**Retour:**
```json
{
  "layers": [2, 4, 1],
  "activation": "sigmoid",
  "learning_rate": 0.5
}
```

### `get_weights()`
Retourne les poids du réseau pour visualisation.

```rust
pub fn get_weights(&self, layer: usize) -> String
```

**Retour:** Matrice de poids au format JSON

### `get_activations()`
Retourne les activations de chaque couche pour une entrée donnée.

**XOR:**
```rust
pub fn get_activations(&self, x1: f64, x2: f64) -> String
```

**Iris:**
```rust
pub fn get_activations(&self, sepal_length: f64, sepal_width: f64, petal_length: f64, petal_width: f64) -> String
```

**Retour:** Array JSON des activations de chaque couche

## Notes Importantes

### Confidence
- **Format:** Toujours en POURCENTAGE (0-100)
- **Calcul XOR:** `(raw - 0.5).abs() * 2.0 * 100.0`
- **Calcul Iris:** `max_probability * 100.0`
- **Usage HTML:** Utiliser directement `confidence.toFixed(1)`, PAS de multiplication par 100

### Probabilités
- **Format:** Toujours entre 0 et 1
- **Usage HTML:** Multiplier par 100 pour affichage en pourcentage

### Cohérence
Les deux modules (XOR et Iris) suivent exactement la même structure d'API pour faciliter l'intégration et la maintenance.
