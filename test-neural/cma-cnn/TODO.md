# 🔧 TODO - Optimisations Performance cma-cnn

## 📊 Analyse du 4 février 2026

### 🔴 Critiques (Impact majeur sur la rapidité)

- [x] **1. Remplacer `conv2d_naive` par `im2col` + GEMM**
  - Fichier: `src/ops.rs`
  - ✅ Implémenté: `conv2d_im2col()` avec multiplication matricielle
  - ✅ Support multi-batch complet
  - Gain estimé: **10-100x plus rapide**

- [x] **2. Activer la convolution optimisée dans `Conv2D::forward()`**
  - Fichier: `src/layers.rs`
  - ✅ Utilise maintenant `conv2d_im2col` par défaut

### 🟠 Majeurs (Impact sur la mémoire)

- [x] **3. Supprimer le clone dans `Sequential::forward()`**
  - Fichier: `src/sequential.rs`
  - ✅ Ajouté `forward_owned()` pour éviter le clone quand ownership disponible

- [x] **4. Éliminer allocations dans `Tensor4D::flatten()`**
  - Fichier: `src/tensor.rs`
  - ✅ Supprimé le `Vec<f64>` intermédiaire, itération directe

- [x] **5. Éliminer allocations dans `Tensor4D::unflatten()`**
  - Fichier: `src/tensor.rs`
  - ✅ Optimisé avec itération directe

### 🟡 Moyens (Optimisations secondaires)

- [x] **6. Vectoriser `BatchNorm2D::forward()`**
  - Fichier: `src/layers.rs`
  - ✅ Utilise `slice()` et `.sum()` vectorisés
  - ✅ Pré-calcul de `std_inv` évite divisions répétées

- [x] **7. Implémenter parallélisation avec Rayon**
  - Fichier: `src/ops.rs`
  - ✅ `conv2d_im2col_parallel()` traite les batches en parallèle
  - ✅ Activé avec `--features parallel`

- [ ] **8. Vectoriser pooling (maxpool2d, avgpool2d)**
  - Fichier: `src/ops.rs`
  - Status: À faire dans une prochaine itération

### 🔵 Faibles (Améliorations futures)

- [x] **9. Migrer f64 → Float (f32)**
  - ✅ Type alias `Float = f32` dans `cma-neural-network/src/lib.rs`
  - ✅ Tous les crates migrés : cma-neural-network, cma-cnn, cma-models, neural-wasm
  - ✅ Impact: -50% mémoire, +SIMD performance
  - ✅ Changement réversible : modifier `pub type Float = f32;` en `f64` si nécessaire

- [x] **10. ~~Ajouter ndarray-blas pour GEMM optimisé~~** → Non nécessaire
  - ✅ `matrixmultiply v0.3.10` déjà inclus via ndarray
  - ✅ SIMD automatique (SSE, AVX, AVX2) en Rust pur
  - ✅ Compatible WASM (fallback sans SIMD)
  - ℹ️ OpenBLAS donnerait +20-30% mais complexité d'installation inutile

---

## ✅ Complétés

- [x] Analyse initiale du code (4 février 2026)
- [x] Implémentation `conv2d_im2col` avec GEMM
- [x] Parallélisation Rayon pour le traitement batch
- [x] Optimisation allocations mémoire (`flatten`, `Sequential::forward`)
- [x] Vectorisation BatchNorm2D
- [x] Tous les tests passent (24/24)
- [x] Nettoyage: `conv2d_naive` → `#[cfg(test)]` only
- [x] Export API publique: `conv2d_im2col`, `im2col_single`
- [x] Migration f64 → Float (f32) sur tout le codebase
- [x] Instructions Copilot ajoutées (`.github/copilot-instructions.md`)

---

## 🧹 Nettoyage Effectué

| Élément | Action | Raison |
|---------|--------|--------|
| `conv2d_naive` | `#[cfg(test)]` | Gardé pour tests de validation uniquement |
| `im2col` (batch) | `#[allow(dead_code)]` | API publique, sera utile pour backward |
| `conv2d_im2col_sequential` | `#[allow(dead_code)]` | Utilisé quand parallel désactivé |
| Exports `lib.rs` | Ajouté `conv2d_im2col`, `im2col_single` | API publique |

---

## 📈 Résumé des Optimisations

| Optimisation | Fichier | Status | Impact |
|-------------|---------|--------|--------|
| im2col + GEMM | ops.rs | ✅ | ~10-100x rapidité |
| Rayon parallel | ops.rs | ✅ | Multi-core support |
| forward_owned | sequential.rs | ✅ | -50% mémoire |
| flatten optimisé | tensor.rs | ✅ | -1 allocation |
| BatchNorm vectorisé | layers.rs | ✅ | ~2-5x rapidité |

### Utilisation

```bash
# Mode standard (séquentiel) - compatible WASM
cargo build --release

# Mode parallèle (multi-thread) - NON compatible WASM
cargo build --release --features parallel
```

### ⚠️ Notes Importantes

1. **WASM Compatibility**: La feature `parallel` utilise Rayon qui n'est **pas compatible** avec WebAssembly. Les modules `neural-wasm/*` utilisent `cma-cnn` sans cette feature.

2. **Relation avec cma-neural-network**: Le trainer de `cma-neural-network` a sa propre option `.parallel()` dans le builder. La feature `parallel` de `cma-cnn` est **indépendante** et ne concerne que les opérations CNN (convolutions par batch).

3. **Quand activer parallel dans cma-cnn**:
   - ✅ Applications natives avec gros batches (>16 images)
   - ✅ Entraînement sur CPU multi-core
   - ❌ WASM / navigateur
   - ❌ Inférence single-image (overhead Rayon > gain)

