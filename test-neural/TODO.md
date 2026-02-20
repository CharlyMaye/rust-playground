# 🎯 TODO - Corrections CNN Training

## 📊 Résumé du Problème

| Modèle | Accuracy Actuelle | Accuracy Attendue | Status |
|--------|-------------------|-------------------|--------|
| XOR | ~100% | 100% | ✅ OK |
| Iris | 93.33% | 95-98% | ⚠️ Acceptable |
| LeNet | En cours | 98.5%+ | 🔄 |
| **ResNet** | **11.33%** | **99.5%+** | ❌ CRITIQUE |
| **VGG** | **~10%** | **99%+** | ❌ CRITIQUE |

**Cause principale:** Loss bloquée à ~2.302 = `-ln(0.1)` = random chance (10 classes)

---

## ✅ Étapes à Réaliser

### Phase 1: BatchNorm2D - Mise à jour des running stats
**Fichier:** `cma-cnn/src/layers.rs`
**Priorité:** 🔴 CRITIQUE

- [ ] **1.1** Modifier `BatchNorm2D` pour utiliser `RefCell` ou `Cell` pour les running stats
- [ ] **1.2** Ajouter la mise à jour EMA dans `forward_sequential()`:
  ```rust
  if self.training {
      self.running_mean[c] = (1.0 - self.momentum) * self.running_mean[c] 
                            + self.momentum * mean;
      self.running_var[c] = (1.0 - self.momentum) * self.running_var[c] 
                           + self.momentum * var;
  }
  ```
- [ ] **1.3** Faire de même dans `forward_parallel()`
- [ ] **1.4** Tester que les running stats sont bien mises à jour

---

### Phase 2: ResidualBlock - Ajouter eval_mode/train_mode
**Fichier:** `cma-models/src/resnet.rs`
**Priorité:** 🔴 CRITIQUE

- [ ] **2.1** Ajouter méthode `eval_mode(&mut self)` à `ResidualBlock`:
  ```rust
  pub fn eval_mode(&mut self) {
      self.bn1.eval_mode();
      self.bn2.eval_mode();
      if let Some((_, ref mut bn)) = self.downsample {
          bn.eval_mode();
      }
  }
  ```
- [ ] **2.2** Ajouter méthode `train_mode(&mut self)` à `ResidualBlock`

---

### Phase 3: ResNet - Ajouter eval_mode/train_mode
**Fichier:** `cma-models/src/resnet.rs`
**Priorité:** 🔴 CRITIQUE

- [ ] **3.1** Ajouter méthode `eval_mode(&mut self)` à `ResNet`:
  ```rust
  pub fn eval_mode(&mut self) {
      self.stem_bn.eval_mode();
      for stage in &mut self.stages {
          for block in stage {
              block.eval_mode();
          }
      }
  }
  ```
- [ ] **3.2** Ajouter méthode `train_mode(&mut self)` à `ResNet`
- [ ] **3.3** Ajouter les mêmes méthodes à `ResNet18` si utilisé

---

### Phase 4: Scripts de Training - Appeler eval_mode avant extraction
**Fichiers:** 
- `neural-wasm/mnist-resnet/src/train_resnet.rs`
- `neural-wasm/mnist-vgg/src/train_vgg.rs`

**Priorité:** 🟡 IMPORTANT

- [ ] **4.1** Dans `train_resnet.rs`, ajouter avant extraction des features:
  ```rust
  // Passer en mode eval pour des stats BatchNorm stables
  resnet.eval_mode();
  ```
- [ ] **4.2** Dans `train_vgg.rs`, ajouter:
  ```rust
  cnn.eval_mode();
  ```

---

### Phase 5: Tests et Validation
**Priorité:** 🟡 IMPORTANT

- [ ] **5.1** Recompiler avec `cargo build --release`
- [ ] **5.2** Supprimer les anciens modèles:
  ```bash
  rm -f neural-wasm/mnist-resnet/src/resnet_model.bin
  rm -f neural-wasm/mnist-vgg/src/vgg_model.bin
  ```
- [ ] **5.3** Relancer les trainings
- [ ] **5.4** Vérifier que la loss descend en dessous de 2.0 dès les premières epochs
- [ ] **5.5** Vérifier accuracy > 50% après quelques epochs

---

### Phase 6 (Optionnel): Améliorations Futures
**Priorité:** 🟢 OPTIONNEL

- [ ] **6.1** Implémenter backpropagation dans le CNN (end-to-end training)
- [ ] **6.2** Ajouter data augmentation
- [ ] **6.3** Ajouter learning rate warmup
- [ ] **6.4** Pre-training ou transfer learning

---

## 📁 Fichiers à Modifier

| Ordre | Fichier | Type de Modif |
|-------|---------|---------------|
| 1 | `cma-cnn/src/layers.rs` | Logique BatchNorm |
| 2 | `cma-models/src/resnet.rs` | Ajouter méthodes |
| 3 | `neural-wasm/mnist-resnet/src/train_resnet.rs` | Appel eval_mode |
| 4 | `neural-wasm/mnist-vgg/src/train_vgg.rs` | Appel eval_mode |

---

## ⚠️ Points d'Attention

1. **Mutabilité:** `BatchNorm2D::forward()` prend `&self`, pas `&mut self`
   - Solution: Utiliser `Cell<Float>` ou `RefCell` pour les running stats
   - Ou: Changer la signature en `&mut self`

2. **Compatibilité:** Vérifier que les modifications ne cassent pas les autres modèles (LeNet, AlexNet)

3. **Tests:** Les running stats doivent être différentes de leur valeur initiale après training

---

## 🧪 Commandes de Test

```bash
# Recompiler
cd /workspace/test-neural
cargo build --release

# Supprimer anciens modèles
rm -f neural-wasm/mnist-resnet/src/resnet_model.bin
rm -f neural-wasm/mnist-vgg/src/vgg_model.bin

# Relancer training ResNet
cargo run --release --bin train_resnet 2>&1 | tee neural-wasm/mnist-resnet/src/training.log

# Relancer training VGG
cargo run --release --bin train_vgg 2>&1 | tee neural-wasm/mnist-vgg/src/training.log
```

---

## ✅ Critères de Succès

- [ ] ResNet accuracy > 95%
- [ ] VGG accuracy > 95%
- [ ] Loss descend progressivement (pas bloquée à 2.30)
- [ ] Running stats de BatchNorm != valeurs initiales
