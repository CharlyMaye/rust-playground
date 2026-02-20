# 📘 Spécifications Techniques : Système Autograd

> **Document de référence** pour l'implémentation d'un système de différentiation automatique en Rust.

---

## 📚 Table des Matières

1. [Vue d'ensemble de l'architecture](#1-vue-densemble-de-larchitecture)
2. [Architecture Hybride : Statique/Dynamique + TrainableLayer](#2-architecture-hybride)
3. [Structure du Tensor](#3-structure-du-tensor)
4. [Système de Gradient Functions](#4-système-de-gradient-functions)
5. [Moteur de Backpropagation](#5-moteur-de-backpropagation)
6. [Implémentation des Opérations](#6-implémentation-des-opérations)
7. [Modules et Couches](#7-modules-et-couches)
8. [Optimizers](#8-optimizers)
9. [Gestion Mémoire](#9-gestion-mémoire)
10. [API Publique](#10-api-publique)
11. [Patterns et Bonnes Pratiques](#11-patterns-et-bonnes-pratiques)

---

## 1. Vue d'ensemble de l'architecture

### 1.1 Principes fondamentaux

Un système autograd repose sur trois piliers :

1. **Tensor** : Structure de données qui peut tracker ses opérations
2. **GradFunction** : Sait calculer le gradient d'une opération spécifique
3. **Engine** : Orchestre le backward pass à travers le graphe

```
┌─────────────────────────────────────────────────────────────────┐
│                         AUTOGRAD                                │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Tensor    │───▶│  GradFn     │───▶│   Engine    │        │
│  │  (données)  │    │ (backward)  │    │ (parcours)  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│    requires_grad      backward()      topological_sort         │
│    grad_fn            inputs          accumulate_grad          │
│    grad               name                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Hybride

Cette section présente une architecture flexible permettant de :
- Choisir entre **mode dynamique** (eager) et **mode statique** (traced/compiled)
- Séparer les couches entraînables des couches stateless via **TrainableLayer**

### 2.1 Pourquoi une architecture hybride ?

| Mode | Avantages | Inconvénients | Cas d'usage |
|------|-----------|---------------|-------------|
| **Dynamique (Eager)** | Flexibilité, debug facile, conditions dynamiques | Plus lent, plus de mémoire | Recherche, prototypage, RNN |
| **Statique (Traced)** | Optimisé, compilation possible, déploiement | Pas de conditions dynamiques | Production, mobile, WASM |

**L'idéal** : Pouvoir basculer de l'un à l'autre selon le contexte.

### 2.2 Hiérarchie des traits : Layer vs TrainableLayer

```rust
/// Trait de base pour TOUTES les couches
/// 
/// Les couches stateless (ReLU, MaxPool, Dropout) n'implémentent que ce trait.
/// Simple, léger, pas de backward nécessaire pour ces opérations.
pub trait Layer: Send + Sync {
    /// Forward pass uniquement
    fn forward(&self, input: &Tensor) -> Tensor;
    
    /// Nom de la couche (debug)
    fn name(&self) -> &'static str;
    
    /// Cette couche a-t-elle des paramètres entraînables ?
    fn is_trainable(&self) -> bool { false }
}

/// Trait pour les couches avec des paramètres entraînables
/// 
/// Étend Layer avec le backward et la gestion des paramètres.
/// Implémenté par : Linear, Conv2D, BatchNorm, Embedding, etc.
pub trait TrainableLayer: Layer {
    /// Backward pass - calcule les gradients
    fn backward(&self, grad_output: &Tensor, cache: &ForwardCache) -> BackwardResult;
    
    /// Retourne les paramètres entraînables
    fn parameters(&self) -> Vec<&Parameter>;
    
    /// Retourne les paramètres mutables (pour l'optimizer)
    fn parameters_mut(&mut self) -> Vec<&mut Parameter>;
    
    /// Remet les gradients à zéro
    fn zero_grad(&mut self) {
        for param in self.parameters_mut() {
            param.zero_grad();
        }
    }
    
    /// Nombre total de paramètres
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

/// Résultat du backward
pub struct BackwardResult {
    /// Gradient à propager vers la couche précédente
    pub grad_input: Tensor,
    /// Gradients des paramètres (stockés dans les Parameter)
    pub param_grads_computed: bool,
}

/// Cache du forward pour le backward
pub struct ForwardCache {
    pub input: Tensor,
    pub output: Tensor,
    pub extra: Option<Box<dyn std::any::Any + Send + Sync>>,
}
```

### 2.3 Implémentations des deux types de couches

#### Couches stateless (Layer uniquement)

```rust
/// ReLU - pas de paramètres, pas de backward spécifique nécessaire
pub struct ReLU;

impl Layer for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        relu(input)  // L'autograd gère le backward via grad_fn
    }
    
    fn name(&self) -> &'static str { "ReLU" }
}

/// MaxPool2D - pas de paramètres
pub struct MaxPool2D {
    kernel_size: usize,
    stride: usize,
}

impl Layer for MaxPool2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        max_pool2d(input, self.kernel_size, self.stride)
    }
    
    fn name(&self) -> &'static str { "MaxPool2D" }
}

/// Dropout - comportement différent train/eval mais pas de paramètres
pub struct Dropout {
    p: Float,
    training: bool,
}

impl Layer for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        if self.training {
            dropout(input, self.p)
        } else {
            input.clone()
        }
    }
    
    fn name(&self) -> &'static str { "Dropout" }
}
```

#### Couches entraînables (TrainableLayer)

```rust
/// Linear - a des paramètres (weight, bias)
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let y = matmul(input, &self.weight);
        match &self.bias {
            Some(b) => add(&y, b),
            None => y,
        }
    }
    
    fn name(&self) -> &'static str { "Linear" }
    fn is_trainable(&self) -> bool { true }
}

impl TrainableLayer for Linear {
    fn backward(&self, grad_output: &Tensor, cache: &ForwardCache) -> BackwardResult {
        // ∂L/∂input = grad_output @ weightᵀ
        let grad_input = matmul(grad_output, &transpose(&self.weight));
        
        // ∂L/∂weight = inputᵀ @ grad_output (accumulé dans weight.grad)
        let grad_weight = matmul(&transpose(&cache.input), grad_output);
        self.weight.accumulate_grad(grad_weight.data());
        
        // ∂L/∂bias = sum(grad_output, axis=0)
        if let Some(ref bias) = self.bias {
            let grad_bias = grad_output.sum_axis(0);
            bias.accumulate_grad(grad_bias.data());
        }
        
        BackwardResult {
            grad_input,
            param_grads_computed: true,
        }
    }
    
    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }
}

/// Conv2D - a des paramètres (weight, bias)
pub struct Conv2D {
    weight: Parameter,
    bias: Option<Parameter>,
    stride: usize,
    padding: usize,
}

impl Layer for Conv2D {
    fn forward(&self, input: &Tensor) -> Tensor {
        conv2d(input, &self.weight, self.bias.as_ref(), self.stride, self.padding)
    }
    
    fn name(&self) -> &'static str { "Conv2D" }
    fn is_trainable(&self) -> bool { true }
}

impl TrainableLayer for Conv2D {
    fn backward(&self, grad_output: &Tensor, cache: &ForwardCache) -> BackwardResult {
        // Backward de la convolution
        let grad_input = conv2d_backward_input(
            grad_output, &self.weight, cache.input.shape(), self.stride, self.padding
        );
        
        let grad_weight = conv2d_backward_weight(
            grad_output, &cache.input, self.weight.shape(), self.stride, self.padding
        );
        self.weight.accumulate_grad(&grad_weight);
        
        if let Some(ref bias) = self.bias {
            let grad_bias = grad_output.sum_axes(&[0, 2, 3]);
            bias.accumulate_grad(&grad_bias);
        }
        
        BackwardResult {
            grad_input,
            param_grads_computed: true,
        }
    }
    
    fn parameters(&self) -> Vec<&Parameter> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias { params.push(b); }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias { params.push(b); }
        params
    }
}
```

### 2.4 Mode d'exécution : Eager vs Traced

```rust
/// Mode d'exécution du graphe
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExecutionMode {
    /// Graphe dynamique - reconstruit à chaque forward
    Eager,
    /// Graphe statique - tracé une fois, réutilisé
    Traced,
}

/// Configuration globale ou par modèle
pub struct ExecutionContext {
    mode: ExecutionMode,
    /// En mode Traced, le graphe compilé
    traced_graph: Option<TracedGraph>,
}

impl ExecutionContext {
    pub fn eager() -> Self {
        Self { mode: ExecutionMode::Eager, traced_graph: None }
    }
    
    pub fn traced() -> Self {
        Self { mode: ExecutionMode::Traced, traced_graph: None }
    }
}
```

### 2.5 Traçage du graphe (mode statique)

```rust
/// Graphe tracé pour le mode statique
pub struct TracedGraph {
    /// Séquence d'opérations enregistrées
    operations: Vec<TracedOp>,
    /// Shapes des entrées attendues
    input_shapes: Vec<Vec<usize>>,
    /// Optimisations appliquées
    optimized: bool,
}

/// Opération tracée
pub struct TracedOp {
    op_type: OpType,
    inputs: Vec<TensorId>,
    output: TensorId,
    /// Paramètres de l'opération (stride, padding, etc.)
    config: OpConfig,
}

impl TracedGraph {
    /// Trace un modèle avec un input exemple
    pub fn trace<M: Module>(model: &M, example_input: &Tensor) -> Self {
        // Active le mode traçage
        let _guard = TracingGuard::new();
        
        // Exécute le forward - les opérations sont enregistrées
        let _output = model.forward(example_input);
        
        // Récupère le graphe tracé
        let operations = get_traced_operations();
        
        Self {
            operations,
            input_shapes: vec![example_input.shape().to_vec()],
            optimized: false,
        }
    }
    
    /// Applique des optimisations au graphe
    pub fn optimize(&mut self) {
        if self.optimized { return; }
        
        // Fusion d'opérations (Conv + BN + ReLU → ConvBNReLU)
        self.fuse_operations();
        
        // Élimination des opérations redondantes
        self.eliminate_redundant();
        
        // Pré-allocation des buffers
        self.preallocate_buffers();
        
        self.optimized = true;
    }
    
    /// Exécute le graphe tracé
    pub fn execute(&self, inputs: &[Tensor]) -> Tensor {
        assert_eq!(inputs.len(), self.input_shapes.len());
        
        let mut tensors: HashMap<TensorId, Tensor> = HashMap::new();
        
        // Enregistre les inputs
        for (i, input) in inputs.iter().enumerate() {
            tensors.insert(TensorId::Input(i), input.clone());
        }
        
        // Exécute chaque opération
        for op in &self.operations {
            let result = self.execute_op(op, &tensors);
            tensors.insert(op.output, result);
        }
        
        // Retourne le dernier output
        tensors.remove(&self.operations.last().unwrap().output).unwrap()
    }
}
```

### 2.6 Module avec mode configurable

```rust
/// Container de modèle avec mode d'exécution configurable
pub struct Model<M: Module> {
    inner: M,
    context: ExecutionContext,
}

impl<M: Module> Model<M> {
    pub fn new(inner: M) -> Self {
        Self {
            inner,
            context: ExecutionContext::eager(),
        }
    }
    
    /// Passe en mode eager (dynamique)
    pub fn eager_mode(&mut self) {
        self.context.mode = ExecutionMode::Eager;
    }
    
    /// Passe en mode tracé (statique)
    /// 
    /// Nécessite un exemple d'input pour tracer le graphe
    pub fn trace(&mut self, example_input: &Tensor) {
        let graph = TracedGraph::trace(&self.inner, example_input);
        self.context.mode = ExecutionMode::Traced;
        self.context.traced_graph = Some(graph);
    }
    
    /// Optimise le graphe tracé
    pub fn optimize(&mut self) {
        if let Some(ref mut graph) = self.context.traced_graph {
            graph.optimize();
        }
    }
    
    /// Forward adaptatif selon le mode
    pub fn forward(&self, input: &Tensor) -> Tensor {
        match self.context.mode {
            ExecutionMode::Eager => {
                // Mode dynamique standard
                self.inner.forward(input)
            }
            ExecutionMode::Traced => {
                // Mode statique - utilise le graphe pré-compilé
                self.context.traced_graph
                    .as_ref()
                    .expect("Graph not traced. Call trace() first.")
                    .execute(&[input.clone()])
            }
        }
    }
    
    /// Backward (uniquement en mode eager pour l'instant)
    pub fn backward(&mut self, loss: &Tensor) {
        match self.context.mode {
            ExecutionMode::Eager => {
                loss.backward();
            }
            ExecutionMode::Traced => {
                // Pour le backward en mode tracé, on peut :
                // 1. Revenir en mode eager temporairement
                // 2. Avoir un graphe backward pré-compilé (plus complexe)
                panic!("Backward in traced mode not yet implemented. Use eager_mode() for training.");
            }
        }
    }
}
```

### 2.7 Exemple d'utilisation complète

```rust
use autograd::prelude::*;

// Définir un modèle avec les deux types de couches
struct ConvNet {
    conv1: Conv2D,      // TrainableLayer
    relu1: ReLU,        // Layer (stateless)
    pool1: MaxPool2D,   // Layer (stateless)
    conv2: Conv2D,      // TrainableLayer
    relu2: ReLU,        // Layer (stateless)
    pool2: MaxPool2D,   // Layer (stateless)
    flatten: Flatten,   // Layer (stateless)
    fc: Linear,         // TrainableLayer
}

impl Module for ConvNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.conv1.forward(x);
        let x = self.relu1.forward(&x);
        let x = self.pool1.forward(&x);
        let x = self.conv2.forward(&x);
        let x = self.relu2.forward(&x);
        let x = self.pool2.forward(&x);
        let x = self.flatten.forward(&x);
        self.fc.forward(&x)
    }
    
    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters().into_iter().cloned());
        params.extend(self.conv2.parameters().into_iter().cloned());
        params.extend(self.fc.parameters().into_iter().cloned());
        params
    }
    
    // ... train/eval
}

fn main() {
    let net = ConvNet::new();
    let mut model = Model::new(net);
    let mut optimizer = Adam::new(model.inner.parameters(), 0.001);
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 1 : Training en mode EAGER (dynamique)
    // ═══════════════════════════════════════════════════════════
    model.eager_mode();
    
    for epoch in 0..10 {
        for (x_batch, y_batch) in train_loader.iter() {
            // Forward
            let output = model.forward(&x_batch);
            let loss = cross_entropy(&output, &y_batch);
            
            // Backward (fonctionne uniquement en mode eager)
            optimizer.zero_grad();
            model.backward(&loss);
            optimizer.step();
        }
    }
    
    // ═══════════════════════════════════════════════════════════
    // PHASE 2 : Inference en mode TRACED (statique)
    // ═══════════════════════════════════════════════════════════
    
    // Crée un exemple pour tracer le graphe
    let example = Tensor::zeros(&[1, 1, 28, 28], false);
    
    // Trace et optimise
    model.trace(&example);
    model.optimize();
    
    // Inference rapide avec le graphe pré-compilé
    let predictions = no_grad(|| {
        test_images.iter().map(|img| {
            model.forward(img)  // Utilise le graphe tracé
        }).collect::<Vec<_>>()
    });
    
    // ═══════════════════════════════════════════════════════════
    // SWITCH : Revenir en mode eager si besoin (fine-tuning)
    // ═══════════════════════════════════════════════════════════
    model.eager_mode();
    // ... continue training
}
```

### 2.8 Résumé de l'architecture hybride

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE HYBRIDE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TRAITS DE COUCHES                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                        Layer                                 │   │
│  │  - forward()                                                 │   │
│  │  - name()                                                    │   │
│  │  - is_trainable() → false par défaut                        │   │
│  │                                                              │   │
│  │  Implémenté par : ReLU, MaxPool, Dropout, Flatten, etc.     │   │
│  └───────────────────────────┬─────────────────────────────────┘   │
│                              │ extends                              │
│  ┌───────────────────────────▼─────────────────────────────────┐   │
│  │                    TrainableLayer                            │   │
│  │  - backward()                                                │   │
│  │  - parameters() / parameters_mut()                          │   │
│  │  - zero_grad()                                               │   │
│  │                                                              │   │
│  │  Implémenté par : Linear, Conv2D, BatchNorm, Embedding      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MODES D'EXÉCUTION                                                  │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐   │
│  │     EAGER (Dynamique)   │    │     TRACED (Statique)       │   │
│  ├─────────────────────────┤    ├─────────────────────────────┤   │
│  │ ✓ Graphe à la volée     │    │ ✓ Graphe pré-compilé        │   │
│  │ ✓ Conditions dynamiques │    │ ✓ Optimisations (fusion)    │   │
│  │ ✓ Debug facile          │    │ ✓ Mémoire pré-allouée       │   │
│  │ ✓ Training complet      │    │ ✓ Déploiement (WASM, etc.)  │   │
│  │                         │    │                             │   │
│  │ Utilisé pour :          │    │ Utilisé pour :              │   │
│  │ - Recherche             │    │ - Production                │   │
│  │ - Prototypage           │    │ - Inference                 │   │
│  │ - Training              │    │ - Mobile/Embarqué           │   │
│  └─────────────────────────┘    └─────────────────────────────┘   │
│              │                              │                       │
│              └──────────┬───────────────────┘                       │
│                         ▼                                           │
│              ┌─────────────────────┐                               │
│              │   Model<M>          │                               │
│              │   - eager_mode()    │                               │
│              │   - trace(example)  │                               │
│              │   - optimize()      │                               │
│              │   - forward()       │ ← Adaptatif selon mode        │
│              │   - backward()      │                               │
│              └─────────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.9 Avantages de cette architecture

| Aspect | Bénéfice |
|--------|----------|
| **Séparation Layer/TrainableLayer** | Code plus clair, couches stateless légères |
| **Backward compatible** | Les couches sans paramètres n'ont pas besoin de backward manuel |
| **Mode Eager** | Flexibilité totale pour le training et la recherche |
| **Mode Traced** | Performance optimale pour l'inférence |
| **Switch runtime** | Passer de l'un à l'autre sans modifier le modèle |
| **Optimisations statiques** | Fusion d'opérations, pré-allocation mémoire |
| **WASM friendly** | Mode tracé idéal pour le déploiement web |

### 2.10 Considérations d'implémentation

```rust
// Détection automatique du mode selon le contexte
fn should_use_traced_mode() -> bool {
    // En WASM, préférer le mode tracé
    #[cfg(target_arch = "wasm32")]
    return true;
    
    // Sinon, vérifier la config
    #[cfg(not(target_arch = "wasm32"))]
    return std::env::var("AUTOGRAD_MODE").map(|v| v == "traced").unwrap_or(false);
}

// Macro pour définir facilement des couches stateless
macro_rules! impl_stateless_layer {
    ($name:ident, $forward_fn:expr) => {
        pub struct $name;
        
        impl Layer for $name {
            fn forward(&self, input: &Tensor) -> Tensor {
                $forward_fn(input)
            }
            fn name(&self) -> &'static str { stringify!($name) }
        }
    };
}

// Usage
impl_stateless_layer!(ReLU, relu);
impl_stateless_layer!(Sigmoid, sigmoid);
impl_stateless_layer!(Tanh, tanh);
impl_stateless_layer!(Flatten, flatten);
```

---

## 3. Structure du Tensor

```
autograd/
├── src/
│   ├── lib.rs                 # Point d'entrée, exports publics
│   │
│   ├── tensor/
│   │   ├── mod.rs             # Tensor principal
│   │   ├── create.rs          # Constructeurs (zeros, ones, randn, etc.)
│   │   ├── ops.rs             # Surcharge des opérateurs (+, -, *, /)
│   │   └── indexing.rs        # Slicing, indexation
│   │
│   ├── autograd/
│   │   ├── mod.rs             # Exports autograd
│   │   ├── grad_fn.rs         # Trait GradFunction
│   │   ├── engine.rs          # Backward engine
│   │   └── context.rs         # Contexte no_grad, enable_grad
│   │
│   ├── ops/
│   │   ├── mod.rs             # Exports opérations
│   │   ├── basic.rs           # Add, Sub, Mul, Div, Neg
│   │   ├── matmul.rs          # Multiplication matricielle
│   │   ├── reduction.rs       # Sum, Mean, Max, Min
│   │   ├── conv.rs            # Convolution 2D
│   │   ├── pool.rs            # MaxPool, AvgPool, GlobalAvgPool
│   │   ├── norm.rs            # BatchNorm, LayerNorm
│   │   ├── activation.rs      # ReLU, Sigmoid, Tanh, Softmax, etc.
│   │   └── reshape.rs         # View, Flatten, Transpose, Permute
│   │
│   ├── nn/
│   │   ├── mod.rs             # Exports modules NN
│   │   ├── module.rs          # Trait Module
│   │   ├── parameter.rs       # Parameter wrapper
│   │   ├── linear.rs          # Couche linéaire (Dense)
│   │   ├── conv.rs            # Module Conv2D
│   │   ├── pool.rs            # Modules de pooling
│   │   ├── norm.rs            # Modules de normalisation
│   │   ├── activation.rs      # Modules d'activation
│   │   ├── dropout.rs         # Dropout, Dropout2D
│   │   └── sequential.rs      # Container séquentiel
│   │
│   ├── optim/
│   │   ├── mod.rs             # Exports optimizers
│   │   ├── optimizer.rs       # Trait Optimizer
│   │   ├── sgd.rs             # SGD, SGD with momentum
│   │   ├── adam.rs            # Adam, AdamW
│   │   └── rmsprop.rs         # RMSprop
│   │
│   └── utils/
│       ├── mod.rs             # Exports utilitaires
│       ├── grad_check.rs      # Vérification numérique des gradients
│       └── serialization.rs   # Sauvegarde/chargement des modèles
```

---

## 2. Structure du Tensor

### 2.1 Définition principale

```rust
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::sync::Arc;

/// Type numérique (f32 pour l'efficacité mémoire, f64 si plus de précision nécessaire)
pub type Float = f32;

/// Tensor avec support autograd
/// 
/// # Invariants
/// - Si `requires_grad` est true, le tensor participe au graphe de calcul
/// - Si `grad_fn` est Some, le tensor est le résultat d'une opération (non-leaf)
/// - Si `is_leaf` est true et `requires_grad` est true, le gradient sera conservé après backward
/// 
/// # Thread Safety
/// - `data` : partagé via Arc (lecture seule après création)
/// - `grad` : modifié via RefCell (single-threaded pendant backward)
#[derive(Clone)]
pub struct Tensor {
    /// Données du tenseur (N-dimensionnel)
    data: Arc<ArrayD<Float>>,
    
    /// Gradient accumulé (même shape que data)
    grad: Arc<RefCell<Option<ArrayD<Float>>>>,
    
    /// Ce tenseur nécessite-t-il un gradient ?
    requires_grad: bool,
    
    /// Fonction pour calculer le gradient lors du backward
    /// None pour les tenseurs leaf (créés directement, pas par une opération)
    grad_fn: Option<Arc<dyn GradFunction>>,
    
    /// Est-ce un tenseur créé directement par l'utilisateur ?
    is_leaf: bool,
    
    /// Conserver le gradient même si non-leaf ?
    retain_grad: bool,
}
```

### 2.2 Constructeurs

```rust
impl Tensor {
    /// Crée un tensor à partir de données existantes
    /// 
    /// # Example
    /// ```rust
    /// use ndarray::array;
    /// let t = Tensor::from_data(array![1.0, 2.0, 3.0].into_dyn(), true);
    /// assert!(t.requires_grad());
    /// assert!(t.is_leaf());
    /// ```
    pub fn from_data(data: ArrayD<Float>, requires_grad: bool) -> Self {
        Self {
            data: Arc::new(data),
            grad: Arc::new(RefCell::new(None)),
            requires_grad,
            grad_fn: None,
            is_leaf: true,
            retain_grad: false,
        }
    }
    
    /// Crée un tensor résultat d'une opération (usage interne)
    pub(crate) fn from_op(
        data: ArrayD<Float>,
        requires_grad: bool,
        grad_fn: Option<Arc<dyn GradFunction>>,
    ) -> Self {
        Self {
            data: Arc::new(data),
            grad: Arc::new(RefCell::new(None)),
            requires_grad,
            grad_fn,
            is_leaf: false,
            retain_grad: false,
        }
    }
    
    /// Tensor rempli de zéros
    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        Self::from_data(ArrayD::zeros(IxDyn(shape)), requires_grad)
    }
    
    /// Tensor rempli de uns
    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        Self::from_data(ArrayD::ones(IxDyn(shape)), requires_grad)
    }
    
    /// Tensor avec valeurs aléatoires ~ N(0, 1)
    pub fn randn(shape: &[usize], requires_grad: bool) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let size: usize = shape.iter().product();
        let data: Vec<Float> = (0..size)
            .map(|_| {
                // Box-Muller transform pour distribution normale
                let u1: Float = rng.random();
                let u2: Float = rng.random();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();
        
        Self::from_data(
            ArrayD::from_shape_vec(IxDyn(shape), data).unwrap(),
            requires_grad,
        )
    }
    
    /// Tensor avec valeurs aléatoires ~ U(low, high)
    pub fn uniform(shape: &[usize], low: Float, high: Float, requires_grad: bool) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let size: usize = shape.iter().product();
        let data: Vec<Float> = (0..size)
            .map(|_| rng.random::<Float>() * (high - low) + low)
            .collect();
        
        Self::from_data(
            ArrayD::from_shape_vec(IxDyn(shape), data).unwrap(),
            requires_grad,
        )
    }
    
    /// Xavier/Glorot initialization
    pub fn xavier_uniform(shape: &[usize], requires_grad: bool) -> Self {
        let fan_in = shape[..shape.len()-1].iter().product::<usize>();
        let fan_out = shape[shape.len()-1];
        let limit = (6.0 / (fan_in + fan_out) as Float).sqrt();
        Self::uniform(shape, -limit, limit, requires_grad)
    }
    
    /// Kaiming/He initialization (pour ReLU)
    pub fn kaiming_uniform(shape: &[usize], requires_grad: bool) -> Self {
        let fan_in: usize = shape[..shape.len()-1].iter().product();
        let limit = (6.0 / fan_in as Float).sqrt();
        Self::uniform(shape, -limit, limit, requires_grad)
    }
}
```

### 2.3 Accesseurs

```rust
impl Tensor {
    /// Accès aux données (lecture seule)
    #[inline]
    pub fn data(&self) -> &ArrayD<Float> {
        &self.data
    }
    
    /// Shape du tensor
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
    
    /// Nombre de dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }
    
    /// Nombre total d'éléments
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }
    
    /// Le tensor nécessite-t-il un gradient ?
    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    /// Est-ce un tensor leaf ?
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }
    
    /// Récupère le gradient (si calculé)
    pub fn grad(&self) -> Option<ArrayD<Float>> {
        self.grad.borrow().clone()
    }
    
    /// Référence vers grad_fn
    pub fn grad_fn(&self) -> Option<&Arc<dyn GradFunction>> {
        self.grad_fn.as_ref()
    }
}
```

### 2.4 Méthodes de contrôle

```rust
impl Tensor {
    /// Demande à conserver le gradient même si non-leaf
    pub fn retain_grad(&mut self) {
        self.retain_grad = true;
    }
    
    /// Remet le gradient à zéro
    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = None;
    }
    
    /// Accumule un gradient (usage interne)
    pub(crate) fn accumulate_grad(&self, grad: &ArrayD<Float>) {
        let mut grad_ref = self.grad.borrow_mut();
        match &mut *grad_ref {
            Some(existing) => *existing = &*existing + grad,
            None => *grad_ref = Some(grad.clone()),
        }
    }
    
    /// Détache du graphe de calcul
    /// 
    /// Retourne un nouveau tensor sans grad_fn, utile pour :
    /// - Stopper la propagation du gradient
    /// - Créer une copie "figée" pour l'inférence
    pub fn detach(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            grad: Arc::new(RefCell::new(None)),
            requires_grad: false,
            grad_fn: None,
            is_leaf: true,
            retain_grad: false,
        }
    }
    
    /// Clone profond (copie les données)
    pub fn deep_clone(&self) -> Self {
        Self::from_data((*self.data).clone(), self.requires_grad)
    }
    
    /// Active/désactive requires_grad
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }
}
```

---

## 3. Système de Gradient Functions

### 3.1 Trait GradFunction

```rust
use std::fmt::Debug;
use std::sync::Arc;

/// Trait pour les fonctions de gradient
/// 
/// Chaque opération différentiable implémente ce trait.
/// Le trait est object-safe pour stockage dynamique dans grad_fn.
pub trait GradFunction: Send + Sync + Debug {
    /// Calcule les gradients par rapport aux inputs
    /// 
    /// # Arguments
    /// * `grad_output` - ∂L/∂y où y est la sortie de cette opération
    /// 
    /// # Returns
    /// Vecteur de gradients, un par input. None si l'input n'a pas besoin de gradient.
    /// 
    /// # Chain Rule
    /// Pour y = f(x₁, x₂, ...) :
    /// ∂L/∂xᵢ = ∂L/∂y × ∂y/∂xᵢ
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>>;
    
    /// Retourne les références vers les inputs
    fn inputs(&self) -> Vec<&Tensor>;
    
    /// Nom de l'opération (debug)
    fn name(&self) -> &'static str;
}
```

### 3.2 Exemples d'implémentation

#### Addition

```rust
/// y = a + b
/// ∂y/∂a = 1, ∂y/∂b = 1
#[derive(Debug)]
pub struct AddBackward {
    input_a: Tensor,
    input_b: Tensor,
}

impl GradFunction for AddBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let grad_a = if self.input_a.requires_grad() {
            Some(handle_broadcast_backward(grad_output, self.input_a.shape()))
        } else {
            None
        };
        
        let grad_b = if self.input_b.requires_grad() {
            Some(handle_broadcast_backward(grad_output, self.input_b.shape()))
        } else {
            None
        };
        
        vec![grad_a, grad_b]
    }
    
    fn inputs(&self) -> Vec<&Tensor> {
        vec![&self.input_a, &self.input_b]
    }
    
    fn name(&self) -> &'static str { "AddBackward" }
}
```

#### Multiplication

```rust
/// y = a * b
/// ∂y/∂a = b, ∂y/∂b = a
#[derive(Debug)]
pub struct MulBackward {
    input_a: Tensor,
    input_b: Tensor,
}

impl GradFunction for MulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let grad_a = if self.input_a.requires_grad() {
            let grad = grad_output.data() * self.input_b.data();
            Some(handle_broadcast_backward(
                &Tensor::from_data(grad, false),
                self.input_a.shape(),
            ))
        } else {
            None
        };
        
        let grad_b = if self.input_b.requires_grad() {
            let grad = grad_output.data() * self.input_a.data();
            Some(handle_broadcast_backward(
                &Tensor::from_data(grad, false),
                self.input_b.shape(),
            ))
        } else {
            None
        };
        
        vec![grad_a, grad_b]
    }
    
    fn inputs(&self) -> Vec<&Tensor> {
        vec![&self.input_a, &self.input_b]
    }
    
    fn name(&self) -> &'static str { "MulBackward" }
}
```

#### Multiplication matricielle

```rust
/// y = A @ B  (matmul)
/// ∂y/∂A = grad_output @ Bᵀ
/// ∂y/∂B = Aᵀ @ grad_output
#[derive(Debug)]
pub struct MatMulBackward {
    input_a: Tensor,  // Shape: [..., m, k]
    input_b: Tensor,  // Shape: [..., k, n]
}

impl GradFunction for MatMulBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // grad_output shape: [..., m, n]
        
        let grad_a = if self.input_a.requires_grad() {
            // ∂L/∂A = grad_output @ Bᵀ
            let b_t = transpose_last_two(&self.input_b);
            let grad = matmul_impl(grad_output.data(), b_t.data());
            Some(Tensor::from_data(grad, false))
        } else {
            None
        };
        
        let grad_b = if self.input_b.requires_grad() {
            // ∂L/∂B = Aᵀ @ grad_output
            let a_t = transpose_last_two(&self.input_a);
            let grad = matmul_impl(a_t.data(), grad_output.data());
            Some(Tensor::from_data(grad, false))
        } else {
            None
        };
        
        vec![grad_a, grad_b]
    }
    
    fn inputs(&self) -> Vec<&Tensor> {
        vec![&self.input_a, &self.input_b]
    }
    
    fn name(&self) -> &'static str { "MatMulBackward" }
}
```

### 3.3 Gestion du broadcasting

```rust
/// Réduit le gradient pour correspondre à la shape originale après broadcasting
fn handle_broadcast_backward(grad: &Tensor, target_shape: &[usize]) -> Tensor {
    let grad_shape = grad.shape();
    
    if grad_shape == target_shape {
        return grad.clone();
    }
    
    let mut result = grad.data().clone();
    
    // Dimensions supplémentaires à gauche
    while result.ndim() > target_shape.len() {
        result = result.sum_axis(ndarray::Axis(0));
    }
    
    // Dimensions broadcastées (taille 1 → taille n)
    for (i, (&grad_dim, &target_dim)) in 
        result.shape().iter().zip(target_shape).enumerate().rev() 
    {
        if target_dim == 1 && grad_dim > 1 {
            result = result.sum_axis(ndarray::Axis(i));
            result = result.insert_axis(ndarray::Axis(i));
        }
    }
    
    Tensor::from_data(result, false)
}
```

---

## 4. Moteur de Backpropagation

### 4.1 Contexte d'exécution

```rust
use std::cell::RefCell;

thread_local! {
    static GRAD_ENABLED: RefCell<bool> = RefCell::new(true);
}

/// Vérifie si le gradient est activé
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| *g.borrow())
}

/// Désactive temporairement le calcul des gradients
/// 
/// # Example
/// ```rust
/// let x = Tensor::randn(&[3, 3], true);
/// 
/// // Avec gradient : grad_fn sera défini
/// let y = relu(&x);
/// assert!(y.grad_fn().is_some());
/// 
/// // Sans gradient : grad_fn sera None
/// let z = no_grad(|| relu(&x));
/// assert!(z.grad_fn().is_none());
/// ```
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    GRAD_ENABLED.with(|g| {
        let prev = *g.borrow();
        *g.borrow_mut() = false;
        let result = f();
        *g.borrow_mut() = prev;
        result
    })
}

/// Active temporairement le calcul des gradients
pub fn enable_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    GRAD_ENABLED.with(|g| {
        let prev = *g.borrow();
        *g.borrow_mut() = true;
        let result = f();
        *g.borrow_mut() = prev;
        result
    })
}
```

### 4.2 Backward Engine

```rust
impl Tensor {
    /// Lance la backpropagation depuis ce tensor
    /// 
    /// # Panics
    /// - Si requires_grad est false
    /// - Si le tensor n'est pas scalaire (utiliser backward_with_grad sinon)
    pub fn backward(&self) {
        assert!(
            self.requires_grad,
            "backward() appelé sur un tensor sans requires_grad"
        );
        assert_eq!(
            self.numel(), 1,
            "backward() ne supporte que les scalaires. \
             Utilisez backward_with_grad() pour les tensors non-scalaires."
        );
        
        // Initialise le gradient de sortie à 1.0
        let initial_grad = ArrayD::ones(self.shape());
        self.backward_with_grad(&Tensor::from_data(initial_grad, false));
    }
    
    /// Backward avec un gradient explicite
    pub fn backward_with_grad(&self, grad_output: &Tensor) {
        assert!(
            self.requires_grad,
            "backward() appelé sur un tensor sans requires_grad"
        );
        assert_eq!(
            grad_output.shape(), self.shape(),
            "grad_output shape doit correspondre au tensor"
        );
        
        // Accumule le gradient initial
        self.accumulate_grad(grad_output.data());
        
        // Tri topologique inverse
        let sorted_nodes = self.topological_sort();
        
        // Backward pass
        for node in sorted_nodes {
            if let Some(ref grad_fn) = node.grad_fn {
                let node_grad = node.grad.borrow();
                if let Some(ref grad) = *node_grad {
                    let grad_tensor = Tensor::from_data(grad.clone(), false);
                    
                    // Calcule les gradients des inputs
                    let input_grads = grad_fn.backward(&grad_tensor);
                    
                    // Accumule dans les inputs
                    for (input, input_grad) in grad_fn.inputs().iter().zip(input_grads) {
                        if let Some(grad) = input_grad {
                            if input.requires_grad() {
                                input.accumulate_grad(grad.data());
                            }
                        }
                    }
                }
            }
            
            // Libère le gradient si non-leaf et retain_grad est false
            if !node.is_leaf && !node.retain_grad {
                *node.grad.borrow_mut() = None;
            }
        }
    }
    
    /// Tri topologique du graphe
    fn topological_sort(&self) -> Vec<Tensor> {
        use std::collections::HashSet;
        
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        
        fn visit(
            node: &Tensor,
            visited: &mut HashSet<usize>,
            sorted: &mut Vec<Tensor>,
        ) {
            let node_id = Arc::as_ptr(&node.data) as usize;
            
            if visited.contains(&node_id) {
                return;
            }
            visited.insert(node_id);
            
            if let Some(ref grad_fn) = node.grad_fn {
                for input in grad_fn.inputs() {
                    visit(input, visited, sorted);
                }
            }
            
            sorted.push(node.clone());
        }
        
        visit(self, &mut visited, &mut sorted);
        sorted.reverse(); // Du output vers les inputs
        sorted
    }
}
```

### 4.3 Construction conditionnelle du graphe

```rust
/// Helper pour créer une GradFunction uniquement si nécessaire
pub(crate) fn maybe_grad_fn<G: GradFunction + 'static>(
    inputs: &[&Tensor],
    builder: impl FnOnce() -> G,
) -> Option<Arc<dyn GradFunction>> {
    if !is_grad_enabled() {
        return None;
    }
    
    if inputs.iter().any(|t| t.requires_grad()) {
        Some(Arc::new(builder()))
    } else {
        None
    }
}

/// Détermine requires_grad du résultat
pub(crate) fn compute_requires_grad(inputs: &[&Tensor]) -> bool {
    is_grad_enabled() && inputs.iter().any(|t| t.requires_grad())
}
```

---

## 5. Implémentation des Opérations

### 5.1 Opérations de base

```rust
/// Addition de tensors
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.data() + b.data();
    
    let grad_fn = maybe_grad_fn(&[a, b], || AddBackward {
        input_a: a.clone(),
        input_b: b.clone(),
    });
    
    Tensor::from_op(result, compute_requires_grad(&[a, b]), grad_fn)
}

/// Multiplication élément par élément
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    let result = a.data() * b.data();
    
    let grad_fn = maybe_grad_fn(&[a, b], || MulBackward {
        input_a: a.clone(),
        input_b: b.clone(),
    });
    
    Tensor::from_op(result, compute_requires_grad(&[a, b]), grad_fn)
}

/// Multiplication matricielle
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let result = matmul_impl(a.data(), b.data());
    
    let grad_fn = maybe_grad_fn(&[a, b], || MatMulBackward {
        input_a: a.clone(),
        input_b: b.clone(),
    });
    
    Tensor::from_op(result, compute_requires_grad(&[a, b]), grad_fn)
}
```

### 5.2 Activations

```rust
/// ReLU: max(0, x)
pub fn relu(input: &Tensor) -> Tensor {
    let result = input.data().mapv(|x| x.max(0.0));
    
    let grad_fn = maybe_grad_fn(&[input], || ReluBackward {
        input: input.clone(),
    });
    
    Tensor::from_op(result.into_dyn(), compute_requires_grad(&[input]), grad_fn)
}

#[derive(Debug)]
struct ReluBackward { input: Tensor }

impl GradFunction for ReluBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // ∂ReLU/∂x = 1 si x > 0, 0 sinon
        let mask = self.input.data().mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        let grad = grad_output.data() * &mask;
        vec![Some(Tensor::from_data(grad, false))]
    }
    
    fn inputs(&self) -> Vec<&Tensor> { vec![&self.input] }
    fn name(&self) -> &'static str { "ReluBackward" }
}

/// Sigmoid: 1 / (1 + exp(-x))
pub fn sigmoid(input: &Tensor) -> Tensor {
    let result = input.data().mapv(|x| 1.0 / (1.0 + (-x).exp()));
    
    let grad_fn = maybe_grad_fn(&[input], || SigmoidBackward {
        output: result.clone(),
    });
    
    Tensor::from_op(result.into_dyn(), compute_requires_grad(&[input]), grad_fn)
}

#[derive(Debug)]
struct SigmoidBackward { output: ArrayD<Float> }

impl GradFunction for SigmoidBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // ∂σ/∂x = σ × (1 - σ)
        let grad = grad_output.data() * &self.output * &(1.0 - &self.output);
        vec![Some(Tensor::from_data(grad, false))]
    }
    
    fn inputs(&self) -> Vec<&Tensor> { vec![] }
    fn name(&self) -> &'static str { "SigmoidBackward" }
}

/// Tanh
pub fn tanh(input: &Tensor) -> Tensor {
    let result = input.data().mapv(|x| x.tanh());
    
    let grad_fn = maybe_grad_fn(&[input], || TanhBackward {
        output: result.clone(),
    });
    
    Tensor::from_op(result.into_dyn(), compute_requires_grad(&[input]), grad_fn)
}

#[derive(Debug)]
struct TanhBackward { output: ArrayD<Float> }

impl GradFunction for TanhBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // ∂tanh/∂x = 1 - tanh²(x)
        let grad = grad_output.data() * &(1.0 - &self.output * &self.output);
        vec![Some(Tensor::from_data(grad, false))]
    }
    
    fn inputs(&self) -> Vec<&Tensor> { vec![] }
    fn name(&self) -> &'static str { "TanhBackward" }
}

/// Softmax sur une dimension
pub fn softmax(input: &Tensor, dim: i32) -> Tensor {
    let dim = normalize_dim(dim, input.ndim());
    
    // Stabilité numérique : soustrait le max
    let max_vals = input.data().map_axis(Axis(dim), |lane| {
        lane.iter().cloned().fold(Float::NEG_INFINITY, Float::max)
    });
    let shifted = input.data() - &max_vals.insert_axis(Axis(dim));
    let exp_vals = shifted.mapv(|x| x.exp());
    let sum_exp = exp_vals.sum_axis(Axis(dim));
    let result = &exp_vals / &sum_exp.insert_axis(Axis(dim));
    
    let grad_fn = maybe_grad_fn(&[input], || SoftmaxBackward {
        output: result.clone(),
        dim,
    });
    
    Tensor::from_op(result.into_dyn(), compute_requires_grad(&[input]), grad_fn)
}

#[derive(Debug)]
struct SoftmaxBackward { output: ArrayD<Float>, dim: usize }

impl GradFunction for SoftmaxBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // ∂L/∂x = softmax × (∂L/∂y - sum(∂L/∂y × softmax))
        let dot = (grad_output.data() * &self.output).sum_axis(Axis(self.dim));
        let grad = &self.output * &(grad_output.data() - &dot.insert_axis(Axis(self.dim)));
        vec![Some(Tensor::from_data(grad, false))]
    }
    
    fn inputs(&self) -> Vec<&Tensor> { vec![] }
    fn name(&self) -> &'static str { "SoftmaxBackward" }
}
```

### 5.3 Convolution 2D

```rust
/// Convolution 2D
/// 
/// # Arguments
/// * `input` - [batch, in_channels, height, width]
/// * `weight` - [out_channels, in_channels, kH, kW]
/// * `bias` - Option<[out_channels]>
/// * `stride` - Pas de déplacement
/// * `padding` - Zero-padding
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Tensor {
    assert_eq!(input.ndim(), 4, "Input must be 4D [N, C, H, W]");
    assert_eq!(weight.ndim(), 4, "Weight must be 4D [Out, In, kH, kW]");
    assert_eq!(
        input.shape()[1], weight.shape()[1],
        "Input channels must match weight in_channels"
    );
    
    let result = conv2d_forward_im2col(
        input.data(),
        weight.data(),
        bias.map(|b| b.data()),
        stride,
        padding,
    );
    
    let mut inputs: Vec<&Tensor> = vec![input, weight];
    if let Some(b) = bias {
        inputs.push(b);
    }
    
    let grad_fn = maybe_grad_fn(&inputs, || Conv2DBackward {
        input: input.clone(),
        weight: weight.clone(),
        bias: bias.cloned(),
        stride,
        padding,
    });
    
    Tensor::from_op(result, compute_requires_grad(&inputs), grad_fn)
}

#[derive(Debug)]
struct Conv2DBackward {
    input: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl GradFunction for Conv2DBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let mut grads = Vec::new();
        
        // ∂L/∂input via col2im
        if self.input.requires_grad() {
            let grad_input = conv2d_backward_input(
                grad_output.data(),
                self.weight.data(),
                self.input.shape(),
                self.stride,
                self.padding,
            );
            grads.push(Some(Tensor::from_data(grad_input, false)));
        } else {
            grads.push(None);
        }
        
        // ∂L/∂weight
        if self.weight.requires_grad() {
            let grad_weight = conv2d_backward_weight(
                grad_output.data(),
                self.input.data(),
                self.weight.shape(),
                self.stride,
                self.padding,
            );
            grads.push(Some(Tensor::from_data(grad_weight, false)));
        } else {
            grads.push(None);
        }
        
        // ∂L/∂bias = sum(grad_output) sur (N, H, W)
        if let Some(ref bias) = self.bias {
            if bias.requires_grad() {
                let grad_bias = grad_output.data()
                    .sum_axis(Axis(0))  // batch
                    .sum_axis(Axis(1))  // height
                    .sum_axis(Axis(1)); // width
                grads.push(Some(Tensor::from_data(grad_bias.into_dyn(), false)));
            } else {
                grads.push(None);
            }
        }
        
        grads
    }
    
    fn inputs(&self) -> Vec<&Tensor> {
        let mut inputs = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            inputs.push(b);
        }
        inputs
    }
    
    fn name(&self) -> &'static str { "Conv2DBackward" }
}
```

### 5.4 Max Pooling

```rust
/// Max Pooling 2D
pub fn max_pool2d(input: &Tensor, kernel_size: usize, stride: usize) -> Tensor {
    assert_eq!(input.ndim(), 4, "Input must be 4D [N, C, H, W]");
    
    let (result, indices) = maxpool2d_forward_with_indices(
        input.data(),
        kernel_size,
        stride,
    );
    
    let grad_fn = maybe_grad_fn(&[input], || MaxPool2DBackward {
        input_shape: input.shape().to_vec(),
        indices,
        kernel_size,
        stride,
    });
    
    Tensor::from_op(result, compute_requires_grad(&[input]), grad_fn)
}

#[derive(Debug)]
struct MaxPool2DBackward {
    input_shape: Vec<usize>,
    indices: ArrayD<usize>,
    kernel_size: usize,
    stride: usize,
}

impl GradFunction for MaxPool2DBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Le gradient passe uniquement par les positions max
        let grad_input = maxpool2d_backward_with_indices(
            grad_output.data(),
            &self.indices,
            &self.input_shape,
            self.kernel_size,
            self.stride,
        );
        
        vec![Some(Tensor::from_data(grad_input, false))]
    }
    
    fn inputs(&self) -> Vec<&Tensor> { vec![] }
    fn name(&self) -> &'static str { "MaxPool2DBackward" }
}
```

### 5.5 Batch Normalization

```rust
/// Batch Normalization 2D
/// 
/// y = gamma × (x - μ) / √(σ² + ε) + beta
pub fn batch_norm2d(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    running_mean: Option<&mut ArrayD<Float>>,
    running_var: Option<&mut ArrayD<Float>>,
    training: bool,
    momentum: Float,
    eps: Float,
) -> Tensor {
    let channels = input.shape()[1];
    assert_eq!(gamma.numel(), channels);
    assert_eq!(beta.numel(), channels);
    
    if training {
        // Calcule statistiques sur le batch actuel
        let (result, mean, var, normalized) = batchnorm_forward_training(
            input.data(),
            gamma.data(),
            beta.data(),
            eps,
        );
        
        // Met à jour running stats si fournis
        if let (Some(rm), Some(rv)) = (running_mean, running_var) {
            *rm = &*rm * (1.0 - momentum) + &mean * momentum;
            *rv = &*rv * (1.0 - momentum) + &var * momentum;
        }
        
        let grad_fn = maybe_grad_fn(&[input, gamma, beta], || BatchNorm2DBackward {
            input: input.clone(),
            gamma: gamma.clone(),
            normalized,
            std_inv: var.mapv(|v| 1.0 / (v + eps).sqrt()),
            eps,
        });
        
        Tensor::from_op(result, compute_requires_grad(&[input, gamma, beta]), grad_fn)
    } else {
        // Inference : utilise running stats
        let rm = running_mean.expect("running_mean requis en mode inference");
        let rv = running_var.expect("running_var requis en mode inference");
        
        let result = batchnorm_forward_inference(
            input.data(),
            gamma.data(),
            beta.data(),
            rm,
            rv,
            eps,
        );
        
        Tensor::from_data(result, false)
    }
}

#[derive(Debug)]
struct BatchNorm2DBackward {
    input: Tensor,
    gamma: Tensor,
    normalized: ArrayD<Float>,
    std_inv: ArrayD<Float>,
    eps: Float,
}

impl GradFunction for BatchNorm2DBackward {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        let shape = self.input.shape();
        let n = (shape[0] * shape[2] * shape[3]) as Float;
        
        // ∂L/∂gamma = sum(grad × normalized) sur (N, H, W)
        let grad_gamma = (grad_output.data() * &self.normalized)
            .sum_axis(Axis(0)).sum_axis(Axis(1)).sum_axis(Axis(1));
        
        // ∂L/∂beta = sum(grad) sur (N, H, W)
        let grad_beta = grad_output.data()
            .sum_axis(Axis(0)).sum_axis(Axis(1)).sum_axis(Axis(1));
        
        // ∂L/∂x (formule complète)
        let grad_input = if self.input.requires_grad() {
            let gamma_bc = broadcast_to_nchw(self.gamma.data(), shape);
            let std_inv_bc = broadcast_to_nchw(&self.std_inv, shape);
            
            // Simplification de la formule BatchNorm backward
            let term1 = grad_output.data() * &gamma_bc * &std_inv_bc;
            let term2 = broadcast_to_nchw(&grad_gamma, shape) * &self.normalized / n;
            let term3 = broadcast_to_nchw(&grad_beta, shape) / n;
            
            Some(Tensor::from_data(&term1 - &term2 - &term3, false))
        } else {
            None
        };
        
        vec![
            grad_input,
            Some(Tensor::from_data(grad_gamma.into_dyn(), false)),
            Some(Tensor::from_data(grad_beta.into_dyn(), false)),
        ]
    }
    
    fn inputs(&self) -> Vec<&Tensor> {
        vec![&self.input, &self.gamma]
    }
    
    fn name(&self) -> &'static str { "BatchNorm2DBackward" }
}
```

---

## 6. Modules et Couches

### 6.1 Trait Module

```rust
/// Trait pour les modules de réseau de neurones
pub trait Module {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Tensor;
    
    /// Retourne les paramètres du module
    fn parameters(&self) -> Vec<Parameter>;
    
    /// Met le module en mode training
    fn train(&mut self);
    
    /// Met le module en mode evaluation
    fn eval(&mut self);
}
```

### 6.2 Parameter

```rust
/// Wrapper pour un tensor trainable
#[derive(Clone)]
pub struct Parameter {
    tensor: Tensor,
}

impl Parameter {
    pub fn new(tensor: Tensor) -> Self {
        let mut t = tensor;
        t.set_requires_grad(true);
        Self { tensor: t }
    }
    
    pub fn data(&self) -> &ArrayD<Float> { self.tensor.data() }
    pub fn grad(&self) -> Option<ArrayD<Float>> { self.tensor.grad() }
    pub fn zero_grad(&self) { self.tensor.zero_grad(); }
    
    pub fn set_data(&mut self, data: ArrayD<Float>) {
        self.tensor = Tensor::from_data(data, true);
    }
}

impl std::ops::Deref for Parameter {
    type Target = Tensor;
    fn deref(&self) -> &Self::Target { &self.tensor }
}
```

### 6.3 Exemples de modules

```rust
/// Couche linéaire (Dense)
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        Self {
            weight: Parameter::new(Tensor::kaiming_uniform(&[in_features, out_features], true)),
            bias: if bias {
                Some(Parameter::new(Tensor::zeros(&[out_features], true)))
            } else {
                None
            },
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let y = matmul(input, &self.weight);
        match &self.bias {
            Some(b) => add(&y, b),
            None => y,
        }
    }
    
    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
    
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

/// Module Conv2D
pub struct Conv2d {
    weight: Parameter,
    bias: Option<Parameter>,
    stride: usize,
    padding: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        Self {
            weight: Parameter::new(Tensor::kaiming_uniform(
                &[out_channels, in_channels, kernel_size, kernel_size],
                true,
            )),
            bias: if bias {
                Some(Parameter::new(Tensor::zeros(&[out_channels], true)))
            } else {
                None
            },
            stride,
            padding,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        conv2d(
            input,
            &self.weight,
            self.bias.as_ref().map(|b| &**b),
            self.stride,
            self.padding,
        )
    }
    
    fn parameters(&self) -> Vec<Parameter> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
    
    fn train(&mut self) {}
    fn eval(&mut self) {}
}
```

---

## 7. Optimizers

### 7.1 Trait Optimizer

```rust
pub trait Optimizer {
    /// Effectue une étape d'optimisation
    fn step(&mut self);
    
    /// Remet tous les gradients à zéro
    fn zero_grad(&self);
    
    /// Learning rate actuel
    fn get_lr(&self) -> Float;
    
    /// Modifie le learning rate
    fn set_lr(&mut self, lr: Float);
}
```

### 7.2 SGD

```rust
pub struct SGD {
    params: Vec<Parameter>,
    lr: Float,
    momentum: Float,
    weight_decay: Float,
    velocities: Vec<Option<ArrayD<Float>>>,
}

impl SGD {
    pub fn new(params: Vec<Parameter>, lr: Float) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            velocities: vec![None; n],
        }
    }
    
    pub fn momentum(mut self, m: Float) -> Self { self.momentum = m; self }
    pub fn weight_decay(mut self, wd: Float) -> Self { self.weight_decay = wd; self }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(mut grad) = param.grad() {
                // Weight decay
                if self.weight_decay > 0.0 {
                    grad = grad + &(param.data() * self.weight_decay);
                }
                
                // Momentum
                if self.momentum > 0.0 {
                    let v = match &self.velocities[i] {
                        Some(v) => v * self.momentum + &grad,
                        None => grad.clone(),
                    };
                    self.velocities[i] = Some(v.clone());
                    grad = v;
                }
                
                // Update
                let new_data = param.data() - &(&grad * self.lr);
                param.set_data(new_data);
            }
        }
    }
    
    fn zero_grad(&self) {
        for p in &self.params { p.zero_grad(); }
    }
    
    fn get_lr(&self) -> Float { self.lr }
    fn set_lr(&mut self, lr: Float) { self.lr = lr; }
}
```

### 7.3 Adam

```rust
pub struct Adam {
    params: Vec<Parameter>,
    lr: Float,
    betas: (Float, Float),
    eps: Float,
    weight_decay: Float,
    m: Vec<ArrayD<Float>>,
    v: Vec<ArrayD<Float>>,
    t: usize,
}

impl Adam {
    pub fn new(params: Vec<Parameter>, lr: Float) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            m: vec![ArrayD::zeros(IxDyn(&[])); n],
            v: vec![ArrayD::zeros(IxDyn(&[])); n],
            t: 0,
        }
    }
    
    pub fn betas(mut self, b1: Float, b2: Float) -> Self { self.betas = (b1, b2); self }
    pub fn eps(mut self, e: Float) -> Self { self.eps = e; self }
    pub fn weight_decay(mut self, wd: Float) -> Self { self.weight_decay = wd; self }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        self.t += 1;
        
        let (b1, b2) = self.betas;
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        
        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = param.grad() {
                // Init m, v si nécessaire
                if self.m[i].is_empty() {
                    self.m[i] = ArrayD::zeros(grad.dim());
                    self.v[i] = ArrayD::zeros(grad.dim());
                }
                
                // Weight decay (AdamW)
                let grad = if self.weight_decay > 0.0 {
                    grad + &(param.data() * self.weight_decay)
                } else {
                    grad
                };
                
                // m = β₁m + (1-β₁)g
                self.m[i] = &self.m[i] * b1 + &grad * (1.0 - b1);
                
                // v = β₂v + (1-β₂)g²
                self.v[i] = &self.v[i] * b2 + &grad.mapv(|g| g * g) * (1.0 - b2);
                
                // Bias correction
                let m_hat = &self.m[i] / bc1;
                let v_hat = &self.v[i] / bc2;
                
                // θ = θ - lr × m̂ / (√v̂ + ε)
                let update = &m_hat / &(v_hat.mapv(|v| v.sqrt()) + self.eps);
                let new_data = param.data() - &(&update * self.lr);
                param.set_data(new_data);
            }
        }
    }
    
    fn zero_grad(&self) {
        for p in &self.params { p.zero_grad(); }
    }
    
    fn get_lr(&self) -> Float { self.lr }
    fn set_lr(&mut self, lr: Float) { self.lr = lr; }
}
```

---

## 8. Gestion Mémoire

### 8.1 Gradient Checkpointing

```rust
/// Réexécute le forward pendant le backward pour économiser la mémoire
/// 
/// Trade-off : ~50% moins de mémoire, ~30% plus de calcul
pub fn checkpoint<F>(input: &Tensor, f: F) -> Tensor
where
    F: Fn(&Tensor) -> Tensor + Clone + 'static,
{
    // Forward sans gradient
    let output = no_grad(|| f(input));
    
    let grad_fn = maybe_grad_fn(&[input], || CheckpointBackward {
        input: input.clone(),
        forward_fn: f,
    });
    
    Tensor::from_op(output.data().clone(), input.requires_grad(), grad_fn)
}

struct CheckpointBackward<F: Fn(&Tensor) -> Tensor> {
    input: Tensor,
    forward_fn: F,
}

impl<F: Fn(&Tensor) -> Tensor + 'static> GradFunction for CheckpointBackward<F> {
    fn backward(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        // Réexécute le forward avec gradient
        let mut input_with_grad = self.input.deep_clone();
        input_with_grad.set_requires_grad(true);
        
        let output = (self.forward_fn)(&input_with_grad);
        output.backward_with_grad(grad_output);
        
        vec![input_with_grad.grad().map(|g| Tensor::from_data(g, false))]
    }
    
    fn inputs(&self) -> Vec<&Tensor> { vec![&self.input] }
    fn name(&self) -> &'static str { "CheckpointBackward" }
}
```

### 8.2 Bonnes pratiques mémoire

```rust
// ✅ Désactiver le gradient pour l'inférence
let prediction = no_grad(|| model.forward(&input));

// ✅ Détacher pour couper le graphe
let features = encoder.forward(&x).detach();
let output = decoder.forward(&features);

// ✅ Libérer le gradient après usage
optimizer.zero_grad();  // Avant le nouveau backward

// ✅ Utiliser checkpoint pour les gros modèles
let output = checkpoint(&x, |x| heavy_computation(x));
```

---

## 9. API Publique

### 9.1 Prelude

```rust
/// Réexporte les types les plus utilisés
pub mod prelude {
    pub use crate::tensor::Tensor;
    pub use crate::autograd::{no_grad, enable_grad};
    pub use crate::nn::{Module, Parameter, Linear, Conv2d, Sequential};
    pub use crate::optim::{Optimizer, Adam, SGD};
    pub use crate::ops::*;
}
```

### 9.2 Exemple complet

```rust
use autograd::prelude::*;

// Définir un modèle
struct MLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl MLP {
    fn new(input_size: usize, hidden: usize, output_size: usize) -> Self {
        Self {
            fc1: Linear::new(input_size, hidden, true),
            fc2: Linear::new(hidden, hidden, true),
            fc3: Linear::new(hidden, output_size, true),
        }
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = relu(&self.fc1.forward(x));
        let x = relu(&self.fc2.forward(x));
        self.fc3.forward(&x)
    }
    
    fn parameters(&self) -> Vec<Parameter> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p.extend(self.fc3.parameters());
        p
    }
    
    fn train(&mut self) {}
    fn eval(&mut self) {}
}

fn main() {
    let model = MLP::new(784, 256, 10);
    let mut optimizer = Adam::new(model.parameters(), 0.001);
    
    // Training loop
    for epoch in 0..10 {
        for (x_batch, y_batch) in data_loader.iter() {
            // Forward
            let output = model.forward(&x_batch);
            let loss = cross_entropy_loss(&output, &y_batch);
            
            // Backward
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            println!("Loss: {:.4}", loss.data()[[0]]);
        }
    }
    
    // Inference
    let prediction = no_grad(|| model.forward(&test_input));
}
```

---

## 10. Patterns et Bonnes Pratiques

### 10.1 Patterns de conception utilisés

| Pattern | Usage |
|---------|-------|
| **Builder** | Configuration des optimizers : `Adam::new(...).betas(...).eps(...)` |
| **Factory** | Création de tensors : `Tensor::zeros()`, `Tensor::randn()` |
| **Visitor** | Parcours du graphe : `topological_sort()` |
| **Flyweight** | Partage des données : `Arc<ArrayD<Float>>` |
| **Strategy** | Différentes GradFunction par opération |

### 10.2 Checklist de debugging

```rust
// 1. Vérifier les gradients numériquement
fn check_gradient<F>(f: F, x: &Tensor, eps: Float) -> bool
where F: Fn(&Tensor) -> Tensor
{
    let analytical = {
        let y = f(x);
        y.backward();
        x.grad().unwrap()
    };
    
    let numerical = numerical_gradient(&f, x, eps);
    
    let diff = (&analytical - &numerical).mapv(|v| v.abs()).sum();
    diff < 1e-4
}

// 2. Afficher le graphe de calcul
fn print_graph(t: &Tensor, indent: usize) {
    let prefix = " ".repeat(indent);
    if let Some(gf) = t.grad_fn() {
        println!("{}{}", prefix, gf.name());
        for input in gf.inputs() {
            print_graph(input, indent + 2);
        }
    } else {
        println!("{}Leaf[{:?}]", prefix, t.shape());
    }
}

// 3. Vérifier les shapes
assert_eq!(output.shape(), expected_shape, "Shape mismatch!");

// 4. Vérifier que les gradients sont calculés
assert!(param.grad().is_some(), "Gradient not computed!");
```

### 10.3 Erreurs communes

| Erreur | Cause | Solution |
|--------|-------|----------|
| Gradients None | `no_grad` actif ou `requires_grad=false` | Vérifier le contexte et les flags |
| Gradient incorrect | Erreur dans backward | Utiliser gradient checking |
| OOM | Graphe trop grand | Utiliser `checkpoint()` ou `detach()` |
| Gradients qui explosent | Learning rate trop grand | Réduire LR, utiliser gradient clipping |

---

## 📚 Références

- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [JAX Autodiff cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)

---

*Document technique - Version 1.0*
