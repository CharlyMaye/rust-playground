# Guide de Configuration de la Visualisation de Réseaux de Neurones

Ce guide explique comment configurer la visualisation des réseaux de neurones dans l'application Angular. Le système utilise un **Builder Pattern** qui permet une configuration déclarative, composable et type-safe.

---

## Table des matières

1. [Démarrage rapide](#démarrage-rapide)
2. [Utilisation du composant](#utilisation-du-composant)
3. [Presets disponibles](#presets-disponibles)
4. [API du Builder](#api-du-builder)
5. [Configuration par couche](#configuration-par-couche)
6. [Configuration des connexions](#configuration-des-connexions)
7. [Taille des neurones](#taille-des-neurones)
8. [Canvas et viewport](#canvas-et-viewport)
9. [Interactions](#interactions)
10. [Level of Detail (LOD)](#level-of-detail-lod)
11. [Rendu](#rendu)
12. [Auto-configuration](#auto-configuration)
13. [Exemples complets](#exemples-complets)

---

## Démarrage rapide

### Méthode 1 : Utiliser un preset (recommandé)

```html
<app-configurable-network
  [architecture]="architecture"
  [weights]="weights"
  preset="small-network"
/>
```

### Méthode 2 : Configuration personnalisée

```typescript
import { NetworkVisualizationBuilder } from './config';

// Dans votre composant
readonly customConfig = NetworkVisualizationBuilder.create()
  .forLayer(0, { representation: 'heatmap', shape: [28, 28] })
  .forHiddenLayers({ representation: 'bar' })
  .forOutputLayer({ representation: 'neurons' })
  .withConnections('none')
  .build();
```

```html
<app-configurable-network
  [architecture]="architecture"
  [weights]="weights"
  [config]="customConfig"
/>
```

---

## Utilisation du composant

Le composant `<app-configurable-network>` accepte les inputs suivants :

| Input | Type | Description |
|-------|------|-------------|
| `architecture` | `NetworkArchitecture` | Structure du réseau (couches, tailles, activations) |
| `weights` | `LayerWeights[]` | Poids des connexions |
| `preset` | `PresetName` | Nom d'un preset prédéfini |
| `config` | `VisualizationConfig` | Configuration personnalisée (via Builder) |
| `autoConfig` | `boolean` | Active l'auto-configuration (défaut: `true`) |
| `debug` | `boolean` | Active le mode debug (défaut: `false`) |

### Priorité de configuration

1. **config** (priorité la plus haute) - Si fourni, utilisé directement
2. **preset** - Si fourni, génère la configuration correspondante
3. **autoConfig** - Si `true`, analyse le réseau et choisit automatiquement
4. **défaut** - Configuration par défaut si rien n'est spécifié

---

## Presets disponibles

### `small-network`
Pour les petits réseaux (XOR, AND, OR).

- **Neurones** : Affichage individuel
- **Connexions** : Toutes visibles
- **Zoom/Pan** : Désactivé
- **Taille canvas** : 500×280px

```html
<app-configurable-network preset="small-network" ... />
```

### `medium-network`
Pour les réseaux moyens (Iris, petits MLP).

- **Neurones** : Affichage individuel
- **Connexions** : Filtrées par poids (seuil > 0.15)
- **Zoom** : Activé (0.5x - 3x)
- **Taille canvas** : 600×400px

### `mnist`
Optimisé pour MNIST (784-128-64-10).

- **Couche d'entrée** : Heatmap 28×28 en niveaux de gris
- **Couches cachées** : Représentation en barre
- **Couche de sortie** : Neurones individuels
- **Connexions** : Affichées au survol uniquement
- **Zoom/Pan** : Activé
- **Renderer** : WebGL

### `cifar`
Pour CIFAR-10 (images 32×32 RGB).

- **Couche d'entrée** : Heatmap 32×32×3
- **Connexions** : Désactivées
- **Renderer** : WebGL

### `large-mlp`
Pour les grands réseaux MLP.

- **Représentation** : Barres pour toutes les couches
- **Connexions** : Désactivées
- **Canvas adaptatif** : Jusqu'à 1200×800px

### `cnn`
Pour les réseaux convolutifs.

- **Layout** : Hiérarchique
- **Représentation** : Heatmaps
- **Connexions** : Désactivées
- **Renderer** : WebGL

### `architecture-only`
Vue simplifiée pour très grands réseaux.

- **Représentation** : Couches collapsées
- **Connexions** : Aucune
- **Interactions** : Clic pour détails

### `interactive`
Exploration détaillée complète.

- **LOD activé** : 3 niveaux de détail selon le zoom
- **Connexions** : Au survol
- **Zoom/Pan** : Complet (0.1x - 10x)

### `presentation`
Pour démos et présentations.

- **Neurones échantillonnés** : Max 15 par couche
- **Connexions** : Filtrées (les plus fortes)
- **Pas d'interaction** : Statique

### `debug`
Mode développement.

- **Tout affiché** : Neurones, connexions, valeurs
- **Debug activé** : Logs et infos supplémentaires
- **Zoom max** : 20x

---

## API du Builder

### Création d'un Builder

```typescript
import { NetworkVisualizationBuilder } from './config';

// Création vide
const builder = NetworkVisualizationBuilder.create();

// À partir d'un preset
const builder = NetworkVisualizationBuilder.fromPreset('mnist');

// Auto-configuration basée sur l'architecture
const builder = NetworkVisualizationBuilder.forNetwork(architecture);
```

### Chaînage fluent

Toutes les méthodes retournent `this` pour permettre le chaînage :

```typescript
const config = NetworkVisualizationBuilder.create()
  .usePreset('medium-network')
  .withConnections('strong')
  .withConnectionThreshold(0.2)
  .forOutputLayer({ showValues: true })
  .withDebug(true)
  .build();
```

---

## Configuration par couche

### Représentations disponibles

| Type | Description | Cas d'usage |
|------|-------------|-------------|
| `'neurons'` | Cercles individuels pour chaque neurone | Petites couches (< 50) |
| `'sampled'` | Sous-ensemble de neurones | Couches moyennes |
| `'bar'` | Barre verticale colorée | Couches > 100 neurones |
| `'heatmap'` | Grille 2D de couleurs | Images (28×28, 32×32) |
| `'histogram'` | Distribution des activations | Analyse statistique |
| `'stats'` | Min/max/moyenne uniquement | Vue très compacte |
| `'collapsed'` | Rectangle simple avec dimensions | Très grands réseaux |

### Configuration d'une couche spécifique

```typescript
// Par index (0 = input, -1 = output)
.forLayer(0, {
  representation: 'heatmap',
  shape: [28, 28],
  colorScheme: 'grayscale'
})

// Dernière couche (output)
.forOutputLayer({
  representation: 'neurons',
  showValues: true
})

// Première couche (input)
.forInputLayer({
  representation: 'heatmap',
  shape: [28, 28]
})
```

### Configuration des couches cachées

```typescript
.forHiddenLayers({
  representation: 'bar',
  showLabel: true
})
```

### Règle automatique par taille

```typescript
// Couches > 100 neurones → représentation en barre
.forLargeLayers(100, {
  representation: 'bar',
  showValues: false
})
```

### Options de couche

```typescript
interface LayerConfig {
  representation: LayerRepresentation;
  sampleCount?: number;      // Pour 'sampled'
  shape?: number[];          // Pour 'heatmap' [rows, cols]
  bins?: number;             // Pour 'histogram'
  showLabel?: boolean;       // Afficher le label
  showValues?: boolean;      // Afficher les valeurs
  colorScheme?: 'default' | 'grayscale' | 'viridis' | 'coolwarm';
}
```

---

## Configuration des connexions

### Stratégies de connexions

| Stratégie | Description |
|-----------|-------------|
| `'all'` | Toutes les connexions |
| `'strong'` | Seulement poids > seuil |
| `'sampled'` | Échantillon aléatoire |
| `'on-hover'` | Affichées au survol d'un neurone |
| `'none'` | Aucune connexion |

### Exemples

```typescript
// Afficher seulement les connexions fortes
.withConnections('strong')
.withConnectionThreshold(0.15)

// Limiter le nombre de connexions
.withConnections('sampled')
.withConnectionSampling(500)

// Connexions au survol
.withConnections('on-hover')

// Désactiver les connexions
.withConnections('none')
```

### Configuration complète

```typescript
.withConnectionConfig({
  strategy: 'strong',
  threshold: 0.2,
  maxCount: 1000,
  opacity: 0.6,
  opacityByWeight: true,
  strokeWidth: 1.5,
  widthByWeight: false
})
```

---

## Taille des neurones

### Stratégies

| Stratégie | Description |
|-----------|-------------|
| `'fixed'` | Taille fixe pour tous |
| `'adaptive'` | S'adapte au nombre de neurones |
| `'by-activation'` | Taille proportionnelle à l'activation |

### Exemples

```typescript
// Taille fixe
.withFixedNeuronSize(40)

// Taille adaptative avec bornes
.withNeuronSize('adaptive')
.withNeuronSizeBounds(4, 40)

// Configuration complète
.withNeuronSizeConfig({
  strategy: 'adaptive',
  minSize: 2,
  maxSize: 50,
  fixedSize: 30  // Utilisé si strategy='fixed'
})
```

---

## Canvas et viewport

### Taille du canvas

```typescript
// Taille fixe
.withCanvasSize(800, 600)

// Remplit le conteneur
.withFillContainer()
.withFillContainer(1200, 800)  // Avec limites max

// Ratio d'aspect
.withAspectRatio(16/9)
.withAspectRatio('auto')
```

### Configuration complète

```typescript
.withCanvasConfig({
  sizeStrategy: 'fixed' | 'adaptive' | 'fill-container',
  width: 800,
  height: 600,
  aspectRatio: 16/9,
  maxWidth: 1200,
  maxHeight: 800
})
```

---

## Interactions

### Zoom

```typescript
// Activer/désactiver
.withZoom(true)
.withZoom(false)

// Configuration détaillée
.withZoom({
  enabled: true,
  min: 0.5,
  max: 5,
  step: 0.1,
  initial: 1.0
})
```

### Pan (défilement)

```typescript
.withPan(true)
.withPan({
  enabled: true,
  constrained: true  // Limite aux bords du réseau
})
```

### Survol (hover)

| Comportement | Description |
|--------------|-------------|
| `'none'` | Aucun effet |
| `'highlight'` | Surligne le neurone |
| `'connections'` | Affiche les connexions du neurone |
| `'details'` | Affiche popup avec détails |

```typescript
.withHover('connections')
```

### Clic

| Comportement | Description |
|--------------|-------------|
| `'none'` | Aucun effet |
| `'focus'` | Zoom sur le neurone/couche |
| `'expand'` | Développe une couche collapsée |
| `'info'` | Affiche informations complètes |

```typescript
.withClick('focus')
```

---

## Level of Detail (LOD)

Le LOD permet d'adapter automatiquement la visualisation selon le niveau de zoom.

```typescript
.withLOD([
  {
    zoomRange: [0, 0.5],
    layerConfig: { representation: 'collapsed' },
    connectionConfig: { strategy: 'none' }
  },
  {
    zoomRange: [0.5, 2],
    layerConfig: { representation: 'bar' },
    connectionConfig: { strategy: 'none' }
  },
  {
    zoomRange: [2, 10],
    layerConfig: { representation: 'neurons' },
    connectionConfig: { strategy: 'on-hover' }
  }
])
```

### Désactiver le LOD

```typescript
.withoutLOD()
```

---

## Rendu

### Type de renderer

| Type | Description |
|------|-------------|
| `'auto'` | Choix automatique selon complexité |
| `'canvas2d'` | Rendu Canvas 2D (petits réseaux) |
| `'webgl'` | Rendu WebGL (grands réseaux) |

```typescript
.withRenderer('webgl')
```

### Options de rendu

```typescript
.withAntialias(true)
.withDebug(true)
.withWebGLThreshold(1000)  // Bascule vers WebGL si > 1000 neurones
```

### Configuration complète

```typescript
.withRenderingConfig({
  renderer: 'auto',
  antialias: true,
  debug: false,
  webglThreshold: 500
})
```

---

## Auto-configuration

Le Builder peut analyser automatiquement un réseau et choisir la meilleure configuration :

```typescript
const config = NetworkVisualizationBuilder
  .forNetwork(architecture)
  .build();
```

### Logique d'auto-configuration

| Caractéristique | Preset appliqué |
|-----------------|-----------------|
| ≤ 20 neurones, ≤ 50 connexions | `small-network` |
| ≤ 100 neurones, ≤ 1000 connexions | `medium-network` |
| Couche de 784 neurones | `mnist` |
| Couche ≥ 100 neurones | `large-mlp` |
| Autres | `medium-network` |

---

## Exemples complets

### Réseau XOR (2-2-1)

```typescript
// Preset simple
preset = 'small-network';

// Ou configuration personnalisée équivalente
config = NetworkVisualizationBuilder.create()
  .withLayoutStrategy('column')
  .withDefaultLayerConfig({
    representation: 'neurons',
    showLabel: true,
    showValues: true
  })
  .withConnections('all')
  .withConnectionOpacity(0.7, true)
  .withFixedNeuronSize(40)
  .withCanvasSize(500, 280)
  .withZoom(false)
  .withPan(false)
  .build();
```

### Réseau Iris (4-7-7-3)

```typescript
config = NetworkVisualizationBuilder.fromPreset('medium-network')
  .withConnectionThreshold(0.2)
  .forOutputLayer({ showValues: true })
  .build();
```

### Réseau MNIST (784-128-64-10)

```typescript
config = NetworkVisualizationBuilder.create()
  .forInputLayer({
    representation: 'heatmap',
    shape: [28, 28],
    colorScheme: 'grayscale'
  })
  .forHiddenLayers({
    representation: 'bar',
    showLabel: true
  })
  .forOutputLayer({
    representation: 'neurons',
    showValues: true
  })
  .withConnections('on-hover')
  .withRenderer('webgl')
  .withZoom({ enabled: true, min: 0.3, max: 5 })
  .withPan(true)
  .build();
```

### Mode présentation

```typescript
config = NetworkVisualizationBuilder.fromPreset('presentation')
  .forLargeLayers(30, {
    representation: 'sampled',
    sampleCount: 10
  })
  .withConnections('strong')
  .withConnectionThreshold(0.3)
  .withFixedNeuronSize(50)
  .build();
```

### Mode exploration interactive

```typescript
config = NetworkVisualizationBuilder.fromPreset('interactive')
  .withLOD([
    { zoomRange: [0, 0.5], layerConfig: { representation: 'collapsed' } },
    { zoomRange: [0.5, 2], layerConfig: { representation: 'bar' } },
    { zoomRange: [2, 10], layerConfig: { representation: 'neurons' } }
  ])
  .withFillContainer(1600, 1000)
  .withDebug(false)
  .build();
```

---

## Résumé des options

| Catégorie | Méthodes principales |
|-----------|---------------------|
| **Layout** | `withLayoutStrategy()`, `withSpacing()` |
| **Couches** | `forLayer()`, `forInputLayer()`, `forOutputLayer()`, `forHiddenLayers()`, `forLargeLayers()` |
| **Connexions** | `withConnections()`, `withConnectionThreshold()`, `withConnectionSampling()` |
| **Neurones** | `withNeuronSize()`, `withFixedNeuronSize()`, `withNeuronSizeBounds()` |
| **Canvas** | `withCanvasSize()`, `withFillContainer()`, `withAspectRatio()` |
| **Interactions** | `withZoom()`, `withPan()`, `withHover()`, `withClick()` |
| **LOD** | `withLOD()`, `addLODLevel()`, `withoutLOD()` |
| **Rendu** | `withRenderer()`, `withAntialias()`, `withDebug()` |
| **Presets** | `usePreset()` |

---

## Architecture technique

Le système de visualisation utilise une architecture **Content-First** :

1. **Layout Calculator** : Calcule les positions en coordonnées naturelles
2. **Viewport** : Calcule l'échelle pour adapter au canvas
3. **Renderer** : Applique une transformation uniforme pour le rendu

Cette approche garantit que le layout est indépendant de la taille d'affichage et permet le zoom/pan sans recalcul.

### Fichiers principaux

```
config/
├── visualization-config.ts   # Types et interfaces
├── visualization-builder.ts  # Builder Pattern API
├── presets.ts                # 10 presets prédéfinis
└── index.ts                  # Exports publics

renderers/
├── configurable-layout-calculator.ts  # Calcul des positions
├── configurable-canvas2d-renderer.ts  # Rendu Canvas 2D
├── configurable-webgl-renderer.ts     # Rendu WebGL
└── index.ts                           # Exports publics
```
