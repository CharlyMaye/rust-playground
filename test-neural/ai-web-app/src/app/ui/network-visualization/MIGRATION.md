# Migration Guide: SVG → Canvas2D Renderer

Guide pour migrer du composant `NeuralNetworkModelVizualizer` (SVG) vers le nouveau `NetworkVisualization` (Canvas2D).

## Pourquoi Migrer ?

### Problèmes avec l'Ancien Système (SVG)

- ❌ **Performance**: 109K+ éléments DOM pour MNIST (très lent)
- ❌ **Scalabilité**: Ne passe pas à l'échelle pour grands réseaux
- ❌ **Complexité**: Logique de calcul et rendu mélangées
- ❌ **Pas extensible**: Impossible d'ajouter WebGL/WebGPU facilement

### Avantages du Nouveau Système (Canvas2D)

- ✅ **Performance**: 60 FPS avec 109K éléments
- ✅ **Scalabilité**: Peut gérer des réseaux beaucoup plus grands
- ✅ **Modulaire**: Séparation claire calcul/rendu
- ✅ **Extensible**: Architecture prête pour WebGL/WebGPU
- ✅ **Maintenable**: Code plus simple et organisé

## Migration Étape par Étape

### Option 1: Migration Rapide (Recommandée)

Utilisez l'adaptateur pour convertir vos données existantes.

#### Avant (SVG)

```typescript
// mnist-digit.ts
import { NeuralNetworkModelVizualizer } from '../../ui/neural-network-model-vizualizer';

@Component({
  imports: [NeuralNetworkModelVizualizer],
  template: `
    <app-neural-network-model-vizualizer
      [activations]="activations()"
      [weights]="weights()"
    />
  `
})
export class MnistDigit {
  public readonly activations = computed(() => {
    // ... calcul des activations
  });

  public readonly weights = this.wasmService.mnistWeights;
}
```

#### Après (Canvas2D)

```typescript
// mnist-digit.ts
import { NetworkVisualization } from '../../ui/network-visualization/network-visualization';
import { activationToArchitecture, neuralNetworkLayersToWeights } from '../../ui/network-visualization/adapter';

@Component({
  imports: [NetworkVisualization],
  template: `
    <app-network-visualization
      [architecture]="networkArchitecture()"
      [weights]="networkWeights()"
    />
  `
})
export class MnistDigit {
  // Convertir les activations existantes
  public readonly networkArchitecture = computed(() => {
    const acts = this.activations();
    if (!acts) return null;
    return activationToArchitecture(acts);
  });

  // Convertir les poids existants
  public readonly networkWeights = computed(() => {
    const wts = this.weights();
    if (!wts) return null;
    return neuralNetworkLayersToWeights(wts);
  });

  // Garder vos computed existants
  public readonly activations = computed(() => {
    // ... même code qu'avant
  });

  public readonly weights = this.wasmService.mnistWeights;
}
```

### Option 2: Migration Progressive (Coexistence)

Utilisez les deux composants en parallèle pendant la transition.

```typescript
@Component({
  imports: [
    NeuralNetworkModelVizualizer, // Ancien (SVG)
    NetworkVisualization,          // Nouveau (Canvas2D)
  ],
  template: `
    <!-- Ancien visualiseur (commenté mais gardé pour référence) -->
    <!-- 
    <app-neural-network-model-vizualizer
      [activations]="activations()"
      [weights]="weights()"
    />
    -->

    <!-- Nouveau visualiseur -->
    <app-network-visualization
      [architecture]="networkArchitecture()"
      [weights]="networkWeights()"
    />
  `
})
```

## Cas d'Usage Spécifiques

### XOR Network (Petit Réseau)

```typescript
// xor.component.ts
public readonly networkArchitecture = computed(() => {
  const acts = this.activations();
  if (!acts) return null;
  return activationToArchitecture(acts);
});

public readonly networkWeights = computed(() => {
  const wts = this.weights();
  if (!wts) return null;
  return neuralNetworkLayersToWeights(wts);
});
```

**Note**: Même pour les petits réseaux, le nouveau système offre une meilleure architecture.

### MNIST Network (Grand Réseau)

```typescript
// mnist-digit.component.ts
public readonly networkArchitecture = computed(() => {
  const acts = this.activations();
  if (!acts) return null;
  return activationToArchitecture(acts);
});

public readonly networkWeights = computed(() => {
  const wts = this.weights();
  if (!wts) return null;
  return neuralNetworkLayersToWeights(wts);
});
```

**Amélioration**: Passage de ~10 FPS à 60 FPS !

### Iris Network (Réseau Moyen)

```typescript
// iris.component.ts - même pattern
```

## Comparaison des Templates

### Ancien (SVG)

```html
<app-neural-network-model-vizualizer
  [activations]="activations()"
  [weights]="weights()"
/>
```

### Nouveau (Canvas2D)

```html
<app-network-visualization
  [architecture]="networkArchitecture()"
  [weights]="networkWeights()"
  [debug]="false"
/>
```

## Différences d'API

| Ancien (SVG) | Nouveau (Canvas2D) | Notes |
|--------------|-------------------|-------|
| `[activations]` | `[architecture]` | Convertir avec `activationToArchitecture()` |
| `[weights]` | `[weights]` | Convertir avec `neuralNetworkLayersToWeights()` |
| N/A | `[debug]` | Nouvelle option pour le debug |

## Checklist de Migration

### Pour Chaque Page/Composant

- [ ] Importer `NetworkVisualization` au lieu de `NeuralNetworkModelVizualizer`
- [ ] Importer les fonctions d'adaptation depuis `adapter.ts`
- [ ] Créer le computed `networkArchitecture` avec `activationToArchitecture()`
- [ ] Créer le computed `networkWeights` avec `neuralNetworkLayersToWeights()`
- [ ] Mettre à jour le template pour utiliser `<app-network-visualization>`
- [ ] Tester le rendu et la performance
- [ ] Supprimer l'ancien composant du template (optionnel)

### Composants à Migrer

- [ ] `mnist-digit.component.ts`
- [ ] `xor.component.ts` (si existant)
- [ ] `iris.component.ts` (si existant)
- [ ] Autres pages utilisant la visualisation

## Test de la Migration

### 1. Vérifier le Rendu

Comparez visuellement l'ancien et le nouveau rendu côte à côte.

```typescript
// Temporairement, affichez les deux
<app-neural-network-model-vizualizer ... />
<app-network-visualization ... />
```

### 2. Tester la Performance

Ouvrez les DevTools et vérifiez:
- FPS (devrait être 60 FPS constant)
- Temps de rendu (devrait être ~16ms)
- Utilisation mémoire (devrait être plus basse)

### 3. Tester l'Interactivité

- Le dessin de chiffres fonctionne
- Les prédictions se mettent à jour
- Le composant réagit aux changements de données

## Dépannage

### "Cannot find module './renderers'"

Solution: Vérifiez que tous les fichiers du dossier `renderers/` sont présents.

### "Canvas is undefined"

Solution: Le canvas n'est pas initialisé. Vérifiez que `@ViewChild` est configuré avec `static: false`.

### Performance toujours faible

Solution:
1. Vérifiez que vous utilisez bien le Canvas2D renderer (check console logs)
2. Activez le mode debug pour voir les stats
3. Considérez réduire `maxConnections` pour les très grands réseaux

### Couleurs incorrectes

Solution: Vérifiez que les variables CSS sont définies dans `styles.scss`:
```css
--nn-positive, --nn-negative, --nn-neutral, --nn-stroke, --nn-label
```

## Rollback (Si Nécessaire)

Si vous devez revenir en arrière:

```typescript
// 1. Restaurer l'import
import { NeuralNetworkModelVizualizer } from '../../ui/neural-network-model-vizualizer';

// 2. Restaurer le template
<app-neural-network-model-vizualizer
  [activations]="activations()"
  [weights]="weights()"
/>

// 3. Supprimer les computed d'adaptation
// Gardez vos computed originaux
```

## Support & Questions

Si vous rencontrez des problèmes:

1. Consultez [README.md](./README.md) pour l'utilisation de base
2. Consultez [RENDERING_ARCHITECTURE.md](./RENDERING_ARCHITECTURE.md) pour l'architecture
3. Vérifiez les erreurs dans la console du navigateur
4. Activez le mode debug: `[debug]="true"`

## Prochaines Étapes

Après la migration:

1. **Supprimez l'ancien code** (optionnel): Une fois que tout fonctionne, vous pouvez supprimer `neural-network-model-vizualizer/`
2. **Optimisez**: Ajustez les configurations pour vos besoins spécifiques
3. **Étendez**: Ajoutez WebGL si nécessaire pour des réseaux encore plus grands
4. **Personnalisez**: Ajoutez des interactions (hover, click, zoom)

## Timeline Suggérée

- **Jour 1**: Migrer XOR (petit réseau, test simple)
- **Jour 2**: Migrer MNIST (grand réseau, test de performance)
- **Jour 3**: Migrer Iris et autres
- **Jour 4**: Tests approfondis et optimisations
- **Jour 5**: Cleanup et documentation

---

**Bonne migration! 🚀**
