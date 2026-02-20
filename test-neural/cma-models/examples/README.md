# Exemples cma-models

Ce dossier contient des exemples détaillés des architectures CNN historiques, reproduisant les innovations clés de chaque paper fondateur.

## Exécution des exemples

Depuis la racine du répertoire `test-neural` :

```bash
# LeNet-5 (LeCun et al., 1998)
cargo run -p cma-models --example lenet5_paper

# AlexNet (Krizhevsky et al., 2012)
cargo run -p cma-models --example alexnet_paper

# VGG-16/19 (Simonyan & Zisserman, 2014)
cargo run -p cma-models --example vgg_paper

# ResNet (He et al., 2015)
cargo run -p cma-models --example resnet_paper

# EfficientNet (Tan & Le, 2019)
cargo run -p cma-models --example efficientnet_paper
```

## Compilation en mode release (plus rapide)

```bash
cargo run -p cma-models --example resnet_paper --release
```

## Liste des exemples disponibles

| Exemple | Architecture | Paper | Points Clés |
|---------|-------------|-------|-------------|
| `lenet5_paper` | LeNet-5 | LeCun 1998 | Première architecture CNN moderne, Conv → Pool → FC |
| `alexnet_paper` | AlexNet | Krizhevsky 2012 | ReLU, Dropout, GPU training, ImageNet winner |
| `vgg_paper` | VGG-16/19 | Simonyan 2014 | Philosophie des filtres 3×3, profondeur |
| `resnet_paper` | ResNet-18/34/50 | He 2015 | Skip connections, entraînement 152 couches |
| `efficientnet_paper` | EfficientNet-B0 | Tan 2019 | Compound scaling, MBConv, Squeeze-Excitation |

## Structure des exemples

Chaque exemple inclut :

1. **Citation du paper original** (BibTeX)
2. **Diagramme ASCII** de l'architecture
3. **Configuration(s)** pour différents datasets (ImageNet, CIFAR-10, etc.)
4. **Démonstration** de forward pass
5. **Contexte historique** et impact
6. **Comparaison** avec le paper original

## Voir aussi

- [cma-neural-network](../../cma-neural-network/) - Couches Dense, activations, optimiseurs
- [cma-cnn](../../cma-cnn/) - Couches CNN (Conv2D, Pooling, BatchNorm)
- [ANALYSE_ARCHITECTURE_IMAGE.md](../../ANALYSE_ARCHITECTURE_IMAGE.md) - Roadmap complète
