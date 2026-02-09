# 📖 Documentation Autograd

Ce répertoire contient la documentation complète sur les systèmes de différentiation automatique (autograd) pour les réseaux de neurones.

## 📂 Structure

| Fichier | Description | Public cible |
|---------|-------------|--------------|
| [01_VULGARISATION.md](01_VULGARISATION.md) | Guide d'introduction à l'autograd | Débutants, étudiants |
| [02_SPECIFICATIONS_TECHNIQUES.md](02_SPECIFICATIONS_TECHNIQUES.md) | Spécifications techniques complètes | Développeurs expérimentés |

## 🎯 Par où commencer ?

### Si vous êtes nouveau sur le sujet

1. Commencez par [01_VULGARISATION.md](01_VULGARISATION.md) pour comprendre les concepts fondamentaux
2. Les analogies (recette, GPS, jeu vidéo) vous aideront à visualiser le fonctionnement

### Si vous êtes développeur

1. Lisez [02_SPECIFICATIONS_TECHNIQUES.md](02_SPECIFICATIONS_TECHNIQUES.md) pour les spécifications détaillées

## 📚 Contenu des documents

### 01 - Vulgarisation
- Fonctionnement des réseaux de neurones
- Forward et Backward pass expliqués simplement
- Pourquoi l'autograd est nécessaire
- Graphe de calcul et règle de chaîne
- Graphe statique vs dynamique
- Analogies du quotidien
- Glossaire complet

### 02 - Spécifications Techniques
- Architecture complète d'un système autograd
- Structure du Tensor avec tracking de gradient
- Trait GradFunction et implémentations
- Moteur de backpropagation
- Opérations : basiques, activations, conv2d, pooling, batchnorm
- Modules et couches (Linear, Conv2d, etc.)
- Optimizers (SGD, Adam)
- Gestion mémoire et checkpointing
- API publique et patterns de conception

## 🔗 Ressources externes

- [PyTorch Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [micrograd](https://github.com/karpathy/micrograd) - Autograd minimal par Andrej Karpathy
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

---

*Documentation générale sur l'autograd*
