# GitHub Workflows

Ce répertoire contient les workflows CI/CD du projet.

## Workflows disponibles

### 1. **CI.yml** - Intégration Continue (à la demande + sur PRs)

**Déclenchement :**
- Automatiquement sur les **pull requests** vers `main` et `develop`
- À la demande via `workflow_dispatch` (bouton "Run workflow" sur GitHub)

**Actions :**
- Setup Rust + wasm-pack
- Build WASM avec le script `test-neural/neural-wasm/build_all.sh`
- Cache des dépendances Cargo pour accélérer les builds

**Statut obligatoire :**
Ce workflow doit passer avec succès avant de pouvoir merger une PR vers `main`.

### 2. **Deploy.yml** - Déploiement sur GitHub Pages

**Déclenchement :**
- Automatiquement quand du code est pushé sur `main` ou `develop`
- À la demande via `workflow_dispatch`

**Actions :**
- Build WASM (identique au CI)
- Déploiement sur GitHub Pages avec structure en sous-chemins :
  - **`main`** → `https://votresite.github.io/` (production)
  - **`develop`** → `https://votresite.github.io/next/` (préversion)

**Ajustement des chemins :**
Le script `test-neural/adjust-paths.sh` ajoute automatiquement le préfixe `/next` à tous les chemins des ressources (HTML, CSS, JS, WASM) pour la branche `develop`.

---

## Workflow Git recommandé

### Pour ajouter des fonctionnalités :

```bash
# 1. Créer une branche feature
git checkout -b feature/ma-feature

# 2. Faire vos commits
git add .
git commit -m "Ajout de ma feature"

# 3. Pousser et créer une PR
git push origin feature/ma-feature
gh pr create --base develop --head feature/ma-feature --title "Ajout: ma feature"
```

### Pour merger develop dans main (sortie en production) :

```bash
# 1. Pousser develop
git checkout develop
git push origin develop

# 2. Créer une PR develop → main
gh pr create --base main --head develop --title "Release: fusion develop → main"
```

**Vérifications automatiques :**
- Le CI s'exécutera automatiquement
- Tant qu'il ne passe pas ❌, tu **ne pourras pas merger**
- Une fois qu'il passe ✅, l'option "Merge" devient disponible
- Après le merge, le deploy s'exécute et publie en production

---

## Configuration des branch rules (à faire sur GitHub)

Pour rendre le CI obligatoire :

1. Va sur **Settings** du repository
2. **Branches** → **Branch protection rules**
3. Ajoute une règle pour `main` :
   - ✅ Require status checks to pass before merging
   - Sélectionne le check **"Build test-neural"** (du workflow CI)
   - ✅ Require branches to be up to date before merging

---

## Notes importantes

- Les caches Cargo sont conservés entre les runs pour accélérer les builds
- Le script `adjust-paths.sh` doit être exécutable (fichier `.sh`)
- Les fichiers HTML doivent utiliser des chemins relatifs pour que l'ajustement fonctionne
- Les deux workflows partagent la même étape de build WASM pour cohérence
