# 🧠 Neural XOR - Web Demo

Interface web interactive pour tester le réseau neuronal XOR via WebAssembly.

## 🚀 Lancement

### Prérequis

Assurez-vous que le module WASM est compilé :

```bash
cd ../neural-wasm && ./build.sh
```

### Démarrer le serveur

Depuis la racine du projet :

```bash
npx http-server -p 8080 -c-1 -o /www/index.html --host 0.0.0.0
```

| Option | Description |
|--------|-------------|
| `-p 8080` | Port du serveur |
| `-c-1` | Désactive le cache (utile en développement) |
| `-o /www/index.html` | Ouvre automatiquement le navigateur |
| `--host 0.0.0.0` | Écoute sur toutes les interfaces (nécessaire pour conteneur/Docker) |

### Accès

- **Local** : http://localhost:8080/www/index.html
- **Réseau** : http://<IP_CONTENEUR>:8080/www/index.html

## 📖 Fonctionnalités

- ⚡ **Prédiction interactive** : Cliquez sur les boutons pour changer les entrées
- 📊 **Table de vérité** : Visualisation des 4 combinaisons XOR
- 🎯 **Confiance** : Affichage du niveau de certitude du modèle
- 🔧 **Info modèle** : Architecture et précision du réseau

## 🔗 API JavaScript

```javascript
import init, { XorNetwork } from '../neural-wasm/pkg/neural_wasm.js';

await init();
const network = new XorNetwork();

// Prédiction binaire (0 ou 1)
network.predict(0, 1);  // → 1

// Valeur brute (0.0 - 1.0)
network.predict_raw(0, 1);  // → 0.9987

// Confiance en pourcentage
network.confidence(0, 1);  // → 99.7

// Tester toutes les combinaisons
network.test_all();  // → JSON array

// Info du modèle
network.model_info();  // → "XOR Network: 2 → [8] → 1"
```
