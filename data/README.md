# Web Data Files

## Pourquoi ce dossier ?

Ces fichiers CSV sont **optionnels** et servent uniquement à enrichir l'interface web.

## ⚠️ Important

- **Le modèle ne dépend PAS de ces fichiers**
- Le modèle est déjà entraîné et embarqué dans le WASM
- Ces CSV sont pour les **tests dans le navigateur uniquement**

## Utilisation

### Option 1: Presets hardcodés (ACTUEL) ✅

Simple, rapide, suffisant pour une démo:

```javascript
const presets = {
    setosa: { sepalLength: 5.1, sepalWidth: 3.5, ... }
};
```

### Option 2: Charger un CSV

Pour avoir tous les vrais samples de test:

```javascript
async function loadTestSamples() {
    const response = await fetch('data/iris_test.csv');
    const text = await response.text();
    const lines = text.split('\n').slice(1); // Skip header
    
    return lines.map(line => {
        const [sl, sw, pl, pw, species] = line.split(',');
        return {
            sepalLength: parseFloat(sl),
            sepalWidth: parseFloat(sw),
            petalLength: parseFloat(pl),
            petalWidth: parseFloat(pw),
            species: species.trim()
        };
    }).filter(s => s.sepalLength); // Remove empty lines
}
```

## Recommandation

**Pour une première version**: Gardez les presets hardcodés.

**Pour une version avancée**: Chargez un CSV si vous voulez:
- Permettre à l'utilisateur de tester sur tous les 150 samples
- Afficher des statistiques détaillées
- Permettre l'upload de datasets custom
