# Iris Classifier Module

Classification des fleurs Iris en 3 espèces (Setosa, Versicolor, Virginica).

## 📊 Dataset

Le module utilise maintenant **les vraies données Iris** provenant du [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

- **Fichier**: `data/iris.csv`
- **Samples**: 150 (50 par classe)
- **Features**: 4 (longueur/largeur sépale et pétale)
- **Source originale**: Fisher, R.A. (1936) "The use of multiple measurements in taxonomic problems"

### Format CSV

```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.3,3.3,6.0,2.5,virginica
...
```

## 🚀 Entraînement

```bash
# Depuis /workspace/test-neural/neural-wasm/iris
cargo run --bin train_iris
```

Le script:
1. ✅ Charge les vraies données depuis `data/iris.csv`
2. ✅ Split train/validation (80/20)
3. ✅ Entraîne le réseau de neurones
4. ✅ Évalue la précision sur le test set
5. ✅ Sauvegarde le modèle dans `src/iris_model.json`

## 📦 Build WASM

```bash
# Depuis /workspace/test-neural/neural-wasm/iris
./build.sh
```

## 🔄 Ajouter d'autres datasets

Pour utiliser un autre dataset CSV:

1. **Créer le fichier CSV** dans `data/`
2. **Modifier `train_iris.rs`**:
   ```rust
   let data = load_iris_from_csv("data/votre_dataset.csv")?;
   ```
3. **Adapter la fonction de parsing** selon vos colonnes

### Exemple pour un dataset custom

```rust
fn load_custom_from_csv(path: &str) -> Result<Vec<(Array1<f64>, Array1<f64>)>, Box<dyn Error>> {
    let mut data = Vec::new();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    
    for result in rdr.records() {
        let record = result?;
        
        // Adapter selon vos colonnes
        let feature1: f64 = record[0].parse()?;
        let feature2: f64 = record[1].parse()?;
        // ...
        
        let label = &record[4];
        let one_hot = match label {
            "class_a" => array![1.0, 0.0, 0.0],
            "class_b" => array![0.0, 1.0, 0.0],
            "class_c" => array![0.0, 0.0, 1.0],
            _ => return Err(format!("Unknown label: {}", label).into()),
        };
        
        data.push((
            array![feature1, feature2, /* ... */],
            one_hot,
        ));
    }
    
    Ok(data)
}
```

## 📚 Autres sources de datasets

- **UCI Repository**: https://archive.ics.uci.edu/ml/datasets.php
- **Kaggle**: https://www.kaggle.com/datasets
- **scikit-learn**: Datasets intégrés (convertir en CSV)
- **OpenML**: https://www.openml.org/

## ✅ Avantages du chargement CSV

✅ **Données réelles** (pas de hardcoding)  
✅ **Facilement modifiables** (pas besoin de recompiler)  
✅ **Réutilisables** entre projets  
✅ **Standard** (compatible avec Python, R, Excel, etc.)  
✅ **Versionnable** avec Git  
