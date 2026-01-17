#!/bin/bash

# Script pour ajuster les chemins relatifs selon la branche

BRANCH=$1
OUTPUT_DIR=$2

# Créer un fichier de configuration avec le base path
if [ "$BRANCH" = "develop" ]; then
  BASE_PATH="/next"
else
  BASE_PATH=""
fi

# Injecter le BASE_PATH dans tous les fichiers HTML
find "$OUTPUT_DIR" -name "*.html" -type f | while read -r file; do
  # Ajouter une balise <script> au début du <head>
  sed -i "/<head>/a\\    <script>window.BASE_PATH = '$BASE_PATH';</script>" "$file"
  
  # Remplacer les chemins relatifs pour les ressources statiques
  if [ "$BRANCH" = "develop" ]; then
    sed -i "s|href=\"|href=\"$BASE_PATH/|g" "$file"
    sed -i "s|src=\"|src=\"$BASE_PATH/|g" "$file"
    # Éviter les chemins en double
    sed -i "s|$BASE_PATH/$BASE_PATH|$BASE_PATH|g" "$file"
  fi
done

echo "Chemins ajustés pour: $BRANCH"
