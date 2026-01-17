#!/bin/bash

# Script pour ajuster les chemins relatifs selon la branche

BRANCH=$1
OUTPUT_DIR=$2

# Valider les paramètres requis
if [ -z "$BRANCH" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <BRANCH> <OUTPUT_DIR>" >&2
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Erreur: le répertoire de sortie '$OUTPUT_DIR' n'existe pas ou n'est pas un répertoire." >&2
  exit 1
fi

# Créer un fichier de configuration avec le base path
if [ "$BRANCH" = "develop" ]; then
  BASE_PATH="/next"
else
  BASE_PATH=""
fi

# Injecter le BASE_PATH dans tous les fichiers HTML
find "$OUTPUT_DIR" -name "*.html" -type f | while read -r file; do
  # Ajouter une balise <script> au début du <head> si elle n'existe pas déjà
  if ! grep -q "window.BASE_PATH" "$file"; then
    sed -i.bak "/<head>/a\\    <script>window.BASE_PATH = '$BASE_PATH';</script>" "$file"
    rm -f "${file}.bak"
  fi
  
  # Remplacer les chemins relatifs pour les ressources statiques
  if [ "$BRANCH" = "develop" ]; then
    # Préfixer uniquement les chemins relatifs (pas ceux commençant par "/", "#" ou "http")
    sed -i.bak "s|href=\"\([^/#][^\"]*\)\"|href=\"$BASE_PATH/\1\"|g" "$file"
    sed -i.bak "s|href=\"$BASE_PATH/http|href=\"http|g" "$file"
    sed -i.bak "s|src=\"\([^/#][^\"]*\)\"|src=\"$BASE_PATH/\1\"|g" "$file"
    sed -i.bak "s|src=\"$BASE_PATH/http|src=\"http|g" "$file"
    rm -f "${file}.bak"
  fi
done

echo "Chemins ajustés pour: $BRANCH"
