#!/bin/bash

# Script pour rendre les chemins des ressources relatifs (./...) afin
# que le site fonctionne correctement lorsqu'il est servi sous /<repo>/next/

set -euo pipefail

BRANCH=${1:-}
OUTPUT_DIR=${2:-}

# Valider les paramètres requis
if [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <BRANCH> <OUTPUT_DIR>" >&2
  exit 1
fi

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "Erreur: le répertoire de sortie '$OUTPUT_DIR' n'existe pas ou n'est pas un répertoire." >&2
  exit 1
fi

echo "Adjusting paths in HTML files under: $OUTPUT_DIR (branch=$BRANCH)"

# Parcourir tous les fichiers HTML et convertir les attributs href/src relatifs
# en chemins relatifs explicites (./...), sans toucher aux URLs absolues
# (commençant par "/", "#", "http", "https", "." ou "..").

find "$OUTPUT_DIR" -name "*.html" -type f | while read -r file; do
  echo "Processing: $file"

  # Supprimer toute injection précédente de window.BASE_PATH si elle existe
  sed -i.bak "/window.BASE_PATH/d" "$file" || true
  rm -f "${file}.bak"

  # Préfixer uniquement les href relatifs qui ne commencent pas par "/", "#", ".", ".." ou "http"
  sed -i.bak -E 's|href="([^./#/][^"]*)"|href="./\1"|g' "$file"

  # Faire de même pour les attributs src
  sed -i.bak -E 's|src="([^./#/][^"]*)"|src="./\1"|g' "$file"

  rm -f "${file}.bak"
done

echo "Chemins relatifs appliqués pour: $BRANCH"
