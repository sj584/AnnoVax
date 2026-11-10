#!/usr/bin/env bash
# Fetch the published AnnoVax model and the precomputed embedding caches.
#
#   bash scripts/download_weights.sh            # model + embeddings
#   bash scripts/download_weights.sh model      # just the model
#   bash scripts/download_weights.sh embeddings # just the embeddings
#
# Set ANNOVAX_RECORD to pull from a different Zenodo record (e.g. an older version).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORD="${ANNOVAX_RECORD:-24095960}"
BASE="https://zenodo.org/records/${RECORD}/files"
WHAT="${1:-all}"

for tool in curl unzip md5sum; do
  command -v "$tool" >/dev/null || { echo "[err] '$tool' is required but not installed" >&2; exit 1; }
done

# MD5 as published on the Zenodo record. Zenodo reports md5, not sha256, so this is the
# digest that can actually be checked against the landing page by hand.
md5_of() {
  case "$1" in
    models.zip)     echo "3995d5c771635ec5d8313ede18a91e0c" ;;
    embeddings.zip) echo "26bff23ea752646816cc098b7cdab7c4" ;;
    *)              echo "" ;;
  esac
}

# Each archive already carries its destination directory as the top-level entry
# (models/... and embeddings/...), so both unpack at the repository root rather than
# inside the directory they create.
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

fetch() {  # fetch <file>
  # Assigned one per line: bash expands every word of `local` before it runs, so a later
  # assignment on the same line cannot see an earlier one and trips `set -u`.
  local name="$1"
  local archive="${SCRATCH}/${name}"
  local expected actual
  echo "[get] ${name}"
  curl -fL --progress-bar -o "$archive" "${BASE}/${name}?download=1"

  expected="$(md5_of "$name")"
  if [[ -n "$expected" ]]; then
    actual="$(md5sum "$archive" | cut -d' ' -f1)"
    if [[ "$actual" != "$expected" ]]; then
      echo "[err] ${name}: md5 mismatch" >&2
      echo "      expected ${expected}" >&2
      echo "      got      ${actual}" >&2
      echo "      The download is incomplete or corrupt; delete it and retry." >&2
      exit 1
    fi
    echo "[ok]  ${name} md5 verified"
  fi

  unzip -q -o "$archive" -d "$ROOT"
  rm -f "$archive"
}



require() {  # require <path> <hint>
  [[ -e "$1" ]] || { echo "[err] expected ${1} after unpacking, but it is missing. $2" >&2; exit 1; }
}

if [[ "$WHAT" == "all" || "$WHAT" == "model" ]]; then
  fetch "models.zip"
  require "${ROOT}/models/annovax_K1000/manifest.json" "The archive layout may have changed."
  echo "[ok] model -> ${ROOT}/models/annovax_K1000 ($(find "${ROOT}/models/annovax_K1000/bags" -name 'bag_*.ubj' | wc -l) bags)"
fi

if [[ "$WHAT" == "all" || "$WHAT" == "embeddings" ]]; then
  fetch "embeddings.zip"
  for enc in plgdl__esmc_6b plgdl__biomistral_7b vaxign__esmc_6b vaxign__biomistral_7b; do
    require "${ROOT}/embeddings/${enc}.npz" "The archive layout may have changed."
  done
  echo "[ok] embeddings -> ${ROOT}/embeddings"
  echo "[note] These four caches cover the esmc_biomistral, esmc and biomistral arms."
  echo "       The esm2_3b and biobert ablation arms are not published; build those"
  echo "       caches with scripts/extract_embedding.py if you need them."
fi
