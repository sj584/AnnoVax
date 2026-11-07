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
RECORD="${ANNOVAX_RECORD:-ZENODO_RECORD_ID}"
BASE="https://zenodo.org/records/${RECORD}/files"
WHAT="${1:-all}"

if [[ "$RECORD" == "ZENODO_RECORD_ID" ]]; then
  cat >&2 <<'MSG'
The Zenodo record is not set yet.

Until it is published, build the inputs locally instead:
  python scripts/extract_embedding.py -i <preprocessed.csv> -c plgdl   # embeddings
  python scripts/train.py --save-models models/annovax_K1000           # model
MSG
  exit 1
fi

fetch() {  # fetch <file> <target-dir>
  local name="$1" dest="$2"
  mkdir -p "$dest"
  echo "[get] ${name}"
  curl -fL --progress-bar -o "${dest}/${name}" "${BASE}/${name}?download=1"
  if [[ -f "${ROOT}/checksums.sha256" ]]; then
    (cd "$dest" && grep " ${name}\$" "${ROOT}/checksums.sha256" | sha256sum -c -)
  fi
}

if [[ "$WHAT" == "all" || "$WHAT" == "model" ]]; then
  fetch "annovax_K1000_models.tar.xz" "${ROOT}/models"
  tar -xJf "${ROOT}/models/annovax_K1000_models.tar.xz" -C "${ROOT}/models"
  rm -f "${ROOT}/models/annovax_K1000_models.tar.xz"
  echo "[ok] model -> ${ROOT}/models/annovax_K1000"
fi

if [[ "$WHAT" == "all" || "$WHAT" == "embeddings" ]]; then
  fetch "annovax_embeddings.tar" "${ROOT}/embeddings"
  tar -xf "${ROOT}/embeddings/annovax_embeddings.tar" -C "${ROOT}/embeddings"
  rm -f "${ROOT}/embeddings/annovax_embeddings.tar"
  echo "[ok] embeddings -> ${ROOT}/embeddings"
fi
