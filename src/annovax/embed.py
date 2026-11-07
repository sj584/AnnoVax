"""Sequences and annotation text -> the mean-pooled .npz caches the model is trained on.

Each encoder writes one file holding exactly `ids` (str) and `X` (float32, (n, D)), which
is the format `data.EmbeddingRepo` reads. Row order is never relied on -- lookups go
through the IDs -- but every ID in a split must be present in every cache of its arm, or
loading raises rather than filling a zero row.

Two encoders make the released model:

* ESM-C 6B over the amino-acid sequence. Served remotely: no local weights are published
  for the 6B checkpoint, and the remote model returns one hidden layer at a time rather
  than a pooled vector, so the layer index is explicit and the pooling happens here.
* BioMistral-7B over the protein's InterProScan annotation text (`desc_nodup`).

ESM-2 3B and BioBERT are the ablation controls (paper Table 2). Note BioBERT takes the
CLS token where everything else mean-pools -- that is what the published cache contains.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: Layer whose hidden states become the sequence features. The manuscript's 80th layer.
ESMC_LAYER = 80
ESMC_MODEL = "esmc-6b-2024-12"
#: The legacy forge.evolutionaryscale.ai host was retired; this is the current endpoint.
ESMC_URL = "https://biohub.ai"

BIOMISTRAL_MODEL = "BioMistral/BioMistral-7B"
ESM2_3B_MODEL = "facebook/esm2_t36_3B_UR50D"
BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.2"

#: Both text encoders truncate here, as in the manuscript.
TEXT_MAX_TOKENS = 512

#: Text column embedded by the biomedical language models.
TEXT_COLUMN = "desc_nodup"


def save_cache(path, ids, vectors, *, expected_dim: int | None = None) -> Path:
    """Write one encoder's cache. Whole-file write, so a crash cannot leave it half-built."""
    path = Path(path)
    X = np.asarray(vectors, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {X.shape}")
    if len(ids) != len(X):
        raise ValueError(f"{len(ids)} ids against {len(X)} rows")
    if expected_dim is not None and X.shape[1] != expected_dim:
        raise ValueError(f"{path.name}: expected width {expected_dim}, got {X.shape[1]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray([str(i) for i in ids]), X=X)
    print(f"[cache] {path} {X.shape}")
    return path


def _text_of(row) -> str:
    """The annotation string for one protein, with the published empty-case fallback."""
    from .preprocess import EMPTY_ANNOTATION

    value = row.get(TEXT_COLUMN)
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return EMPTY_ANNOTATION
    return str(value)


# --------------------------------------------------------------------------- sequences

def esmc_client(token: str | None = None):
    """Remote ESM-C client. Token from --token, $ESM_API_KEY or $ESM_FORGE_TOKEN."""
    from esm.sdk import client

    token = token or os.environ.get("ESM_API_KEY") or os.environ.get("ESM_FORGE_TOKEN")
    if not token:
        raise RuntimeError(
            "ESM-C 6B is served remotely and needs an API token: set $ESM_API_KEY, or pass "
            "--token. Readers without one can download the published embeddings instead "
            "(scripts/download_weights.sh).")
    return client(model=ESMC_MODEL, url=ESMC_URL, token=token)


def embed_sequences_esmc(frame, *, token: str | None = None, layer: int = ESMC_LAYER):
    """Mean-pooled (2560,) per protein. Returns (ids, X, failed_ids).

    One request per protein, so a failure is reported and skipped rather than aborting a
    run of thousands. Any protein that fails here is missing from the cache, and loading a
    split that contains it will stop the run -- which is the intended behaviour.
    """
    from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

    model = esmc_client(token)
    config = LogitsConfig(return_hidden_states=True, ith_hidden_layer=layer)

    ids, vectors, failed = [], [], []
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        protein_id, sequence = str(row.ID), str(row.Sequence)
        try:
            tensor = model.encode(ESMProtein(sequence=sequence))
            if isinstance(tensor, ESMProteinError):
                raise RuntimeError(f"encode failed: {tensor}")
            output = model.logits(tensor, config)
            if isinstance(output, ESMProteinError):
                raise RuntimeError(f"logits failed: {output}")

            emb = output.embeddings if output.embeddings is not None else output.hidden_states
            # The remote model returns [n_layers, B, L, D]; collapse to (L, D).
            while emb.dim() > 2:
                emb = emb[0]
            residues = emb[1:-1, :]          # drop BOS/EOS
            if residues.shape[0] != len(sequence):
                raise RuntimeError(f"{residues.shape[0]} residues against {len(sequence)} aa")
            ids.append(protein_id)
            vectors.append(residues.float().mean(dim=0).cpu().numpy())
        except Exception as error:          # noqa: BLE001 -- one bad protein must not end the run
            print(f"[ERR] {protein_id}: {error}", flush=True)
            failed.append(protein_id)
        if position % 100 == 0:
            print(f"[esmc] {position}/{len(frame)} ({len(failed)} failed)", flush=True)
    return ids, np.asarray(vectors, dtype=np.float32), failed


def embed_sequences_esm2(frame, *, device: str = "cuda", batch_size: int = 1):
    """ESM-2 3B mean-pooled (2560,) per protein -- the ablation control."""
    import torch
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(ESM2_3B_MODEL)
    model = EsmModel.from_pretrained(ESM2_3B_MODEL).to(device).eval()

    ids, vectors = [], []
    for position, row in enumerate(frame.itertuples(index=False), start=1):
        inputs = tokenizer(str(row.Sequence), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state[0]
        ids.append(str(row.ID))
        vectors.append(hidden[1:-1, :].float().mean(dim=0).cpu().numpy())
        if position % 100 == 0:
            print(f"[esm2] {position}/{len(frame)}", flush=True)
    return ids, np.asarray(vectors, dtype=np.float32), []


# ------------------------------------------------------------------------------- text

def _load_text_model(model_name: str, device: str, dtype=None):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    model = AutoModel.from_pretrained(model_name, **kwargs).to(device).eval()
    return tokenizer, model, torch


def embed_text_biomistral(frame, *, device: str = "cuda"):
    """Attention-masked mean pool of the last hidden state -> (4096,) per protein."""
    import torch

    tokenizer, model, torch_mod = _load_text_model(BIOMISTRAL_MODEL, device, torch.float16)
    ids, vectors = [], []
    for position, row in enumerate(frame.to_dict("records"), start=1):
        inputs = tokenizer(_text_of(row), return_tensors="pt", truncation=True,
                           max_length=TEXT_MAX_TOKENS, padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch_mod.no_grad():
            hidden = model(**inputs).last_hidden_state[0]
        mask = inputs["attention_mask"][0].unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(dim=0) / mask.sum()
        ids.append(str(row["ID"]))
        vectors.append(pooled.cpu().numpy())
        if position % 200 == 0:
            print(f"[biomistral] {position}/{len(frame)}", flush=True)
    return ids, np.asarray(vectors, dtype=np.float32), []


def embed_text_biobert(frame, *, device: str = "cuda"):
    """BioBERT CLS token -> (768,). CLS, not mean pool: that is what the cache holds."""
    tokenizer, model, torch_mod = _load_text_model(BIOBERT_MODEL, device)
    ids, vectors = [], []
    for position, row in enumerate(frame.to_dict("records"), start=1):
        inputs = tokenizer(_text_of(row), return_tensors="pt", truncation=True,
                           max_length=TEXT_MAX_TOKENS, padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch_mod.no_grad():
            hidden = model(**inputs).last_hidden_state
        ids.append(str(row["ID"]))
        vectors.append(hidden[0, 0, :].float().cpu().numpy())
        if position % 200 == 0:
            print(f"[biobert] {position}/{len(frame)}", flush=True)
    return ids, np.asarray(vectors, dtype=np.float32), []


#: encoder -> (callable, output width, which column it reads)
ENCODERS = {
    "esmc_6b": (embed_sequences_esmc, 2560, "Sequence"),
    "biomistral_7b": (embed_text_biomistral, 4096, TEXT_COLUMN),
    "esm2_3b": (embed_sequences_esm2, 2560, "Sequence"),
    "biobert": (embed_text_biobert, 768, TEXT_COLUMN),
}
