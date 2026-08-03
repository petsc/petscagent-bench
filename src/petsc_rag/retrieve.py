"""Query-time retrieval against the FAISS index built by build_index.py.

Two-stage retrieval:
  1. Vector search over the FAISS index returns `k_initial` candidates.
  2. Optional cross-encoder reranker (flashrank) reads (query, chunk) pairs and
     reorders them by a finer-grained relevance score, returning the top `k`.

The reranker fixes the failure mode where the literal best-fit tutorial is in
the top-K candidates but ranked below less-relevant neighbors — e.g., the
Robertson_ODE problem retrieves `ts/tutorials/ex8.c` (which IS Robertson) at
rank 5 with vector search alone.

Lazy-loads everything on first call so importing this module is free — matters
because the outer agent imports this even when RAG is disabled by config.
"""

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

# Sits next to this file, populated by build_index.py. Callers can override
# via the `index_dir` parameter on retrieve() — used for A/B testing against
# alternate corpora.
DEFAULT_INDEX_DIR = Path(__file__).parent / "index"
DEFAULT_RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"  # flashrank small CPU model

# Module-level lazy cache. Keyed on `dir` so calling retrieve() with a
# different index_dir reloads instead of returning stale state.
_state: dict[str, Any] = {"loaded": False}


def _load(index_dir: Path) -> None:
    """Load FAISS index + metadata + embedder on first call for this dir.

    The embedder model name is pulled from the store — we always encode
    queries with the same model that built the index, otherwise the vector
    space is nonsense.
    """
    if _state.get("loaded") and _state.get("dir") == index_dir:
        return
    faiss_path = index_dir / "faiss.bin"
    store_path = index_dir / "store.pkl"
    if not faiss_path.exists() or not store_path.exists():
        raise FileNotFoundError(
            f"RAG index not found at {index_dir}. "
            "Run `python -m petsc_rag.build_index` first."
        )
    # Deferred import: sentence_transformers pulls in torch, which is a heavy
    # startup cost we don't want to pay when RAG is disabled.
    from sentence_transformers import SentenceTransformer

    with open(store_path, "rb") as f:
        store = pickle.load(f)
    _state.update(
        loaded=True,
        dir=index_dir,
        index=faiss.read_index(str(faiss_path)),
        chunks=store["chunks"],
        metas=store["metas"],
        model=SentenceTransformer(store["model_name"]),
    )


def _get_reranker(model_name: str = DEFAULT_RERANKER_MODEL):
    """Lazy-construct and cache the flashrank reranker (only used when rerank=True)."""
    cached = _state.get("reranker")
    if cached is not None and _state.get("reranker_name") == model_name:
        return cached
    from flashrank import Ranker  # imported lazily so non-rerank users skip the dep

    ranker = Ranker(model_name=model_name)
    _state["reranker"] = ranker
    _state["reranker_name"] = model_name
    return ranker


def retrieve(
    query: str,
    k: int = 4,
    index_dir: Path = DEFAULT_INDEX_DIR,
    rerank: bool = False,
    k_initial: int | None = None,
) -> list[dict]:
    """Return top-k tutorials most similar to `query`, highest score first.

    When `rerank=True`, vector search returns `k_initial` (default 2*k+4)
    candidates, then a cross-encoder reranker picks the top `k` from those.
    Each returned dict gains a `vector_score` and a `rerank_score` field so we
    can see what the reranker changed.

    Return-dict shape (per hit):
        path, subsystem, chars, text  — from the store's metas + chunk
        vector_score, vector_rank     — always present
        score                         — the score the caller should treat as "the" score
        rerank_score                  — present only when rerank=True
    """
    _load(index_dir)

    # Vector stage: fetch enough candidates for whichever stage picks final k.
    n_vector = k_initial if rerank else k
    if rerank and n_vector is None:
        n_vector = 2 * k + 4

    # Same-model encode + normalize means inner-product = cosine similarity,
    # matching how the index was built.
    qv = _state["model"].encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    scores, idxs = _state["index"].search(qv, max(n_vector, k))

    # Assemble the initial pool. FAISS returns -1 for slots when the index
    # has fewer vectors than requested; skip those.
    initial: list[dict] = []
    for rank, idx in enumerate(idxs[0]):
        if idx < 0:
            continue
        initial.append(
            {
                "vector_score": float(scores[0][rank]),
                "vector_rank": rank,
                "text": _state["chunks"][idx],
                **_state["metas"][idx],
            }
        )

    if not rerank:
        # Single-stage: vector score IS the score.
        return [{**h, "score": h["vector_score"]} for h in initial[:k]]

    # Two-stage: rerank the initial pool with the cross-encoder.
    # Cross-encoders score (query, passage) pairs directly instead of
    # embedding each independently — much slower per pair, but only run on
    # a small pool, and much better at fine-grained relevance.
    from flashrank import RerankRequest

    ranker = _get_reranker()
    passages = [
        {"id": str(i), "text": h["text"], "meta": h["path"]}
        for i, h in enumerate(initial)
    ]
    req = RerankRequest(query=query, passages=passages)
    ranked = ranker.rerank(req)  # list of {id, score, text, meta}, score-sorted desc

    # Map reranker output back to the original hit dicts (with their metas)
    # instead of the reranker's slimmer passage dicts.
    out = []
    for r in ranked[:k]:
        src = initial[int(r["id"])]
        out.append(
            {
                **src,
                "score": float(r["score"]),
                "rerank_score": float(r["score"]),
            }
        )
    return out


def format_for_prompt(hits: list[dict]) -> str:
    """Render retrieved hits as a single string ready to splice into a system prompt.

    Empty-string on no hits so callers can `prompt += format_for_prompt(hits)`
    unconditionally. The similarity score is printed for the LLM's benefit —
    higher scores let it weight which tutorial to lean on hardest.
    """
    if not hits:
        return ""
    parts = ["REFERENCE PETSc EXAMPLES (similar to the requested problem):"]
    for h in hits:
        parts.append(f"\n--- {h['path']} (similarity={h['score']:.3f}) ---\n{h['text']}")
    return "\n".join(parts)
