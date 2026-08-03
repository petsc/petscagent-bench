"""Build a FAISS index over the PETSc tutorial corpus.

One-time offline step. Walks the PETSc tutorial tree, embeds each `.c` file
with a local SentenceTransformer model, and writes a FAISS index plus a
metadata pickle that `retrieve.py` loads at query time.

Corpus: every file matching `tutorials/ex*.c` under `--petsc-root`. That
pattern intentionally *excludes* files under `tests/` and non-`ex*`
tutorials — the ex-numbered tutorials are the canonical PETSc examples that
teach one concept at a time, so they're the highest-signal-per-token thing to
put in the LLM's context. `build_index_expanded.py` is a one-off variant that
sweeps tutorials + tests for A/B experiments (not shipped).

Output layout (both files required by retrieve.py):
    <out>/faiss.bin    — inner-product FAISS index of normalized embeddings
    <out>/store.pkl    — {"chunks": [...], "metas": [...], "model_name": str}

Rebuild whenever `$PETSC_DIR` changes; the index bakes in the on-disk
tutorial text at build time, not a reference to it.

Usage:
    python -m petsc_rag.build_index
    python -m petsc_rag.build_index --petsc-root /path/to/petsc --out src/petsc_rag/index
"""

import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Homebrew-installed PETSc on Apple Silicon. Override with --petsc-root on
# other machines (linux clusters, spack installs, etc.).
DEFAULT_PETSC_ROOT = Path(
    "/opt/homebrew/Cellar/petsc/3.24.6/share/petsc/examples/src"
)
DEFAULT_OUT = Path(__file__).parent / "index"
# Small, CPU-friendly sentence embedder. 384-dim, ~90MB. Good enough for
# code-tutorial retrieval; upgrading buys precision at ~5x latency.
DEFAULT_MODEL = "all-MiniLM-L6-v2"
MAX_CHARS = 8000  # truncate very long tutorials; most are well under this


def collect_tutorials(petsc_root: Path) -> list[tuple[Path, str, str]]:
    """Return [(absolute_path, relative_path, subsystem)] for every tutorial .c.

    Only `tutorials/ex*.c` files — the numbered canonical examples. The
    subsystem is taken from the first path component under `petsc_root`
    (e.g. "ksp/tutorials/ex2.c" -> "ksp"), which is what the tutorials tree
    is organized by.
    """
    out = []
    for c_file in sorted(petsc_root.rglob("tutorials/ex*.c")):
        rel = c_file.relative_to(petsc_root)
        # rel looks like "ksp/tutorials/ex2.c" -> subsystem = "ksp"
        subsystem = rel.parts[0] if rel.parts else "unknown"
        out.append((c_file, str(rel), subsystem))
    return out


def main() -> None:
    """CLI entry point: collect -> embed -> write faiss.bin + store.pkl."""
    p = argparse.ArgumentParser()
    p.add_argument("--petsc-root", type=Path, default=DEFAULT_PETSC_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    if not args.petsc_root.exists():
        raise SystemExit(f"PETSc tutorial root not found: {args.petsc_root}")

    args.out.mkdir(parents=True, exist_ok=True)

    tutorials = collect_tutorials(args.petsc_root)
    print(f"Found {len(tutorials)} tutorials under {args.petsc_root}")

    # Read every tutorial into memory; truncate at MAX_CHARS so a single
    # pathologically long file can't dominate the embedding batch. Metas
    # carry the on-disk relative path (used at retrieval time to cite the
    # source) plus the pre-truncation length (useful for debugging).
    chunks: list[str] = []
    metas: list[dict] = []
    for abs_path, rel_path, subsystem in tutorials:
        try:
            text = abs_path.read_text(errors="ignore")
        except OSError as e:
            print(f"  skip {rel_path}: {e}")
            continue
        chunks.append(text[:MAX_CHARS])
        metas.append({"path": rel_path, "subsystem": subsystem, "chars": len(text)})

    # Normalize embeddings so inner-product search on IndexFlatIP is
    # equivalent to cosine similarity — cheaper and more numerically stable
    # than IndexFlatL2 with post-hoc normalization.
    print(f"Embedding {len(chunks)} chunks with {args.model}...")
    model = SentenceTransformer(args.model)
    vectors = model.encode(
        chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # Flat exact search. The corpus is small (~hundreds of tutorials), so
    # there's no reason to bother with IVF/HNSW approximate indices.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # Write index and store separately: FAISS owns its binary format, and
    # the pickle carries everything retrieve.py needs to reconstruct hits
    # (chunk text, per-hit metadata, and the embedder model name so
    # queries embed with the same model that built the index).
    faiss_path = args.out / "faiss.bin"
    store_path = args.out / "store.pkl"
    faiss.write_index(index, str(faiss_path))
    with open(store_path, "wb") as f:
        pickle.dump(
            {"chunks": chunks, "metas": metas, "model_name": args.model},
            f,
        )

    print(f"Wrote {faiss_path} ({vectors.shape[0]} vectors, dim={vectors.shape[1]})")
    print(f"Wrote {store_path}")


if __name__ == "__main__":
    main()
