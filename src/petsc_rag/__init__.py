"""PETSc RAG subsystem — retrieval-augmented context for the code-gen agents.

Two independent indices, both consumed by `purple_agent_v2/petsc_agent.py`
(and optionally by `green_agent/agent.py`):

- Tutorial-text FAISS index (see `build_index.py`, `retrieve.py`): embeds every
  `tutorials/ex*.c` under `$PETSC_DIR` with a SentenceTransformer, stores
  vectors in `index/faiss.bin` + `index/store.pkl`. At query time
  `retrieve.retrieve(query)` returns the top-K most similar tutorials, with an
  optional cross-encoder rerank stage. Used to inject "here is a similar
  problem someone already solved in PETSc" into the first-turn system prompt.

- Header-signature index (see `headers.py`): parses every `.h` under
  `$PETSC_DIR/include`, extracts `PETSC_EXTERN PetscErrorCode Foo(...)`
  declarations, and lets the fix-it loop look up canonical signatures for
  symbols the compiler complained about. Used to correct wrong-arity /
  wrong-type API calls in the second turn of the self-verify loop.

Both indices are lazy-loaded on first use so `import petsc_rag` stays cheap
in downstream code.
"""
