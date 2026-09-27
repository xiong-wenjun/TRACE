# Pinned A-MEM integration

Source: https://github.com/agiresearch/A-mem
Revision: ceffb860f0712bbae97b184d440df62bc910ca8d
License: MIT (retained in LICENSE). UPSTREAM.json records original file hashes.

The adapter executes the upstream MemoryNote and AgenticMemorySystem, including
analyze_content, process_memory and search_agentic. It is an integration of the
published library, not a reproduction of the separate paper evaluation repository.

Compatibility changes to memory_system.py:

- Heavy imports are lazy. Constructor injection supplies a namespace-specific
  retriever and an explicit model transport; the default implementation remains
  available. Injected construction never resets the global Chroma client.
- Neighbor prompts and updates use stable note IDs. Upstream enumerated retrieved
  neighbors but applied their positions to global insertion order, which could
  update a different note. Prompts otherwise retain the upstream content/schema.
- Optional strict_errors propagates model, parsing and retrieval failures rather
  than returning an apparently successful empty result.
- Consolidation uses the injected retriever factory. Its scoped Chroma adapter
  upserts existing IDs rather than leaving stale index metadata after add().

Adapter-specific choices outside the vendor file:

- Every write explicitly calls native analyze_content before add_note. The pinned
  library does not call analysis from add_note itself.
- External embeddings and generation endpoints are explicit. Formation uses
  temperature 0, a recorded seed, max_tokens=2048 and evo_threshold=100 by default.
  These are TRACE harness settings, not the original paper's fixed protocol.
- Snapshot imports copy note attributes, IDs, the native index metadata and
  embeddings without additional LLM formation or embedding calls. Links to notes
  excluded from the destination are removed. Namespace reset only removes that
  namespace's collection.
- Retrieval preserves native search_agentic IDs/order, renders current native
  content/context/keywords/tags, and maps distance d to 1/(1+max(d,0)) for the common
  higher-is-better interface. This avoids exposing outdated Chroma metadata before
  the native consolidation threshold. It does not rerank the native result.
- SourceLedger retains a conservative formation-input scope. All source IDs must
  be admitted before a merged/evolved record is imported. This can reject more
  records than exact dependency tracking would; it is not a causal attribution.

retrievers.py and llm_controller.py are retained unmodified. Runtime transport
injection avoids requiring LiteLLM for this integration. No benchmark gain follows
from unit tests or a transport smoke; A-MEM paper results remain pending.
