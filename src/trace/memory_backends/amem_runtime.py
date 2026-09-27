"""Scoped Chroma and explicit model transports for the pinned A-MEM engine.

Optional packages are imported only when this backend is selected.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from ..providers import OpenAICompatibleEmbeddingClient, OpenAICompatibleProvider
from ..vendor.amem_official.memory_system import AgenticMemorySystem


def build_system_factory(*, storage_dir: Path, generation_base_url: str,
                         generation_model: str, generation_api_key: str,
                         embedding_base_url: str, embedding_model: str,
                         embedding_api_key: str, embedding_dimensions: int,
                         timeout: float = 180, retries: int = 3,
                         evo_threshold: int = 100, seed: int = 731,
                         max_tokens: int = 2048, temperature: float = 0.0):
    import chromadb
    from chromadb.config import Settings
    from chromadb.api.types import EmbeddingFunction
    from ..vendor.amem_official.retrievers import ChromaRetriever

    if min(embedding_dimensions, evo_threshold, max_tokens) < 1:
        raise ValueError("A-MEM dimensions, evolution threshold and token limit must be positive")
    client = chromadb.PersistentClient(path=str(storage_dir / "chroma"),
                                      settings=Settings(anonymized_telemetry=False))
    embedding = OpenAICompatibleEmbeddingClient(
        base_url=embedding_base_url, model=embedding_model, api_key=embedding_api_key,
        timeout=timeout, retries=max(1, retries))
    counts = dict(completed_generation_calls=0, failed_generation_calls=0,
                  reported_prompt_tokens=0, reported_completion_tokens=0,
                  reported_total_tokens=0, embedded_texts=0)

    class ExternalEmbedding(EmbeddingFunction):
        def __call__(self, input):
            vectors = embedding.embed(input)
            if any(len(v) != embedding_dimensions for v in vectors):
                raise RuntimeError("A-MEM embedding width differs from the declared native width")
            counts["embedded_texts"] += len(input)
            return vectors

        @staticmethod
        def name():
            return "trace_amem_explicit_embedding"

        def get_config(self):
            return {"model": embedding_model, "dimensions": embedding_dimensions}

    embedding_function = ExternalEmbedding()

    class ScopedRetriever(ChromaRetriever):
        def __init__(self, namespace):
            self.client = client
            self.embedding_function = embedding_function
            self.collection_name = "trace-amem-" + hashlib.sha256(namespace.encode()).hexdigest()[:48]
            self._open()

        def _open(self):
            self.collection = client.get_or_create_collection(
                name=self.collection_name, embedding_function=self.embedding_function)

        def reset_namespace(self):
            client.delete_collection(self.collection_name)
            self._open()

        def add_document(self, document, metadata, doc_id):
            # Same serialization as upstream; upsert makes consolidation update
            # existing note IDs without ever resetting another collection.
            values = {k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                      for k, v in metadata.items()}
            self.collection.upsert(ids=[doc_id], documents=[document], metadatas=[values])

        def search(self, query, k=5):
            count = self.collection.count()
            if count == 0:
                return {key: [[]] for key in ("ids", "documents", "metadatas", "distances")}
            return super().search(query, min(k, count))

        def snapshot(self):
            rows = self.collection.get(include=["documents", "metadatas", "embeddings"])
            return {identifier: {"document": rows["documents"][i],
                                 "metadata": deepcopy(rows["metadatas"][i]),
                                 "embedding": list(map(float, rows["embeddings"][i]))}
                    for i, identifier in enumerate(rows["ids"])}

        def restore(self, identifier, row):
            self.collection.add(ids=[identifier], documents=[row["document"]],
                                metadatas=[row["metadata"]], embeddings=[row["embedding"]])

    class CompletionTransport:
        def get_completion(self, prompt, response_format, **kwargs):
            provider = OpenAICompatibleProvider(
                generation_base_url=generation_base_url, generation_model=generation_model,
                generation_api_key=generation_api_key, timeout=timeout,
                retries=max(1, retries), chat_response_format=response_format)
            try:
                result = provider.complete(prompt, system="Respond with the requested JSON object.",
                                           seed=seed, max_tokens=max_tokens, temperature=temperature)
                counts["completed_generation_calls"] += 1
                counts["reported_prompt_tokens"] += result.prompt_tokens
                counts["reported_completion_tokens"] += result.completion_tokens
                counts["reported_total_tokens"] += result.total_tokens
                json.loads(result.text)  # Never silently accept malformed native formation.
                return result.text
            except Exception:
                counts["failed_generation_calls"] += 1
                raise

    def factory(namespace):
        return AgenticMemorySystem(
            model_name=embedding_model, llm_model=generation_model,
            evo_threshold=evo_threshold, strict_errors=True,
            llm_controller=SimpleNamespace(llm=CompletionTransport()),
            retriever_factory=lambda **_: ScopedRetriever(namespace))

    def receipt():
        return {"generation_model": generation_model, "embedding_model": embedding_model,
                "embedding_dimensions": embedding_dimensions, "evo_threshold": evo_threshold,
                "seed": seed, "max_tokens": max_tokens, "temperature": temperature,
                "counter_scope": "driver_lifetime_cumulative",
                "usage_scope": "successful formation responses; excludes unreported retry and embedding tokens",
                **counts}

    return factory, receipt
