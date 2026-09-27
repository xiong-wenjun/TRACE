"""Factory and CLI wiring for pluggable memory backends."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from ..mas_return_pipeline import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MAX_CHARS,
    EpisodeLongTermMemory,
    MemoryEmbeddingProvider,
)
from .base import ExternalEpisodeLongTermMemory, NativeMemoryDriver


ASVM_BACKEND = "asvm-vector"  # Legacy receipts and resumed runs remain compatible.
SCOPEDMEM_BACKEND = "scopedmem"
AMEM_BACKEND = "amem"
MEM0_BACKEND = "mem0"
MEMOBASE_BACKEND = "memobase"
MEMORY_BACKENDS = (SCOPEDMEM_BACKEND, ASVM_BACKEND, MEM0_BACKEND, MEMOBASE_BACKEND, AMEM_BACKEND)


@dataclass
class MemoryBackendFactory:
    backend: str
    embedding_provider: MemoryEmbeddingProvider | None
    embedding_dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS
    embedding_max_chars: int = DEFAULT_EMBEDDING_MAX_CHARS
    native_driver: NativeMemoryDriver | None = None

    def build_episode(self, episode_id: str) -> Any:
        if self.backend in {ASVM_BACKEND, SCOPEDMEM_BACKEND}:
            return EpisodeLongTermMemory.build(
                episode_id,
                embedding_provider=self.embedding_provider,
                embedding_dimensions=self.embedding_dimensions,
                embedding_max_chars=self.embedding_max_chars,
            )
        if self.native_driver is None:
            raise RuntimeError("external memory factory has no native driver")
        return ExternalEpisodeLongTermMemory(
            shadow=EpisodeLongTermMemory.build(episode_id),
            driver=self.native_driver,
        )

    def record(self) -> dict[str, object]:
        if self.native_driver is None:
            return {
                "backend": self.backend,
                "backend_id": "asvm_vector_v1",
                "native_external_backend": False,
            }
        return {
            "backend": self.backend,
            "backend_id": self.native_driver.backend_id,
            "upstream_revision": self.native_driver.upstream_revision,
            "native_external_backend": True,
        }


def add_memory_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--memory-backend",
        choices=MEMORY_BACKENDS,
        default=ASVM_BACKEND,
        help="Storage/retrieval backend below Return governance.",
    )
    parser.add_argument(
        "--memory-backend-storage-dir",
        type=Path,
        help="Process-local persistent state for native OSS backends.",
    )
    parser.add_argument(
        "--memory-backend-embedding-dimensions",
        type=int,
        help=(
            "Native backend vector width. This may differ from the ASVM "
            "projection width when an embedding server returns a fixed "
            "full-size vector."
        ),
    )
    parser.add_argument("--amem-evo-threshold", type=int, default=100)
    parser.add_argument("--amem-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--memobase-base-url",
        default=os.environ.get("TRACE_MEMOBASE_BASE_URL"),
        help="Self-hosted Memobase endpoint, with or without /api/v1.",
    )
    parser.add_argument(
        "--memobase-api-key-env",
        default="TRACE_MEMOBASE_API_KEY",
    )


def build_memory_backend_factory(
    args: argparse.Namespace,
    *,
    generation_api_key: str,
    embedding_provider: MemoryEmbeddingProvider | None,
    default_storage_dir: Path,
) -> MemoryBackendFactory:
    backend = str(args.memory_backend)
    if backend in {ASVM_BACKEND, SCOPEDMEM_BACKEND}:
        return MemoryBackendFactory(
            backend=backend,
            embedding_provider=embedding_provider,
            embedding_dimensions=args.embedding_dimensions,
            embedding_max_chars=args.embedding_max_chars,
        )

    storage_root = (
        args.memory_backend_storage_dir or default_storage_dir
    ).resolve()
    shard_dir = storage_root / (
        f"shard_{int(args.shard_index):02d}_of_"
        f"{int(args.shard_count):02d}"
    )
    shard_dir.mkdir(parents=True, exist_ok=True)

    if backend in {MEM0_BACKEND, AMEM_BACKEND}:
        if not args.embedding_base_url or not args.embedding_model:
            raise ValueError(
                "Mem0/A-MEM requires embedding-base-url and embedding-model"
            )
        native_dimensions = (
            args.memory_backend_embedding_dimensions
            or args.embedding_dimensions
        )
        if native_dimensions < 1:
            raise ValueError(
                "memory-backend-embedding-dimensions must be positive"
            )
        embedding_key = (
            os.environ.get(args.embedding_api_key_env)
            if args.embedding_api_key_env
            else None
        ) or "local"
        if backend == MEM0_BACKEND:
            from .mem0 import Mem0Driver
            driver_class = Mem0Driver
            extra = {}
        else:
            from .amem import AMemDriver
            driver_class = AMemDriver
            shard_dir = shard_dir / "amem"
            extra = dict(timeout=args.timeout, retries=args.retries,
                         evo_threshold=args.amem_evo_threshold, max_tokens=args.amem_max_tokens,
                         seed=getattr(args, "seed", 731))

        driver: NativeMemoryDriver = driver_class.build(
            storage_dir=shard_dir,
            generation_base_url=args.base_url,
            generation_model=args.model,
            generation_api_key=generation_api_key,
            embedding_base_url=args.embedding_base_url,
            embedding_model=args.embedding_model,
            embedding_api_key=embedding_key,
            embedding_dimensions=native_dimensions,
            **extra,
        )
    elif backend == MEMOBASE_BACKEND:
        if not args.memobase_base_url:
            raise ValueError("Memobase requires --memobase-base-url")
        memobase_key = os.environ.get(args.memobase_api_key_env)
        if not memobase_key:
            raise RuntimeError(
                f"missing API key env: {args.memobase_api_key_env}"
            )
        from .memobase import MemobaseDriver
        from .lineage import SourceLedger

        driver = MemobaseDriver(
            ledger=SourceLedger(shard_dir / "memobase_source_receipts"),
            base_url=args.memobase_base_url,
            api_key=memobase_key,
            timeout=args.timeout,
            retries=args.retries,
        )
    else:
        raise ValueError(f"unknown memory backend: {backend}")

    return MemoryBackendFactory(
        backend=backend,
        embedding_provider=None,
        embedding_dimensions=args.embedding_dimensions,
        embedding_max_chars=args.embedding_max_chars,
        native_driver=driver,
    )


__all__ = [
    "ASVM_BACKEND",
    "SCOPEDMEM_BACKEND",
    "AMEM_BACKEND",
    "MEM0_BACKEND",
    "MEMORY_BACKENDS",
    "MEMOBASE_BACKEND",
    "MemoryBackendFactory",
    "add_memory_backend_arguments",
    "build_memory_backend_factory",
]
