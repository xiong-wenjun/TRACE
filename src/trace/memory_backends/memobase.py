"""Official Memobase REST adapter for RETURN memory experiments.

The upstream Python client currently requires Python 3.11 while the benchmark
host uses Python 3.10.  This module therefore calls the documented REST API
directly; it does not reproduce Memobase extraction or profile logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import json
import time
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..providers import endpoint_origin, _open_external_no_redirect
import uuid

from .base import BackendMemory
from .lineage import SourceLedger, checked_entries, verify_import


MEMOBASE_UPSTREAM_REVISION = "358c16bbc6d687937d79bc2f984a11c3be8da901"
_USER_NAMESPACE = uuid.UUID("f6a7bbdc-4060-5e43-a928-45bb414460d4")


class MemobaseAPIError(RuntimeError):
    """Application-level error returned inside a successful HTTP response."""

    def __init__(self, errno: int, message: str) -> None:
        super().__init__(f"Memobase error {errno}: {message}")
        self.errno = errno
        self.message = message


def _is_not_found(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        return error.code == 404
    return isinstance(error, MemobaseAPIError) and (error.errno == 404 or (
        error.errno == 1 and "user does not exist" in error.message.lower()))


@dataclass
class MemobaseDriver:
    """Native profile formation and retrieval through Memobase's REST API."""

    base_url: str
    api_key: str = field(repr=False)
    timeout: float = 300.0
    retries: int = 3
    backend_id: str = "memobase_oss_v0_0_27"
    upstream_revision: str = MEMOBASE_UPSTREAM_REVISION
    ledger: SourceLedger = field(default_factory=SourceLedger)
    _audit: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        endpoint_origin(self.base_url)
        if any(c in self.api_key for c in ("\r", "\n")):
            raise ValueError("Memobase API key contains invalid header characters")
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.endswith("/api/v1"):
            self.base_url += "/api/v1"
        if not self.api_key:
            raise ValueError("Memobase API key must be non-empty")
        if self.timeout <= 0 or self.retries < 0:
            raise ValueError("invalid Memobase transport settings")

    @staticmethod
    def _user_id(namespace: str) -> str:
        return str(uuid.uuid5(_USER_NAMESPACE, namespace))

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        body = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        request = Request(
            self.base_url + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        last_error: Exception | None = None
        # Mutations may have succeeded before a timeout. Never blindly replay
        # a profile/blob POST; a fresh episode can reset its namespace safely.
        attempts = self.retries + 1 if method in {"GET", "DELETE"} else 1
        for attempt in range(attempts):
            try:
                with _open_external_no_redirect(request, timeout=self.timeout) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, Mapping):
                    raise RuntimeError("Memobase returned a non-object response")
                errno = int(decoded.get("errno", 0) or 0)
                if errno != 0:
                    raise MemobaseAPIError(
                        errno,
                        str(decoded.get("errmsg") or ""),
                    )
                data = decoded.get("data")
                return data if isinstance(data, Mapping) else {}
            except HTTPError as error:
                last_error = error
                if error.code == 404:
                    raise
                if error.code < 500 or attempt + 1 >= attempts:
                    raise
            except (URLError, TimeoutError) as error:
                last_error = error
                if attempt + 1 >= attempts:
                    raise
            time.sleep(min(2.0**attempt, 8.0))
        assert last_error is not None
        raise last_error

    def _ensure_user(self, namespace: str) -> str:
        user_id = self._user_id(namespace)
        try:
            self._request_json("GET", f"/users/{user_id}")
        except (HTTPError, MemobaseAPIError) as error:
            if not _is_not_found(error):
                raise
            self._request_json(
                "POST",
                "/users",
                {"id": user_id, "data": {"trace_namespace": namespace}},
            )
        return user_id

    def _profiles(
        self,
        namespace: str,
        *,
        query: str | None = None,
        limit: int = 1000,
    ) -> tuple[Mapping[str, object], ...]:
        user_id = self._ensure_user(namespace)
        # No token/rank budget on checkpoint reads: the API defaults to all.
        params: dict[str, object] = {}
        if query:
            params["max_token_size"] = 16384
            params["chats_str"] = json.dumps(
                [{"role": "user", "content": query}],
                ensure_ascii=False,
            )
        data = self._request_json(
            "GET",
            f"/users/profile/{user_id}?{urlencode(params)}",
        )
        raw = data.get("profiles", ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise RuntimeError("Memobase profiles must be an array")
        if any(not isinstance(row, Mapping) for row in raw):
            raise RuntimeError("Memobase returned a malformed profile")
        if query is None and len(raw) >= limit:
            raise RuntimeError("Memobase snapshot may exceed capacity; refusing truncation")
        return tuple(
            row for row in raw[: max(1, limit)] if isinstance(row, Mapping)
        )

    def _entry(
        self,
        namespace: str,
        row: Mapping[str, object],
        *,
        score: float,
    ) -> BackendMemory:
        profile_id = str(row.get("id") or "")
        attributes = row.get("attributes")
        metadata = (
            dict(attributes) if isinstance(attributes, Mapping) else {}
        )
        metadata["native_profile_attributes"] = dict(metadata)
        metadata["native_revision"] = str(row.get("updated_at") or "")
        return BackendMemory(
            external_id=profile_id,
            text=str(row.get("content") or "").strip(),
            score=score,
            source_workstate_ids=(),
            metadata=metadata,
        )

    def reset_namespace(self, namespace: str) -> None:
        user_id = self._user_id(namespace)
        try:
            self._request_json("DELETE", f"/users/{user_id}")
        except (HTTPError, MemobaseAPIError) as error:
            if not _is_not_found(error):
                raise
        self.ledger.reset(namespace)
        self._request_json(
            "POST",
            "/users",
            {"id": user_id, "data": {"trace_namespace": namespace}},
        )
        self._audit.append({"operation": "reset", "namespace": namespace})

    def write(
        self,
        namespace: str,
        text: str,
        metadata: Mapping[str, object],
    ) -> tuple[BackendMemory, ...]:
        user_id = self._ensure_user(namespace)
        before = self.list(namespace)
        try:
            return self._write_profile(namespace, text, metadata, before, user_id)
        except Exception:
            self.ledger.fail(namespace)
            raise

    def _write_profile(self, namespace, text, metadata, before, user_id):
        self._request_json(
            "POST",
            f"/blobs/insert/{user_id}?wait_process=true",
            {
                "blob_type": "chat",
                "fields": {"trace_source": dict(metadata)},
                "blob_data": {
                    "messages": [{"role": "user", "content": text}]
                },
            },
        )
        # Memobase buffers short ChatBlobs until its token threshold is met.
        # The official clients expose this operation as ``user.flush(sync=True)``;
        # invoke the corresponding REST endpoint so a benchmark write has a
        # materialized native profile before provenance is reconciled below.
        self._request_json(
            "POST",
            f"/users/buffer/{user_id}/chat?wait_process=true",
        )
        after = checked_entries(tuple(self._entry(namespace, row, score=1.0)
                                      for row in self._profiles(namespace)))
        changed = self.ledger.observe(namespace, before, after, metadata)
        entries = tuple(self.ledger.decorate(namespace, e) for e in after if e.external_id in changed)
        self._audit.append(
            {
                "operation": "native_chat_insert",
                "namespace": namespace,
                "changed_profile_count": len(entries),
                "buffer_flushed_synchronously": True,
            }
        )
        return entries

    def list(self, namespace: str) -> tuple[BackendMemory, ...]:
        self.ledger.check(namespace)
        rows = self._profiles(namespace)
        return checked_entries(tuple(self.ledger.decorate(
            namespace, self._entry(namespace, row, score=1.0)) for row in rows))

    def import_entries(
        self,
        namespace: str,
        entries: Sequence[BackendMemory],
    ) -> tuple[BackendMemory, ...]:
        self.ledger.check(namespace)
        checked_entries(entries)
        try:
            user_id = self._ensure_user(namespace)
            imported: list[BackendMemory] = []
            for index, source in enumerate(entries):
                data = self._request_json(
                    "POST",
                    f"/users/profile/{user_id}",
                    {
                        "content": source.text,
                        "attributes": dict(source.metadata.get("native_profile_attributes") or {
                            "topic": "trace_checkpoint",
                            "sub_topic": f"memory_{index:04d}",
                        }),
                    },
                )
                profile_id = str(data.get("id") or "")
                if not profile_id:
                    self.ledger.fail(namespace)
                    raise RuntimeError("Memobase import returned no profile ID")
                metadata = {
                    **dict(source.metadata),
                    "native_checkpoint_import": True,
                    "source_native_memory_id": source.external_id,
                }
                self.ledger.bind(namespace, BackendMemory(profile_id, source.text),
                                 source.source_workstate_ids,
                                 complete=bool(source.metadata.get("provenance_complete", bool(source.source_workstate_ids))),
                                 metadata=metadata)
                imported.append(
                    BackendMemory(
                        external_id=profile_id,
                        text=source.text,
                        score=1.0,
                        source_workstate_ids=source.source_workstate_ids,
                        metadata=metadata,
                    )
                )
            self._audit.append(
                {
                    "operation": "checkpoint_profile_import",
                    "namespace": namespace,
                    "source_count": len(entries),
                    "result_count": len(imported),
                    "native_inference_repeated": False,
                }
            )
            verify_import(entries, self.list(namespace))
            return tuple(imported)
        except Exception:
            self.ledger.fail(namespace)
            raise

    def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int,
    ) -> tuple[BackendMemory, ...]:
        self.ledger.check(namespace)
        if limit < 1:
            raise ValueError("search limit must be positive")
        rows = self._profiles(namespace, query=query, limit=limit)
        denominator = max(1, len(rows))
        entries = tuple(
            self.ledger.decorate(namespace, self._entry(
                namespace,
                row,
                score=1.0 - (index / (denominator + 1)),
            ))
            for index, row in enumerate(rows)
        )
        self._audit.append(
            {
                "operation": "native_profile_retrieval",
                "namespace": namespace,
                "limit": limit,
                "result_count": len(entries),
            }
        )
        return checked_entries(entries)

    def record(self) -> dict[str, object]:
        counts = Counter(str(item["operation"]) for item in self._audit)
        return {
            "schema_version": "memobase_driver_v2",
            "adapter_revision": "source_complete_snapshots_v2",
            "lineage_policy": "conservative_formation_scope",
            "backend_id": self.backend_id,
            "upstream_revision": self.upstream_revision,
            "transport": "official_rest_api",
            "native_profile_formation": True,
            "native_profile_retrieval": True,
            "branch_import_repeats_inference": False,
            "operation_counts": dict(sorted(counts.items())),
            "api_key_persisted": False,
        }


__all__ = ["MemobaseAPIError", "MemobaseDriver"]
