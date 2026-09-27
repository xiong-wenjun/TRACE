#!/usr/bin/env python3
"""Proxy fixed-width OpenAI-compatible embeddings without matryoshka args.

Some native memory systems always send ``dimensions`` even when it equals a
model's fixed output width.  Qwen3-Embedding rejects that optional field.  The
proxy removes only ``dimensions`` and forwards every other byte-equivalent
request field and response unchanged.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def normalize_embedding_payload(payload: object) -> object:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    normalized.pop("dimensions", None)
    return normalized


class ProxyHandler(BaseHTTPRequestHandler):
    upstream: str
    timeout: float

    def _forward(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length) if length else None
        if body and self.path.rstrip("/").endswith("/embeddings"):
            body = json.dumps(
                normalize_embedding_payload(json.loads(body)),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.casefold() not in {"host", "content-length", "connection"}
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))
        request = Request(
            self.upstream + self.path,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read()
                self.send_response(response.status)
                for key, value in response.headers.items():
                    if key.casefold() not in {
                        "connection",
                        "content-length",
                        "transfer-encoding",
                    }:
                        self.send_header(key, value)
        except HTTPError as error:
            response_body = error.read()
            self.send_response(error.code)
            for key, value in error.headers.items():
                if key.casefold() not in {
                    "connection",
                    "content-length",
                    "transfer-encoding",
                }:
                    self.send_header(key, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    do_GET = _forward
    do_POST = _forward

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18003)
    parser.add_argument(
        "--upstream",
        default="https://dashscope-us.aliyuncs.com/compatible-mode",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    handler = type(
        "ConfiguredProxyHandler",
        (ProxyHandler,),
        {"upstream": args.upstream.rstrip("/"), "timeout": args.timeout},
    )
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
