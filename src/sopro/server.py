from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import http.server
import json
import sys
import threading
import urllib.parse
import webbrowser
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from sopro.model import Reference, SoproTTS


def _static_root() -> Path:
    packaged = Path(__file__).with_name("demo")
    if packaged.is_dir():
        return packaged
    source = Path(__file__).resolve().parents[2] / "demos" / "web"
    if source.is_dir():
        return source
    raise FileNotFoundError("Sopro demo assets are missing")


def _device(value: str) -> str:
    if value != "auto":
        return value
    if sys.platform == "darwin":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _audio_bytes(wav: torch.Tensor) -> bytes:
    return wav.detach().float().cpu().reshape(-1).numpy().astype("<f4", copy=False).tobytes()


class DemoState:
    def __init__(self, model: str, device: str, int8: bool) -> None:
        self.model_name = model
        self.full_device = _device(device)
        self.device = "cpu" if int8 else self.full_device
        self.quantization = "int8" if int8 else None
        self.model: Optional[SoproTTS] = None
        self.model_lock = threading.Lock()
        self.generation_lock = threading.Lock()
        self.references: OrderedDict[str, Reference] = OrderedDict()

    def configure(self, precision: str) -> None:
        if precision not in {"full", "int8"}:
            raise ValueError("precision must be 'full' or 'int8'")
        quantization = "int8" if precision == "int8" else None
        device = "cpu" if quantization else self.full_device
        with self.generation_lock, self.model_lock:
            if quantization == self.quantization and device == self.device:
                return
            self.model = None
            self.references.clear()
            self.quantization = quantization
            self.device = device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mps_backend = getattr(torch.backends, "mps", None)
        mps = getattr(torch, "mps", None)
        if mps_backend and mps_backend.is_available() and hasattr(mps, "empty_cache"):
            mps.empty_cache()

    def load(self) -> SoproTTS:
        if self.model is not None:
            return self.model
        with self.model_lock:
            if self.model is None:
                self.model = SoproTTS.from_pretrained(
                    self.model_name,
                    device=self.device,
                    quantization=self.quantization,
                )
        return self.model

    def info(self, precision: Optional[str] = None) -> Dict[str, Any]:
        if precision is not None:
            self.configure(precision)
        model = self.load()
        return {"sample_rate": model.sample_rate, "device": self.device, "model": self.model_name, "precision": "int8" if self.quantization else "full"}

    def prepare_reference(self, body: bytes, sample_rate: int) -> Dict[str, Any]:
        if len(body) % 4:
            raise ValueError("invalid reference audio")
        key = hashlib.sha256(int(sample_rate).to_bytes(4, "little") + body).hexdigest()
        with self.generation_lock:
            if key in self.references:
                self.references.move_to_end(key)
                return {"id": key, "fromCache": True}
            samples = torch.frombuffer(bytearray(body), dtype=torch.float32)
            reference = self.load().prepare_reference(
                ref_audio=samples,
                sample_rate=sample_rate,
                stream=True,
            )
            self.references[key] = reference
            while len(self.references) > 3:
                self.references.popitem(last=False)
        return {"id": key, "fromCache": False}

    def reference(self, key: str) -> Reference:
        try:
            reference = self.references[key]
        except KeyError as error:
            raise ValueError("reference expired; upload it again") from error
        self.references.move_to_end(key)
        return reference

    def options(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "lang": request.get("language") or None,
            "temperature": request.get("temperature"),
            "top_p": request.get("top_p"),
            "top_k": request.get("top_k"),
        }


class DemoHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: DemoState

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store" if self.path.startswith("/api/") else "no-cache")
        super().end_headers()

    def translate_path(self, path: str) -> str:
        if path.split("?", 1)[0] == "/backend.js":
            return str(_static_root() / "backend-python.js")
        return super().translate_path(path)

    def _body(self, limit: int) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > limit:
            raise ValueError("invalid request size")
        return self.rfile.read(length)

    def _request(self) -> Dict[str, Any]:
        return json.loads(self._body(64 * 1024))

    def _json(self, status: int, value: Dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, error: Exception) -> None:
        client_error = isinstance(error, (KeyError, ValueError, json.JSONDecodeError))
        self._json(400 if client_error else 500, {"error": str(error)})

    def do_GET(self) -> None:
        url = urllib.parse.urlsplit(self.path)
        if url.path != "/api/info":
            super().do_GET()
            return
        try:
            precision = urllib.parse.parse_qs(url.query).get("precision", [None])[0]
            self._json(200, self.state.info(precision))
        except Exception as error:
            self._error(error)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            if path == "/api/reference":
                sample_rate = int(self.headers.get("X-Sopro-Sample-Rate", "0"))
                if sample_rate < 8000 or sample_rate > 192000:
                    raise ValueError("invalid sample rate")
                self._json(200, self.state.prepare_reference(self._body(sample_rate * 4 * 60), sample_rate))
            elif path == "/api/synthesize":
                self._synthesize(self._request())
            elif path == "/api/stream":
                self._stream(self._request())
            else:
                self.send_error(404)
        except Exception as error:
            self._error(error)

    def _seed(self, request: Dict[str, Any]) -> None:
        if request.get("seed") is not None:
            torch.manual_seed(int(request["seed"]))

    def _synthesize(self, request: Dict[str, Any]) -> None:
        with self.state.generation_lock:
            self._seed(request)
            audio = self.state.load().synthesize(
                str(request.get("text", "")),
                ref=self.state.reference(str(request.get("reference", ""))),
                **self.state.options(request),
            )
            body = _audio_bytes(audio)
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, request: Dict[str, Any]) -> None:
        with self.state.generation_lock:
            self._seed(request)
            chunks = self.state.load().stream(
                str(request.get("text", "")),
                ref=self.state.reference(str(request.get("reference", ""))),
                **self.state.options(request),
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for chunk in chunks:
                    body = _audio_bytes(chunk)
                    if not body:
                        continue
                    self.wfile.write(f"{len(body):x}\r\n".encode())
                    self.wfile.write(body)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                self.log_error("stream failed: %s", error)
                self.close_connection = True


def serve(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="soprotts serve",
        description="Run the local Sopro demo",
    )
    parser.add_argument("--model", default="samuel-vitorino/sopro-v2-turbo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--int8", action="store_true", help="use int8 AR weights on CPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)
    state = DemoState(args.model, args.device, args.int8)
    if args.int8 and _device(args.device) != "cpu":
        parser.error("--int8 requires --device cpu")
    DemoHandler.state = state
    handler = functools.partial(DemoHandler, directory=str(_static_root()))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{display_host}:{server.server_port}"
    print(f"Sopro demo: {url}")
    suffix = " (int8)" if args.int8 else ""
    print(f"Model: {args.model}, device: {state.device}{suffix}")
    if not args.no_open:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
