#!/usr/bin/env python3
"""Static demo server with the headers required for WASM threads.

  python serve.py [--port 8123]
"""
import argparse
import functools
import http.server
from pathlib import Path

ROOT = Path(__file__).resolve().parent
class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean == "/backend.js":
            return str(ROOT / "backend-onnx.js")
        return super().translate_path(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cert", default=None)
    parser.add_argument("--key", default=None)
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    scheme = "http"
    if args.cert and args.key:
        import ssl

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(args.cert, args.key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    print(f"{scheme}://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
