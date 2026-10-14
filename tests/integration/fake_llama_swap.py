"""Stand-in for llama-swap's OpenAI-compatible `/v1/chat/completions`, run as its own process by
`test_resilience_llama_swap.py` so the test can kill it mid-request and start it again on the same
port. Every request is appended to `--log`; with `--hang` the server accepts requests and never
answers, like a llama-server busy on a long generation."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True, help="written once listening")
    parser.add_argument("--response", type=Path, required=True, help="JSON body for every 200")
    parser.add_argument("--hang", action="store_true")
    args = parser.parse_args()

    body = args.response.read_bytes()
    lock = threading.Lock()
    never = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive, so the client pools a connection to kill

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            with lock, args.log.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"path": self.path, "model": request.get("model")}) + "\n")
            if args.hang:
                never.wait()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *log_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    args.ready.write_text("ready", encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
