"""Local sink for the in-browser SportsCardsPro scraper.

The scraper runs inside a Chrome tab (Cloudflare blocks plain HTTP), and
POSTs JSON batches here so bulk data lands on disk instead of passing
through the agent. Each POST body is {"kind": str, "rows": [...]} and is
appended as JSON lines to raw/<kind>.jsonl.

    .venv/bin/python research/card_market/receiver.py   # listens on 127.0.0.1:8765
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

RAW = Path(__file__).parent / "raw"
RAW.mkdir(exist_ok=True)
_lock = Lock()

BRIDGE = b"""<!doctype html><title>scrape bridge</title><script>
addEventListener('message', async e => {
  const {seq, kind, rows} = e.data || {};
  if (!kind) return;
  let ok = false;
  try { ok = (await fetch('/', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({kind, rows})})).ok; } catch (err) {}
  e.source.postMessage({bridgeAck: seq, ok}, e.origin);
});
(window.opener || window.parent).postMessage({bridgeReady: true}, '*');
</script>bridge up"""


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        # /<file>.json serves a work list from this folder; anything else
        # returns line counts per kind, so the scraper can resume
        work = Path(__file__).parent / self.path.lstrip("/")
        ctype = "application/json"
        if self.path == "/bridge.html":
            # same-origin relay: a page that can't reach localhost directly
            # (Chrome local-network permission) postMessages batches here
            body, ctype = BRIDGE, "text/html"
        elif self.path.endswith(".json") and work.parent == RAW.parent and work.exists():
            body = work.read_bytes()
        else:
            counts = {p.stem: sum(1 for _ in p.open()) for p in RAW.glob("*.jsonl")}
            body = json.dumps(counts).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n))
        kind = "".join(c for c in payload["kind"] if c.isalnum() or c == "_")
        with _lock, (RAW / f"{kind}.jsonl").open("a") as f:
            for row in payload["rows"]:
                f.write(json.dumps(row) + "\n")
        self.send_response(200)
        self._cors()
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
