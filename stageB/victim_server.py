"""
VICTIM (runs on the Windows host). Threaded web server + a few open 'decoy' ports for nmap to find,
plus a /mark endpoint: the VM calls it at the start/end of every attack phase, and the HOST clock is
logged to attack_log.csv. That gives ground-truth labels with no VM/host clock-skew problems.

  python victim_server.py --bind 192.168.xx.1
"""
import argparse, csv, os, socket, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ap = argparse.ArgumentParser()
ap.add_argument("--bind", required=True, help="host-only adapter IP (VMnet1)")
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--log", default="attack_log.csv")
a = ap.parse_args()

if not os.path.exists(a.log):
    with open(a.log, "w", newline="") as f: csv.writer(f).writerow(["epoch", "phase", "state"])
BIG = b"x" * 150_000

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *args): pass                    # no console spam during floods
    def _send(self, body, code=200):
        self.send_response(code); self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/mark":
            q = parse_qs(u.query); phase = q.get("phase", ["?"])[0]; state = q.get("state", ["?"])[0]
            with open(a.log, "a", newline="") as f: csv.writer(f).writerow([f"{time.time():.3f}", phase, state])
            print(time.strftime("%H:%M:%S"), "MARK", phase, state); return self._send(b"ok")
        if u.path == "/big":  return self._send(BIG)
        if u.path == "/slow": time.sleep(1); return self._send(b"slow ok")
        self._send(b"<html><body><h1>Lab victim</h1><p>hello</p></body></html>")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n); self._send(b"posted")

def decoy(port, banner):
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((a.bind, port)); s.listen(50)
    def loop():
        while True:
            try:
                c, _ = s.accept(); c.sendall(banner); time.sleep(0.2); c.close()
            except OSError: pass
    threading.Thread(target=loop, daemon=True).start()

decoy(2121, b"220 FTP ready\r\n"); decoy(2222, b"SSH-2.0-OpenSSH_8.9\r\n")
srv = ThreadingHTTPServer((a.bind, a.port), H); srv.daemon_threads = True
srv.request_queue_size = 256
print(f"victim up on {a.bind}:{a.port} (+2121, 2222). marks -> {a.log}")
srv.serve_forever()
