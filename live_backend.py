"""
Detector backend: a TCP listener that receives flow records (JSON lines) from ANY sensor
(CSV replay or real packet capture), scales them, classifies them, and keeps results for the dashboard.

Sensor --(JSON over TCP)--> [receiver threads] -> queue -> [worker: scale + predict in micro-batches] -> results
"""
import json, pickle, socket, threading, time, queue
from collections import deque
import numpy as np
from flow_adapter import FlowAdapter

# Verify against the "Classes:" line your preprocessing printed (alphabetical LabelEncoder order).
CLASS_NAMES = {0: "Bot", 1: "DDoS", 2: "DoS", 3: "Infiltration", 4: "Benign", 5: "PortScan"}
BENIGN = 4


class LiveDetector:
    def __init__(self, model_path="model.pkl", scaler_path="scaler_15.json", host="0.0.0.0", port=9009):
        self.model = pickle.load(open(model_path, "rb"))
        sc = json.load(open(scaler_path))
        self.features = sc["features"]
        self.dmin = np.array(sc["data_min"]); rng = np.array(sc["data_max"]) - self.dmin
        self.rng = np.where(rng == 0, 1.0, rng)
        self.host, self.port = host, port
        self.adapter = FlowAdapter.load("adapter.json")   # applied to live-capture flows only
        print("adapter:", "OFF" if self.adapter is None else f"k={self.adapter.k:.1f}")
        self.q = queue.Queue(maxsize=50000)
        self.results = deque(maxlen=20000)      # newest at right
        self.lock = threading.Lock()
        self.total = self.dropped = 0
        self.lat_ms = deque(maxlen=2000)        # per-flow model time
        self.e2e_ms = deque(maxlen=2000)        # sensor timestamp -> verdict
        self.running = False
        self.error = None
        self._server = None
        self.connections = 0

    # ---------- lifecycle ----------
    def start(self):
        if self.running: return
        self.running = True; self.error = None
        try:
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((self.host, self.port)); self._server.listen(8)
        except OSError as e:
            self.running = False; self.error = str(e); return
        threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    def stop(self):
        self.running = False
        try: self._server.close()
        except Exception: pass

    def clear(self):
        with self.lock:
            self.results.clear(); self.total = self.dropped = 0
            self.lat_ms.clear(); self.e2e_ms.clear()

    # ---------- network side ----------
    def _accept(self):
        while self.running:
            try: conn, addr = self._server.accept()
            except OSError: break
            self.connections += 1
            threading.Thread(target=self._client, args=(conn,), daemon=True).start()

    def _client(self, conn):
        buf = b""
        with conn:
            while self.running:
                try: data = conn.recv(65536)
                except OSError: break
                if not data: break
                buf += data
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    if not ln.strip(): continue
                    try: rec = json.loads(ln)
                    except ValueError: continue
                    rec["_rx"] = time.time()
                    try: self.q.put_nowait(rec)
                    except queue.Full: self.dropped += 1

    # ---------- detection ----------
    def _worker(self):
        while self.running:
            try: batch = [self.q.get(timeout=0.5)]
            except queue.Empty: continue
            while len(batch) < 256:
                try: batch.append(self.q.get_nowait())
                except queue.Empty: break
            try:
                raw = np.array([[float((self.adapter(r["features"]) if (self.adapter and r.get("source") == "live") else r["features"])[f])
                                 for f in self.features] for r in batch])
            except (KeyError, ValueError, TypeError):
                continue
            X = (raw - self.dmin) / self.rng
            t0 = time.perf_counter()
            proba = self.model.predict_proba(X)
            dt = (time.perf_counter() - t0) * 1000 / len(batch)
            pred = self.model.classes_[proba.argmax(1)]
            now = time.time()
            ood = ((X < -0.05) | (X > 1.05)).mean(axis=1)     # share of features outside the training range
            with self.lock:
                for r, x, p, pr, o in zip(batch, X, pred, proba, ood):
                    true = r.get("true")
                    self.results.append({
                        "t": r.get("ts", r["_rx"]), "src": r.get("src", ""), "dst": r.get("dst", ""),
                        "dport": r.get("dport", ""), "pred": int(p), "conf": float(pr.max()),
                        "true": None if true is None else int(true), "x": x, "ood": float(o),
                        "source": r.get("source", "?"), "proba": pr,
                    })
                    self.lat_ms.append(dt)
                    self.e2e_ms.append((now - r.get("ts", r["_rx"])) * 1000)
                self.total += len(batch)

    # ---------- dashboard side ----------
    def snapshot(self, n=None):
        with self.lock:
            res = list(self.results) if n is None else list(self.results)[-n:]
            return res, self.total, self.dropped, list(self.lat_ms), list(self.e2e_ms)


_lime = None
def lime_explain(det, x, label, X_res_path="X_res.npy", num_features=10):
    """On-demand LIME for one scaled flow vector (same background data as the main dashboard)."""
    global _lime
    from lime.lime_tabular import LimeTabularExplainer
    if _lime is None:
        _lime = LimeTabularExplainer(np.load(X_res_path), feature_names=det.features,
                                     class_names=[CLASS_NAMES[i] for i in range(6)], mode="classification")
    return _lime.explain_instance(x, det.model.predict_proba, labels=(int(label),), num_features=num_features)