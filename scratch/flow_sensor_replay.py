"""
SENSOR (simulation mode): streams held-out CICIDS2017 test flows to the detector over TCP,
as if a network tap were emitting flow records in real time. Run it in a separate terminal
(or on another machine / VM - just change --host).

  python flow_sensor_replay.py --mode scenario          # scripted demo: benign + attack bursts
  python flow_sensor_replay.py --mode natural --rate 20 # random order, real class mix (mostly benign)
"""
import argparse, json, random, socket, time
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=9009)
ap.add_argument("--csv", default="replay_flows.csv")
ap.add_argument("--mode", choices=["scenario", "natural"], default="scenario")
ap.add_argument("--rate", type=float, default=10, help="flows/sec for natural mode")
ap.add_argument("--loops", type=int, default=0, help="scenario repeats (0 = forever)")
a = ap.parse_args()

df = pd.read_csv(a.csv)
feat_cols = [c for c in df.columns if c not in ("Timestamp", "Source IP", "Source Port",
                                                "Destination IP", "Destination Port", "label_id")]
by_class = {c: g for c, g in df.groupby("label_id")}

# (class_id, seconds, flows_per_sec)   4=Benign 5=PortScan 1=DDoS 2=DoS 0=Bot 3=Infiltration
SCENARIO = [(4, 15, 8), (5, 8, 25), (4, 10, 8), (1, 8, 30), (4, 10, 8),
            (2, 8, 25), (4, 8, 8), (0, 6, 5), (4, 8, 8), (3, 4, 2)]

def record(row, source="replay"):
    return {"ts": time.time(), "src": row.get("Source IP", ""), "dst": row.get("Destination IP", ""),
            "dport": int(row["Destination Port"]) if "Destination Port" in row else 0,
            "features": {c: float(row[c]) for c in feat_cols}, "true": int(row["label_id"]), "source": source}

def send_loop():
    s = socket.create_connection((a.host, a.port))
    print(f"connected to detector {a.host}:{a.port}")
    def push(row):
        s.sendall((json.dumps(record(row)) + "\n").encode())
    if a.mode == "natural":
        while True:
            push(df.sample(1).iloc[0]); time.sleep(1 / a.rate)
    n = 0
    while a.loops == 0 or n < a.loops:
        for cls, secs, rate in SCENARIO:
            if cls not in by_class: continue
            print(f"phase: class {cls} for {secs}s @ {rate}/s")
            end = time.time() + secs
            while time.time() < end:
                push(by_class[cls].sample(1).iloc[0]); time.sleep(random.expovariate(rate))  # Poisson-ish arrivals
        n += 1

try: send_loop()
except (ConnectionRefusedError, OSError) as e:
    print("Could not reach the detector. Start the dashboard page and click 'Start listener' first.", e)
