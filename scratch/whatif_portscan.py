"""
Which feature is stopping your live portscan flows from being called PortScan?

For the live portscan flows already in live_features.csv, overwrite ONE feature (or one group) at a time with
the CICIDS PortScan median and re-predict. Diagnostic only - nothing is saved and nothing is sent to the dashboard.

  python whatif_portscan.py --host-ip 192.168.72.1
"""
import argparse, json, os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from flow_adapter import FlowAdapter

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default="live_features.csv"); ap.add_argument("--log", default="attack_log.csv")
ap.add_argument("--host-ip", required=True); ap.add_argument("--train", default="replay_flows.csv")
ap.add_argument("--model", default="model.pkl"); ap.add_argument("--scaler", default="scaler_15.json")
ap.add_argument("--grace", type=float, default=5.0); ap.add_argument("--no-adapter", action="store_true")
a = ap.parse_args()

NAMES = {0: "Bot", 1: "DDoS", 2: "DoS", 3: "Infiltration", 4: "Benign", 5: "PortScan"}
sc = json.load(open(a.scaler)); feats = sc["features"]
dmin = np.array(sc["data_min"]); rng = np.array(sc["data_max"]) - dmin; rng = np.where(rng == 0, 1, rng)
model = pickle.load(open(a.model, "rb"))

live = pd.read_csv(a.dump); live = live[live["dst"] == a.host_ip]
log = pd.read_csv(a.log); opened, wins = {}, []
for _, r in log.iterrows():
    if r["state"] == "start": opened[r["phase"]] = r["epoch"]
    elif r["state"] == "end" and r["phase"] in opened: wins.append((r["phase"], opened.pop(r["phase"]), r["epoch"]))
mask = np.zeros(len(live), bool)
for ph, s, e in wins:
    if ph.startswith("portscan"): mask |= ((live["ts"] >= s) & (live["ts"] <= e + a.grace)).to_numpy()
L = live[mask][feats].reset_index(drop=True)
if L.empty: sys.exit("no portscan flows found in the dump")

ad = FlowAdapter.load()
if ad and not a.no_adapter:
    L = pd.DataFrame([ad(dict(r)) for _, r in L.iterrows()])[feats]; print(f"adapter ON (k={ad.k})")
else: print("adapter OFF")

train = pd.read_csv(a.train); T = train[train["label_id"] == 5]
print(f"{len(L)} live portscan flows   vs   {len(T)} CICIDS PortScan flows\n")

def predict(df):
    p = model.predict((df[feats].to_numpy(float) - dmin) / rng)
    return pd.Series(p).map(NAMES).value_counts(normalize=True).mul(100).round(1).to_dict()

print("== Feature comparison (raw units) ==")
rows = [{"feature": f, "live_median": L[f].median(), "cic_median": T[f].median(),
         "cic_p5": T[f].quantile(.05), "cic_p95": T[f].quantile(.95),
         "live_in_p5-p95": f"{((L[f] >= T[f].quantile(.05)) & (L[f] <= T[f].quantile(.95))).mean():.0%}"} for f in feats]
pd.set_option("display.width", 220); pd.set_option("display.float_format", lambda v: f"{v:,.4g}")
print(pd.DataFrame(rows).to_string(index=False))

print("\n== Baseline ==\n", predict(L))
print("\n== Replace ONE feature with the CICIDS PortScan median ==")
for f in feats:
    X = L.copy(); X[f] = T[f].median(); print(f"{f:28s} -> {predict(X)}")

groups = {"windows": [f for f in feats if "Init_Win" in f],
          "lengths": [f for f in feats if "Length" in f or "Segment" in f],
          "timing": [f for f in feats if "IAT" in f or "/s" in f]}
print("\n== Replace a GROUP ==")
for g, cols in groups.items():
    X = L.copy()
    for c in cols: X[c] = T[c].median()
    print(f"{g:10s} -> {predict(X)}")
X = L.copy()
for c in feats: X[c] = T[c].median()
print(f"{'ALL 15':10s} -> {predict(X)}   (sanity check: should be ~100% PortScan)")
