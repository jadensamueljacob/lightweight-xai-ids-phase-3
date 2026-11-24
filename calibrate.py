"""
Fits the calibration (adapter.json) from the BENIGN phases of a live capture, then evaluates offline how the
predictions change per phase type with and without it.

  python stageB\\calibrate.py --dump live_features.csv --log attack_log.csv --host-ip 192.168.72.1 --train replay_flows.csv

Honest-use rules: fit on a benign calibration run, evaluate on a DIFFERENT run. Never tune k by looking at
attack detection - that would be fitting the demo, not calibrating the environment.
"""
import argparse, json, pickle, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from flow_adapter import FlowAdapter

NAMES = {0: "Bot", 1: "DDoS", 2: "DoS", 3: "Infiltration", 4: "Benign", 5: "PortScan"}
ap = argparse.ArgumentParser()
ap.add_argument("--dump", default="live_features.csv"); ap.add_argument("--log", default="attack_log.csv")
ap.add_argument("--host-ip", required=True); ap.add_argument("--train", default="replay_flows.csv")
ap.add_argument("--model", default="model.pkl"); ap.add_argument("--scaler", default="scaler_15.json")
ap.add_argument("--grace", type=float, default=5.0); ap.add_argument("--no-write", action="store_true")
ap.add_argument("--k", type=float, help="use this fixed k instead of refitting")
a = ap.parse_args()

live = pd.read_csv(a.dump); live = live[live["dst"] == a.host_ip]
log = pd.read_csv(a.log); opened, wins = {}, []
for _, r in log.iterrows():
    if r["state"] == "start": opened[r["phase"]] = r["epoch"]
    elif r["state"] == "end" and r["phase"] in opened: wins.append((r["phase"], opened.pop(r["phase"]), r["epoch"]))
live["phase"] = None
for ph, s, e in wins: live.loc[(live["ts"] >= s) & (live["ts"] <= e + a.grace), "phase"] = ph
live = live.dropna(subset=["phase"]).copy(); live["kind"] = live["phase"].str.split("_").str[0]

train = pd.read_csv(a.train); cic_b = train[train["label_id"] == 4]; live_b = live[live["kind"] == "benign"]
ratios = {f: cic_b[f].median() / live_b[f].median() for f in ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max"] if live_b[f].median() > 0}
k = float(np.median(list(ratios.values())))
if a.k is not None: k = a.k; a.no_write = True
print("per-feature CICIDS/live benign median ratios:", {f: round(v, 1) for f, v in ratios.items()})
print(f"=> time_scale k = {k:.1f}  (fitted from {len(live_b)} benign flows only)\n")
if not a.no_write:
    json.dump({"time_scale": k}, open("adapter.json", "w"), indent=1); print("saved adapter.json\n")

model = pickle.load(open(a.model, "rb")); sc = json.load(open(a.scaler)); feats = sc["features"]
dmin = np.array(sc["data_min"]); rng = np.array(sc["data_max"]) - dmin; rng = np.where(rng == 0, 1, rng)
ad = FlowAdapter({"time_scale": k})
def predict(df):
    X = (df[feats].to_numpy(float) - dmin) / rng
    return pd.Series(model.predict(X), index=df.index).map(NAMES)
adapted = pd.DataFrame([ad({f: row[f] for f in feats}) for _, row in live.iterrows()], index=live.index)
for name, df in [("without adapter", live), ("with adapter", live.assign(**adapted[feats].to_dict("series")))]:
    print(f"--- {name}: % predicted class per phase type ---")
    print((pd.crosstab(live["kind"], predict(df), normalize="index") * 100).round(1).to_string(), "\n")
