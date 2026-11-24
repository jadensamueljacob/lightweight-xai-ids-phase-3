"""
Diagnostic: do the flows my live sensor produces LOOK like the CICIDS2017 flows the model learned?

  1. run the sensor with:  python flow_sensor_live.py --iface "..." --dump live_features.csv
  2. run vm_attacks.sh (benign / portscan / dos_syn are enough)
  3. python stageB\\compare_features.py --dump live_features.csv --log attack_log.csv --host-ip 192.168.72.1 --train replay_flows.csv
"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dump", default="live_features.csv"); ap.add_argument("--log", default="attack_log.csv")
ap.add_argument("--host-ip", required=True); ap.add_argument("--train", default="replay_flows.csv")
ap.add_argument("--grace", type=float, default=5.0)
a = ap.parse_args()
NAMES = {0: "Bot", 1: "DDoS", 2: "DoS", 3: "Infiltration", 4: "Benign", 5: "PortScan"}

live = pd.read_csv(a.dump); live = live[live["dst"] == a.host_ip]
log = pd.read_csv(a.log); opened, wins = {}, []
for _, r in log.iterrows():
    if r["state"] == "start": opened[r["phase"]] = r["epoch"]
    elif r["state"] == "end" and r["phase"] in opened: wins.append((r["phase"], opened.pop(r["phase"]), r["epoch"]))
live["phase"] = None
for ph, s, e in wins: live.loc[(live["ts"] >= s) & (live["ts"] <= e + a.grace), "phase"] = ph
live = live.dropna(subset=["phase"])
live["kind"] = live["phase"].str.split("_").str[0]
kind_to_class = {"benign": 4, "portscan": 5, "dos": 2}

train = pd.read_csv(a.train)
feats = [c for c in live.columns if c in train.columns and c not in ("label_id",)]
print("live flows per kind:", live["kind"].value_counts().to_dict())
print("packets per flow (median fwd/bwd):", live.groupby("kind")[["n_fwd", "n_bwd"]].median().to_dict("index"), "\n")

for kind, cid in kind_to_class.items():
    L = live[live["kind"] == kind]
    if L.empty: continue
    T = train[train["label_id"] == cid]
    print(f"=== live '{kind}' ({len(L)} flows)  vs  CICIDS {NAMES[cid]} ({len(T)} test flows) ===")
    rows = []
    for f in feats:
        lo, hi = T[f].quantile(0.05), T[f].quantile(0.95)
        inside = ((L[f] >= lo) & (L[f] <= hi)).mean()
        rows.append({"feature": f, "live_median": L[f].median(), "cic_median": T[f].median(),
                     "cic_p5": lo, "cic_p95": hi, "live_in_cic_5-95%": f"{inside:.0%}"})
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:,.4g}")
    print(pd.DataFrame(rows).to_string(index=False)); print()