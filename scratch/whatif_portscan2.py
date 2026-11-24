"""
Round 2: use REAL CICIDS PortScan rows (not medians) and the tree's own decision path.

  python whatif_portscan2.py --host-ip 192.168.72.1

Prints:
  1. model recall on CICIDS PortScan rows, split by TCP window type (is the 1024-window scan style learned at all?)
  2. the live portscan flow vs the nearest CORRECTLY-classified CICIDS PortScan flow
  3. the tree's decision path for both, so you see the exact split where they diverge
  4. swapping each feature from that nearest CICIDS flow into your live flows
Diagnostic only - nothing is saved.
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
model = pickle.load(open(a.model, "rb")); tree = model.tree_
scale = lambda df: (df[feats].to_numpy(float) - dmin) / rng

live = pd.read_csv(a.dump); live = live[live["dst"] == a.host_ip]
log = pd.read_csv(a.log); opened, wins = {}, []
for _, r in log.iterrows():
    if r["state"] == "start": opened[r["phase"]] = r["epoch"]
    elif r["state"] == "end" and r["phase"] in opened: wins.append((r["phase"], opened.pop(r["phase"]), r["epoch"]))
mask = np.zeros(len(live), bool)
for ph, s, e in wins:
    if ph.startswith("portscan"): mask |= ((live["ts"] >= s) & (live["ts"] <= e + a.grace)).to_numpy()
L = live[mask][feats].reset_index(drop=True)
if L.empty: sys.exit("no portscan flows found")
ad = FlowAdapter.load()
if ad and not a.no_adapter: L = pd.DataFrame([ad(dict(r)) for _, r in L.iterrows()])[feats]

train = pd.read_csv(a.train); T = train[train["label_id"] == 5].reset_index(drop=True)
predT = model.predict(scale(T)); ok = predT == 5
wf, wb = "Init_Win_bytes_forward", "Init_Win_bytes_backward"

print(f"\n1) model recall on {len(T)} CICIDS PortScan rows: {ok.mean():.1%}")
for name, m in [("fwd window == 1024", T[wf] == 1024), ("fwd window != 1024", T[wf] != 1024),
                ("fwd 1024 AND bwd 0", (T[wf] == 1024) & (T[wb] == 0))]:
    if m.sum(): print(f"   {name:22s} n={int(m.sum()):4d}  recall={(predT[m] == 5).mean():.1%}")
print(f"   live scan: fwd window values = {L[wf].value_counts().head(3).to_dict()}, bwd = {L[wb].value_counts().head(3).to_dict()}")

rep_s = np.median(scale(L), axis=0); rep_raw = rep_s * rng + dmin
good = T[ok].reset_index(drop=True)
near = good.iloc[np.linalg.norm(scale(good) - rep_s, axis=1).argmin()]
print("\n2) live (median) vs nearest correctly-classified CICIDS PortScan flow")
pd.set_option("display.float_format", lambda v: f"{v:,.4g}"); pd.set_option("display.width", 200)
print(pd.DataFrame({"feature": feats, "live": rep_raw, "nearest_cic": [near[f] for f in feats]}).to_string(index=False))

def path(xs):
    x_raw = xs * rng + dmin; lines = []
    for n in model.decision_path(xs.reshape(1, -1)).indices:
        if tree.children_left[n] == tree.children_right[n]:
            lines.append(f"-> LEAF: {NAMES[int(model.classes_[tree.value[n].argmax()])]}")
        else:
            f = tree.feature[n]; thr = tree.threshold[n] * rng[f] + dmin[f]
            lines.append(f"{feats[f]} {'<=' if xs[f] <= tree.threshold[n] else '> '} {thr:,.4g}   (value {x_raw[f]:,.4g})")
    return lines
pl, pc = path(rep_s), path(scale(pd.DataFrame([near]))[0])
print("\n3) tree decision paths   (LIVE | nearest CICIDS PortScan)")
for i in range(max(len(pl), len(pc))):
    l = pl[i] if i < len(pl) else ""; c = pc[i] if i < len(pc) else ""
    print(f"{l:62s}| {c}{'' if l == c else '   <-- differs'}")

print("\n4) swap ONE feature in all live flows with that nearest CICIDS flow's value")
for f in feats:
    X = L.copy(); X[f] = near[f]
    p = pd.Series(model.predict(scale(X))).map(NAMES).value_counts(normalize=True).mul(100).round(1).to_dict()
    print(f"   {f:28s} -> {p}")
