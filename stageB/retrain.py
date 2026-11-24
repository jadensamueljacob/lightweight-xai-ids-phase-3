r"""
Lab-adapted model: CICIDS training data + a small amount of YOUR labelled lab flows.

  1) look at the options:      python stageB\retrain.py --dump live_features.csv --log attack_log.csv --host-ip 192.168.72.1 --grid
  2) train with chosen values: python stageB\retrain.py --dump live_features.csv --log attack_log.csv --host-ip 192.168.72.1 --depth 14 --lab-weight 20
  3) FINAL test on a second, fresh capture run (never used for training or tuning):
        python stageB\retrain.py --host-ip 192.168.72.1 --final-dump live_features_run2.csv --final-log attack_log_run2.csv

Lab flows are split BY PHASE REPETITION per attack tool (benign / portscan / dos_syn / dos_slowhttp ...): the last
repetition(s) of each tool are held out and never trained on. model.pkl is never touched; the result is model_lab.pkl.
Honest framing: lab results show adaptation to YOUR environment and tools, not generalisation to unseen attacks.
"""
import argparse, json, pickle, re, itertools
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score

NAMES = {0: "Bot", 1: "DDoS", 2: "DoS", 3: "Infiltration", 4: "Benign", 5: "PortScan"}
ap = argparse.ArgumentParser()
ap.add_argument("--dump", default="live_features.csv"); ap.add_argument("--log", default="attack_log.csv")
ap.add_argument("--host-ip", required=True); ap.add_argument("--vm-ip")
ap.add_argument("--model", default="model.pkl"); ap.add_argument("--scaler", default="scaler_15.json")
ap.add_argument("--out", default="model_lab.pkl")
ap.add_argument("--cap", type=int, default=400, help="max lab flows kept per phase")
ap.add_argument("--lab-weight", type=float, default=20.0); ap.add_argument("--depth", type=int, default=14)
ap.add_argument("--grid", action="store_true", help="compare several depth/weight settings (does not save a model)")
ap.add_argument("--final-dump"); ap.add_argument("--final-log")
ap.add_argument("--grace", type=float, default=5.0)
a = ap.parse_args()

orig = pickle.load(open(a.model, "rb")); sc = json.load(open(a.scaler)); F = sc["features"]
dmin = np.array(sc["data_min"]); rng = np.array(sc["data_max"]) - dmin; rng = np.where(rng == 0, 1, rng)
scale = lambda d: (d[F].to_numpy(float) - dmin) / rng
CW = {0: 1, 1: 1, 2: 1, 3: 3, 4: 1, 5: 1}

def load_lab(dump, log_path):
    live = pd.read_csv(dump); live = live[live["dst"] == a.host_ip]
    if a.vm_ip: live = live[live["src"] == a.vm_ip]
    log = pd.read_csv(log_path); opened, wins = {}, []
    for _, r in log.iterrows():
        if r["state"] == "start": opened[r["phase"]] = r["epoch"]
        elif r["state"] == "end" and r["phase"] in opened: wins.append((r["phase"], opened.pop(r["phase"]), r["epoch"]))
    live["phase"] = None
    for ph, s, e in wins: live.loc[(live["ts"] >= s) & (live["ts"] <= e + a.grace), "phase"] = ph
    live = live.dropna(subset=["phase"]).copy()
    live["grp"] = live["phase"].str.replace(r"_\d+$", "", regex=True)          # benign / portscan / dos_syn / dos_slowhttp ...
    live["kind"] = live["grp"].str.split("_").str[0]
    live = live[live["kind"].isin(["benign", "portscan", "dos"])]
    # Ground truth: the attacker only ever targets the victim service (port 8000), so any other flow inside a benign or
    # DoS window (DNS lookups, system chatter) is genuine BACKGROUND traffic -> label Benign, keep it as its own group.
    bg = live["kind"].isin(["benign", "dos"]) & (live["dport"] != 8000)
    live["grp"] = np.where(bg, "background", live["grp"])
    live["label"] = np.where(bg, 4, live["kind"].map({"benign": 4, "portscan": 5, "dos": 2}))
    live = pd.concat([g.sample(min(len(g), a.cap), random_state=42) for _, g in live.groupby(["phase", "grp"])])
    return live, {ph: s for ph, s, e in wins}

def report(models, lab, title):
    print(f"\n=== {title} ===")
    rows = []
    for name, m in models.items():
        p = m.predict(scale(lab)); row = {"model": name}
        for g, idx in lab.groupby("grp").indices.items():
            pg = p[idx]; row[g] = f"{np.mean(pg == 4):.0%} benign" if g in ("benign", "background") else f"{np.mean(pg != 4):.0%} detected"
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))

# ---------------- final test on a second run ----------------
if a.final_dump:
    lab2, _ = load_lab(a.final_dump, a.final_log)
    X_cic_te, y_cic_te = np.load("X_test_selected.npy"), np.load("y_test.npy")
    lab_model = pickle.load(open(a.out, "rb"))
    report({"original": orig, "lab-adapted": lab_model}, lab2, f"FINAL TEST on fresh run ({len(lab2)} flows, never trained/tuned on)")
    print("\n(benign column: higher is better; attack columns: higher is better)")
    raise SystemExit

# ---------------- CICIDS base ----------------
X_res = np.load("X_res.npy"); y_res = np.repeat([0, 1, 2, 3, 4, 5], [2000, 30000, 30000, 500, 50000, 2000])
chk = (orig.predict(X_res) == y_res).mean()
print(f"CICIDS base: {X_res.shape}, rebuilt labels agree with original model on {chk:.1%} (expect >99%)")
assert chk > 0.95, "y_res reconstruction failed - tell Claude"
X_cic_te, y_cic_te = np.load("X_test_selected.npy"), np.load("y_test.npy")

live, order = load_lab(a.dump, a.log)
test_phases = set()
for grp, g in live[live["grp"] != "background"].groupby("grp"):      # background flows follow their phase's train/test side
    phs = sorted(g["phase"].unique(), key=lambda p: order[p])
    n_test = len(phs) // 2 if grp == "benign" else (1 if len(phs) > 1 else 0)
    test_phases |= set(phs[len(phs) - n_test:])
    print(f"  {grp:13s}: {len(phs)} phase(s) -> train {phs[:len(phs)-n_test]}  test {phs[len(phs)-n_test:]}")
lab_tr, lab_te = live[~live["phase"].isin(test_phases)], live[live["phase"].isin(test_phases)]
print(f"lab train {len(lab_tr)} flows {lab_tr['label'].value_counts().to_dict()} | lab test {len(lab_te)} flows")

def train(depth, weight):
    X = np.vstack([X_res, scale(lab_tr)]); y = np.concatenate([y_res, lab_tr["label"].to_numpy()])
    w = np.concatenate([np.ones(len(X_res)), np.full(len(lab_tr), weight)])
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, class_weight=CW, random_state=42).fit(X, y, sample_weight=w)

def cic_f1(m): return f1_score(y_cic_te, m.predict(X_cic_te), average="macro")

if a.grid:
    print(f"\noriginal model: CICIDS macro F1 {cic_f1(orig):.3f}")
    rows = []
    for depth, weight in itertools.product([10, 12, 14, 16], [5, 20, 50]):
        m = train(depth, weight); p = m.predict(scale(lab_te)); row = {"depth": depth, "weight": weight, "leaves": m.get_n_leaves(), "CIC macroF1": round(cic_f1(m), 3)}
        for g, idx in lab_te.groupby("grp").indices.items():
            row[g] = round(np.mean(p[idx] == 4), 2) if g in ("benign", "background") else round(np.mean(p[idx] != 4), 2)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nbenign column = share correctly benign; attack columns = detection rate. Pick a row, then re-run with --depth/--lab-weight.")
    print("NOTE: choosing a row by looking at these numbers tunes on the held-out phases - confirm on a fresh run with --final-dump.")
    raise SystemExit

new = train(a.depth, a.lab_weight)
pickle.dump(new, open(a.out, "wb")); print(f"\nsaved {a.out} (depth={new.get_depth()}, leaves={new.get_n_leaves()}, lab-weight={a.lab_weight})")
print("\n=== CICIDS held-out test set (379,713 flows) ===")
for name, m in [("original", orig), ("lab-adapted", new)]:
    p = m.predict(X_cic_te)
    print(f"{name:12s} macro F1 {f1_score(y_cic_te, p, average='macro'):.3f} | Infiltration recall {recall_score(y_cic_te, p, labels=[3], average=None, zero_division=0)[0]:.2f} | PortScan recall {recall_score(y_cic_te, p, labels=[5], average=None, zero_division=0)[0]:.2f}")
report({"original": orig, "lab-adapted": new}, lab_tr, "Lab TRAIN phases (fit check)")
report({"original": orig, "lab-adapted": new}, lab_te, "Lab HELD-OUT phases (never trained on)")