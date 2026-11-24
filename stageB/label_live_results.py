"""
Labels the dashboard's exported live_results.csv using attack_log.csv (host-clock phase markers)
and prints an honest evaluation of the REAL-CAPTURE run.

  python label_live_results.py --results live_results.csv --log attack_log.csv --host-ip 192.168.xx.1 [--vm-ip 192.168.xx.y]
"""
import argparse
import numpy as np, pandas as pd

NAMES = {0: "Bot", 1: "DDoS", 2: "DoS", 3: "Infiltration", 4: "Benign", 5: "PortScan"}
ap = argparse.ArgumentParser()
ap.add_argument("--results", default="live_results.csv"); ap.add_argument("--log", default="attack_log.csv")
ap.add_argument("--host-ip", required=True); ap.add_argument("--vm-ip")
ap.add_argument("--victim-port", type=int, default=8000, help="the port the attacker targets; other flows in benign/DoS windows are background (benign)")
ap.add_argument("--grace", type=float, default=5.0, help="sec after a phase ends in which its late-closing flows still count")
a = ap.parse_args()

res = pd.read_csv(a.results)
if "source" in res: res = res[res["source"] == "live"]
res = res[res["dst"] == a.host_ip]                               # only attacker -> victim flows
if a.vm_ip: res = res[res["src"] == a.vm_ip]

log = pd.read_csv(a.log); opened, wins = {}, []
for _, r in log.iterrows():
    if r["state"] == "start": opened[r["phase"]] = r["epoch"]
    elif r["state"] == "end" and r["phase"] in opened: wins.append((r["phase"], opened.pop(r["phase"]), r["epoch"]))

def true_id(phase):
    p = phase.lower()
    return 4 if p.startswith("benign") else 5 if p.startswith("portscan") else 2 if p.startswith("dos") else None

res["phase"] = None
for ph, s, e in wins:
    m = (res["t"] >= s) & (res["t"] <= e + a.grace)
    res.loc[m, "phase"] = ph
res = res.dropna(subset=["phase"]).copy()
res["true"] = res["phase"].map(true_id).astype(int)
# Ground truth: the attacker only targets the victim port, so any other flow inside a benign/DoS window (DNS, system chatter)
# is background traffic, i.e. Benign - not an attack that the model "missed".
bg = res["true"].isin([4, 2]) & (res["dport"] != a.victim_port)
res.loc[bg, "true"] = 4
res["grp"] = np.where(bg, "background", res["phase"])
res.to_csv("labelled_live_results.csv", index=False)
print(f"{len(res)} labelled flows in {len(wins)} phases\n")

print("== Per phase: share of flows predicted as each class ==")
tab = (res.groupby(["grp", res["pred"].map(NAMES)]).size().unstack(fill_value=0))
print((tab.div(tab.sum(axis=1), axis=0) * 100).round(1).join(res.groupby("grp")["ood"].mean().round(3).rename("meanOOD")).join(tab.sum(axis=1).rename("n")))

bad = res[(res["true"] == 4) & (res["pred"] != 4)]
if len(bad):
    print("\n== Benign-phase flows predicted as attack, by destination port ==")
    print(bad.groupby(["dport", bad["pred"].map(NAMES)]).size().sort_values(ascending=False).head(8).to_string())
att, ben = res[res["true"] != 4], res[res["true"] == 4]
svc = ben[ben["dport"] == 8000]
if len(svc): print(f"\nvictim-service view: false-alarm rate on benign flows to port 8000 only: {(svc['pred'] != 4).mean():.1%} ({len(svc)} flows)")
print("\n== Binary view (attack vs benign) ==")
if len(att): print(f"attack detection rate (flagged non-benign): {(att['pred'] != 4).mean():.1%}  ({len(att)} attack flows)")
if len(ben): print(f"false-alarm rate on benign:                {(ben['pred'] != 4).mean():.1%}  ({len(ben)} benign flows)")

print("\n== Exact-class view ==")
from sklearn.metrics import classification_report, confusion_matrix
labs = sorted(set(res["true"]) | set(res["pred"]))
print(classification_report(res["true"], res["pred"], labels=labs, target_names=[NAMES[i] for i in labs], zero_division=0))
cm = confusion_matrix(res["true"], res["pred"], labels=labs)
print(pd.DataFrame(cm, index=[f"true {NAMES[i]}" for i in labs], columns=[f"pred {NAMES[i]}" for i in labs]))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 5)); ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
ax.set_xticklabels([NAMES[i] for i in labs], rotation=30); ax.set_yticklabels([NAMES[i] for i in labs])
for i in range(len(labs)):
    for j in range(len(labs)): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True (from phase markers)"); ax.set_title("Real-capture confusion matrix")
plt.tight_layout(); plt.savefig("live_capture_confusion.png", dpi=150)
print("\nsaved labelled_live_results.csv and live_capture_confusion.png")