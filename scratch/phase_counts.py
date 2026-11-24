"""Prints how many captured flows fall inside each logged phase window.
Run from the project folder:  python phase_counts.py
"""
import pandas as pd

HOST = "192.168.72.1"
GRACE = 5.0

d = pd.read_csv("live_features.csv")
l = pd.read_csv("attack_log.csv")
d = d[d["dst"] == HOST]

print(f"flows to {HOST}: {len(d)}")
print(f"dump  ts range: {d.ts.min():.0f} -> {d.ts.max():.0f}  ({d.ts.max() - d.ts.min():.0f}s)")
print(f"log   ts range: {l.epoch.min():.0f} -> {l.epoch.max():.0f}\n")

opened = {}
print(f"{'phase':18s} {'start':>12s} {'end':>12s} {'secs':>6s} {'flows':>7s}")
for r in l.itertuples():
    if r.state == "start":
        opened[r.phase] = r.epoch
    elif r.phase in opened:
        s = opened.pop(r.phase)
        e = r.epoch
        n = int(((d.ts >= s) & (d.ts <= e + GRACE)).sum())
        print(f"{r.phase:18s} {s:12.0f} {e:12.0f} {e - s:6.0f} {n:7d}")

if opened:
    print("\nphases with a start but no end:", list(opened))
