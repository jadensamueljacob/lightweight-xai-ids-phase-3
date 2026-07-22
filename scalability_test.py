# scalability_test.py
import numpy as np
import pickle
import time
import psutil
import matplotlib.pyplot as plt

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
X_test_selected = np.load("X_test_selected.npy")

scenarios = {
    "Low\n100/s":     100,
    "Medium\n500/s":  500,
    "High\n1000/s":   1000,
    "Burst\n5000/s":  5000,
}

results = {}

for label, rate in scenarios.items():
    n = min(rate, len(X_test_selected))
    latencies, cpus, mems = [], [], []
    print(f"Running {label.strip()} ...")

    for i in range(n):
        row = X_test_selected[i].reshape(1, -1)
        t0  = time.perf_counter()
        model.predict(row)
        t1  = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        cpus.append(psutil.cpu_percent(interval=None))
        mems.append(psutil.virtual_memory().percent)

    results[label] = {
        "latency": np.mean(latencies),
        "cpu":     np.mean(cpus),
        "mem":     np.mean(mems),
    }
    print(f"  Avg latency: {np.mean(latencies):.4f} ms | CPU: {np.mean(cpus):.1f}% | Mem: {np.mean(mems):.1f}%")

# Plot
labels  = list(results.keys())
latency = [results[s]["latency"] for s in labels]
cpu     = [results[s]["cpu"]     for s in labels]
mem     = [results[s]["mem"]     for s in labels]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(labels, latency, color="steelblue")
axes[0].set_title("Avg Prediction Latency (ms)")
axes[0].set_ylabel("Milliseconds")

axes[1].bar(labels, cpu, color="darkorange")
axes[1].set_title("Avg CPU Utilisation (%)")
axes[1].set_ylabel("Percent")

axes[2].bar(labels, mem, color="seagreen")
axes[2].set_title("Avg Memory Usage (%)")
axes[2].set_ylabel("Percent")

plt.suptitle("XAI-IDS Scalability Test Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("scalability_results.png", dpi=150)
plt.show()
print("Saved: scalability_results.png")