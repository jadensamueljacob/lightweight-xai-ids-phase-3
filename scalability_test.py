
import numpy as np
import pickle
import time
import os
import psutil
import pandas as pd
import matplotlib.pyplot as plt

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
X_test_selected = np.load("X_test_selected.npy")

process = psutil.Process(os.getpid())

# ── Renamed: these are BATCH SIZES, not enforced arrival rates ──
scenarios = {
    "100 samples": 100,
    "500 samples": 500,
    "1000 samples": 1000,
    "5000 samples": 5000,
}

results = {}

for label, n_requested in scenarios.items():
    n = min(n_requested, len(X_test_selected))
    latencies, cpus = [], []
    mem_before = process.memory_info().rss / (1024 * 1024)

    print(f"Running batch: {label} ...")

    # warm-up CPU sampling so the first reading is not zero/garbage
    psutil.cpu_percent(interval=None)

    t_start = time.perf_counter()
    for i in range(n):
        row = X_test_selected[i].reshape(1, -1)
        t0 = time.perf_counter()
        model.predict(row)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    t_end = time.perf_counter()
    cpu_after = psutil.cpu_percent(interval=0.1)  # one clean sample per scenario
    mem_after = process.memory_info().rss / (1024 * 1024)

    total_time_s = t_end - t_start
    throughput = n / total_time_s if total_time_s > 0 else 0

    results[label] = {
        "n": n,
        "avg_latency_ms": np.mean(latencies),
        "max_latency_ms": np.max(latencies),
        "total_time_s": total_time_s,
        "throughput_flows_per_sec": throughput,
        "cpu_percent": cpu_after,
        "process_mem_mb": mem_after,
        "mem_delta_mb": mem_after - mem_before,
    }

    print(f"  Avg latency: {np.mean(latencies):.4f} ms | "
          f"Throughput: {throughput:.1f} flows/sec | "
          f"CPU: {cpu_after:.1f}% | Process Mem: {mem_after:.1f} MB")

# ── Save results as CSV ──────────────────────────────────
results_df = pd.DataFrame(results).T
results_df.index.name = "Scenario"
results_df.to_csv("scalability_results.csv")
print("Saved: scalability_results.csv")

# ── Plot ──────────────────────────────────────────────────
labels = list(results.keys())
latency = [results[s]["avg_latency_ms"] for s in labels]
throughput = [results[s]["throughput_flows_per_sec"] for s in labels]
cpu = [results[s]["cpu_percent"] for s in labels]
mem = [results[s]["process_mem_mb"] for s in labels]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].bar(labels, latency, color="steelblue")
axes[0].set_title("Avg Prediction Latency (ms)")
axes[0].set_ylabel("Milliseconds")

axes[1].bar(labels, throughput, color="mediumpurple")
axes[1].set_title("Effective Throughput (flows/sec)")
axes[1].set_ylabel("Flows per second")

axes[2].bar(labels, cpu, color="darkorange")
axes[2].set_title("CPU Utilisation (%) — single clean sample")
axes[2].set_ylabel("Percent")

axes[3].bar(labels, mem, color="seagreen")
axes[3].set_title("Process Memory (MB)")
axes[3].set_ylabel("MB")

plt.suptitle("XAI-IDS Batch Inference Benchmark Results", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("scalability_results.png", dpi=150)
plt.show()
print("Saved: scalability_results.png")
