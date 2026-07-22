
import streamlit as st
import numpy as np
import pickle
import time
import psutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, accuracy_score
import os

# ── Config ─────────────────────────────────────────────────
st.set_page_config(page_title="XAI-IDS Dashboard", layout="wide")

CLASS_NAMES = {
    0: "Brute Force / Web Attacks",
    1: "FTP / SSH Patator",
    2: "DoS / DDoS",
    3: "Infiltration",
    4: "Benign",
    5: "PortScan"
}
ALERT_CLASSES = {0, 1, 2, 3, 5}  # everything except Benign (4)
MINORITY_CLASS = 3  # Infiltration — the class with only 11 real samples

# ── Load artefacts ─────────────────────────────────────────
@st.cache_resource
def load_all():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("final_features.pkl", "rb") as f:
        final_features = pickle.load(f)

    X_test_selected = np.load("X_test_selected.npy")
    X_res = np.load("X_res.npy")
    y_test = np.load("y_test.npy")

    lime_explainer = LimeTabularExplainer(
        X_res,
        feature_names=final_features,
        class_names=[CLASS_NAMES[i] for i in range(6)],
        mode='classification'
    )
    return model, X_test_selected, y_test, final_features, lime_explainer

model, X_test_selected, y_test, final_features, lime_explainer = load_all()

# ── Session state init ────────��─────────────────────────────
if "log" not in st.session_state:
    st.session_state.log = []
if "attack_rows" not in st.session_state:
    st.session_state.attack_rows = []       # store (flow_idx, pred, confidence) for on-demand LIME
if "latencies" not in st.session_state:
    st.session_state.latencies = []
if "explain_latencies" not in st.session_state:
    st.session_state.explain_latencies = []
if "sim_done" not in st.session_state:
    st.session_state.sim_done = False

process = psutil.Process(os.getpid())

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("⚙️ Simulation Controls")
flows_per_second = st.sidebar.slider("Flows per second", 1, 50, 5)
max_flows = st.sidebar.slider("Total flows to simulate", 50, 2000, 200)
start = st.sidebar.button("▶ Start Simulation")
reset = st.sidebar.button("⟲ Reset")

if reset:
    st.session_state.log = []
    st.session_state.attack_rows = []
    st.session_state.latencies = []
    st.session_state.explain_latencies = []
    st.session_state.sim_done = False
    st.rerun()

# ── Main header ─────────────────────────────────────────────
st.title("🛡️ Explainable AI — Intrusion Detection System")
st.markdown(
    "Simulated real-time replay of CICIDS2017 test flows through the trained "
    "Decision Tree. Detection runs continuously; explanations are generated "
    "**offline, on demand** — decoupled from the detection loop."
)

tab_live, tab_explain, tab_perf, tab_eval = st.tabs(
    ["📡 Live Detection", "🔍 Alert Explanations", "⚡ Performance", "📊 Evaluation"]
)

# ══════════════════════════════════════════════════════════
# TAB 1 — LIVE DETECTION
# ══════════════════════════════════════════════════════════
with tab_live:
    m1, m2, m3, m4 = st.columns(4)
    cpu_box = m1.empty()
    mem_box = m2.empty()
    latency_box = m3.empty()
    alert_box = m4.empty()

    st.markdown("---")
    class_counter_placeholder = st.empty()

    st.subheader("Live Traffic Feed (last 20 flows)")
    feed_placeholder = st.empty()

    st.subheader("Class Distribution So Far")
    dist_placeholder = st.empty()

    if start:
        st.session_state.log = []
        st.session_state.attack_rows = []
        st.session_state.latencies = []
        delay = 1.0 / flows_per_second
        n = min(max_flows, len(X_test_selected))

        for i in range(n):
            row = X_test_selected[i].reshape(1, -1)

            t0 = time.perf_counter()
            pred = int(model.predict(row)[0])
            prob = model.predict_proba(row)[0]
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000
            st.session_state.latencies.append(latency_ms)

            is_attack = pred in ALERT_CLASSES
            class_name = CLASS_NAMES[pred]
            confidence = float(max(prob))
            true_class = CLASS_NAMES[int(y_test[i])]
            correct = (pred == int(y_test[i]))

            if is_attack:
                st.session_state.attack_rows.append(
                    {"flow_idx": i, "pred": pred, "confidence": confidence}
                )

            cpu = psutil.cpu_percent(interval=None)
            mem_mb = process.memory_info().rss / (1024 * 1024)

            cpu_box.metric("🖥️ CPU", f"{cpu:.1f}%")
            mem_box.metric("💾 Process Memory", f"{mem_mb:.1f} MB")
            latency_box.metric("⚡ Avg Detection Latency", f"{np.mean(st.session_state.latencies):.3f} ms")
            alert_box.metric("🚨 Alerts", str(len(st.session_state.attack_rows)))

            st.session_state.log.append({
                "Flow #": i + 1,
                "True Class": true_class,
                "Predicted": class_name,
                "Confidence": f"{confidence:.2f}",
                "Result": "✅ Correct" if correct else "❌ Incorrect",
                "Status": "🚨 ATTACK" if is_attack else "✅ Benign",
                "Latency (ms)": f"{latency_ms:.4f}"
            })

            df = pd.DataFrame(st.session_state.log[-20:])

            def highlight_result(row):
                color = "background-color: #d4f8d4" if row["Result"] == "✅ Correct" else "background-color: #f8d4d4"
                return [color] * len(row)

            feed_placeholder.dataframe(
                df.style.apply(highlight_result, axis=1), use_container_width=True
            )

            full_log_df = pd.DataFrame(st.session_state.log)
            class_wise = full_log_df["Predicted"].value_counts().reindex(
                [CLASS_NAMES[k] for k in range(6)], fill_value=0
            )
            dist_placeholder.bar_chart(class_wise)

            with class_counter_placeholder.container():
                cols = st.columns(6)
                for idx, cname in CLASS_NAMES.items():
                    cols[idx].metric(cname.split(" / ")[0], int(class_wise[cname]))

            time.sleep(delay)

        st.session_state.sim_done = True
        st.success(f"✅ Simulation complete — {n} flows processed, "
                    f"{len(st.session_state.attack_rows)} attacks detected.")

    if st.session_state.log and not start:
        df = pd.DataFrame(st.session_state.log[-20:])
        feed_placeholder.dataframe(df, use_container_width=True)
        full_log_df = pd.DataFrame(st.session_state.log)
        class_wise = full_log_df["Predicted"].value_counts().reindex(
            [CLASS_NAMES[k] for k in range(6)], fill_value=0
        )
        dist_placeholder.bar_chart(class_wise)

    if st.session_state.log:
        csv = pd.DataFrame(st.session_state.log).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export results as CSV", csv, "simulation_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════
# TAB 2 — ALERT EXPLANATIONS (ON-DEMAND LIME)
# ══════════════════════════════════════════════════════════
with tab_explain:
    st.subheader("Select an alert to explain")
    st.caption("Explanations are generated offline, only when requested — "
               "this keeps the detection loop fast and decouples XAI cost from live inference.")

    if not st.session_state.attack_rows:
        st.info("No attacks detected yet. Run a simulation in the Live Detection tab first.")
    else:
        options = [
            f"Flow #{r['flow_idx']+1} | {CLASS_NAMES[r['pred']]} | Confidence {r['confidence']:.2f}"
            for r in st.session_state.attack_rows
        ]
        selected = st.selectbox("Flagged alerts", options)
        sel_idx = options.index(selected)
        flow_idx = st.session_state.attack_rows[sel_idx]["flow_idx"]
        pred = st.session_state.attack_rows[sel_idx]["pred"]
        confidence = st.session_state.attack_rows[sel_idx]["confidence"]

        if st.button("🧠 Generate LIME Explanation"):
            t0 = time.perf_counter()
            exp = lime_explainer.explain_instance(
                X_test_selected[flow_idx],
                model.predict_proba,
                num_features=10
            )
            t1 = time.perf_counter()
            explain_ms = (t1 - t0) * 1000
            st.session_state.explain_latencies.append(explain_ms)

            fig = exp.as_pyplot_figure()
            plt.title(f"Flow #{flow_idx+1} | Predicted: {CLASS_NAMES[pred]} | Confidence: {confidence:.2f}")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.metric("⏱️ Explanation generation time", f"{explain_ms:.1f} ms")
            st.caption(
                f"Detection for this flow took ~{np.mean(st.session_state.latencies):.3f} ms — "
                f"explanation took {explain_ms:.1f} ms, generated separately and asynchronously "
                f"from the live detection pipeline."
            )

# ══════════════════════════════════════════════════════════
# TAB 3 — PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab_perf:
    if st.session_state.latencies:
        lat_df = pd.DataFrame({
            "Flow #": range(1, len(st.session_state.latencies) + 1),
            "Latency (ms)": st.session_state.latencies
        })
        st.subheader("Prediction Latency Over Time")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(lat_df["Flow #"], lat_df["Latency (ms)"], color="steelblue", linewidth=1)
        ax.set_xlabel("Flow #")
        ax.set_ylabel("Latency (ms)")
        ax.set_ylim(0, max(lat_df["Latency (ms)"].max() * 1.2, 1))
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Latency", f"{np.mean(st.session_state.latencies):.3f} ms")
        c2.metric("Max Latency", f"{np.max(st.session_state.latencies):.3f} ms")
        throughput = 1000 / np.mean(st.session_state.latencies) if np.mean(st.session_state.latencies) > 0 else 0
        c3.metric("Effective Throughput", f"{throughput:.0f} flows/sec")

        if st.session_state.explain_latencies:
            st.subheader("Explanation (LIME) Latency")
            e1, e2 = st.columns(2)
            e1.metric("Avg Explanation Time", f"{np.mean(st.session_state.explain_latencies):.1f} ms")
            e2.metric("Explanations Generated", len(st.session_state.explain_latencies))
    else:
        st.info("Run a simulation first to see performance metrics.")

# ══════════════════════════════════════════════════════════
# TAB 4 — EVALUATION
# ══════════════════════════════════════════════════════════
with tab_eval:
    if st.session_state.log:
        full_df = pd.DataFrame(st.session_state.log)
        n = len(full_df)

        true_idx = [int(y_test[i]) for i in range(n)]
        pred_idx = [list(CLASS_NAMES.values()).index(row) if False else None for row in full_df["Predicted"]]
        name_to_idx = {v: k for k, v in CLASS_NAMES.items()}
        pred_idx = [name_to_idx[p] for p in full_df["Predicted"]]

        acc = accuracy_score(true_idx, pred_idx)
        macro_f1 = f1_score(true_idx, pred_idx, average="macro", zero_division=0)
        weighted_f1 = f1_score(true_idx, pred_idx, average="weighted", zero_division=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc*100:.2f}%")
        c2.metric("Macro F1", f"{macro_f1:.3f}")
        c3.metric("Weighted F1", f"{weighted_f1:.3f}")

        st.subheader("Confusion Matrix")
        labels = list(range(6))
        cm = confusion_matrix(true_idx, pred_idx, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels([CLASS_NAMES[i] for i in range(6)], rotation=45, ha="right")
        ax.set_yticklabels([CLASS_NAMES[i] for i in range(6)])
        for i in range(6):
            for j in range(6):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Per-Class Precision / Recall / F1")
        prec, rec, f1, support = precision_recall_fscore_support(
            true_idx, pred_idx, labels=labels, zero_division=0
        )
        report_df = pd.DataFrame({
            "Class": [CLASS_NAMES[i] for i in labels],
            "Precision": prec, "Recall": rec, "F1": f1, "Support": support
        })
        st.dataframe(report_df, use_container_width=True)

        st.subheader("🔬 Minority-Class Spotlight — Infiltration")
        infil_support = int(support[MINORITY_CLASS])
        infil_f1 = f1[MINORITY_CLASS]
        infil_recall = rec[MINORITY_CLASS]
        i1, i2, i3 = st.columns(3)
        i1.metric("Infiltration samples seen", infil_support)
        i2.metric("Infiltration F1", f"{infil_f1:.3f}")
        i3.metric("Infiltration Recall", f"{infil_recall:.3f}")
        st.caption(
            "This class has only 11 real training samples in CICIDS2017 — most surveyed "
            "papers report only weighted/overall accuracy, which hides performance here. "
            "We report it explicitly to demonstrate our imbalance-handling pipeline."
        )
    else:
        st.info("Run a simulation first to see evaluation metrics.")
