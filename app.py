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

# ── Load artefacts ─────────────────────────────────────────
@st.cache_resource
def load_all():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("final_features.pkl", "rb") as f:
        final_features = pickle.load(f)

    X_test_selected = np.load("X_test_selected.npy")
    X_res           = np.load("X_res.npy")
    y_test          = np.load("y_test.npy")

    lime_explainer = LimeTabularExplainer(
        X_res,                          # same background used in notebook
        feature_names=final_features,
        class_names=[CLASS_NAMES[i] for i in range(6)],
        mode='classification'
    )
    return model, X_test_selected, y_test, final_features, lime_explainer

model, X_test_selected, y_test, final_features, lime_explainer = load_all()

# ── Sidebar ────────────────────────────────────────────────
st.sidebar.title("⚙️ Simulation Controls")
flows_per_second = st.sidebar.slider("Flows per second", 1, 50, 5)
max_flows        = st.sidebar.slider("Total flows to simulate", 50, 2000, 200)
explain_on       = st.sidebar.checkbox("Generate LIME for attacks", value=True)
max_explanations = st.sidebar.slider("Max LIME explanations to show", 1, 10, 3)

start = st.sidebar.button("▶ Start Simulation")

# ── Main layout ────────────────────────────────────────────
st.title("🛡️ Explainable AI — Intrusion Detection System")
st.markdown("Streaming CICIDS2017 test flows through the trained Decision Tree in real time.")

# Metric boxes
m1, m2, m3, m4 = st.columns(4)
cpu_box      = m1.empty()
mem_box      = m2.empty()
latency_box  = m3.empty()
alert_box    = m4.empty()

st.markdown("---")
st.subheader("📡 Live Traffic Feed (last 20 flows)")
feed_placeholder = st.empty()

st.subheader("📊 Class Distribution So Far")
dist_placeholder = st.empty()

st.subheader("🔍 LIME Explanations (attacks only)")
lime_placeholder = st.empty()

# ── Simulation ─────────────────────────────────────────────
if start:
    log          = []
    lime_figs    = []
    latencies    = []
    alert_count  = 0
    delay        = 1.0 / flows_per_second

    n = min(max_flows, len(X_test_selected))

    for i in range(n):
        row  = X_test_selected[i].reshape(1, -1)

        # Predict
        t0   = time.perf_counter()
        pred = int(model.predict(row)[0])
        prob = model.predict_proba(row)[0]
        t1   = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        is_attack  = pred in ALERT_CLASSES
        class_name = CLASS_NAMES[pred]
        confidence = float(max(prob))

        if is_attack:
            alert_count += 1

        # System metrics
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        # Update metric boxes
        cpu_box.metric("🖥️ CPU", f"{cpu:.1f}%")
        mem_box.metric("💾 Memory", f"{mem:.1f}%")
        latency_box.metric("⚡ Avg Latency", f"{np.mean(latencies):.3f} ms")
        alert_box.metric("🚨 Alerts", str(alert_count))

        # Log
        log.append({
            "Flow #":       i + 1,
            "True Class":   CLASS_NAMES[int(y_test[i])],
            "Predicted":    class_name,
            "Confidence":   f"{confidence:.2f}",
            "Status":       "🚨 ATTACK" if is_attack else "✅ Benign",
            "Latency (ms)": f"{latency_ms:.4f}"
        })

        # Live table — last 20 rows
        df = pd.DataFrame(log[-20:])
        feed_placeholder.dataframe(df, use_container_width=True)

        # Class distribution bar chart
        dist_df = pd.DataFrame(log)["Predicted"].value_counts().reset_index()
        dist_df.columns = ["Class", "Count"]
        dist_placeholder.bar_chart(dist_df.set_index("Class"))

        # LIME for attacks
        if is_attack and explain_on and len(lime_figs) < max_explanations:
            exp = lime_explainer.explain_instance(
                X_test_selected[i],
                model.predict_proba,
                num_features=10
            )
            fig = exp.as_pyplot_figure()
            plt.title(f"Flow #{i+1} | Predicted: {class_name} | Confidence: {confidence:.2f}")
            plt.tight_layout()
            lime_figs.append(fig)

            with lime_placeholder.container():
                for fig in lime_figs:
                    st.pyplot(fig)
                    plt.close(fig)

        time.sleep(delay)

    # ── Final summary ──────────────────────────────────────
    st.success(f"✅ Simulation complete — {n} flows processed, {alert_count} attacks detected.")
    st.markdown("---")
    st.subheader("📋 Full Results Table")
    full_df = pd.DataFrame(log)
    st.dataframe(full_df, use_container_width=True)

    st.subheader("📈 Final Class Distribution")
    st.bar_chart(full_df["Predicted"].value_counts())

    st.subheader("📉 Prediction Latency Over Time")
    st.line_chart(latencies)

    # Accuracy on simulated flows
    true_labels = [CLASS_NAMES[int(y_test[i])] for i in range(n)]
    pred_labels = [row["Predicted"] for row in log]
    correct     = sum(t == p for t, p in zip(true_labels, pred_labels))
    st.metric("Accuracy on simulated flows", f"{correct/n*100:.2f}%")
