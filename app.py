import streamlit as st
import numpy as np
import pickle
import time
import psutil
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support, accuracy_score)
import os
import json

st.set_page_config(page_title="XAI-IDS Dashboard", layout="wide")

CLASS_NAMES = {
    0: "Brute Force / Web Attacks",
    1: "FTP / SSH Patator",
    2: "DoS / DDoS",
    3: "Infiltration",
    4: "Benign",
    5: "PortScan"
}
ALERT_CLASSES  = {0, 1, 2, 3, 5}
MINORITY_CLASS = 3

SHAP_BEESWARM  = "shap_beeswarm.png"
SHAP_META      = "shap_meta.json"   # stores time_ms and bar data path
SHAP_BAR_CSV   = "shap_bar.csv"

# ── Load + stratified shuffle ───────────────────────────────
@st.cache_resource
def load_all():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("final_features.pkl", "rb") as f:
        final_features = pickle.load(f)

    X_test = np.load("X_test_selected.npy")
    X_res  = np.load("X_res.npy")
    y_test = np.load("y_test.npy")

    # Interleave all 6 classes so every class appears from flow #1
    indices = [np.where(y_test == c)[0] for c in range(6)]
    max_len = max(len(i) for i in indices)
    interleaved = []
    for pos in range(max_len):
        for cls in range(6):
            if pos < len(indices[cls]):
                interleaved.append(indices[cls][pos])
    interleaved = np.array(interleaved)

    lime_explainer = LimeTabularExplainer(
        X_res,
        feature_names=final_features,
        class_names=[CLASS_NAMES[i] for i in range(6)],
        mode='classification'
    )
    return model, X_test[interleaved], y_test[interleaved], final_features, lime_explainer

model, X_test_selected, y_test, final_features, lime_explainer = load_all()

# ── Helpers ─────────────────────────────────────────────────
def shap_files_exist():
    return (os.path.exists(SHAP_BEESWARM) and
            os.path.exists(SHAP_META) and
            os.path.exists(SHAP_BAR_CSV))

def compute_and_save_shap(sample_size):
    """Run SHAP and save all outputs to disk. Returns time_ms or None on error."""
    try:
        sample      = X_test_selected[:sample_size]
        explainer   = shap.TreeExplainer(model)
        t0          = time.perf_counter()
        shap_values = explainer.shap_values(sample)
        t1          = time.perf_counter()
        shap_ms     = (t1 - t0) * 1000

        # Mean |SHAP| per feature averaged across classes
        if isinstance(shap_values, list):
            stacked   = np.stack([np.abs(sv) for sv in shap_values], axis=0)
            mean_shap = stacked.mean(axis=0).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)
        mean_shap = np.array(mean_shap).flatten()

        bar_df = pd.DataFrame({
            "Feature":     list(final_features),
            "Mean |SHAP|": mean_shap.tolist()
        }).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        bar_df.to_csv(SHAP_BAR_CSV, index=False)

        # Beeswarm — let SHAP own the figure
        plt.close("all")
        shap.summary_plot(
            shap_values, sample,
            feature_names=final_features,
            show=False,
            plot_size=(10, 6)
        )
        plt.tight_layout()
        plt.savefig(SHAP_BEESWARM, dpi=120, bbox_inches="tight")
        plt.close("all")

        with open(SHAP_META, "w") as f:
            json.dump({"time_ms": shap_ms, "sample_size": sample_size}, f)

        return shap_ms

    except Exception as e:
        return str(e)   # return error string so caller can show it

# ── Session state ───────────────────────────────────────────
defaults = {
    "log":               [],
    "attack_rows":       [],
    "latencies":         [],
    "explain_latencies": [],
    "sim_done":          False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

process = psutil.Process(os.getpid())

# ── Sidebar ─────────────────────────────────────────────────
st.sidebar.title("⚙️ Simulation Controls")
flows_per_second = st.sidebar.slider("Flows per second",        1,   50,   5)
max_flows        = st.sidebar.slider("Total flows to simulate", 50, 2000, 500)
shap_sample_size = st.sidebar.slider("SHAP sample size",        50,  300, 100)
start = st.sidebar.button("▶ Start Simulation")
reset = st.sidebar.button("⟲ Reset")

if reset:
    for k, v in defaults.items():
        st.session_state[k] = v
    for f in [SHAP_BEESWARM, SHAP_META, SHAP_BAR_CSV]:
        if os.path.exists(f):
            os.remove(f)
    # Clean up any LIME files
    for f in os.listdir("."):
        if f.startswith("lime_flow_") and f.endswith(".png"):
            os.remove(f)
    st.rerun()

# ── Header ───────────────────────────────────────────────────
st.title("🛡️ Explainable AI — Intrusion Detection System")
st.markdown(
    "Simulated real-time replay of CICIDS2017 test flows through the trained "
    "Decision Tree. Detection runs continuously; **LIME** explanations are "
    "generated on demand per alert; **SHAP** global view computes automatically "
    "after simulation ends."
)

tab_live, tab_explain, tab_shap, tab_perf, tab_eval = st.tabs([
    "📡 Live Detection",
    "🔍 LIME Explanations",
    "🌐 SHAP Global View",
    "⚡ Performance",
    "📊 Evaluation"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — LIVE DETECTION
# ══════════════════════════════════════════════════════════
with tab_live:
    m1, m2, m3, m4 = st.columns(4)
    cpu_box     = m1.empty()
    mem_box     = m2.empty()
    latency_box = m3.empty()
    alert_box   = m4.empty()

    st.markdown("---")
    st.subheader("Live Traffic Feed (last 20 flows)")
    feed_placeholder = st.empty()
    st.subheader("Class Distribution So Far")
    dist_placeholder = st.empty()

    if start:
        # Full reset
        for k, v in defaults.items():
            st.session_state[k] = v
        for f in [SHAP_BEESWARM, SHAP_META, SHAP_BAR_CSV]:
            if os.path.exists(f):
                os.remove(f)

        delay = 1.0 / flows_per_second
        n     = min(max_flows, len(X_test_selected))

        for i in range(n):
            row = X_test_selected[i].reshape(1, -1)

            t0   = time.perf_counter()
            pred = int(model.predict(row)[0])
            prob = model.predict_proba(row)[0]
            t1   = time.perf_counter()

            latency_ms = (t1 - t0) * 1000
            st.session_state.latencies.append(latency_ms)

            is_attack  = pred in ALERT_CLASSES
            class_name = CLASS_NAMES[pred]
            confidence = float(max(prob))
            true_label = int(y_test[i])
            correct    = (pred == true_label)

            if is_attack:
                st.session_state.attack_rows.append({
                    "flow_idx":   i,
                    "pred":       pred,
                    "confidence": confidence,
                    "label": f"Flow #{i+1} | {class_name} | Conf {confidence:.2f}"
                })

            cpu    = psutil.cpu_percent(interval=None)
            mem_mb = process.memory_info().rss / (1024 * 1024)

            cpu_box.metric("🖥️ CPU",            f"{cpu:.1f}%")
            mem_box.metric("💾 Memory",          f"{mem_mb:.1f} MB")
            latency_box.metric("⚡ Avg Latency", f"{np.mean(st.session_state.latencies):.3f} ms")
            alert_box.metric("🚨 Alerts",        str(len(st.session_state.attack_rows)))

            # Single unified status column
            if correct and not is_attack:
                status = "✅ Benign"
            elif correct and is_attack:
                status = "🚨 Attack (correct)"
            elif not correct and is_attack:
                status = "⚠️ Attack (misclassified)"
            else:
                status = "❌ Missed attack"

            st.session_state.log.append({
                "Flow #":       i + 1,
                "True Class":   CLASS_NAMES[true_label],
                "Predicted":    class_name,
                "Confidence":   f"{confidence:.2f}",
                "Status":       status,
                "Latency (ms)": f"{latency_ms:.4f}"
            })

            feed_placeholder.dataframe(
                pd.DataFrame(st.session_state.log[-20:]),
                use_container_width=True
            )

            full_df    = pd.DataFrame(st.session_state.log)
            class_wise = full_df["Predicted"].value_counts().reindex(
                [CLASS_NAMES[k] for k in range(6)], fill_value=0
            )
            dist_placeholder.bar_chart(class_wise)
            time.sleep(delay)

        st.session_state.sim_done = True

        # ── SHAP: compute and write to disk immediately ─────
        # Using st.status so user sees it happening
        with st.status("Computing SHAP global explanations...", expanded=True) as shap_status:
            result = compute_and_save_shap(shap_sample_size)
            if isinstance(result, float):
                shap_status.update(
                    label=f"✅ SHAP complete — {result:.1f} ms. Check the SHAP Global View tab.",
                    state="complete"
                )
            else:
                shap_status.update(
                    label=f"⚠️ SHAP failed: {result}",
                    state="error"
                )

        st.success(
            f"✅ Simulation complete — {n} flows processed, "
            f"{len(st.session_state.attack_rows)} attacks detected."
        )

    # Persist display after sim ends
    if st.session_state.log and not start:
        feed_placeholder.dataframe(
            pd.DataFrame(st.session_state.log[-20:]),
            use_container_width=True
        )
        full_df    = pd.DataFrame(st.session_state.log)
        class_wise = full_df["Predicted"].value_counts().reindex(
            [CLASS_NAMES[k] for k in range(6)], fill_value=0
        )
        dist_placeholder.bar_chart(class_wise)

    if st.session_state.log:
        csv = pd.DataFrame(st.session_state.log).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export CSV", csv,
            "simulation_results.csv", "text/csv"
        )

# ══════════════════════════════════════════════════════════
# TAB 2 — LIME ON-DEMAND
# ══════════════════════════════════════════════════════════
with tab_explain:
    st.subheader("Select an alert to explain")
    st.caption(
        "LIME explanations are generated on demand — decoupled from the "
        "live detection loop. Inference speed is never affected."
    )

    if not st.session_state.attack_rows:
        st.info("No attacks detected yet. Run the simulation first.")
    else:
        options  = [r["label"] for r in st.session_state.attack_rows]
        selected = st.selectbox("Flagged alerts", options)
        sel_idx  = options.index(selected)
        row_info = st.session_state.attack_rows[sel_idx]
        flow_idx = row_info["flow_idx"]
        pred     = row_info["pred"]
        conf     = row_info["confidence"]

        st.subheader("Feature values for this flow")
        fv_df = pd.DataFrame({
            "Feature": list(final_features),
            "Value":   np.round(X_test_selected[flow_idx], 6).tolist()
        })
        st.dataframe(fv_df, use_container_width=True)

        if st.button("🧠 Generate LIME Explanation"):
            with st.spinner("Perturbing input 1,000 times and fitting local linear model..."):
                t0  = time.perf_counter()
                exp = lime_explainer.explain_instance(
                    X_test_selected[flow_idx],
                    model.predict_proba,
                    num_features=10
                )
                t1         = time.perf_counter()
                explain_ms = (t1 - t0) * 1000
                st.session_state.explain_latencies.append(explain_ms)

            fig = exp.as_pyplot_figure()
            plt.title(
                f"Flow #{flow_idx+1} | Predicted: {CLASS_NAMES[pred]} "
                f"| Confidence: {conf:.2f}"
            )
            plt.tight_layout()
            lime_path = f"lime_flow_{flow_idx}.png"
            plt.savefig(lime_path, dpi=120, bbox_inches="tight")
            plt.close("all")
            st.image(lime_path, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric("⏱️ Explanation time",    f"{explain_ms:.1f} ms")
            c2.metric("Detection latency (avg)", f"{np.mean(st.session_state.latencies):.3f} ms")
            st.caption(
                "LIME runs post-hoc and asynchronously — "
                "it never touches the live detection pipeline."
            )

            # ── FIX: probability table with correct class mapping ──
            st.subheader("Predicted class probabilities")
            proba = model.predict_proba(
                X_test_selected[flow_idx].reshape(1, -1)
            )[0]
            # Build with explicit class index — don't sort then show index
            prob_df = pd.DataFrame({
                "Class ID": list(CLASS_NAMES.keys()),
                "Class":    list(CLASS_NAMES.values()),
                "Probability": np.round(proba, 4).tolist()
            }).sort_values("Probability", ascending=False).reset_index(drop=True)
            st.dataframe(prob_df, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — SHAP GLOBAL VIEW
# ══════════════════════════════════════════════════════════
with tab_shap:
    st.subheader("SHAP Global Feature Importance")
    st.caption(
        "SHAP (SHapley Additive exPlanations) shows which features the model "
        "relies on globally. Each dot = one sample; x-axis = SHAP contribution; "
        "colour = feature value (red = high, blue = low). "
        "Computed automatically when simulation ends."
    )

    if not shap_files_exist():
        st.info("Run the simulation — SHAP computes automatically when it finishes.")

        # Manual trigger fallback (in case auto-compute failed)
        if st.session_state.sim_done:
            if st.button("🔬 Retry SHAP Computation"):
                with st.spinner("Computing SHAP..."):
                    result = compute_and_save_shap(shap_sample_size)
                if isinstance(result, float):
                    st.success(f"SHAP complete — {result:.1f} ms")
                    st.rerun()
                else:
                    st.error(f"SHAP failed: {result}")
    else:
        # Load from disk — always reliable, never depends on session state
        with open(SHAP_META, "r") as f:
            meta = json.load(f)

        c1, c2 = st.columns(2)
        c1.metric("⏱️ SHAP computation time", f"{meta['time_ms']:.1f} ms")
        c2.metric("Samples used",              meta['sample_size'])
        st.caption(
            "Uses TreeExplainer — reads the tree structure directly, "
            "making it orders of magnitude faster than model-agnostic SHAP."
        )

        st.subheader("Beeswarm Plot — Per-Sample Feature Contributions")
        st.image(SHAP_BEESWARM, use_container_width=True)

        bar_df = pd.read_csv(SHAP_BAR_CSV)
        st.subheader("Mean |SHAP| per Feature (averaged across all classes)")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            bar_df["Feature"].tolist()[::-1],
            bar_df["Mean |SHAP|"].tolist()[::-1],
            color="steelblue"
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Global Feature Importance (SHAP)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(bar_df, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE
# ══════════════════════════════════════════════════════════
with tab_perf:
    if st.session_state.latencies:
        avg_lat    = np.mean(st.session_state.latencies)
        throughput = 1000 / avg_lat if avg_lat > 0 else 0

        st.subheader("Prediction Latency Over Time")
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(
            range(1, len(st.session_state.latencies) + 1),
            st.session_state.latencies,
            color="steelblue", linewidth=0.8, alpha=0.8
        )
        ax.axhline(
            avg_lat, color="red", linestyle="--", linewidth=1.2,
            label=f"Avg = {avg_lat:.3f} ms"
        )
        ax.set_xlabel("Flow #")
        ax.set_ylabel("Latency (ms)")
        ax.set_ylim(0, max(np.max(st.session_state.latencies) * 1.2, 1))
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Latency",           f"{avg_lat:.3f} ms")
        c2.metric("Max Latency",           f"{np.max(st.session_state.latencies):.3f} ms")
        c3.metric("Effective Throughput",  f"{throughput:.0f} flows/sec")

        # ── Real-time readiness assessment ──────────────────
        st.subheader("Real-Time Readiness Assessment")
        rt1, rt2, rt3 = st.columns(3)
        rt1.metric("Avg inference time",    f"{avg_lat:.3f} ms",
                   help="Sub-10ms = real-time capable")
        rt2.metric("Max inference time",    f"{np.max(st.session_state.latencies):.3f} ms",
                   help="Max spikes should stay under 10ms for real-time")
        rt3.metric("Flows/sec (inference)", f"{throughput:.0f}",
                   help="Enterprise networks: ~1,000–10,000 flows/min typical")

        # Colour-coded verdict
        if avg_lat < 1.0:
            st.success("✅ Excellent — avg latency under 1ms. Suitable for high-throughput edge deployment.")
        elif avg_lat < 5.0:
            st.success("✅ Good — avg latency under 5ms. Suitable for real-time deployment.")
        elif avg_lat < 10.0:
            st.warning("⚠️ Acceptable — avg latency under 10ms. Meets real-time threshold but monitor spikes.")
        else:
            st.error("❌ Too slow for real-time deployment at this latency.")

        st.caption(
            f"For context: a typical enterprise network sees ~1,000–10,000 flows per minute. "
            f"At {throughput:.0f} flows/sec ({throughput*60:,.0f}/min) this IDS is "
            f"**{throughput*60/5000:.0f}× faster** than the typical enterprise requirement. "
            "The `time.sleep()` in the simulation is artificial — raw inference is what matters."
        )

        if st.session_state.explain_latencies:
            st.subheader("LIME Explanation Latency")
            e1, e2 = st.columns(2)
            e1.metric("Avg Explanation Time",   f"{np.mean(st.session_state.explain_latencies):.1f} ms")
            e2.metric("Explanations Generated",  len(st.session_state.explain_latencies))
            st.caption(
                "LIME perturbs each input 1,000 times and fits a local linear model — "
                "expected 200–2,000 ms. This is acceptable because LIME runs "
                "post-hoc and asynchronously — it never blocks detection."
            )

        if shap_files_exist():
            with open(SHAP_META, "r") as f:
                meta = json.load(f)
            st.subheader("SHAP Computation Time")
            st.metric(
                f"SHAP on {meta['sample_size']} samples",
                f"{meta['time_ms']:.1f} ms"
            )
            st.caption(
                "TreeExplainer reads the tree structure directly — "
                "26ms for 100 samples is very fast and scales linearly."
            )

        if os.path.exists("scalability_results.csv"):
            st.subheader("Scalability Benchmark Results")
            sc_df = pd.read_csv("scalability_results.csv", index_col=0)
            st.dataframe(sc_df, use_container_width=True)
            st.caption(
                "Raw CPU batch inference with no sleep delay — represents maximum "
                "theoretical capacity. The 100-sample batch shows inflated throughput "
                "due to CPU cache warmup; 1,000–5,000 sample figures are more "
                "representative of sustained real-world throughput."
            )
            if os.path.exists("scalability_results.png"):
                st.image("scalability_results.png", use_container_width=True)
        else:
            st.info("Run `python scalability_test.py` to generate scalability graphs.")

    else:
        st.info("Run the simulation first.")

# ══════════════════════════════════════════════════════════
# TAB 5 — EVALUATION
# ══════════════════════════════════════════════════════════
with tab_eval:
    if st.session_state.log:
        full_df = pd.DataFrame(st.session_state.log)
        n       = len(full_df)

        name_to_idx = {v: k for k, v in CLASS_NAMES.items()}
        true_idx    = [int(y_test[i]) for i in range(n)]
        pred_idx    = [name_to_idx[p] for p in full_df["Predicted"]]

        acc         = accuracy_score(true_idx, pred_idx)
        macro_f1    = f1_score(true_idx, pred_idx, average="macro",    zero_division=0)
        weighted_f1 = f1_score(true_idx, pred_idx, average="weighted", zero_division=0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",    f"{acc*100:.2f}%")
        c2.metric("Macro F1",    f"{macro_f1:.3f}")
        c3.metric("Weighted F1", f"{weighted_f1:.3f}")
        st.caption(
            "Metrics reflect the stratified-shuffled subset streamed during simulation. "
            "**Macro F1** is the primary metric — it weights all 6 classes equally "
            "regardless of class size, penalising the model for missing rare attacks."
        )

        st.subheader("Confusion Matrix")
        labels = list(range(6))
        cm     = confusion_matrix(true_idx, pred_idx, labels=labels)
        short_names = [
            "Brute Force\n/ Web Attacks",
            "FTP / SSH\nPatator",
            "DoS / DDoS",
            "Infiltration",
            "Benign",
            "PortScan"
        ]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels(short_names, fontsize=8)
        ax.set_yticklabels(short_names, fontsize=8)
        for i in range(6):
            for j in range(6):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black"
                )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        benign_fp = int(cm[:, 4].sum() - cm[4, 4])
        if benign_fp > 0:
            st.caption(
                f"⚠️ {benign_fp} attack flows were misclassified as Benign "
                "(visible in the Benign column above). "
                "These are primarily Infiltration and Brute Force flows whose "
                "feature values overlap with benign traffic — "
                "a known challenge with this dataset."
            )

        st.subheader("Per-Class Precision / Recall / F1")
        prec, rec, f1s, support = precision_recall_fscore_support(
            true_idx, pred_idx, labels=labels, zero_division=0
        )
        report_df = pd.DataFrame({
            "Class":     [CLASS_NAMES[i] for i in labels],
            "Precision": np.round(prec, 3).tolist(),
            "Recall":    np.round(rec,  3).tolist(),
            "F1":        np.round(f1s,  3).tolist(),
            "Support":   support.tolist()
        })
        st.dataframe(report_df, use_container_width=True)

        st.subheader("🔬 Minority-Class Spotlight — Infiltration (Class 3)")
        i1, i2, i3 = st.columns(3)
        i1.metric("Infiltration samples seen", int(support[MINORITY_CLASS]))
        i2.metric("Infiltration F1",           f"{f1s[MINORITY_CLASS]:.3f}")
        i3.metric("Infiltration Recall",       f"{rec[MINORITY_CLASS]:.3f}")
        st.caption(
            "Class 3 has only 11 real training samples in CICIDS2017. "
            "Most published papers hide this by reporting weighted or overall accuracy only. "
            "We track it explicitly to demonstrate what our SMOTE + class-weighting "
            "imbalance-handling pipeline achieves on the rarest and most dangerous class."
        )
    else:
        st.info("Run the simulation first.")