import os, sys, time
import numpy as np, pandas as pd, streamlit as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())                       # run `streamlit run app.py` from the project folder
from live_backend import LiveDetector, lime_explain, CLASS_NAMES, BENIGN

st.set_page_config(page_title="Live Network Feed", layout="wide")
st.title("📡 Live Network Feed")
st.caption("Sensors (CSV replay or real packet capture) stream flow records over TCP to this detector. "
           "Detection runs continuously; LIME is on demand and never blocks detection.")

@st.cache_resource
def get_detector():
    return LiveDetector(os.environ.get("XAI_MODEL", "model.pkl"), "scaler_15.json", host="0.0.0.0", port=9009)

det = get_detector()

with st.sidebar:
    st.header("Live feed controls")
    if not det.running:
        if st.button("▶ Start listener (port 9009)"): det.start(); st.rerun()
    else:
        if st.button("■ Stop listener"): det.stop(); st.rerun()
    if st.button("⟲ Clear results"): det.clear()
    thr = st.slider("Alert confidence threshold", 0.50, 1.00, 0.70, 0.05)
    st.write("Listener:", "🟢 running" if det.running else "🔴 stopped")
    st.write("Model:", os.environ.get("XAI_MODEL", "model.pkl"))
    st.write("Adapter:", "OFF" if det.adapter is None else f"ON (k={det.adapter.k:.0f})")
    if det.error: st.error(det.error)

def is_alert(r): return r["pred"] != BENIGN and r["conf"] >= thr


@st.fragment(run_every=1)
def monitor():
    res, total, dropped, lat, e2e = det.snapshot()
    now = time.time()
    last10 = [r for r in res if now - r["t"] <= 10]
    alerts = [r for r in res if is_alert(r)]
    c = st.columns(5)
    c[0].metric("Flows received", f"{total:,}")
    c[1].metric("Flows/sec (10 s)", f"{len(last10)/10:.1f}")
    c[2].metric("Alerts", f"{len(alerts):,}")
    c[3].metric("Model time / flow", f"{np.mean(lat):.3f} ms" if lat else "-")
    c[4].metric("Sensor→verdict", f"{np.median(e2e):.0f} ms" if e2e else "-")
    if dropped: st.warning(f"{dropped} flows dropped (queue full) — detector can't keep up with this rate")
    if not res:
        st.info("Waiting for a sensor to connect. Start `flow_sensor_replay.py` or `flow_sensor_live.py`."); return

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Live flow feed (last 25)")
        rows = []
        for r in res[-25:][::-1]:
            status = "🚨 ALERT" if is_alert(r) else ("🟡 suppressed" if r["pred"] != BENIGN else "✅ benign")
            row = {"Time": time.strftime("%H:%M:%S", time.localtime(r["t"])),
                   "Source": r["src"], "→ Dest": f'{r["dst"]}:{r["dport"]}',
                   "Predicted": CLASS_NAMES[r["pred"]], "Conf": round(r["conf"], 2),
                   "Status": status, "OOD": f'{r["ood"]:.0%}'}
            if r["true"] is not None: row["True"] = CLASS_NAMES[r["true"]]
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Last 60 s by predicted class")
        win = [r for r in res if now - r["t"] <= 60]
        if win:
            df = pd.DataFrame({"t": [int(r["t"]) // 5 * 5 for r in win],
                               "cls": [CLASS_NAMES[r["pred"]] for r in win]})
            piv = df.groupby(["t", "cls"]).size().unstack(fill_value=0)
            piv.index = [time.strftime("%H:%M:%S", time.localtime(t)) for t in piv.index]
            st.bar_chart(piv)
    labelled = [r for r in res if r["true"] is not None]
    if labelled:
        acc = np.mean([r["pred"] == r["true"] for r in labelled])
        st.caption(f"Replay mode: {len(labelled)} labelled flows, running accuracy {acc:.1%}. "
                   "Real-capture flows have no ground truth (True column is empty). "
                   "OOD = share of features outside the training range (high OOD ⇒ treat verdict with caution).")
    else:
        st.caption("OOD = share of features outside the training range (high OOD ⇒ treat verdict with caution).")

monitor()

st.markdown("---")
st.subheader("🔍 Explain an alert (LIME, on demand)")
res, *_ = det.snapshot()
alerts = [r for r in res if is_alert(r)][-50:][::-1]
if not alerts:
    st.info("No alerts yet. This list refreshes when you interact with the page.")
else:
    labels = [f'{time.strftime("%H:%M:%S", time.localtime(r["t"]))} | {r["src"]}→{r["dst"]}:{r["dport"]} | '
              f'{CLASS_NAMES[r["pred"]]} | conf {r["conf"]:.2f}' for r in alerts]
    pick = st.selectbox("Recent alerts", labels)
    r = alerts[labels.index(pick)]
    st.dataframe(pd.DataFrame({"Feature": det.features, "Scaled value": np.round(r["x"], 4)}),
                 use_container_width=True, hide_index=True)
    if st.button("🧠 Generate LIME explanation"):
        t0 = time.perf_counter()
        exp = lime_explain(det, r["x"], r["pred"])
        ms = (time.perf_counter() - t0) * 1000
        fig = exp.as_pyplot_figure(label=r["pred"]); plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.metric("Explanation time", f"{ms:.0f} ms")

if res:
    csv = pd.DataFrame([{k: v for k, v in r.items() if k not in ("x", "proba")} for r in res]).to_csv(index=False)
    st.download_button("⬇️ Export live results CSV", csv, "live_results.csv", "text/csv")