# XAI-IDS Dashboard

Explainable AI-based Intrusion Detection System — real-time traffic simulation with offline SHAP/LIME explanations, built on CICIDS-2017.

## What This Does

- Simulates real-time network flow detection using a trained Decision Tree model
- Flags attacks live (Brute Force, FTP/SSH Patator, DoS/DDoS, Infiltration, PortScan)
- Generates SHAP (global) and LIME (per-alert) explanations offline, decoupled from detection
- Tracks performance metrics (latency, throughput, CPU/memory) and evaluation metrics (accuracy, F1, confusion matrix)

## Prerequisites

- Python 3.9–3.11 installed
- pip installed and working

## Setup Instructions

### 1. Clone / Download the Project

Make sure you have the full project folder, including:
```
xai_ids_dashboard/
├── app.py
├── scalability_test.py
├── verify.py
├── model.pkl
├── final_features.pkl
├── X_test_selected.npy
├── X_res.npy
└── y_test.npy
```

If any of the `.pkl` or `.npy` files are missing, the dashboard will show a clear error on launch telling you which files are missing — get those from whoever trained the model before proceeding.

### 2. Install Dependencies

Run this once (works on Windows, Mac, Linux):

```bash
pip install streamlit lime shap scikit-learn imbalanced-learn numpy pandas psutil matplotlib
```

> Tip: If you're using a virtual environment (recommended), activate it first:
> ```bash
> python -m venv venv
> venv\Scripts\activate     # Windows
> source venv/bin/activate    # Mac/Linux
> ```

### 3. Verify Your Setup

Run the verification script to confirm all required files and libraries are correctly in place before launching the dashboard:

```bash
python verify.py
```

If this throws any errors, fix them before moving to the next step — it usually means a missing file or a library version mismatch.

### 4. (Optional) Run the Scalability Benchmark

This generates `scalability_results.csv` and `scalability_results.png`, which show up automatically in the dashboard's Performance tab:

```bash
python scalability_test.py
```

This step is optional but recommended before your first demo — it can take a minute or two to run.

### 5. Launch the Dashboard

```bash
cd xai_ids_dashboard
streamlit run app.py
```

This will open the dashboard automatically in your browser at `http://localhost:8501`. If it doesn't open automatically, copy that URL into your browser manually.

## Using the Dashboard

1. **Live Detection tab** — set flows/sec and total flows in the sidebar, click "▶ Start Simulation"
2. **LIME Explanations tab** — pick any flagged alert, click "Generate LIME Explanation" for a per-instance breakdown
3. **SHAP Global View tab** — auto-computes after simulation ends; shows global feature importance
4. **Performance tab** — latency, throughput, CPU/memory, and real-time readiness verdict
5. **Evaluation tab** — accuracy, macro/weighted F1, confusion matrix, per-class metrics (including the Infiltration minority-class spotlight)

Use the "⟲ Reset" button in the sidebar to clear all results and start a fresh simulation run.

## Troubleshooting

| Issue | Fix |
|---|---|
| "Required files are missing" error on launch | Ensure `model.pkl`, `final_features.pkl`, `X_test_selected.npy`, `X_res.npy`, `y_test.npy` are in the same folder as `app.py` |
| SHAP tab shows an error | Re-run the simulation — SHAP recomputes automatically each time. If it persists, check your `shap` library version (`pip show shap`) |
| Dashboard doesn't open in browser | Manually visit `http://localhost:8501` |
| `ModuleNotFoundError` | Re-run the pip install command above — a package may not have installed correctly |
| Port already in use | Run `streamlit run app.py --server.port 8502` instead |

## Notes

- Detection latency is sub-1ms per flow — well within real-time thresholds for enterprise deployment
- SHAP and LIME run asynchronously/on-demand — they never block or slow down live detection
- This is a simulation using CICIDS-2017 test flows; it does not connect to a live network interface
