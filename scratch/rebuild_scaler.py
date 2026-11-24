"""
STEP 1 - run ONCE (Colab or locally, wherever the 5 CICIDS CSVs are).

Your notebook never saved the MinMaxScaler, and the live pipeline needs it to turn raw
flow values into the 0-1 values the model was trained on. This script reproduces your
preprocessing exactly (same cleaning, labels, split seed) and saves:

  scaler_15.json     -> min/max of the 15 selected features (fit on TRAIN rows only)
  replay_flows.csv   -> the held-out TEST flows (raw values + IPs/ports/timestamps) for replay
                        (so the replay demo never shows flows the model trained on)

It then VERIFIES itself against your existing X_test_selected.npy / y_test.npy.
"""
import json, pickle, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ---- EDIT THESE -------------------------------------------------------------
CSV_DIR   = "."                       # folder containing the 5 CSVs
FILES = [                             # SAME ORDER as your Colab glob printed (order affects the split!)
    "Thursday-WorkingHours-AfternoonInfilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-AfternoonPortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
]
FEATURE_NAMES_PKL  = "feature_names.pkl"     # all 77 names (from preprocessing)
FINAL_FEATURES_PKL = "final_features.pkl"    # your 15
X_TEST_SEL = "X_test_selected.npy"; Y_TEST = "y_test.npy"   # for verification
# -----------------------------------------------------------------------------

feature_names  = pickle.load(open(FEATURE_NAMES_PKL, "rb"))
final_features = pickle.load(open(FINAL_FEATURES_PKL, "rb"))

dfs = [pd.read_csv(f"{CSV_DIR}/{f}", encoding="cp1252", low_memory=False) for f in FILES]
df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()

ID_COLS = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Timestamp", "Protocol"]
keep_ids = [c for c in ID_COLS if c in df.columns]
feat_cols = [c for c in df.columns if c not in keep_ids + ["Label"]]
assert feat_cols == feature_names, "Column order differs from feature_names.pkl"

# same cleaning as before (duplicates judged on features+label only, like the original)
df = df.drop_duplicates(subset=feat_cols + ["Label"])
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + ["Label"]).reset_index(drop=True)

label_mapping = {'BENIGN':'Normal','DDoS':'DDoS','DoS Hulk':'DoS','DoS GoldenEye':'DoS','DoS slowloris':'DoS',
 'DoS Slowhttptest':'DoS','Heartbleed':'DoS','Bot':'Bot','PortScan':'PortScan','Infiltration':'Infiltration',
 'Web Attack Brute Force':'WebAttack','Web Attack XSS':'WebAttack','Web Attack Sql Injection':'WebAttack'}
df["Label"] = df["Label"].map(label_mapping).fillna("Other")
df = df[df["Label"] != "Other"].reset_index(drop=True)
le = LabelEncoder(); y = le.fit_transform(df["Label"])
print("Class id -> name :", dict(enumerate(le.classes_)))          # <-- USE THIS for CLASS_NAMES

idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.3, random_state=42, stratify=y)

X = df[feat_cols].to_numpy(dtype=np.float64)
sc = MinMaxScaler().fit(X[idx_train])
sel = [feature_names.index(f) for f in final_features]

# ---- verification against what you trained/tested with ----
try:
    y_test_saved = np.load(Y_TEST); X_saved = np.load(X_TEST_SEL)
    same_y = len(y_test_saved) == len(idx_test) and (y[idx_test] == y_test_saved).all()
    same_x = same_y and np.allclose(sc.transform(X[idx_test])[:, sel], X_saved, atol=1e-6)
    print("test labels identical :", same_y, "| scaled features identical :", same_x)
    if not same_y:
        print("!! Split differs from your Colab run -> check FILES order. Do not use these outputs.")
        sys.exit(1)
except FileNotFoundError:
    print("(skipped verification: X_test_selected.npy / y_test.npy not found)")

json.dump({"features": list(final_features),
           "data_min": sc.data_min_[sel].tolist(),
           "data_max": sc.data_max_[sel].tolist()}, open("scaler_15.json", "w"), indent=1)

test = df.iloc[idx_test].copy(); test["label_id"] = y[idx_test]
cols = [c for c in ["Timestamp","Source IP","Source Port","Destination IP","Destination Port"] if c in test.columns]
test[cols + ["label_id"] + list(final_features)].to_csv("replay_flows.csv", index=False)
print(f"saved scaler_15.json and replay_flows.csv ({len(test)} test flows)")
