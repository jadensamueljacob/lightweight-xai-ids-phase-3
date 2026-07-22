import numpy as np
import pickle

print("Loading files...")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("final_features.pkl", "rb") as f:
    final_features = pickle.load(f)

X_test_selected = np.load("X_test_selected.npy")
X_res = np.load("X_res.npy")
y_test = np.load("y_test.npy")

print(f"model expects: {model.n_features_in_} features")
print(f"final_features count: {len(final_features)}")
print(f"X_test_selected: {X_test_selected.shape}")
print(f"X_res shape: {X_res.shape}")
print(f"y_test shape: {y_test.shape}")

# ── Original checks ──────────────────────────────────────
assert model.n_features_in_ == 15, "❌ model feature count wrong"
assert len(final_features) == 15, "❌ final_features count wrong"
assert X_test_selected.shape[1] == 15, "❌ X_test_selected column count wrong"
assert X_res.shape[1] == 15, "❌ X_res column count wrong"

# ── New checks (added) ───────────────────────────────────
# 1. X_test_selected and y_test must have matching lengths
assert X_test_selected.shape[0] == y_test.shape[0], \
    "❌ X_test_selected and y_test row counts do not match"

# 2. Labels must only be from the expected 6 classes
valid_labels = {0, 1, 2, 3, 4, 5}
unique_labels = set(np.unique(y_test).astype(int).tolist())
assert unique_labels.issubset(valid_labels), \
    f"❌ y_test contains unexpected labels: {unique_labels - valid_labels}"

# 3. Model must be able to run a live prediction without error
try:
    sample_row = X_test_selected[0].reshape(1, -1)
    pred = model.predict(sample_row)
    proba = model.predict_proba(sample_row)
    assert pred.shape[0] == 1, "❌ model.predict() returned unexpected shape"
    assert proba.shape[1] == 6, "❌ model.predict_proba() returned wrong number of classes"
except Exception as e:
    raise AssertionError(f"❌ Live prediction sanity check failed: {e}")

print("\n✅ All checks passed (including label range, shape match, and live prediction). Safe to run dashboard.")
