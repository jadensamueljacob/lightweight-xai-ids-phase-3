import numpy as np
import pickle

print("Loading files...")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("final_features.pkl", "rb") as f:
    final_features = pickle.load(f)

X_test_selected = np.load("X_test_selected.npy")
X_res           = np.load("X_res.npy")
y_test          = np.load("y_test.npy")

print(f"model expects:        {model.n_features_in_} features")
print(f"final_features count: {len(final_features)}")
print(f"X_test_selected:      {X_test_selected.shape}")
print(f"X_res shape:          {X_res.shape}")
print(f"y_test shape:         {y_test.shape}")

# All checks
assert model.n_features_in_ == 15,          "❌ model feature count wrong"
assert len(final_features)  == 15,          "❌ final_features count wrong"
assert X_test_selected.shape[1] == 15,      "❌ X_test_selected column count wrong"
assert X_res.shape[1] == 15,               "❌ X_res column count wrong"

print("\n✅ All checks passed. Safe to run dashboard.")