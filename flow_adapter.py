"""
Environment calibration layer applied to LIVE flows only (never to the CICIDS replay) before scaling.
Everything here is fitted from BENIGN calibration traffic - never from attack flows or attack labels.

Currently: one global time-dilation factor k. The lab network (VMware, microsecond RTT) is ~k times faster than the
real LAN the dataset was captured on, so every inter-arrival/duration-based value is scaled by k and rates by 1/k.
That preserves the relationships between the features (Packets/s = packets / duration) - it just changes the clock.
"""
import json, os

TIME_FEATURES = ["Fwd IAT Total", "Fwd IAT Mean", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max"]
RATE_FEATURES = ["Fwd Packets/s"]

class FlowAdapter:
    def __init__(self, cfg):
        self.k = float(cfg.get("time_scale", 1.0))
    @classmethod
    def load(cls, path="adapter.json"):
        return cls(json.load(open(path))) if os.path.exists(path) else None
    def __call__(self, feats):
        out = dict(feats)
        for f in TIME_FEATURES:
            if f in out: out[f] = out[f] * self.k
        for f in RATE_FEATURES:
            if f in out: out[f] = out[f] / self.k
        return out
