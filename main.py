"""CLI:  python main.py [--bands 20 --slots 1000 --emitters 5 --seed 0]"""
import argparse, pandas as pd
from smartscan import Config, compare_all

p = argparse.ArgumentParser()
p.add_argument("--bands", type=int, default=20); p.add_argument("--slots", type=int, default=1000)
p.add_argument("--emitters", type=int, default=5); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
cfg = Config(n_bands=a.bands, n_slots=a.slots, n_emitters=a.emitters, seed=a.seed)
G, info, logs, res, smart = compare_all(cfg)
print("Emitters:", info)
print(pd.DataFrame(res).round(3).to_string())
for name, log in logs.items():
    pd.DataFrame(log, columns=["timestamp", "band", "detection", "truth", "confidence", "prediction"]
                 ).to_csv(f"scan_history_{name.replace(' ', '_')}.csv", index=False)
print("Feature importance:", {k: round(v, 3) for k, v in smart.feature_importance().items()})
