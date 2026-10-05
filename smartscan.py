"""SmartScan core: RF environment, receiver, schedulers, ML predictor, metrics."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier

EMITTER_TYPES = ("continuous", "periodic", "random", "burst", "agile")
FEATURES = ["hit_rate", "recent_activity", "time_since_hit", "recent_misses", "obs_age", "periodicity", "last_obs"]


@dataclass
class Config:
    n_bands: int = 20
    n_slots: int = 1000
    n_emitters: int = 5
    pd: float = 0.95
    pfa: float = 0.02
    seed: int = 0
    emitter_types: tuple = ("periodic", "random", "burst", "agile", "continuous")
    weights: tuple = (0.70, 0.00, 0.15, 0.15)  # w1 P, w2 R, w3 A, w4 U
    warmup: int = 40          # fixed-sweep exploration slots before ML kicks in
    retrain_every: int = 25
    tracked_penalty: float = 0.5  # score multiplier for bands whose last scan was a hit


# ---------------------------------------------------------------- environment
def generate_ground_truth(cfg: Config):
    """Band x Time matrix (1 = transmission). Hidden from schedulers."""
    rng = np.random.default_rng(cfg.seed)
    B, T = cfg.n_bands, cfg.n_slots
    G = np.zeros((B, T), dtype=np.uint8)
    info = []
    for i in range(cfg.n_emitters):
        kind = cfg.emitter_types[i % len(cfg.emitter_types)]
        band = int(rng.integers(B))
        if kind == "continuous":
            G[band, :] = 1
        elif kind == "periodic":
            period, on, off = int(rng.integers(8, 20)), 2, int(rng.integers(0, 8))
            t = np.arange(T)
            G[band, ((t + off) % period) < on] = 1
        elif kind == "random":
            G[band, rng.random(T) < 0.3] = 1
        elif kind == "burst":
            t = 0
            while t < T:
                t += int(rng.geometric(0.03))
                G[band, t:t + int(rng.integers(3, 7))] = 1
        elif kind == "agile":
            hops = rng.choice(B, size=min(3, B), replace=False)
            t = 0
            while t < T:
                dwell = int(rng.integers(5, 16))
                G[int(rng.choice(hops)), t:t + dwell] = 1
                t += dwell
        info.append((f"E{i+1}", kind, band))
    return G, info


# ------------------------------------------------------------------- receiver
class Receiver:
    def __init__(self, G, cfg: Config):
        self.G, self.cfg = G, cfg
        self.rng = np.random.default_rng(cfg.seed + 1)

    def scan(self, band, t):
        present = bool(self.G[band, t])
        u = self.rng.random()
        hit = (u < self.cfg.pd) if present else (u < self.cfg.pfa)
        conf = float(np.clip(self.rng.normal(0.9 if hit else 0.2, 0.06), 0, 1))
        return int(hit), conf


# ------------------------------------------------------------- observation memory
class Memory:
    def __init__(self, B):
        self.B = B
        self.n = np.zeros(B); self.hits = np.zeros(B)
        self.last_hit = np.full(B, -1); self.last_scan = np.full(B, -1); self.last_obs = np.zeros(B)
        self.recent = [deque(maxlen=5) for _ in range(B)]
        self.onsets = [deque(maxlen=8) for _ in range(B)]

    def update(self, b, t, obs):
        self.n[b] += 1; self.hits[b] += obs; self.recent[b].append(obs)
        if obs:
            if self.last_hit[b] < 0 or t - self.last_hit[b] > 2:
                self.onsets[b].append(t)
            self.last_hit[b] = t
        self.last_scan[b] = t; self.last_obs[b] = obs

    def features(self, t):
        F = np.zeros((self.B, len(FEATURES)))
        for b in range(self.B):
            r = self.recent[b]
            F[b, 0] = self.hits[b] / self.n[b] if self.n[b] else 0
            F[b, 1] = np.mean(r) if r else 0
            F[b, 2] = t - self.last_hit[b] if self.last_hit[b] >= 0 else t + 1
            F[b, 3] = len(r) - sum(r)
            F[b, 4] = t - self.last_scan[b] if self.last_scan[b] >= 0 else t + 1
            F[b, 6] = self.last_obs[b]
            o = list(self.onsets[b])
            if len(o) >= 3:
                d = np.diff(o); period = max(np.median(d), 1)
                phase = (t - o[-1]) % period
                due = 1 - min(phase, period - phase) / (period / 2)
                F[b, 5] = due / (1 + np.std(d) / period)
        return F


# ----------------------------------------------------------------- schedulers
class FixedScheduler:
    name = "Fixed"
    def __init__(self, cfg): self.B = cfg.n_bands
    def next_band(self, t, mem): return t % self.B
    def update(self, *a): pass


class RandomScheduler:
    name = "Random"
    def __init__(self, cfg):
        self.B, self.rng = cfg.n_bands, np.random.default_rng(cfg.seed + 2)
    def next_band(self, t, mem): return int(self.rng.integers(self.B))
    def update(self, *a): pass


class SmartScheduler:
    """Closed loop: features -> Random Forest P(tx) -> priority score -> next band."""
    name = "Smart ML"

    def __init__(self, cfg):
        self.cfg, self.B = cfg, cfg.n_bands
        self.X, self.y = [], []
        self.model, self.last_features, self.last_pred = None, None, None
        self.pred_log = []

    def _train(self):
        y = np.array(self.y)
        if len(y) < 30 or y.min() == y.max():
            return
        self.model = RandomForestClassifier(n_estimators=40, min_samples_leaf=3,
                                            random_state=self.cfg.seed, n_jobs=1)
        self.model.fit(np.array(self.X), y)

    def predict(self, F):
        if self.model is None:
            return F[:, 0] * (1 - F[:, 6])            # cold-start heuristic
        return self.model.predict_proba(F)[:, 1]

    def next_band(self, t, mem):
        if t < self.cfg.warmup:                        # phase 1: exploration sweep
            b = t % self.B
            self.last_features = mem.features(t)
            self.last_pred = self.predict(self.last_features)
            return b
        F = mem.features(t)
        P = self.predict(F)
        w1, w2, w3, w4 = self.cfg.weights
        age = F[:, 4] / (F[:, 4].max() + 1e-9)
        unc = 1 / np.sqrt(1 + mem.n)
        score = w1 * P + w2 * F[:, 1] * (1 - F[:, 6]) + w3 * age + w4 * unc
        # intercept-aware: a band we just caught is already being tracked -> low marginal value
        score = np.where(mem.last_obs > 0, score * self.cfg.tracked_penalty, score)
        self.last_features, self.last_pred = F, P
        return int(np.argmax(score))

    def update(self, t, b, obs, mem_unused=None):
        # target = NEW interception: hit on a band that was not already being hit
        label = int(obs and not self.last_features[b][6])
        self.X.append(self.last_features[b]); self.y.append(label)
        self.pred_log.append((self.last_pred[b], label))
        if t % self.cfg.retrain_every == 0:
            self._train()

    def feature_importance(self):
        return dict(zip(FEATURES, self.model.feature_importances_)) if self.model else {}


# ---------------------------------------------------------------- closed loop
def run_simulation(cfg: Config, G, scheduler):
    rx, mem = Receiver(G, cfg), Memory(cfg.n_bands)
    log = []
    for t in range(cfg.n_slots):
        b = scheduler.next_band(t, mem)
        obs, conf = rx.scan(b, t)
        mem.update(b, t, obs)
        scheduler.update(t, b, obs)
        pred = float(scheduler.last_pred[b]) if getattr(scheduler, "last_pred", None) is not None else np.nan
        log.append((t, b, obs, int(G[b, t]), conf, pred))
    return log, compute_metrics(G, log)


def compute_metrics(G, log):
    a = np.array(log)
    obs, truth = a[:, 2].astype(bool), a[:, 3].astype(bool)
    tp, fn = (obs & truth).sum(), (~obs & truth).sum()
    fp, tn = (obs & ~truth).sum(), (~obs & ~truth).sum()
    tpm = np.zeros_like(G, dtype=bool)
    for t, b, o, tr, *_ in log:
        if o and tr: tpm[int(b), int(t)] = True
    events = intercepted = 0; delays = []
    for b in range(G.shape[0]):
        row = np.concatenate(([0], G[b].astype(int), [0]))
        starts, ends = np.where(np.diff(row) == 1)[0], np.where(np.diff(row) == -1)[0]
        for s, e in zip(starts, ends):
            events += 1
            idx = np.where(tpm[b, s:e])[0]
            if len(idx): intercepted += 1; delays.append(idx[0])
    return {
        "Detection prob (Pd)": tp / max(tp + fn, 1),
        "False alarm rate (Pfa)": fp / max(fp + tn, 1),
        "Useful scan rate": truth.mean(),
        "Interception rate": intercepted / max(events, 1),
        "Avg intercept delay (slots)": float(np.mean(delays)) if delays else float("nan"),
        "Events": events, "Intercepted": intercepted,
    }


def compare_all(cfg: Config):
    G, info = generate_ground_truth(cfg)
    results, logs, smart = {}, {}, None
    for cls in (FixedScheduler, RandomScheduler, SmartScheduler):
        s = cls(cfg)
        logs[s.name], results[s.name] = run_simulation(cfg, G, s)
        if cls is SmartScheduler: smart = s
    return G, info, logs, results, smart
