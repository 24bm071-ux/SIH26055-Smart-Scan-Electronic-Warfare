"""Dashboard:  streamlit run app.py"""
import numpy as np, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from smartscan import Config, compare_all, EMITTER_TYPES

st.set_page_config(page_title="SmartScan", layout="wide")
st.title("Smart Scan Strategy Simulator")

with st.sidebar:
    st.header("Simulation setup")
    bands = st.slider("Frequency bands", 5, 50, 20)
    slots = st.slider("Duration (slots)", 200, 3000, 1000, 100)
    emitters = st.slider("Emitters", 1, 12, 5)
    pd_ = st.slider("Detection probability", 0.5, 1.0, 0.95, 0.01)
    pfa = st.slider("False alarm probability", 0.0, 0.2, 0.02, 0.01)
    types = st.multiselect("Emitter types", EMITTER_TYPES, default=list(EMITTER_TYPES))
    seed = st.number_input("Seed", 0, 9999, 0)
    st.subheader("Priority weights")
    w = [st.slider(n, 0.0, 1.0, v, 0.05) for n, v in
         [("w1 predicted P", .70), ("w2 recent activity", .0), ("w3 obs. age", .15), ("w4 uncertainty", .15)]]
    run = st.button("RUN SIMULATION", type="primary")

if run or "out" not in st.session_state:
    cfg = Config(bands, slots, emitters, pd_, pfa, int(seed), tuple(types or EMITTER_TYPES), tuple(w))
    with st.spinner("Simulating..."):
        st.session_state.out = compare_all(cfg)
G, info, logs, res, smart = st.session_state.out

t1, t2, t3, t4 = st.tabs(["Performance", "Live spectrum", "ML prediction", "Scheduler history"])
with t1:
    df = pd.DataFrame(res).drop(["Events", "Intercepted"])
    st.dataframe(df.style.format("{:.3f}"), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.bar_chart(df.loc[["Detection prob (Pd)", "Interception rate", "Useful scan rate"]])
    c2.bar_chart(df.loc[["Avg intercept delay (slots)"]])
    st.caption("Emitters: " + ", ".join(f"{i} ({k}, band {b+1})" for i, k, b in info))
with t2:
    name = st.radio("Scheduler", list(logs), horizontal=True, index=2)
    n = st.slider("Slots shown", 50, G.shape[1], min(200, G.shape[1]))
    a = np.array(logs[name]); a = a[a[:, 0] < n]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.imshow(G[:, :n], aspect="auto", cmap="Greys", origin="lower", interpolation="nearest")
    ax.scatter(a[:, 0], a[:, 1], c=np.where(a[:, 2] == 1, "lime", "red"), s=10, alpha=.8)
    ax.set_xlabel("Time slot"); ax.set_ylabel("Band"); ax.set_title("Ground truth (grey) + scans (green=hit, red=miss)")
    st.pyplot(fig)
with t3:
    imp = smart.feature_importance()
    if imp: st.bar_chart(pd.Series(imp, name="importance"))
    pl = np.array(smart.pred_log)
    if len(pl):
        acc = ((pl[:, 0] > .5) == pl[:, 1].astype(bool)).mean()
        st.metric("Online prediction accuracy", f"{acc:.1%}")
    st.metric("Scans concentrated on top-3 bands (Smart)",
              f"{pd.Series(np.array(logs['Smart ML'])[:, 1]).value_counts(normalize=True).head(3).sum():.1%}")
with t4:
    name = st.selectbox("Scheduler", list(logs), index=2)
    h = pd.DataFrame(logs[name], columns=["timestamp", "band", "detection", "truth", "confidence", "prediction"])
    h["band"] += 1
    st.dataframe(h, use_container_width=True, height=400)
    st.download_button("Download scan_history.csv", h.to_csv(index=False), f"scan_history_{name}.csv")
