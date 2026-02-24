
"""
app.py — Streamlit GUI for Word Ladder Search.
Run with: python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Word Ladder Search", page_icon="🤖", layout="wide")
st.title(" Word Ladder Search in Semantic Embedding Space")

EMBEDDING_FILE = "glove.100d.20000.txt"

@st.cache_resource(show_spinner="Loading GloVe embeddings…")
def load_embeddings(path):
    from Embeddings import WordEmbeddings
    return WordEmbeddings(path)

if not os.path.exists(EMBEDDING_FILE):
    st.error(f"**{EMBEDDING_FILE}** not found. Place it in the same folder as app.py.")
    st.stop()

emb = load_embeddings(EMBEDDING_FILE)

from search.bfs    import bfs
from search.dfs    import dfs
from search.ucs    import ucs
from search.greedy import greedy
from search.A_star  import a_star

ALGORITHMS = {"BFS": bfs, "DFS": dfs, "UCS": ucs, "Greedy": greedy, "A*": a_star}

ALGO_DESC = {
    "BFS":    "Breadth-First Search — shortest hop-count path, uninformed.",
    "DFS":    "Depth-First Search — deep dive with depth limit, uninformed.",
    "UCS":    "Uniform Cost Search — min total cosine-distance path, uninformed.",
    "Greedy": "Greedy Best-First — fast but suboptimal, informed (h only).",
    "A*":     "A* — optimal + informed, balances g(n) + h(n).",
}

WORD_PAIRS = [
    ("doped", "leather"), ("shoves", "caressing"), ("salvatorian", "planetside"),
    ("panah", "holytown"), ("shindo", "ramadhin"), ("treatement", "muursepp"),
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Controls")
start_word  = st.sidebar.text_input("Start Word", value="king")
goal_word   = st.sidebar.text_input("Goal Word",  value="queen")
algorithm   = st.sidebar.selectbox("Algorithm", list(ALGORITHMS.keys()))
st.sidebar.caption(ALGO_DESC[algorithm])
k           = st.sidebar.slider("Neighbors (k)", 5, 50, 20)
depth_limit = st.sidebar.slider("DFS Depth Limit", 3, 20, 10)
run_btn     = st.sidebar.button("▶  Run Search", type="primary", use_container_width=True)
batch_btn   = st.sidebar.button("📊 Run All Pairs × All Algorithms", use_container_width=True)

# ── Single search ─────────────────────────────────────────────────────────────
if run_btn:
    sw, gw = start_word.strip().lower(), goal_word.strip().lower()

    if not emb.contains(sw):
        st.error(f"❌ '{sw}' is not in the vocabulary.")
    elif not emb.contains(gw):
        st.error(f"❌ '{gw}' is not in the vocabulary.")
    else:
        fn = ALGORITHMS[algorithm]
        with st.spinner(f"Running {algorithm}…"):
            result = fn(emb, sw, gw, k=k, depth_limit=depth_limit) if algorithm == "DFS" else fn(emb, sw, gw, k=k)

        st.markdown(f"### {algorithm}: `{sw}` → `{gw}`")

        if result.found:
            st.success(f"✅ Path found in **{result.path_length}** step(s)")
            st.code(" → ".join(result.path), language=None)
        else:
            st.warning("❌ No path found within search limits.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Path Steps",      result.path_length if result.found else "—")
        m2.metric("Nodes Expanded",  result.nodes_expanded)
        m3.metric("Search Time (s)", f"{result.runtime:.4f}")

# ── Batch experiment ──────────────────────────────────────────────────────────
elif batch_btn:
    st.subheader("📊 Batch Experimental Results")
    rows = []
    prog = st.progress(0)
    total = len(WORD_PAIRS) * len(ALGORITHMS)
    done = 0
    for sw, gw in WORD_PAIRS:
        for alg_name, fn in ALGORITHMS.items():
            if not emb.contains(sw) or not emb.contains(gw):
                rows.append({"Start": sw, "Goal": gw, "Algorithm": alg_name,
                             "Found": "OOV", "Steps": "—", "Nodes Expanded": "—", "Time (s)": "—"})
            else:
                result = fn(emb, sw, gw, k=20, depth_limit=10) if alg_name == "DFS" else fn(emb, sw, gw, k=20)
                rows.append({"Start": sw, "Goal": gw, "Algorithm": alg_name,
                             "Found": "✅" if result.found else "❌",
                             "Steps": str(result.path_length) if result.found else "—",
                             "Nodes Expanded": result.nodes_expanded,
                             "Time (s)": f"{result.runtime:.4f}"})
            done += 1
            prog.progress(done / total)
    prog.empty()
    st.dataframe(pd.DataFrame(rows), width='stretch')

# ── Vocabulary checker ────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔍 Vocabulary Checker"):
    check = st.text_input("Check if a word is in the GloVe vocabulary:")
    if check:
        w = check.strip().lower()
        st.success(f"✅ '{w}' is in vocabulary.") if emb.contains(w) else st.error(f"❌ '{w}' NOT in vocabulary.")