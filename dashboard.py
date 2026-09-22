"""
TRAFFIC-INTEL — Streamlit dashboard for Malicious Traffic Pattern Clustering
Run with:  streamlit run dashboard.py
Expects clustered_traffic_results.csv (from your Colab Stage 9 export)
in the SAME FOLDER as this file.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# =========================================================
# 0. PAGE CONFIG + DARK SOC THEME
# =========================================================
st.set_page_config(page_title="TRAFFIC-INTEL", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #05070a; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #0b0f16; }
    div[data-testid="stMetricValue"] { color: #e6edf3; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetric"] { background-color: #0b0f16; border: 1px solid #1a2230;
                                   border-radius: 8px; padding: 10px; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
    .cluster-card { background: #0e131c; border: 1px solid #1a2230; border-radius: 8px;
                    padding: 10px 12px; margin-bottom: 8px; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 1. LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("clustered_traffic_results.csv")
    return df


df = load_data()

FEATURES = ['Flow Duration', 'Total Fwd Packets', 'Flow Bytes/s', 'Flow IAT Mean',
            'Fwd Packet Length Mean', 'SYN Flag Count', 'Down/Up Ratio']

# Real evaluation metrics from your Colab Stage 8 run (corrected sample).
# Update these three dicts if you rerun Stage 8 and get different numbers.
METRICS = {
    "K-Means": {"Silhouette": 0.384, "Davies-Bouldin": 0.827, "Calinski-Harabasz": 5514.487},
    "Hierarchical": {"Silhouette": 0.348, "Davies-Bouldin": 0.954, "Calinski-Harabasz": 5102.269},
}

# =========================================================
# 2. HEADER
# =========================================================
st.markdown("### TRAFFIC-INTEL <span style='color:#66717f; font-size:16px;'>/ cluster console</span>",
            unsafe_allow_html=True)
st.caption(
    "MALICIOUS TRAFFIC PATTERN CLUSTERING — K-MEANS & HIERARCHICAL (real CICIDS2017 sample, n={:,})".format(len(df)))

c1, c2, c3, c4, c5 = st.columns(5)
algo_key = "kmeans_cluster"  # updated below once sidebar renders
c1.metric("Flows Plotted", f"{len(df):,}")
c2.metric("Clusters (k)", df["kmeans_cluster"].nunique())
c3.metric("Silhouette (K-Means)", f"{METRICS['K-Means']['Silhouette']:.3f}")
c4.metric("Davies-Bouldin (K-Means)", f"{METRICS['K-Means']['Davies-Bouldin']:.3f}")
c5.metric("Calinski-Harabasz (K-Means)", f"{METRICS['K-Means']['Calinski-Harabasz']:.0f}")

st.markdown("---")

# =========================================================
# 3. SIDEBAR — algorithm selector + cluster legend
# =========================================================
st.sidebar.markdown("#### ALGORITHM")
algo_choice = st.sidebar.radio("Algorithm", ["K-Means", "Hierarchical"], horizontal=True, label_visibility="collapsed")
cluster_col = "kmeans_cluster" if algo_choice == "K-Means" else "hierarchical_cluster"

st.sidebar.markdown("#### EVALUATION SCORES")
m = METRICS[algo_choice]
st.sidebar.write(f"Silhouette: **{m['Silhouette']:.3f}**")
st.sidebar.write(f"Davies-Bouldin: **{m['Davies-Bouldin']:.3f}**")
st.sidebar.write(f"Calinski-Harabasz: **{m['Calinski-Harabasz']:.1f}**")

st.sidebar.markdown("#### CLUSTER LEGEND")
st.sidebar.caption("Clusters group traffic by volume/duration behavior — "
                   "NOT a direct attack classifier. See Panel B for ground-truth attack fingerprints.")

CLUSTER_COLORS = ["#2dd4c8", "#ff3b5c", "#ffb020", "#b56bff", "#4d8dff", "#ff8a5c", "#7ee787"]
cluster_ids = sorted(df[cluster_col].unique())

for i, cid in enumerate(cluster_ids):
    sub = df[df[cluster_col] == cid]
    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
    top_label = sub["true_label"].value_counts(normalize=True).idxmax()
    top_pct = sub["true_label"].value_counts(normalize=True).max() * 100
    st.sidebar.markdown(f"""
    <div class="cluster-card">
        <span style="color:{color}; font-weight:700;">● Cluster {cid}</span>
        &nbsp; n={len(sub):,} ({len(sub) / len(df) * 100:.1f}%)<br>
        <span style="color:#66717f; font-size:11px;">Dominant label: {top_label} ({top_pct:.1f}%)</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("#### FILTER BY TRUE LABEL")
label_filter = st.sidebar.multiselect("Filter", options=sorted(df["true_label"].unique()),
                                      default=[], label_visibility="collapsed",
                                      help="Leave empty to show all traffic")
plot_df = df if not label_filter else df[df["true_label"].isin(label_filter)]

# =========================================================
# 4. PANEL A — 3D cluster visualization (real PCA + real cluster assignments)
# =========================================================
st.markdown(f"#### PANEL A — Unsupervised Clusters ({algo_choice}, PCA→3D projection)")
st.caption("PCA explains 60.4% of total variance across 3 components — proximity in this view is "
           "approximate, not exact. Colors = cluster ID, not attack severity.")

fig = go.Figure()
for i, cid in enumerate(cluster_ids):
    sub = plot_df[plot_df[cluster_col] == cid]
    color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
    fig.add_trace(go.Scatter3d(
        x=sub["pca_x"], y=sub["pca_y"], z=sub["pca_z"], mode="markers",
        name=f"Cluster {cid}",
        marker=dict(size=3, color=color, opacity=0.75),
        customdata=sub.index,
        hovertemplate="Cluster %{text}<extra></extra>", text=[str(cid)] * len(sub),
    ))

fig.update_layout(
    template="plotly_dark", paper_bgcolor="#05070a", plot_bgcolor="#05070a",
    scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3", bgcolor="#05070a"),
    legend=dict(bgcolor="#0b0f16"), margin=dict(l=0, r=0, t=10, b=0), height=560,
)

# Native Streamlit click-selection — no extra package needed
event = st.plotly_chart(fig, width="stretch", on_select="rerun",
                        selection_mode="points", key="cluster_plot")

# =========================================================
# 5. FLOW INSPECTOR — updates when a point is clicked above
# =========================================================
left, right = st.columns([2, 1])

with right:
    st.markdown("#### FLOW INSPECTOR")
    points = event.selection.get("points", []) if event and event.selection else []
    if not points:
        st.info("Click any point in the 3D plot to inspect that flow.")
    else:
        row_idx = points[0]["customdata"][0]  # customdata comes back as a list, e.g. [1234]
        row = df.loc[row_idx]
        cid = row[cluster_col]
        color = CLUSTER_COLORS[cluster_ids.index(cid) % len(CLUSTER_COLORS)]
        st.markdown(f"<span style='color:{color}; font-weight:700;'>Cluster {cid}</span> ({algo_choice})",
                    unsafe_allow_html=True)
        for feat in FEATURES:
            st.write(f"**{feat}:** {row[feat]:.2f}")
        st.markdown(f"**Known label (ground truth):** `{row['true_label']}`")
        st.caption("Note: label shown for reference only — was NOT used during clustering.")

with left:
    st.markdown("#### CLUSTER SIZE COMPARISON")
    size_df = df[cluster_col].value_counts().sort_index()
    st.bar_chart(size_df)

st.markdown("---")

# =========================================================
# 6. PANEL B — ground-truth attack fingerprints (NOT from clustering)
# =========================================================
st.markdown("#### PANEL B — Known Attack-Type Traffic Profiles")
st.caption("Built directly from the dataset's ground-truth `Label` column, shown for comparison "
           "only. This is NOT the output of the unsupervised model above — the clustering did not "
           "reliably separate these categories on its own (see report for why).")

profile = df.groupby("true_label")[FEATURES].mean().round(2)
st.dataframe(profile.style.background_gradient(cmap="viridis", axis=0), width="stretch")

st.markdown("#### K-MEANS vs HIERARCHICAL — EVALUATION COMPARISON")
metrics_df = pd.DataFrame(METRICS).T
st.dataframe(metrics_df.style.format("{:.3f}"), width="stretch")

# =========================================================
# 7. DOWNLOAD
# =========================================================
st.markdown("---")
st.download_button("Download clustered dataset (CSV)", df.to_csv(index=False),
                   file_name="clustered_traffic_results.csv", mime="text/csv")
