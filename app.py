import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation · Clustering Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e4dc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f0f17 !important;
    border-right: 1px solid #1e1e2e;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stMultiselect label {
    color: #8b8fa8 !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'Space Mono', monospace;
}

/* Header */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
    background: linear-gradient(135deg, #e8e4dc 0%, #c4a882 50%, #e8e4dc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4a4a6a;
    text-transform: uppercase;
    letter-spacing: 0.2em;
}

/* Metric cards */
.metric-card {
    background: #0f0f17;
    border: 1px solid #1e1e2e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #c4a882, #8b5cf6);
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a4a6a;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #c4a882;
}

/* Section headers */
.section-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #4a4a6a;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-left: 2px solid #c4a882;
    padding-left: 0.75rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e8e4dc;
    margin-bottom: 0.25rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e1e2e;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4a4a6a !important;
    background: transparent !important;
    border: none !important;
    padding: 0.75rem 1.25rem !important;
}
.stTabs [aria-selected="true"] {
    color: #c4a882 !important;
    border-bottom: 2px solid #c4a882 !important;
}

/* Divider */
hr { border-color: #1e1e2e; }

/* Selectbox / Slider */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #0f0f17 !important;
    border-color: #1e1e2e !important;
    color: #e8e4dc !important;
}

/* Plotly charts background */
.js-plotly-plot .plotly { background: transparent !important; }

/* Info box */
.info-box {
    background: #0f0f17;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #6b6b8a;
    line-height: 1.7;
}

/* Upload area */
.upload-zone {
    background: #0f0f17;
    border: 1px dashed #2e2e4e;
    border-radius: 16px;
    padding: 3rem;
    text-align: center;
    transition: border-color 0.3s;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2e2e4e; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,10,15,0.6)",
    font=dict(family="Space Mono, monospace", color="#8b8fa8", size=11),
    xaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e", tickfont=dict(color="#4a4a6a")),
    yaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e", tickfont=dict(color="#4a4a6a")),
    margin=dict(l=40, r=20, t=40, b=40),
    colorway=["#c4a882","#8b5cf6","#06b6d4","#f59e0b","#10b981","#f43f5e","#84cc16","#e879f9"],
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e1e2e", font=dict(color="#6b6b8a")),
)

CLUSTER_COLORS = ["#c4a882","#8b5cf6","#06b6d4","#f59e0b","#10b981","#f43f5e","#84cc16","#e879f9",
                  "#fb923c","#38bdf8","#a3e635","#f472b6"]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n = 2000

    age_groups = np.random.choice([25, 35, 45, 55, 65], n, p=[0.2,0.25,0.25,0.2,0.1])
    age = age_groups + np.random.randint(-5, 6, n)

    gender = np.random.choice(["Male","Female"], n, p=[0.48,0.52])

    spending_score = np.clip(
        np.where(age < 35, np.random.normal(65, 20, n),
        np.where(age < 50, np.random.normal(50, 25, n),
                            np.random.normal(35, 20, n))), 1, 100).astype(int)

    annual_income = np.clip(
        age * 1200 + np.random.normal(0, 15000, n) +
        np.where(gender=="Male", 5000, -2000), 15000, 200000).astype(int)

    profession_map = {
        25: ["Student","Artist","Healthcare"],
        35: ["Engineer","Doctor","Lawyer"],
        45: ["Executive","Doctor","Engineer"],
        55: ["Executive","Lawyer","Healthcare"],
        65: ["Retired","Healthcare","Artist"]
    }
    profession = [np.random.choice(profession_map[g]) for g in age_groups]
    work_exp = np.clip(age - np.random.randint(22, 26, n), 0, 40)
    family_size = np.random.choice([1,2,3,4,5,6], n, p=[0.1,0.3,0.25,0.2,0.1,0.05])

    df = pd.DataFrame({
        "CustomerID": range(1, n+1),
        "Gender": gender,
        "Ever_Married": np.where(age > 30, np.random.choice(["Yes","No"], n, p=[0.7,0.3]),
                                  np.random.choice(["Yes","No"], n, p=[0.2,0.8])),
        "Age": age,
        "Graduated": np.random.choice(["Yes","No"], n, p=[0.65,0.35]),
        "Profession": profession,
        "Work_Experience": work_exp,
        "Spending_Score": np.random.choice(["Low","Average","High"], n, p=[0.3,0.4,0.3]),
        "Family_Size": family_size,
        "Annual_Income": annual_income,
        "Spending_Score_Num": spending_score,
        "Segmentation": np.random.choice(["A","B","C","D"], n)
    })
    return df

@st.cache_data
def preprocess(df, feature_cols):
    df_proc = df[feature_cols].copy()
    le = LabelEncoder()
    for col in df_proc.select_dtypes(include="object").columns:
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    df_proc = df_proc.fillna(df_proc.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_proc)
    return X_scaled, df_proc

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem;'>
        <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#4a4a6a;
                    text-transform:uppercase;letter-spacing:0.2em;margin-bottom:0.5rem;'>System</div>
        <div style='font-size:1.1rem;font-weight:800;color:#e8e4dc;'>◈ Clustering Lab</div>
    </div>
    <hr style='border-color:#1e1e2e;margin:0 0 1.5rem;'>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Données**")
    data_source = st.radio("Source", ["Données démo", "Charger CSV"], label_visibility="collapsed")

    df_raw = None
    if data_source == "Charger CSV":
        uploaded = st.file_uploader("Fichier CSV", type=["csv"])
        if uploaded:
            df_raw = pd.read_csv(uploaded)
            st.success(f"✓ {len(df_raw)} lignes chargées")
    else:
        df_raw = load_sample_data()
        st.caption("▸ Dataset synthétique (2000 clients)")

    st.markdown("<hr style='border-color:#1e1e2e;'>", unsafe_allow_html=True)

    if df_raw is not None:
        num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df_raw.select_dtypes(include="object").columns.tolist()
        exclude = ["CustomerID","Segmentation","Spending_Score_Num"] if "CustomerID" in df_raw.columns else []
        default_features = [c for c in (num_cols + cat_cols) if c not in exclude][:7]

        st.markdown("**⚙ Features**")
        feature_cols = st.multiselect("Variables", num_cols + cat_cols,
                                       default=default_features, label_visibility="collapsed")

        st.markdown("<hr style='border-color:#1e1e2e;'>", unsafe_allow_html=True)
        st.markdown("**🔬 Algorithme**")
        algorithm = st.selectbox("Méthode", ["KMeans","CAH (Agglomeratif)","DBSCAN"],
                                  label_visibility="collapsed")

        if algorithm == "KMeans":
            n_clusters = st.slider("Clusters k", 2, 10, 4)
            init_method = st.selectbox("Init", ["k-means++","random"])
        elif algorithm == "CAH (Agglomeratif)":
            n_clusters = st.slider("Clusters k", 2, 10, 4)
            linkage_method = st.selectbox("Linkage", ["ward","complete","average","single"])
        else:
            eps = st.slider("eps (rayon)", 0.1, 3.0, 0.5, 0.05)
            min_samples = st.slider("min_samples", 2, 20, 5)

        st.markdown("<hr style='border-color:#1e1e2e;'>", unsafe_allow_html=True)
        st.markdown("**📐 Réduction dim.**")
        dim_method = st.selectbox("Méthode", ["PCA","t-SNE","UMAP (via PCA)"],
                                   label_visibility="collapsed")
        n_components_viz = st.radio("Dimensions", ["2D","3D"], horizontal=True)
        n_comp = 2 if n_components_viz == "2D" else 3

        if dim_method == "t-SNE":
            perplexity = st.slider("Perplexité", 5, 50, 30)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if df_raw is None or not feature_cols:
    st.markdown("""
    <div style='padding:4rem;text-align:center;'>
        <div class='hero-title'>Customer<br>Segmentation</div>
        <div class='hero-sub' style='margin-top:1rem;'>Clustering · Réduction de Dimensions · Analyse</div>
        <div style='margin-top:3rem;color:#4a4a6a;font-family:Space Mono,monospace;font-size:0.75rem;'>
            ← Chargez un dataset dans le panneau latéral
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Compute clusters ──────────────────────────────────────────────────────────
with st.spinner("Calcul en cours…"):
    X_scaled, df_proc = preprocess(df_raw, feature_cols)

    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, init=init_method, n_init=10, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif algorithm == "CAH (Agglomeratif)":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = model.fit_predict(X_scaled)
    else:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)

    df_result = df_raw.copy()
    df_result["Cluster"] = labels.astype(str)
    unique_labels = sorted(set(labels))
    n_noise = (labels == -1).sum()
    n_real_clusters = len([l for l in unique_labels if l != -1])

    # Metrics
    valid_mask = labels != -1
    if valid_mask.sum() > 1 and len(set(labels[valid_mask])) > 1:
        sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
        ch  = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
        db  = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
    else:
        sil = ch = db = 0.0

    # Dimensionality reduction
    if dim_method == "PCA" or dim_method == "UMAP (via PCA)":
        reducer = PCA(n_components=n_comp, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
        explained = getattr(reducer, "explained_variance_ratio_", None)
    else:
        reducer = TSNE(n_components=n_comp, perplexity=perplexity, random_state=42, n_iter=300)
        X_reduced = reducer.fit_transform(X_scaled)
        explained = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:2rem 0 1.5rem;'>
    <div class='hero-title'>Customer<br>Segmentation</div>
    <div class='hero-sub' style='margin-top:0.75rem;'>{algorithm} · {dim_method} · {len(df_raw)} clients</div>
</div>
""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    ("Clusters", n_real_clusters),
    ("Clients", f"{len(df_raw):,}"),
    ("Silhouette", f"{sil:.3f}"),
    ("Calinski-H", f"{ch:.0f}"),
    ("Davies-B", f"{db:.3f}"),
]
for col, (label, val) in zip([c1,c2,c3,c4,c5], cards):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "◈  Visualisation", "◎  Profils clusters", "◉  Méthode coude", "◐  Dendrogram / PCA", "◑  Données"
])

# ══════════════════ TAB 1 : Visualisation ════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown(f"<div class='section-tag'>Réduction · {dim_method}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Visualisation {n_components_viz} des clusters</div>", unsafe_allow_html=True)

        color_map = {str(l): CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, l in enumerate(unique_labels)}

        if n_comp == 2:
            df_plot = pd.DataFrame({
                "Dim 1": X_reduced[:, 0],
                "Dim 2": X_reduced[:, 1],
                "Cluster": labels.astype(str),
            })
            fig = px.scatter(df_plot, x="Dim 1", y="Dim 2", color="Cluster",
                             color_discrete_map=color_map,
                             opacity=0.7, height=480)
        else:
            df_plot = pd.DataFrame({
                "Dim 1": X_reduced[:, 0],
                "Dim 2": X_reduced[:, 1],
                "Dim 3": X_reduced[:, 2],
                "Cluster": labels.astype(str),
            })
            fig = px.scatter_3d(df_plot, x="Dim 1", y="Dim 2", z="Dim 3",
                                color="Cluster", color_discrete_map=color_map,
                                opacity=0.6, height=480)

        fig.update_traces(marker=dict(size=4 if n_comp==2 else 3))
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        if explained is not None and len(explained) >= 2:
            st.markdown(f"""
            <div class='info-box'>
            ▸ Variance expliquée — PC1: <strong style='color:#c4a882'>{explained[0]:.1%}</strong> 
            · PC2: <strong style='color:#8b5cf6'>{explained[1]:.1%}</strong>
            {f"· PC3: <strong style='color:#06b6d4'>{explained[2]:.1%}</strong>" if len(explained)>2 else ""}
            · Total: <strong style='color:#e8e4dc'>{sum(explained[:n_comp]):.1%}</strong>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-tag'>Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Taille des clusters</div>", unsafe_allow_html=True)

        cluster_counts = pd.Series(labels).value_counts().sort_index()
        df_counts = pd.DataFrame({"Cluster": cluster_counts.index.astype(str),
                                   "Count": cluster_counts.values})
        df_counts["Pct"] = (df_counts["Count"] / len(labels) * 100).round(1)

        fig2 = go.Figure(go.Bar(
            x=df_counts["Cluster"],
            y=df_counts["Count"],
            marker_color=[color_map.get(str(c), "#c4a882") for c in df_counts["Cluster"]],
            text=df_counts["Pct"].astype(str) + "%",
            textposition="outside",
            textfont=dict(color="#8b8fa8", size=10),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=240, showlegend=False,
                           xaxis_title="Cluster", yaxis_title="Clients")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-tag'>Métriques</div>", unsafe_allow_html=True)

        metrics_info = [
            ("Silhouette Score", f"{sil:.4f}", "[-1, 1] → plus c'est proche de 1, mieux c'est"),
            ("Calinski-Harabasz", f"{ch:.1f}", "Plus la valeur est élevée, mieux c'est"),
            ("Davies-Bouldin", f"{db:.4f}", "Plus la valeur est faible, mieux c'est"),
        ]
        for name, val, hint in metrics_info:
            st.markdown(f"""
            <div style='padding:0.6rem 0;border-bottom:1px solid #1e1e2e;'>
                <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#4a4a6a;'>{name}</div>
                <div style='font-size:1.2rem;font-weight:700;color:#c4a882;'>{val}</div>
                <div style='font-size:0.65rem;color:#2e2e4e;font-family:Space Mono,monospace;'>{hint}</div>
            </div>
            """, unsafe_allow_html=True)

        if n_noise > 0:
            st.warning(f"⚠ DBSCAN: {n_noise} points de bruit (cluster -1)")

# ══════════════════ TAB 2 : Profils clusters ═════════════════════════════════
with tab2:
    st.markdown("<div class='section-tag'>Analyse · Profils</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Caractéristiques par cluster</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    num_feature_cols = [c for c in feature_cols if df_raw[c].dtype in [np.float64, np.int64, float, int]]

    if num_feature_cols:
        df_profile = df_result[df_result["Cluster"] != "-1"].copy()
        df_grouped = df_profile.groupby("Cluster")[num_feature_cols].mean().round(2)

        # Radar chart
        fig_radar = go.Figure()
        scaler_viz = StandardScaler()
        df_scaled_viz = pd.DataFrame(
            scaler_viz.fit_transform(df_grouped), columns=df_grouped.columns, index=df_grouped.index
        )

        for i, cluster_id in enumerate(df_scaled_viz.index):
            vals = df_scaled_viz.loc[cluster_id].tolist()
            vals += [vals[0]]
            cats = df_scaled_viz.columns.tolist() + [df_scaled_viz.columns[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill="toself",
                name=f"Cluster {cluster_id}",
                line_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                fillcolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)].replace("#","") and
                          f"rgba({int(CLUSTER_COLORS[i%len(CLUSTER_COLORS)][1:3],16)},"
                          f"{int(CLUSTER_COLORS[i%len(CLUSTER_COLORS)][3:5],16)},"
                          f"{int(CLUSTER_COLORS[i%len(CLUSTER_COLORS)][5:7],16)},0.15)",
            ))

        fig_radar.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
            polar=dict(
                bgcolor="rgba(10,10,15,0.6)",
                radialaxis=dict(visible=True, range=[-2, 2], tickfont=dict(color="#4a4a6a", size=9),
                                gridcolor="#1e1e2e"),
                angularaxis=dict(tickfont=dict(color="#8b8fa8", size=10), gridcolor="#1e1e2e"),
            ),
            height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Heatmap
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-tag'>Heatmap · Moyennes normalisées</div>", unsafe_allow_html=True)

        fig_heat = px.imshow(
            df_scaled_viz.T,
            color_continuous_scale=[[0, "#0a0a0f"], [0.5, "#4a2c6e"], [1, "#c4a882"]],
            aspect="auto", height=300,
        )
        fig_heat.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
                               xaxis=dict(title="Cluster", tickfont=dict(color="#8b8fa8")),
                               yaxis=dict(title="", tickfont=dict(color="#8b8fa8"), gridcolor="transparent"))
        st.plotly_chart(fig_heat, use_container_width=True)

    # Box plots
    st.markdown("<br><div class='section-tag'>Distribution · Box plots</div>", unsafe_allow_html=True)
    if num_feature_cols:
        feat_select = st.selectbox("Variable", num_feature_cols)
        df_box = df_result[df_result["Cluster"] != "-1"]
        fig_box = px.box(df_box, x="Cluster", y=feat_select, color="Cluster",
                         color_discrete_sequence=CLUSTER_COLORS, height=350)
        fig_box.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

# ══════════════════ TAB 3 : Méthode du coude ═════════════════════════════════
with tab3:
    st.markdown("<div class='section-tag'>Optimisation · k</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Méthode du coude & Silhouette</div>", unsafe_allow_html=True)

    k_range = range(2, 11)
    with st.spinner("Calcul pour k=2..10…"):
        inertias, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, km.labels_))

    col_a, col_b = st.columns(2)

    with col_a:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=inertias, mode="lines+markers",
            line=dict(color="#c4a882", width=2),
            marker=dict(color="#c4a882", size=8, symbol="circle"),
            fill="tozeroy", fillcolor="rgba(196,168,130,0.05)",
            name="Inertie",
        ))
        fig_elbow.update_layout(**PLOTLY_LAYOUT, height=320,
                                title=dict(text="Méthode du Coude (Inertie)", font=dict(color="#e8e4dc", size=13)),
                                xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="k"),
                                yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Inertie"))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_b:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=list(k_range), y=sil_scores,
            marker_color=[CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(k_range))],
            text=[f"{s:.3f}" for s in sil_scores], textposition="outside",
            textfont=dict(color="#8b8fa8", size=9),
        ))
        fig_sil.update_layout(**PLOTLY_LAYOUT, height=320,
                               title=dict(text="Score de Silhouette par k", font=dict(color="#e8e4dc", size=13)),
                               xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="k"),
                               yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Silhouette"))
        st.plotly_chart(fig_sil, use_container_width=True)

    best_k = list(k_range)[np.argmax(sil_scores)]
    st.markdown(f"""
    <div class='info-box'>
    ▸ Silhouette max = <strong style='color:#c4a882'>{max(sil_scores):.4f}</strong> pour 
    k = <strong style='color:#c4a882'>{best_k}</strong>
    &nbsp;·&nbsp; Recommandation automatique : <strong style='color:#e8e4dc'>k = {best_k}</strong>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════ TAB 4 : Dendrogram / PCA ═════════════════════════════════
with tab4:
    col_d, col_p = st.columns(2)

    with col_d:
        st.markdown("<div class='section-tag'>CAH · Dendrogramme</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Hiérarchie des clusters</div>", unsafe_allow_html=True)

        sample_idx = np.random.choice(len(X_scaled), min(200, len(X_scaled)), replace=False)
        Z = linkage(X_scaled[sample_idx], method="ward")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dend, ax = plt.subplots(figsize=(8, 4))
        fig_dend.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#0a0a0f")
        dendrogram(Z, ax=ax, truncate_mode="level", p=5,
                   color_threshold=0.7 * max(Z[:, 2]),
                   above_threshold_color="#2e2e4e")
        ax.spines[["top","right","left","bottom"]].set_edgecolor("#1e1e2e")
        ax.tick_params(colors="#4a4a6a", labelsize=7)
        ax.set_xlabel("Echantillons", color="#4a4a6a", fontsize=8)
        ax.set_ylabel("Distance", color="#4a4a6a", fontsize=8)
        ax.yaxis.label.set_color("#4a4a6a")
        plt.tight_layout()
        st.pyplot(fig_dend, use_container_width=True)
        plt.close()

    with col_p:
        st.markdown("<div class='section-tag'>PCA · Variance</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Variance expliquée cumulée</div>", unsafe_allow_html=True)

        max_comp = min(len(feature_cols), len(X_scaled), 10)
        pca_full = PCA(n_components=max_comp, random_state=42)
        pca_full.fit(X_scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)

        fig_pca = go.Figure()
        fig_pca.add_trace(go.Bar(
            x=list(range(1, max_comp+1)),
            y=pca_full.explained_variance_ratio_,
            marker_color="#8b5cf6", name="Par composante", opacity=0.7,
        ))
        fig_pca.add_trace(go.Scatter(
            x=list(range(1, max_comp+1)), y=cum_var,
            mode="lines+markers", name="Cumulée",
            line=dict(color="#c4a882", width=2),
            marker=dict(color="#c4a882", size=7),
            yaxis="y2",
        ))
        fig_pca.add_hline(y=0.90, line_dash="dot", line_color="#f43f5e",
                          annotation_text="90%", annotation_font_color="#f43f5e", yref="y2")
        fig_pca.update_layout(
            **PLOTLY_LAYOUT, height=350,
            yaxis2=dict(overlaying="y", side="right", range=[0, 1.05],
                        tickformat=".0%", tickfont=dict(color="#4a4a6a"),
                        gridcolor="transparent"),
            xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title="Composante"),
            yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Variance expliquée"),
            legend=dict(**PLOTLY_LAYOUT["legend"]),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        # PCA loadings
        if len(feature_cols) <= 10:
            st.markdown("<div class='section-tag' style='margin-top:1rem;'>Loadings · PC1 & PC2</div>", unsafe_allow_html=True)
            loadings = pd.DataFrame(
                pca_full.components_[:2].T,
                columns=["PC1","PC2"],
                index=feature_cols
            )
            fig_load = px.bar(loadings.reset_index(), x="index", y=["PC1","PC2"],
                              barmode="group", color_discrete_sequence=["#c4a882","#8b5cf6"],
                              height=260)
            fig_load.update_layout(**PLOTLY_LAYOUT, showlegend=True,
                                   xaxis=dict(**PLOTLY_LAYOUT["xaxis"], title=""),
                                   yaxis=dict(**PLOTLY_LAYOUT["yaxis"], title="Loading"))
            st.plotly_chart(fig_load, use_container_width=True)

# ══════════════════ TAB 5 : Données ══════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-tag'>Données · Résultats</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Dataset avec labels de clusters</div>", unsafe_allow_html=True)

    c_filter, c_dl = st.columns([3, 1])
    with c_filter:
        cluster_filter = st.multiselect("Filtrer par cluster", sorted(df_result["Cluster"].unique()),
                                         default=sorted(df_result["Cluster"].unique()))
    with c_dl:
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = df_result.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Télécharger CSV", csv_data, "segmentation_resultats.csv",
                           "text/csv", use_container_width=True)

    df_display = df_result[df_result["Cluster"].isin(cluster_filter)]
    st.dataframe(df_display.head(500), use_container_width=True, height=400)

    st.markdown(f"""
    <div class='info-box' style='margin-top:1rem;'>
    ▸ {len(df_display):,} lignes affichées (max 500) · {len(df_result):,} total
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e1e2e;margin-top:3rem;'>
<div style='text-align:center;padding:1.5rem;font-family:Space Mono,monospace;
            font-size:0.65rem;color:#2e2e4e;letter-spacing:0.15em;'>
    CLUSTERING LAB · KMeans · CAH · DBSCAN · PCA · t-SNE
</div>
""", unsafe_allow_html=True)
