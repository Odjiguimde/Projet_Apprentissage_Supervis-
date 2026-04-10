import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import time
import io
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay, precision_recall_curve
)
import joblib

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MalwareShield — ML Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# THEME & CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, .mono { font-family: 'JetBrains Mono', monospace; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #060810;
    border-right: 1px solid #12182e;
}
[data-testid="stSidebar"] * { color: #c8d0f0 !important; }
[data-testid="stSidebar"] label {
    color: #4a5580 !important;
    font-size: .68rem !important;
    letter-spacing: .1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #0c0f1e;
    border: 1px solid #1a2040;
    border-radius: 10px;
    padding: .8rem 1rem;
    transition: border-color .2s;
}
[data-testid="metric-container"]:hover { border-color: #3d5aff; }
[data-testid="metric-container"] label {
    color: #4a5580 !important;
    font-size: .65rem !important;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8edff !important;
    font-size: 1.45rem !important;
    font-weight: 800;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #1a2040;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4a5580;
    border-radius: 0;
    padding: .5rem 1.2rem;
    font-size: .78rem;
    font-weight: 600;
    letter-spacing: .06em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #5b7fff !important;
    border-bottom: 2px solid #5b7fff !important;
}

/* ── Main bg ── */
.main .block-container {
    background: #07091a;
    padding-top: 1rem;
    max-width: 1500px;
}
footer { visibility: hidden; }

/* ── Section labels ── */
.sec-label {
    font-size: .62rem;
    font-weight: 700;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #5b7fff;
    border-left: 3px solid #5b7fff;
    padding-left: .5rem;
    margin: 1.5rem 0 .6rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Cards ── */
.info-card {
    background: #0c0f1e;
    border: 1px solid #1a2040;
    border-radius: 12px;
    padding: .9rem 1.1rem;
    margin: .35rem 0;
    font-size: .84rem;
    color: #9aa5cc;
    line-height: 1.7;
}
.info-card b { color: #5b7fff; }
.info-card.good  { border-left: 3px solid #00d98b; }
.info-card.good  b { color: #00d98b; }
.info-card.warn  { border-left: 3px solid #ffd166; }
.info-card.warn  b { color: #ffd166; }
.info-card.danger{ border-left: 3px solid #ff4d6d; }
.info-card.danger b { color: #ff4d6d; }
.info-card.info  { border-left: 3px solid #5b7fff; }

/* ── Badge ── */
.badge {
    display: inline-block;
    background: #0c0f1e;
    border: 1px solid #1a2040;
    border-radius: 6px;
    padding: .2rem .7rem;
    font-size: .75rem;
    color: #9aa5cc;
    margin: .2rem .2rem .2rem 0;
    font-family: 'JetBrains Mono', monospace;
}
.badge b { color: #5b7fff; }

/* ── Champion banner ── */
.champion-banner {
    background: linear-gradient(135deg, #0d1535 0%, #0c1530 50%, #0d1535 100%);
    border: 1px solid #3d5aff;
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.champion-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #5b7fff, transparent);
}
.champion-banner .trophy { font-size: 2rem; }
.champion-banner .name { font-size: 1.4rem; font-weight: 800; color: #e8edff; }
.champion-banner .sub  { font-size: .78rem; color: #5b7fff; font-family: 'JetBrains Mono', monospace; }

/* ── Model card ── */
.model-card {
    background: #0c0f1e;
    border: 1px solid #1a2040;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: border-color .2s, transform .2s;
}
.model-card:hover { border-color: #3d5aff; transform: translateY(-2px); }
.model-card.champion { border-color: #5b7fff; background: #0d1535; }
.model-card .mc-name { font-size: .85rem; font-weight: 700; color: #e8edff; }
.model-card .mc-val  { font-size: 1.6rem; font-weight: 800; color: #5b7fff; }
.model-card .mc-sub  { font-size: .72rem; color: #4a5580; font-family: 'JetBrains Mono', monospace; }

/* ── Predict result ── */
.pred-malicious {
    background: linear-gradient(135deg, #1a0610, #200a14);
    border: 2px solid #ff4d6d;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.pred-legit {
    background: linear-gradient(135deg, #061a10, #0a2014);
    border: 2px solid #00d98b;
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
}
.pred-title { font-size: 1.6rem; font-weight: 800; margin-bottom: .3rem; }
.pred-conf  { font-size: .9rem; font-family: 'JetBrains Mono', monospace; color: #9aa5cc; }

/* ── Progress bar custom ── */
.prog-bar-container {
    background: #1a2040;
    border-radius: 4px;
    height: 6px;
    margin: .3rem 0;
    overflow: hidden;
}
.prog-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #3d5aff, #5b7fff);
    transition: width .5s ease;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# THEME GRAPHIQUES
# ──────────────────────────────────────────────
PAL  = ["#5b7fff", "#00d98b", "#ffd166", "#ff4d6d", "#c77dff",
        "#64dfdf", "#f4a261", "#a8dadc", "#e76f51", "#06d6a0"]
BG   = "#0c0f1e"
GRID = "#1a2040"
TEXT = "#9aa5cc"
AXBG = "#07091a"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "axes.titlecolor": "#e8edff", "axes.titlesize": 11,
    "axes.titleweight": "600", "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "legend.facecolor": BG, "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT, "grid.color": GRID, "grid.linestyle": "--",
    "grid.linewidth": 0.4, "axes.grid": True, "figure.dpi": 110,
    "font.family": "monospace",
})

def nfig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax

def th(ax, title="", xl="", yl=""):
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID); sp.set_linewidth(0.5)
    if title: ax.set_title(title)
    if xl:    ax.set_xlabel(xl, color=TEXT, fontsize=9)
    if yl:    ax.set_ylabel(yl, color=TEXT, fontsize=9)

def sec(label):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ MalwareShield")
    st.markdown(
        "<small style='color:#4a5580;font-family:JetBrains Mono,monospace;'>"
        "PE Static Analysis · ML Classifier</small>",
        unsafe_allow_html=True)
    st.markdown("---")

    up = st.file_uploader("Charger un CSV", type=["csv"])

    st.markdown("---")
    sec("Paramètres d'entraînement")
    test_size   = st.slider("Taille du test set", 0.1, 0.4, 0.2, 0.05)
    random_seed = st.number_input("Random seed", 0, 999, 42)
    cv_folds    = st.slider("Folds (Cross-Validation)", 3, 10, 5)
    run_grid    = st.checkbox("Optimisation GridSearchCV", True)

    st.markdown("---")
    sec("Modèles à entraîner")
    use_svm = st.checkbox("Support Vector Machine", True)
    use_rf  = st.checkbox("Random Forest", True)
    use_knn = st.checkbox("K-Nearest Neighbors", True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5580;font-family:JetBrains Mono,monospace;'>"
        "v1.0 · Malware PE Classifier</small>",
        unsafe_allow_html=True)

# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes, filename):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df

@st.cache_data
def preprocess(file_bytes, filename):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    return df

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown(
    "<h1 style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;"
    "color:#e8edff;margin-bottom:.2rem;'>🛡️ MalwareShield — Système Expert de Classification</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='color:#4a5580;font-size:.85rem;font-family:JetBrains Mono,monospace;'>"
    "Analyse Statique de Fichiers PE · Apprentissage Supervisé · SVM · Random Forest · KNN"
    "</p>",
    unsafe_allow_html=True)

# ──────────────────────────────────────────────
# NO FILE → INSTRUCTIONS
# ──────────────────────────────────────────────
if up is None:
    st.markdown("---")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        <div class='info-card info'>
        <b>📂 Comment démarrer</b><br><br>
        1. Chargez votre fichier <b>CSV</b> dans la sidebar gauche<br>
        2. Le fichier doit contenir les colonnes PE suivantes :<br>
        <code>AddressOfEntryPoint · MajorLinkerVersion · MajorImageVersion</code><br>
        <code>MajorOperatingSystemVersion · DllCharacteristics</code><br>
        <code>SizeOfStackReserve · NumberOfSections · ResourceSize</code><br>
        3. La colonne cible doit s'appeler <b>legitimate</b> (1=légitime, 0=malveillant)<br>
        4. L'entraînement démarre automatiquement après le chargement
        </div>
        """, unsafe_allow_html=True)
    with col_info2:
        st.markdown("""
        <div class='info-card'>
        <b>🤖 Modèles disponibles</b><br><br>
        <b>SVM (RBF)</b> — Très précis en haute dimension<br>
        <b>Random Forest</b> — Robuste, importance des features<br>
        <b>KNN</b> — Proximité géométrique, interprétable<br><br>
        <b>⚙️ Optimisation</b><br>
        GridSearchCV 5-fold sur le modèle champion<br><br>
        <b>📊 Métriques</b><br>
        Accuracy · Precision · Recall · F1 · AUC · ROC · PR Curve
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
file_bytes = up.read()
df_raw = load_data(file_bytes, up.name)
df     = preprocess(file_bytes, up.name)

if "legitimate" not in df.columns:
    st.error("❌ Colonne 'legitimate' introuvable. Vérifiez votre fichier.")
    st.stop()

feature_cols = [c for c in df.columns if c != "legitimate"]
X = df[feature_cols]
y = df["legitimate"]

n_total    = len(df)
n_legit    = int(y.sum())
n_malware  = n_total - n_legit
conv_rate  = y.mean() * 100

# ── KPI badges ──
st.markdown(
    f"<span class='badge'><b>{n_total:,}</b> fichiers PE</span>"
    f"<span class='badge'>Légitimes <b>{n_legit:,}</b> ({n_legit/n_total*100:.1f}%)</span>"
    f"<span class='badge'>Malveillants <b>{n_malware:,}</b> ({n_malware/n_total*100:.1f}%)</span>"
    f"<span class='badge'><b>{len(feature_cols)}</b> features</span>"
    f"<span class='badge'>Doublons supprimés <b>{len(df_raw) - len(df):,}</b></span>",
    unsafe_allow_html=True)

# ── KPI columns ──
kpi_cols = st.columns(5)
kpi_cols[0].metric("Total fichiers",   f"{n_total:,}")
kpi_cols[1].metric("Légitimes",        f"{n_legit:,}",   f"{n_legit/n_total*100:.1f}%")
kpi_cols[2].metric("Malveillants",     f"{n_malware:,}",  f"{n_malware/n_total*100:.1f}%")
kpi_cols[3].metric("Équilibre classes",
                   "Équilibré" if 0.35 < y.mean() < 0.65 else "Déséquilibré",
                   f"{min(y.mean(), 1-y.mean())*100:.1f}% minoritaire")
kpi_cols[4].metric("Features PE",      f"{len(feature_cols)}")

st.markdown("---")

# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ──────────────────────────────────────────────
# TRAIN ALL MODELS
# ──────────────────────────────────────────────
@st.cache_data
def train_all_models(_X_train, _X_test, _X_train_sc, _X_test_sc,
                     _y_train, _y_test, use_svm, use_rf, use_knn,
                     cv_folds, run_grid, random_seed):

    results = {}

    # ── SVM ──
    if use_svm:
        svm = SVC(kernel='rbf', C=1.0, gamma='scale',
                  probability=True, random_state=random_seed)
        svm.fit(_X_train_sc, _y_train)
        yp = svm.predict(_X_test_sc)
        ypr = svm.predict_proba(_X_test_sc)[:, 1]
        cv  = cross_val_score(svm, _X_train_sc, _y_train,
                              cv=StratifiedKFold(cv_folds), scoring='f1').mean()
        results['SVM'] = {
            'model': svm, 'y_pred': yp, 'y_proba': ypr,
            'X_tr': _X_train_sc, 'X_te': _X_test_sc,
            'accuracy' : accuracy_score(_y_test, yp),
            'precision': precision_score(_y_test, yp, zero_division=0),
            'recall'   : recall_score(_y_test, yp, zero_division=0),
            'f1'       : f1_score(_y_test, yp, zero_division=0),
            'cv_f1'    : cv,
            'needs_scale': True,
        }

    # ── Random Forest ──
    if use_rf:
        rf = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)
        rf.fit(_X_train, _y_train)
        yp = rf.predict(_X_test)
        ypr = rf.predict_proba(_X_test)[:, 1]
        cv  = cross_val_score(rf, _X_train, _y_train,
                              cv=StratifiedKFold(cv_folds), scoring='f1').mean()
        results['Random Forest'] = {
            'model': rf, 'y_pred': yp, 'y_proba': ypr,
            'X_tr': _X_train, 'X_te': _X_test,
            'accuracy' : accuracy_score(_y_test, yp),
            'precision': precision_score(_y_test, yp, zero_division=0),
            'recall'   : recall_score(_y_test, yp, zero_division=0),
            'f1'       : f1_score(_y_test, yp, zero_division=0),
            'cv_f1'    : cv,
            'needs_scale': False,
        }

    # ── KNN ──
    if use_knn:
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(_X_train_sc, _y_train)
        yp = knn.predict(_X_test_sc)
        ypr = knn.predict_proba(_X_test_sc)[:, 1]
        cv  = cross_val_score(knn, _X_train_sc, _y_train,
                              cv=StratifiedKFold(cv_folds), scoring='f1').mean()
        results['KNN'] = {
            'model': knn, 'y_pred': yp, 'y_proba': ypr,
            'X_tr': _X_train_sc, 'X_te': _X_test_sc,
            'accuracy' : accuracy_score(_y_test, yp),
            'precision': precision_score(_y_test, yp, zero_division=0),
            'recall'   : recall_score(_y_test, yp, zero_division=0),
            'f1'       : f1_score(_y_test, yp, zero_division=0),
            'cv_f1'    : cv,
            'needs_scale': True,
        }

    return results


@st.cache_data
def run_gridsearch(_X_train, _X_test, _X_train_sc, _X_test_sc,
                   _y_train, _y_test, champion_name, cv_folds, random_seed):

    param_grids = {
        'SVM': {
            'model' : SVC(probability=True, random_state=random_seed),
            'params': {
                'C'     : [0.1, 1, 10, 50],
                'gamma' : ['scale', 'auto', 0.01],
                'kernel': ['rbf', 'linear']
            },
            'X_tr': _X_train_sc, 'X_te': _X_test_sc
        },
        'Random Forest': {
            'model' : RandomForestClassifier(random_state=random_seed, n_jobs=-1),
            'params': {
                'n_estimators'     : [100, 200, 300],
                'max_depth'        : [None, 10, 20],
                'min_samples_split': [2, 5],
                'max_features'     : ['sqrt', 'log2']
            },
            'X_tr': _X_train, 'X_te': _X_test
        },
        'KNN': {
            'model' : KNeighborsClassifier(n_jobs=-1),
            'params': {
                'n_neighbors': [3, 5, 7, 11, 15],
                'weights'    : ['uniform', 'distance'],
                'metric'     : ['euclidean', 'manhattan']
            },
            'X_tr': _X_train_sc, 'X_te': _X_test_sc
        }
    }

    if champion_name not in param_grids:
        return None

    cfg = param_grids[champion_name]
    gs = GridSearchCV(
        cfg['model'], cfg['params'],
        cv=StratifiedKFold(cv_folds), scoring='f1',
        n_jobs=-1, verbose=0
    )
    gs.fit(cfg['X_tr'], _y_train)

    bm = gs.best_estimator_
    yp = bm.predict(cfg['X_te'])
    ypr = bm.predict_proba(cfg['X_te'])[:, 1]

    return {
        'best_model'  : bm,
        'best_params' : gs.best_params_,
        'best_cv_f1'  : gs.best_score_,
        'y_pred'      : yp,
        'y_proba'     : ypr,
        'accuracy'    : accuracy_score(_y_test, yp),
        'precision'   : precision_score(_y_test, yp, zero_division=0),
        'recall'      : recall_score(_y_test, yp, zero_division=0),
        'f1'          : f1_score(_y_test, yp, zero_division=0),
        'X_te'        : cfg['X_te'],
    }


# ── Spinner pendant l'entraînement ──
with st.spinner("⚙️  Entraînement des modèles en cours…"):
    results = train_all_models(
        X_train.values, X_test.values,
        X_train_sc, X_test_sc,
        y_train.values, y_test.values,
        use_svm, use_rf, use_knn,
        cv_folds, run_grid, random_seed
    )

if not results:
    st.warning("Aucun modèle sélectionné. Cochez au moins un modèle dans la sidebar.")
    st.stop()

# ── Identify champion ──
metrics_df = pd.DataFrame({
    n: {'Accuracy': r['accuracy'], 'Precision': r['precision'],
        'Recall': r['recall'], 'F1-Score': r['f1'], 'CV F1': r['cv_f1']}
    for n, r in results.items()
}).T.round(4)

champion = metrics_df['F1-Score'].idxmax()

# ── GridSearch ──
grid_res = None
if run_grid:
    with st.spinner(f"🔧  GridSearchCV sur {champion}…"):
        grid_res = run_gridsearch(
            X_train.values, X_test.values,
            X_train_sc, X_test_sc,
            y_train.values, y_test.values,
            champion, cv_folds, random_seed
        )

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tabs = st.tabs([
    "📊 Aperçu & EDA",
    "🤖 Résultats Modèles",
    "📈 Courbes & Matrices",
    "🔧 Optimisation",
    "🌲 Features",
    "🔭 PCA / Clustering",
    "🎯 Prédiction Live",
    "📋 Rapport Final",
])

# ══════════════════════════════════════════════
# TAB 0 — APERÇU & EDA
# ══════════════════════════════════════════════
with tabs[0]:
    sec("Aperçu du dataset")
    c0a, c0b, c0c = st.columns(3)

    with c0a:
        sec("Données brutes")
        st.dataframe(df.head(15), use_container_width=True, height=280)

    with c0b:
        sec("Statistiques descriptives")
        desc = df[feature_cols].describe().T[['mean','std','min','50%','max']].round(2)
        st.dataframe(desc, use_container_width=True, height=280)

    with c0c:
        sec("Distribution des classes")
        vc = y.value_counts().sort_index()
        fig, ax = nfig(4, 3)
        bars = ax.bar(['Malveillant\n(0)', 'Légitime\n(1)'], vc.values,
                      color=[PAL[3], PAL[0]], edgecolor=BG, alpha=.9, width=.5)
        for b, v in zip(bars, vc.values):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
                    f"{v:,}\n({v/n_total*100:.1f}%)",
                    ha='center', fontsize=9, fontweight='700', color='#e8edff')
        th(ax, "Répartition des classes", yl="Effectif")
        ax.grid(axis='x', visible=False)
        st.pyplot(fig, use_container_width=True); plt.close()

    sec("Distribution des features")
    n_feat = len(feature_cols)
    n_cols_plot = min(4, n_feat)
    n_rows_plot = (n_feat + n_cols_plot - 1) // n_cols_plot
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                              figsize=(5*n_cols_plot, 3.5*n_rows_plot))
    axes = np.array(axes).flatten()
    for i, feat in enumerate(feature_cols):
        ax = axes[i]
        ax.set_facecolor(BG)
        for label, color, name in [(1, PAL[0], 'Légitime'), (0, PAL[3], 'Malveillant')]:
            ax.hist(df[df['legitimate'] == label][feat], bins=50,
                    alpha=.6, color=color, label=name, density=True)
        ax.set_title(feat, fontsize=9, fontweight='700')
        ax.legend(fontsize=7)
        th(ax)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    fig.patch.set_facecolor(BG)
    plt.suptitle("Distributions des features par classe",
                 color='#e8edff', fontsize=12, fontweight='700', y=1.01)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    sec("Matrice de corrélation")
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask,
                cmap=sns.diverging_palette(230, 20, as_cmap=True),
                center=0, vmin=-1, vmax=1, annot=True, fmt='.2f',
                annot_kws={'size': 8, 'color': TEXT},
                linewidths=.4, linecolor=GRID, ax=ax,
                cbar_kws={'shrink': .7})
    ax.set_title("Corrélations entre variables", color='#e8edff', fontsize=11, fontweight='700')
    ax.tick_params(colors=TEXT, labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    sec("Boxplots par classe")
    fig2, axes2 = plt.subplots(n_rows_plot, n_cols_plot,
                                figsize=(5*n_cols_plot, 3.5*n_rows_plot))
    axes2 = np.array(axes2).flatten()
    palette = {0: PAL[3], 1: PAL[0]}
    for i, feat in enumerate(feature_cols):
        ax2 = axes2[i]; ax2.set_facecolor(BG)
        sns.boxplot(data=df, x='legitimate', y=feat, palette=palette,
                    ax=ax2, showfliers=False)
        ax2.set_xticklabels(['Malveillant', 'Légitime'])
        ax2.set_title(feat, fontsize=9, fontweight='700')
        ax2.set_xlabel('')
        th(ax2)
    for j in range(i+1, len(axes2)):
        axes2[j].set_visible(False)
    fig2.patch.set_facecolor(BG)
    plt.suptitle("Boxplots par classe", color='#e8edff', fontsize=12, fontweight='700', y=1.01)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True); plt.close()

# ══════════════════════════════════════════════
# TAB 1 — RÉSULTATS MODÈLES
# ══════════════════════════════════════════════
with tabs[1]:
    sec("Classement des modèles")

    # Model cards
    card_cols = st.columns(len(results))
    for col_c, (name, res) in zip(card_cols, results.items()):
        is_champ = (name == champion)
        cls = "model-card champion" if is_champ else "model-card"
        col_c.markdown(
            f"<div class='{cls}'>"
            f"{'🏆 ' if is_champ else ''}<div class='mc-name'>{name}</div>"
            f"<div class='mc-val'>{res['f1']:.4f}</div>"
            f"<div class='mc-sub'>F1-Score</div>"
            f"<div style='margin-top:.5rem;font-size:.75rem;color:#9aa5cc;'>"
            f"Acc: {res['accuracy']:.3f} · Prec: {res['precision']:.3f} · Recall: {res['recall']:.3f}"
            f"</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    sec("Tableau comparatif complet")
    st.dataframe(
        metrics_df.style.format("{:.4f}")
                  .background_gradient(subset=['F1-Score','Accuracy'], cmap='Blues')
                  .highlight_max(subset=['F1-Score'], color='#1a3080'),
        use_container_width=True)

    sec("Comparaison graphique des métriques")
    metric_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV F1']
    x = np.arange(len(metric_list))
    bw = 0.8 / len(results)
    fig, ax = nfig(13, 5)
    for i, (name, res) in enumerate(results.items()):
        vals = [res['accuracy'], res['precision'], res['recall'], res['f1'], res['cv_f1']]
        bars = ax.bar(x + i*bw - (len(results)-1)*bw/2, vals, bw,
                      label=name, color=PAL[i], alpha=.85, edgecolor=BG)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + .004,
                    f"{h:.3f}", ha='center', va='bottom', fontsize=7.5, fontweight='700')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_list, fontsize=10)
    ax.set_ylim(max(0, metrics_df.min().min() - 0.05), 1.03)
    ax.legend(fontsize=10)
    th(ax, "Comparaison des performances", yl="Score")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    sec("Rapports de classification détaillés")
    for name, res in results.items():
        with st.expander(f"📄 Rapport — {name} (F1={res['f1']:.4f})", expanded=(name == champion)):
            report_str = classification_report(
                y_test, res['y_pred'], target_names=['Malveillant', 'Légitime'])
            st.code(report_str, language='text')

# ══════════════════════════════════════════════
# TAB 2 — COURBES & MATRICES
# ══════════════════════════════════════════════
with tabs[2]:
    sec("Matrices de confusion")
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4.5))
    if n_models == 1:
        axes = [axes]
    cmaps = ['Blues', 'Greens', 'Purples', 'Oranges']
    for ax, (name, res), cmap in zip(axes, results.items(), cmaps):
        cm = confusion_matrix(y_test, res['y_pred'])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Malveillant', 'Légitime'])
        disp.plot(ax=ax, cmap=cmap, colorbar=False)
        champ_str = " 🏆" if name == champion else ""
        ax.set_title(f"{name}{champ_str}\nF1={res['f1']:.4f}", fontweight='700', fontsize=11)
        ax.set_xlabel("Prédit", color=TEXT); ax.set_ylabel("Réel", color=TEXT)
        ax.tick_params(colors=TEXT)
    fig.patch.set_facecolor(BG)
    plt.suptitle("Matrices de Confusion", color='#e8edff', fontsize=13, fontweight='700', y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    col_roc, col_pr = st.columns(2)

    with col_roc:
        sec("Courbes ROC")
        fig, ax = nfig(7, 5)
        for i, (name, res) in enumerate(results.items()):
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            roc_auc = auc(fpr, tpr)
            ls = ['-', '--', '-.'][i % 3]
            ax.plot(fpr, tpr, lw=2.5, ls=ls, color=PAL[i],
                    label=f"{name} (AUC={roc_auc:.4f})")
            ax.fill_between(fpr, tpr, alpha=.07, color=PAL[i])
        ax.plot([0,1],[0,1], '--', color=GRID, lw=1.5, label='Aléatoire')
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xlim([-.01, 1]); ax.set_ylim([0, 1.02])
        th(ax, "Courbes ROC", xl="Taux Faux Positifs", yl="Taux Vrais Positifs")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col_pr:
        sec("Courbes Précision-Rappel")
        fig, ax = nfig(7, 5)
        for i, (name, res) in enumerate(results.items()):
            prec_c, rec_c, _ = precision_recall_curve(y_test, res['y_proba'])
            pr_auc = auc(rec_c, prec_c)
            ls = ['-', '--', '-.'][i % 3]
            ax.plot(rec_c, prec_c, lw=2.5, ls=ls, color=PAL[i],
                    label=f"{name} (AUC={pr_auc:.4f})")
        ax.axhline(y=y.mean(), color=GRID, lw=1.5, ls='--', label='Baseline')
        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        th(ax, "Courbes Précision-Rappel", xl="Rappel", yl="Précision")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    sec("Distributions des probabilités de prédiction")
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 3.5))
    if n_models == 1: axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        ax.set_facecolor(BG)
        for label, color, lbl in [(1, PAL[0], 'Légitime'), (0, PAL[3], 'Malveillant')]:
            mask_l = y_test.values == label
            ax.hist(res['y_proba'][mask_l], bins=40, alpha=.6, color=color,
                    label=lbl, density=True)
        ax.axvline(.5, color='white', ls='--', lw=1.2, alpha=.7)
        ax.legend(fontsize=8)
        th(ax, f"Proba — {name}", xl="P(Légitime)", yl="Densité")
    fig.patch.set_facecolor(BG)
    plt.suptitle("Distributions des probabilités", color='#e8edff',
                 fontsize=12, fontweight='700', y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    sec("Cross-Validation F1 par modèle")
    cv_data = {}
    for name, res in results.items():
        Xc = res['X_tr'] if name != 'Random Forest' else X_train.values
        Xc_all = np.vstack([res['X_tr'], res['X_te']]) if name != 'Random Forest' \
                 else np.vstack([X_train.values, X_test.values])
        y_all  = np.concatenate([y_train.values, y_test.values])
        scores = cross_val_score(res['model'], Xc_all, y_all,
                                 cv=StratifiedKFold(cv_folds), scoring='f1')
        cv_data[name] = scores

    fig, ax = nfig(10, 4)
    for i, (name, scores) in enumerate(cv_data.items()):
        ax.boxplot(scores, positions=[i], widths=.4, patch_artist=True,
                   boxprops=dict(facecolor=PAL[i], alpha=.5),
                   medianprops=dict(color='white', lw=2.5),
                   whiskerprops=dict(color=TEXT),
                   capprops=dict(color=TEXT),
                   flierprops=dict(marker='o', color=PAL[i], alpha=.5, ms=4))
        ax.scatter([i]*len(scores), scores, color=PAL[i], alpha=.7, s=30, zorder=5)
    ax.set_xticks(range(len(cv_data)))
    ax.set_xticklabels(list(cv_data.keys()), fontsize=10)
    th(ax, f"Distribution F1 — {cv_folds}-Fold Cross-Validation", yl="F1-Score")
    ax.grid(axis='x', visible=False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════
# TAB 3 — OPTIMISATION
# ══════════════════════════════════════════════
with tabs[3]:
    sec("Optimisation GridSearchCV")

    if not run_grid:
        st.markdown("""<div class='info-card warn'>
        <b>GridSearchCV désactivé</b><br>
        Activez l'option dans la sidebar pour lancer l'optimisation.
        </div>""", unsafe_allow_html=True)
    elif grid_res is None:
        st.warning("Résultats GridSearchCV non disponibles.")
    else:
        # Champion banner
        st.markdown(
            f"<div class='champion-banner'>"
            f"<div class='trophy'>🏆</div>"
            f"<div class='name'>Modèle Champion : {champion}</div>"
            f"<div class='sub'>Meilleurs hyperparamètres trouvés par GridSearchCV ({cv_folds}-Fold)</div>"
            f"</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Best params display
        params_md = " · ".join([f"<b>{k}</b>={v}" for k, v in grid_res['best_params'].items()])
        st.markdown(
            f"<div class='info-card info'><b>⚙️ Hyperparamètres optimaux</b><br>{params_md}</div>",
            unsafe_allow_html=True)

        # Before / After comparison
        f1_base = results[champion]['f1']
        f1_opt  = grid_res['f1']
        gain    = (f1_opt - f1_base) * 100

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("F1 initial",  f"{f1_base:.4f}")
        g2.metric("F1 optimisé", f"{f1_opt:.4f}", f"{gain:+.2f}%")
        g3.metric("Accuracy",    f"{grid_res['accuracy']:.4f}")
        g4.metric("Precision",   f"{grid_res['precision']:.4f}")
        g5.metric("Recall",      f"{grid_res['recall']:.4f}")

        sec("Comparaison Avant / Après GridSearchCV")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.patch.set_facecolor(BG)

        cm_b = confusion_matrix(y_test, results[champion]['y_pred'])
        ConfusionMatrixDisplay(cm_b, display_labels=['Malveillant', 'Légitime']).plot(
            ax=axes[0], cmap='Oranges', colorbar=False)
        axes[0].set_title(f"Avant Optimisation\nF1 = {f1_base:.4f}", fontweight='700')
        axes[0].tick_params(colors=TEXT)

        cm_o = confusion_matrix(y_test, grid_res['y_pred'])
        ConfusionMatrixDisplay(cm_o, display_labels=['Malveillant', 'Légitime']).plot(
            ax=axes[1], cmap='Greens', colorbar=False)
        axes[1].set_title(f"Après Optimisation\nF1 = {f1_opt:.4f}", fontweight='700')
        axes[1].tick_params(colors=TEXT)

        plt.suptitle(f"{champion} — Avant vs Après GridSearchCV",
                     color='#e8edff', fontsize=13, fontweight='700', y=1.02)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        sec("Courbes ROC comparatives")
        fig, ax = nfig(10, 5)
        for (label, yp, color, ls) in [
            ("Avant (base)", results[champion]['y_proba'], PAL[3], '--'),
            ("Après (optimisé)", grid_res['y_proba'], PAL[0], '-')
        ]:
            fpr, tpr, _ = roc_curve(y_test, yp)
            a = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2.5, ls=ls, color=color, label=f"{label} (AUC={a:.4f})")
            ax.fill_between(fpr, tpr, alpha=.08, color=color)
        ax.plot([0,1],[0,1], '--', color=GRID, lw=1.2, label='Aléatoire')
        ax.legend(fontsize=10)
        th(ax, f"ROC — {champion} Avant/Après Optimisation",
           xl="FPR", yl="TPR")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

        sec("Rapport de classification — Modèle Optimisé")
        st.code(classification_report(y_test, grid_res['y_pred'],
                                      target_names=['Malveillant', 'Légitime']),
                language='text')

        sec("Téléchargement du modèle optimisé")
        model_bytes = io.BytesIO()
        joblib.dump(grid_res['best_model'], model_bytes)
        model_bytes.seek(0)
        scaler_bytes = io.BytesIO()
        joblib.dump(scaler, scaler_bytes)
        scaler_bytes.seek(0)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "💾 Télécharger le modèle (.joblib)",
                data=model_bytes.getvalue(),
                file_name="malware_classifier.joblib",
                mime="application/octet-stream",
                use_container_width=True)
        with dl2:
            st.download_button(
                "💾 Télécharger le scaler (.joblib)",
                data=scaler_bytes.getvalue(),
                file_name="scaler.joblib",
                mime="application/octet-stream",
                use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — FEATURES
# ══════════════════════════════════════════════
with tabs[4]:
    sec("Importance des features")

    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = pd.Series(rf_model.feature_importances_,
                                index=feature_cols).sort_values(ascending=False)

        f_a, f_b = st.columns(2)
        with f_a:
            fig, ax = nfig(7, 5)
            bars = ax.barh(importances.index[::-1], importances.values[::-1],
                           color=[PAL[0] if v == importances.max() else PAL[1]
                                  if v > importances.mean() else '#2a3060'
                                  for v in importances.values[::-1]],
                           edgecolor=BG, alpha=.9)
            for bar, val in zip(bars, importances.values[::-1]):
                ax.text(val + .001, bar.get_y() + bar.get_height()/2,
                        f"{val:.4f}", va='center', fontsize=9, fontweight='700')
            th(ax, "Importance des Features (Random Forest)", xl="Score")
            ax.grid(axis='y', visible=False)
            ax.set_xlim(0, importances.max() * 1.2)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        with f_b:
            sec("Importance cumulée")
            cum = importances.cumsum() / importances.sum() * 100
            fig, ax = nfig(6, 5)
            ax.plot(range(1, len(cum)+1), cum.values, color=PAL[0], lw=2.5, marker='o', ms=6)
            ax.axhline(80, color=PAL[2], ls='--', lw=1.5, label='80% seuil')
            ax.axhline(95, color=PAL[3], ls='--', lw=1.5, label='95% seuil')
            ax.fill_between(range(1, len(cum)+1), cum.values, alpha=.1, color=PAL[0])
            ax.set_xticks(range(1, len(cum)+1))
            ax.set_xticklabels(importances.index, rotation=30, ha='right', fontsize=8)
            ax.legend(fontsize=9)
            ax.set_ylim(0, 105)
            th(ax, "Importance cumulée des features", xl="Features", yl="% cumulé")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

        sec("Matrice de corrélation features × cible")
        corr_with_target = df.corr()['legitimate'].drop('legitimate').sort_values(key=abs, ascending=False)
        fig, ax = nfig(9, 4)
        colors_corr = [PAL[0] if v > 0 else PAL[3] for v in corr_with_target.values]
        bars = ax.barh(corr_with_target.index[::-1], corr_with_target.values[::-1],
                       color=colors_corr[::-1], edgecolor=BG, alpha=.85)
        for bar, val in zip(bars, corr_with_target.values[::-1]):
            ax.text(val + (.005 if val >= 0 else -.005),
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va='center', ha='left' if val >= 0 else 'right',
                    fontsize=9, fontweight='700')
        ax.axvline(0, color=GRID, lw=1)
        legend_elems = [
            mpatches.Patch(facecolor=PAL[0], label='Corrélation positive'),
            mpatches.Patch(facecolor=PAL[3], label='Corrélation négative'),
        ]
        ax.legend(handles=legend_elems, fontsize=9)
        th(ax, "Corrélation des features avec la variable cible", xl="Corrélation de Pearson")
        ax.grid(axis='y', visible=False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    else:
        st.markdown("""<div class='info-card warn'>
        Activez le <b>Random Forest</b> dans la sidebar pour voir l'importance des features.
        </div>""", unsafe_allow_html=True)

    sec("Distributions features par classe (violin)")
    fig, axes = plt.subplots(1, len(feature_cols),
                              figsize=(3.5*len(feature_cols), 5))
    if len(feature_cols) == 1: axes = [axes]
    for ax, feat in zip(axes, feature_cols):
        ax.set_facecolor(BG)
        data0 = df[df['legitimate'] == 0][feat].dropna().values
        data1 = df[df['legitimate'] == 1][feat].dropna().values
        parts0 = ax.violinplot([data0], positions=[0], showmeans=True, showmedians=True)
        parts1 = ax.violinplot([data1], positions=[1], showmeans=True, showmedians=True)
        for parts, color in [(parts0, PAL[3]), (parts1, PAL[0])]:
            for pc in parts.get('bodies', []):
                pc.set_facecolor(color); pc.set_alpha(.5)
            for key in ['cmeans', 'cmedians', 'cbars', 'cmaxes', 'cmins']:
                if key in parts:
                    parts[key].set_color(color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Malv.', 'Légit.'], fontsize=8)
        ax.set_title(feat, fontsize=8, fontweight='700')
        th(ax)
    fig.patch.set_facecolor(BG)
    plt.suptitle("Violins des features par classe", color='#e8edff',
                 fontsize=12, fontweight='700', y=1.02)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════
# TAB 5 — PCA / CLUSTERING
# ══════════════════════════════════════════════
with tabs[5]:
    sec("Analyse en Composantes Principales (PCA)")

    pca_2d = PCA(n_components=2, random_state=random_seed)
    X_all_sc = scaler.transform(X)
    X_pca = pca_2d.fit_transform(X_all_sc)
    var_exp = pca_2d.explained_variance_ratio_

    p1, p2 = st.columns(2)
    with p1:
        fig, ax = nfig(7, 5)
        for label, color, name in [(1, PAL[0], 'Légitime'), (0, PAL[3], 'Malveillant')]:
            mask_l = y.values == label
            ax.scatter(X_pca[mask_l, 0], X_pca[mask_l, 1],
                       c=color, label=name, alpha=.35, s=8, edgecolors='none')
        ax.legend(fontsize=10)
        th(ax,
           f"PCA 2D — {var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}% variance",
           xl=f"PC1 ({var_exp[0]*100:.1f}%)",
           yl=f"PC2 ({var_exp[1]*100:.1f}%)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with p2:
        sec("Variance expliquée cumulative")
        pca_full = PCA(random_state=random_seed)
        pca_full.fit(X_all_sc)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
        fig, ax = nfig(6, 5)
        ax.bar(range(1, len(cum_var)+1), pca_full.explained_variance_ratio_*100,
               color=PAL[0], alpha=.6, edgecolor=BG, label='Variance par composante')
        ax.plot(range(1, len(cum_var)+1), cum_var, color=PAL[2], lw=2.5,
                marker='o', ms=6, label='Variance cumulée')
        ax.axhline(90, color=PAL[3], ls='--', lw=1.5, label='90%')
        ax.legend(fontsize=9)
        th(ax, "Variance expliquée par composante PCA",
           xl="Composante", yl="%")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    sec("Décision boundary PCA (modèle champion)")
    best_model_for_vis = grid_res['best_model'] if (run_grid and grid_res) else results[champion]['model']
    pca_train = pca_2d.transform(X_all_sc[:len(X_train)])
    pca_test  = pca_2d.transform(X_all_sc[len(X_train):])

    # Simple scatter avec regions colorées
    fig, ax = nfig(10, 6)
    h_step = .05
    x_min, x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
    y_min, y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_step),
                          np.arange(y_min, y_max, h_step))

    # Train small model on PCA for visualization
    from sklearn.neighbors import KNeighborsClassifier as KNN_vis
    vis_model = KNN_vis(n_neighbors=5)
    vis_model.fit(X_pca, y.values)
    Z = vis_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=.15, cmap='RdYlGn', levels=1)

    for label, color, name in [(1, PAL[0], 'Légitime'), (0, PAL[3], 'Malveillant')]:
        mask_l = y.values == label
        ax.scatter(X_pca[mask_l, 0], X_pca[mask_l, 1],
                   c=color, label=name, alpha=.3, s=6, edgecolors='none')
    ax.legend(fontsize=10)
    th(ax, "Espace PCA — Séparation des classes (KNN boundary)",
       xl=f"PC1 ({var_exp[0]*100:.1f}%)",
       yl=f"PC2 ({var_exp[1]*100:.1f}%)")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════
# TAB 6 — PRÉDICTION LIVE
# ══════════════════════════════════════════════
with tabs[6]:
    sec("Prédiction en temps réel")
    st.markdown("""<div class='info-card info'>
    Entrez les caractéristiques PE d'un fichier pour obtenir une prédiction instantanée.
    Utilise le <b>modèle champion optimisé</b> (ou le meilleur modèle de base si GridSearch désactivé).
    </div>""", unsafe_allow_html=True)

    active_model = grid_res['best_model'] if (run_grid and grid_res) else results[champion]['model']
    needs_scale  = results[champion]['needs_scale'] if champion in results else True

    _opt_label = "(GridSearch optimisé)" if (run_grid and grid_res) else "(modèle de base)"
    _active_f1 = grid_res["f1"] if (run_grid and grid_res) else results[champion]["f1"]
    st.markdown(
        f"<div class='info-card'>"
        f"<b>Modèle actif :</b> {champion} {_opt_label}"
        f" · F1 = {_active_f1:.4f}"
        f"</div>",
        unsafe_allow_html=True)

    st.markdown("---")
    input_cols = st.columns(4)
    sim_vals = {}
    for i, feat in enumerate(feature_cols):
        col_ui = input_cols[i % 4]
        mn  = float(X[feat].min())
        mx  = float(X[feat].max())
        med = float(X[feat].median())
        sim_vals[feat] = col_ui.number_input(
            feat, min_value=mn, max_value=mx, value=med,
            key=f"pred_{feat}", format="%g")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Analyser ce fichier PE", use_container_width=True):
        X_input = np.array([[sim_vals[f] for f in feature_cols]])
        if needs_scale:
            X_input = scaler.transform(X_input)
        pred = active_model.predict(X_input)[0]
        prob = active_model.predict_proba(X_input)[0]

        st.markdown("<br>", unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            conf_pct = prob[0]*100 if pred == 0 else prob[1]*100
            if pred == 0:
                html_pred = (
                    "<div class='pred-malicious'>"
                    "<div class='pred-title' style='color:#ff4d6d;'>🚨 MALVEILLANT</div>"
                    f"<div class='pred-conf'>Confiance : {conf_pct:.1f}%</div>"
                    "<div style='margin-top:.5rem;font-size:.8rem;color:#9aa5cc;'>"
                    "Ce fichier PE présente des caractéristiques malveillantes"
                    "</div></div>"
                )
            else:
                html_pred = (
                    "<div class='pred-legit'>"
                    "<div class='pred-title' style='color:#00d98b;'>✅ LÉGITIME</div>"
                    f"<div class='pred-conf'>Confiance : {conf_pct:.1f}%</div>"
                    "<div style='margin-top:.5rem;font-size:.8rem;color:#9aa5cc;'>"
                    "Ce fichier PE semble légitime"
                    "</div></div>"
                )
            st.markdown(html_pred, unsafe_allow_html=True)

        with res_col2:
            fig, ax = nfig(5, 3.5)
            bars = ax.barh(['Malveillant', 'Légitime'], [prob[0]*100, prob[1]*100],
                           color=[PAL[3], PAL[0]], edgecolor=BG, alpha=.85)
            for bar, val in zip(bars, [prob[0]*100, prob[1]*100]):
                ax.text(val + .5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va='center', fontsize=11, fontweight='700')
            ax.set_xlim(0, 110)
            ax.axvline(50, color=GRID, ls='--', lw=1.2)
            th(ax, "Probabilités de prédiction", xl="%")
            ax.grid(axis='y', visible=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()

    sec("Test en lot — Prédire sur tout le jeu de test")
    if st.button("📊 Prédire sur l'ensemble de test", use_container_width=True):
        X_te_pred = X_test_sc if needs_scale else X_test.values
        probas_all = active_model.predict_proba(X_te_pred)[:, 1]
        preds_all  = active_model.predict(X_te_pred)

        test_results = X_test.copy()
        test_results['Vrai_label']    = y_test.values
        test_results['Prédit']        = preds_all
        test_results['Prob_Légitime'] = probas_all.round(4)
        test_results['Correct']       = (preds_all == y_test.values)
        test_results['Verdict']       = test_results['Prédit'].map({1: '✅ Légitime', 0: '🚨 Malveillant'})

        st.dataframe(test_results, use_container_width=True, height=350)
        csv_out = test_results.to_csv(index=False).encode('utf-8')
        st.download_button("💾 Télécharger les prédictions (CSV)",
                           csv_out, "predictions_test.csv", "text/csv",
                           use_container_width=True)

# ══════════════════════════════════════════════
# TAB 7 — RAPPORT FINAL
# ══════════════════════════════════════════════
with tabs[7]:
    sec("Synthèse complète du projet")

    # Champion banner
    best_f1 = grid_res['f1'] if (run_grid and grid_res) else results[champion]['f1']
    st.markdown(
        f"<div class='champion-banner'>"
        f"<div class='trophy'>🏆</div>"
        f"<div class='name'>{champion}</div>"
        f"<div class='sub'>Modèle champion · F1-Score = {best_f1:.4f}</div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.columns(2)

    with r1:
        medals = ["🥇", "🥈", "🥉"]
        ranking_html = ""
        for i, (mname, mscore) in enumerate(metrics_df["F1-Score"].sort_values(ascending=False).items()):
            medal = medals[i] if i < len(medals) else "▪"
            ranking_html += f"{medal} <b>{mname}</b> — F1={mscore:.4f}<br>"

        html_r1 = (
            "<div class='info-card good'>"
            "<b>📊 Dataset</b><br>"
            f"• {n_total:,} fichiers PE analysés<br>"
            f"• {n_legit:,} légitimes ({n_legit/n_total*100:.1f}%) · {n_malware:,} malveillants<br>"
            f"• {len(feature_cols)} features d'analyse statique<br>"
            f"• Split train/test : {int((1-test_size)*100)}/{int(test_size*100)}%"
            "</div>"
            "<br>"
            "<div class='info-card'>"
            "<b>🏅 Classement des modèles</b><br>"
            f"{ranking_html}"
            "</div>"
        )
        st.markdown(html_r1, unsafe_allow_html=True)

    with r2:
        opt_str = ""
        if run_grid and grid_res:
            gain = (grid_res["f1"] - results[champion]["f1"]) * 100
            params_str = ", ".join([str(k) + "=" + str(v) for k, v in grid_res["best_params"].items()])
            f1_before = results[champion]["f1"]
            f1_after  = grid_res["f1"]
            cv_f1_val = grid_res["best_cv_f1"]
            opt_str = (
                "<div class='info-card good'>"
                "<b>🔧 Optimisation GridSearchCV</b><br>"
                f"• Meilleurs hyperparamètres : {params_str}<br>"
                f"• F1 avant : {f1_before:.4f} → Après : {f1_after:.4f} ({gain:+.2f}%)<br>"
                f"• CV F1 : {cv_f1_val:.4f}"
                "</div>"
            )

        _acc  = grid_res["accuracy"]  if (run_grid and grid_res) else results[champion]["accuracy"]
        _prec = grid_res["precision"] if (run_grid and grid_res) else results[champion]["precision"]
        _rec  = grid_res["recall"]    if (run_grid and grid_res) else results[champion]["recall"]

        html_r2 = (
            "<div class='info-card info'>"
            f"<b>🎯 Métriques finales — {champion}</b><br>"
            f"• Accuracy  : {_acc:.4f}<br>"
            f"• Precision : {_prec:.4f}<br>"
            f"• Recall    : {_rec:.4f}<br>"
            f"• F1-Score  : {best_f1:.4f}"
            "</div>"
            + ("<br>" + opt_str if opt_str else "")
        )
        st.markdown(html_r2, unsafe_allow_html=True)

    sec("Recommandations")
    best_feat = feature_cols[0]
    if 'Random Forest' in results:
        best_feat = pd.Series(
            results['Random Forest']['model'].feature_importances_,
            index=feature_cols).idxmax()

    recos = [
        ("Déployer le modèle en production",
         f"Le {champion} atteint un F1 de {best_f1:.4f} — niveau adapté à une intégration FastAPI/Streamlit."),
        ("Feature la plus discriminante",
         f"<b>{best_feat}</b> est la feature PE la plus importante pour la classification."),
        ("Enrichir le dataset",
         "Ajouter des features PE supplémentaires (imports, exports, entropie) pour améliorer la détection."),
        ("Monitoring en production",
         "Mettre en place un suivi mensuel du drift de données et ré-entraîner périodiquement."),
        ("Analyse dynamique complémentaire",
         "Combiner l'analyse statique avec une sandbox comportementale pour les cas limites (probabilité ≈ 50%)."),
    ]
    for title, detail in recos:
        st.markdown(
            f"<div class='info-card'><b>{title}</b><br>{detail}</div>",
            unsafe_allow_html=True)

    sec("Export")
    ex1, ex2 = st.columns(2)
    with ex1:
        report_data = {
            'Modèle': list(results.keys()),
            'Accuracy':  [r['accuracy'] for r in results.values()],
            'Precision': [r['precision'] for r in results.values()],
            'Recall':    [r['recall'] for r in results.values()],
            'F1-Score':  [r['f1'] for r in results.values()],
            'CV F1':     [r['cv_f1'] for r in results.values()],
        }
        st.download_button(
            "📥 Télécharger le rapport complet (CSV)",
            pd.DataFrame(report_data).to_csv(index=False).encode('utf-8'),
            "malware_model_report.csv", "text/csv",
            use_container_width=True)
    with ex2:
        st.download_button(
            "📥 Télécharger le dataset filtré (CSV)",
            df.to_csv(index=False).encode('utf-8'),
            "dataset_clean.csv", "text/csv",
            use_container_width=True)

    sec("Informations techniques")
    _models_str = " · ".join(results.keys())
    _gs_str = "Oui" if run_grid else "Non"
    _split_str = f"{int((1-test_size)*100)}/{int(test_size*100)}"
    html_tech = (
        "<div class='info-card' style='font-family:JetBrains Mono,monospace;font-size:.8rem;'>"
        "<b>Environnement</b><br>"
        f"Dataset : {up.name} · {n_total:,} lignes · {len(feature_cols)} features<br>"
        f"Split   : {_split_str} · Seed={random_seed} · CV={cv_folds} folds<br>"
        f"Models  : {_models_str}<br>"
        f"Champion: {champion} · GridSearch={_gs_str}<br>"
        "Librairies : scikit-learn · pandas · numpy · matplotlib · seaborn · streamlit"
        "</div>"
    )
    st.markdown(html_tech, unsafe_allow_html=True)
