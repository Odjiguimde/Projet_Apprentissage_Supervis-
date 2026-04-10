"""
app/main.py
-----------
Interface Streamlit de démo pour la classification de malwares.
Lance avec : streamlit run app/main.py
"""

import os
import sys
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ajout du dossier src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_extractor import predict_file, extract_features

# ─────────────────────────────────────────────
#  Configuration de la page
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ Malware Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS personnalisé
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .main-title   { font-size:2.2rem; font-weight:700; color:#1565C0; }
  .subtitle     { color:#546E7A; font-size:1rem; margin-bottom:1.5rem; }
  .card         { background:#F8F9FA; border-radius:12px; padding:1.5rem;
                  border-left:5px solid #1565C0; margin:0.5rem 0; }
  .malware-card { border-left-color:#E53935 !important; background:#FFEBEE; }
  .benin-card   { border-left-color:#43A047 !important; background:#E8F5E9; }
  .metric-val   { font-size:2rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Chargement du modèle (mis en cache)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    try:
        model  = joblib.load(os.path.join(base, "best_model.pkl"))
        scaler = joblib.load(os.path.join(base, "scaler.pkl"))
        meta   = joblib.load(os.path.join(base, "meta.pkl"))
        return model, scaler, meta
    except FileNotFoundError:
        return None, None, None


model, scaler, meta = load_model()

# ─────────────────────────────────────────────
#  Barre latérale
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/1565C0/FFFFFF?text=MalwareML",
             width=200)
    st.markdown("---")
    st.markdown("### ℹ️ À propos")
    st.info(
        "Ce système classe les fichiers Windows PE "
        "(.exe, .dll) comme **Bénins** ou **Malwares** "
        "par analyse statique et apprentissage supervisé."
    )
    st.markdown("---")

    if meta:
        st.markdown("### 🏆 Modèle Actif")
        st.success(f"**{meta['champion']}**")
        st.markdown("**Métriques (test set)**")
        for k, v in meta["metrics"].items():
            st.metric(label=k, value=v)
    else:
        st.warning("⚠️ Aucun modèle chargé.\nLancez d'abord `train.py`.")

    st.markdown("---")
    st.caption("Projet ML — 4A Cycle Ingénieur")

# ─────────────────────────────────────────────
#  En-tête principal
# ─────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ Système Expert de Classification de Malwares</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyse Statique & Apprentissage Supervisé</div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Onglets
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyse de Fichier", "📊 Résultats", "📈 À propos du Modèle"])

with tab1:
    st.markdown("### Uploader un fichier exécutable Windows")
    st.caption("Formats acceptés : `.exe`, `.dll`")

    uploaded = st.file_uploader(
        "Choisir un fichier",
        type=["exe", "dll"],
        help="Le fichier sera analysé statiquement (aucun code exécuté).",
    )

    col_btn, col_space = st.columns([1, 4])
    analyze_btn = col_btn.button("🔬 Analyser", use_container_width=True,
                                 disabled=(uploaded is None or model is None))

    if uploaded and analyze_btn:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=os.path.splitext(uploaded.name)[-1]) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        with st.spinner("Extraction des features en cours..."):
            result = predict_file(tmp_path, model, scaler, meta["feature_names"])
        os.unlink(tmp_path)

        if "error" in result:
            st.error(f"❌ Erreur : {result['error']}")
        else:
            st.session_state["result"] = result
            st.session_state["filename"] = uploaded.name
            st.success("✅ Analyse terminée ! Consultez l'onglet **Résultats**.")

    if model is None:
        st.warning("⚠️ Le modèle n'est pas encore entraîné. Exécutez `python src/train.py` d'abord.")

with tab2:
    if "result" not in st.session_state:
        st.info("Uploader et analyser un fichier dans l'onglet **Analyse de Fichier**.")
    else:
        result   = st.session_state["result"]
        filename = st.session_state.get("filename", "inconnu")
        pred     = result["prediction"]
        proba    = result["probability"]
        features = result["features"]

        # ── Verdict ──────────────────────────────────────────────
        card_class = "malware-card" if pred == "Malware" else "benin-card"
        icon       = "🔴" if pred == "Malware" else "🟢"
        st.markdown(f"""
        <div class="card {card_class}">
          <div style="font-size:1.1rem; color:#555;">Fichier analysé : <b>{filename}</b></div>
          <div class="metric-val" style="margin:0.5rem 0;">{icon} {pred}</div>
          {"<div>Confiance : <b>" + str(round(proba * 100, 1)) + "%</b></div>" if proba else ""}
        </div>
        """, unsafe_allow_html=True)

        # ── Jauge de confiance ────────────────────────────────────
        if proba is not None:
            st.markdown("#### Niveau de confiance")
            fig, ax = plt.subplots(figsize=(7, 0.7))
            color = "#E53935" if pred == "Malware" else "#43A047"
            ax.barh([""], [proba], color=color)
            ax.barh([""], [1 - proba], left=[proba], color="#EEEEEE")
            ax.set_xlim(0, 1)
            ax.axis("off")
            ax.text(proba / 2, 0, f"{proba * 100:.1f}%",
                    ha="center", va="center", color="white", fontweight="bold")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        # ── Features extraites ────────────────────────────────────
        st.markdown("#### Features Statiques Extraites")
        df_feat = pd.DataFrame(
            [(k, v) for k, v in features.items() if k != "MD5"],
            columns=["Feature", "Valeur"]
        )
        st.dataframe(df_feat, use_container_width=True, height=350)

        if "MD5" in features:
            st.caption(f"MD5 : `{features['MD5']}`")

with tab3:
    if meta is None:
        st.warning("Aucun modèle chargé.")
    else:
        st.markdown("### Informations sur le Modèle")
        col1, col2 = st.columns(2)
        col1.metric("Modèle Champion", meta["champion"])
        col1.metric("Nombre de Features", len(meta["feature_names"]))

        for k, v in meta["metrics"].items():
            col2.metric(k, v)

        st.markdown("### Features Utilisées")
        st.write(meta["feature_names"])

        # Images de résultats si disponibles
        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        images = {
            "Comparaison des Modèles": "model_comparison.png",
            "Matrice de Confusion":    "confusion_matrix.png",
            "Courbe ROC":              "roc_curve.png",
            "Courbe Précision-Rappel": "precision_recall_curve.png",
            "Importance des Features": "feature_importance.png",
        }
        for title, fname in images.items():
            path = os.path.join(results_dir, fname)
            if os.path.exists(path):
                st.markdown(f"#### {title}")
                st.image(path)
