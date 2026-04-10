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

""
feature_extractor.py
--------------------
Extraction de features statiques depuis un fichier PE (.exe, .dll)
via la bibliothèque pefile.
"""

import os
import math
import hashlib
from typing import Optional

try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False
    print("[!] pefile non installé — pip install pefile")


# ─────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────
def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = [0] * 256
    for byte in data:
        freq[byte] += 1
    n = len(data)
    entropy = 0.0
    for f in freq:
        if f > 0:
            p = f / n
            entropy -= p * math.log2(p)
    return round(entropy, 6)


def _safe_get(pe_attr, default=0):
    try:
        return pe_attr
    except Exception:
        return default


# ─────────────────────────────────────────────
#  Extraction principale
# ─────────────────────────────────────────────
def extract_features(file_path: str) -> Optional[dict]:
    """
    Extrait les features statiques d'un fichier PE.

    Retourne un dictionnaire feature_name → valeur,
    ou None si le fichier n'est pas un PE valide.
    """
    if not PEFILE_AVAILABLE:
        raise RuntimeError("pefile n'est pas installé.")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    try:
        pe = pefile.PE(file_path)
    except pefile.PEFormatError as e:
        print(f"[!] Fichier PE invalide : {e}")
        return None

    features = {}

    # ── Taille du fichier ───────────────────────────────────────
    features["FileSize"]         = os.path.getsize(file_path)

    # ── En-tête DOS ─────────────────────────────────────────────
    features["e_magic"]          = _safe_get(pe.DOS_HEADER.e_magic)
    features["e_lfanew"]         = _safe_get(pe.DOS_HEADER.e_lfanew)

    # ── FILE_HEADER ─────────────────────────────────────────────
    features["Machine"]          = _safe_get(pe.FILE_HEADER.Machine)
    features["NumberOfSections"] = _safe_get(pe.FILE_HEADER.NumberOfSections)
    features["TimeDateStamp"]    = _safe_get(pe.FILE_HEADER.TimeDateStamp)
    features["Characteristics"]  = _safe_get(pe.FILE_HEADER.Characteristics)

    # ── OPTIONAL_HEADER ─────────────────────────────────────────
    oh = pe.OPTIONAL_HEADER
    features["MajorLinkerVersion"]      = _safe_get(oh.MajorLinkerVersion)
    features["MinorLinkerVersion"]      = _safe_get(oh.MinorLinkerVersion)
    features["SizeOfCode"]              = _safe_get(oh.SizeOfCode)
    features["SizeOfInitializedData"]   = _safe_get(oh.SizeOfInitializedData)
    features["SizeOfUninitializedData"] = _safe_get(oh.SizeOfUninitializedData)
    features["AddressOfEntryPoint"]     = _safe_get(oh.AddressOfEntryPoint)
    features["BaseOfCode"]              = _safe_get(oh.BaseOfCode)
    features["ImageBase"]               = _safe_get(oh.ImageBase)
    features["SectionAlignment"]        = _safe_get(oh.SectionAlignment)
    features["FileAlignment"]           = _safe_get(oh.FileAlignment)
    features["MajorOperatingSystemVersion"] = _safe_get(oh.MajorOperatingSystemVersion)
    features["MajorImageVersion"]       = _safe_get(oh.MajorImageVersion)
    features["MajorSubsystemVersion"]   = _safe_get(oh.MajorSubsystemVersion)
    features["SizeOfImage"]             = _safe_get(oh.SizeOfImage)
    features["SizeOfHeaders"]           = _safe_get(oh.SizeOfHeaders)
    features["CheckSum"]                = _safe_get(oh.CheckSum)
    features["Subsystem"]               = _safe_get(oh.Subsystem)
    features["DllCharacteristics"]      = _safe_get(oh.DllCharacteristics)
    features["SizeOfStackReserve"]      = _safe_get(oh.SizeOfStackReserve)
    features["SizeOfStackCommit"]       = _safe_get(oh.SizeOfStackCommit)
    features["SizeOfHeapReserve"]       = _safe_get(oh.SizeOfHeapReserve)
    features["SizeOfHeapCommit"]        = _safe_get(oh.SizeOfHeapCommit)
    features["LoaderFlags"]             = _safe_get(oh.LoaderFlags)
    features["NumberOfRvaAndSizes"]     = _safe_get(oh.NumberOfRvaAndSizes)

    # ── Sections ─────────────────────────────────────────────────
    section_entropies = []
    section_sizes     = []
    for section in pe.sections:
        try:
            data = section.get_data()
            section_entropies.append(_entropy(data))
            section_sizes.append(len(data))
        except Exception:
            pass

    features["SectionsNb"]            = len(pe.sections)
    features["SectionsMeanEntropy"]   = round(sum(section_entropies) / len(section_entropies), 6) \
                                         if section_entropies else 0.0
    features["SectionsMaxEntropy"]    = round(max(section_entropies), 6) \
                                         if section_entropies else 0.0
    features["SectionsMinEntropy"]    = round(min(section_entropies), 6) \
                                         if section_entropies else 0.0
    features["SectionsMeanRawsize"]   = int(sum(section_sizes) / len(section_sizes)) \
                                         if section_sizes else 0
    features["SectionsMaxRawsize"]    = max(section_sizes) if section_sizes else 0

    # ── Imports ───────────────────────────────────────────────────
    try:
        import_count = sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT)
        dll_count    = len(pe.DIRECTORY_ENTRY_IMPORT)
    except AttributeError:
        import_count = 0
        dll_count    = 0
    features["ImportsNbDLL"]  = dll_count
    features["ImportsNb"]     = import_count

    # ── Exports ───────────────────────────────────────────────────
    try:
        exports_nb = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
    except AttributeError:
        exports_nb = 0
    features["ExportNb"] = exports_nb

    # ── Entropie globale du fichier ───────────────────────────────
    with open(file_path, "rb") as f:
        raw = f.read()
    features["FileEntropy"] = _entropy(raw)

    # ── MD5 (non utilisé pour l'inférence mais utile au logging) ─
    features["MD5"] = hashlib.md5(raw).hexdigest()

    pe.close()
    return features


# ─────────────────────────────────────────────
#  Alignement sur les features du modèle
# ─────────────────────────────────────────────
def align_features(raw_features: dict, model_feature_names: list) -> list:
    """
    Réordonne et complète le vecteur de features pour correspondre
    exactement aux colonnes attendues par le modèle.
    """
    vector = []
    for col in model_feature_names:
        vector.append(raw_features.get(col, 0))
    return vector


# ─────────────────────────────────────────────
#  Inférence directe
# ─────────────────────────────────────────────
def predict_file(file_path: str,
                 model,
                 scaler,
                 feature_names: list) -> dict:
    """
    Pipeline complet : fichier PE → prédiction.

    Retourne :
        {"prediction": "Malware"|"Bénin",
         "probability": float,
         "features": dict}
    """
    import numpy as np

    raw = extract_features(file_path)
    if raw is None:
        return {"error": "Fichier PE invalide ou non parseable."}

    vector   = align_features(raw, feature_names)
    X        = np.array(vector).reshape(1, -1)
    X_scaled = scaler.transform(X)

    label_map = {0: "Bénin", 1: "Malware"}
    pred      = model.predict(X_scaled)[0]
    label     = label_map.get(int(pred), str(pred))

    proba = None
    if hasattr(model, "predict_proba"):
        proba = round(float(model.predict_proba(X_scaled)[0][int(pred)]), 4)

    return {
        "prediction":  label,
        "probability": proba,
        "features":    raw,
    }


# ─────────────────────────────────────────────
#  Test en ligne de commande
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage : python feature_extractor.py <chemin_vers_fichier.exe>")
        sys.exit(1)

    feats = extract_features(sys.argv[1])
    if feats:
        import json
        print(json.dumps({k: v for k, v in feats.items() if k != "MD5"},
                         indent=2))


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
