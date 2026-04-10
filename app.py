import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, RocCurveDisplay)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Malware Classifier Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] { background:#0a0c14; border-right:1px solid #1e2130; }
[data-testid="stSidebar"] * { color:#d4d8f0 !important; }
[data-testid="stSidebar"] label { color:#6b7499 !important; font-size:.72rem !important;
    letter-spacing:.08em; text-transform:uppercase; }

[data-testid="metric-container"] { background:#131627; border:1px solid #1e2540;
    border-radius:12px; padding:.9rem 1.1rem; }
[data-testid="metric-container"] label { color:#6b7499 !important; font-size:.72rem !important;
    text-transform:uppercase; letter-spacing:.07em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#e4e8ff !important;
    font-size:1.55rem !important; font-weight:700; }

.stTabs [data-baseweb="tab-list"] { gap:2px; background:transparent;
    border-bottom:1px solid #1e2540; padding-bottom:0; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#6b7499;
    border-radius:8px 8px 0 0; padding:.45rem 1.1rem; font-size:.82rem; font-weight:500; }
.stTabs [aria-selected="true"] { background:#131627 !important; color:#ef476f !important;
    border-bottom:2px solid #ef476f; }

.sec { font-size:.68rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
    color:#ef476f; border-left:3px solid #ef476f; padding-left:.55rem;
    margin:1.4rem 0 .7rem; }

.kpi-badge { display:inline-block; background:#131627; border:1px solid #1e2540;
    border-radius:8px; padding:.3rem .8rem; font-size:.8rem; color:#a0a8cc;
    margin:.2rem .2rem .2rem 0; }
.kpi-badge b { color:#ef476f; }

.insight { background:#0f1221; border:1px solid #1e2540; border-left:3px solid #ef476f;
    border-radius:10px; padding:.9rem 1.1rem; margin:.4rem 0;
    font-size:.86rem; line-height:1.65; color:#b0b8d8; }
.insight b { color:#ef476f; }
.insight.warn  { border-left-color:#f4a261; } .insight.warn  b { color:#f4a261; }
.insight.good  { border-left-color:#52b788; } .insight.good  b { color:#52b788; }

.main .block-container { background:#080a12; padding-top:1.2rem; max-width:1400px; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# THEME GRAPHIQUES
# ----------------------------------------------------------------------
PAL  = ["#ef476f","#52b788","#7b9cff","#f4a261","#64dfdf","#ffd166"]
BG   = "#131627"
GRID = "#1e2540"
TEXT = "#b0b8d8"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT, "axes.titlecolor": "#e4e8ff", "axes.titlesize": 11,
    "axes.titleweight": "600", "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "legend.facecolor": BG, "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT, "grid.color": GRID, "grid.linestyle": "--",
    "grid.linewidth": 0.5, "axes.grid": True, "figure.dpi": 110,
})

def nfig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax

def th(ax, title="", xl="", yl=""):
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
        sp.set_linewidth(0.5)
    if title: ax.set_title(title)
    if xl:    ax.set_xlabel(xl)
    if yl:    ax.set_ylabel(yl)

def sec(label):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# CHARGEMENT DES DONNEES
# ----------------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # La colonne cible est 'legitimate' : 1 = légitime, 0 = malware
    # On la renomme pour plus de clarté
    df.rename(columns={'legitimate': 'target'}, inplace=True)
    return df

# ----------------------------------------------------------------------
# PREPARATION DES DONNEES
# ----------------------------------------------------------------------
def prepare_data(df, feature_cols):
    X = df[feature_cols].copy()
    y = df['target'].copy()
    
    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# ----------------------------------------------------------------------
# ENTRAINEMENT ET EVALUATION
# ----------------------------------------------------------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return model, metrics, y_pred, y_proba

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Malware Classifier")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Charger le dataset CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success(f"✓ {len(df)} échantillons chargés")
        st.markdown(f"<small>Légitimes: {df['target'].sum()} | Malwares: {len(df)-df['target'].sum()}</small>", unsafe_allow_html=True)
        
        st.markdown("---")
        sec("Configuration du modèle")
        
        # Sélection des features (toutes sauf la cible)
        feature_cols = [c for c in df.columns if c != 'target']
        selected_features = st.multiselect(
            "Variables à utiliser", 
            feature_cols, 
            default=feature_cols
        )
        
        # Choix du modèle
        algorithm = st.selectbox(
            "Algorithme", 
            ["Random Forest", "SVM (Support Vector Machine)", "KNN (K-Nearest Neighbors)"]
        )
        
        # Hyperparamètres simplifiés
        if algorithm == "Random Forest":
            n_estimators = st.slider("Nombre d'arbres", 50, 300, 100, 50)
            max_depth = st.slider("Profondeur max", 5, 30, 15, 5)
            model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
            model = RandomForestClassifier(**model_params)
            
        elif algorithm == "SVM (Support Vector Machine)":
            C = st.select_slider("Paramètre C (régularisation)", options=[0.1, 1, 10, 100], value=1)
            gamma = st.select_slider("Gamma", options=['scale', 'auto', 0.1, 1], value='scale')
            model_params = {"C": C, "gamma": gamma, "random_state": 42, "probability": True}
            model = SVC(**model_params)
            
        else:  # KNN
            n_neighbors = st.slider("Nombre de voisins (k)", 3, 15, 5, 2)
            weights = st.selectbox("Pondération", ["uniform", "distance"])
            model_params = {"n_neighbors": n_neighbors, "weights": weights}
            model = KNeighborsClassifier(**model_params)
        
        st.markdown("---")
        sec("Optimisation")
        optimize = st.checkbox("Activer GridSearchCV (recherche des meilleurs hyperparamètres)")
        
        if optimize and algorithm == "Random Forest":
            st.info("GridSearch explorera différentes combinaisons d'arbres et de profondeur")
        
        train_btn = st.button("🚀 Lancer l'entraînement", use_container_width=True, type="primary")
        
    else:
        st.info("📂 Veuillez charger le fichier DatasetmalwareExtrait.csv")
        df = None
        train_btn = False

# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if df is not None and train_btn and selected_features:
    
    # Préparation des données
    X_scaled, y, scaler = prepare_data(df, selected_features)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Optimisation GridSearch (optionnelle)
    if optimize and algorithm == "Random Forest":
        with st.spinner("Recherche des meilleurs hyperparamètres (GridSearchCV)..."):
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid, cv=5, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            st.success(f"✅ Meilleurs paramètres: {grid_search.best_params_}")
            st.markdown(f"<div class='insight good'>Meilleur F1-Score (validation croisée): <b>{grid_search.best_score_:.4f}</b></div>", unsafe_allow_html=True)
            model = best_model
    else:
        with st.spinner(f"Entraînement du modèle {algorithm}..."):
            model.fit(X_train, y_train)
    
    # Évaluation
    model, metrics, y_pred, y_proba = train_and_evaluate(
        model, X_train, X_test, y_train, y_test
    )
    
    # Sauvegarde du modèle et du scaler pour la prédiction future
    joblib.dump(model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(selected_features, "features.pkl")
    
    # ------------------------------------------------------------------
    # AFFICHAGE DES RESULTATS
    # ------------------------------------------------------------------
    st.markdown("# Classification de Malwares - Résultats")
    st.markdown(
        f"<span class='kpi-badge'><b>{algorithm}</b> · "
        f"{len(X_train)} entraînement · {len(X_test)} test</span>"
        f"<span class='kpi-badge'>Ratio Malwares: <b>{(1-y.mean())*100:.1f}%</b></span>",
        unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1-Score", f"{metrics['f1']*100:.2f}%")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Matrice de confusion", "📈 Courbe ROC", "🔍 Rapport de classification", "💻 Prédiction en ligne"
    ])
    
    with tab1:
        sec("Matrice de confusion")
        fig, ax = nfig(5, 4)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax,
                    annot_kws={"size":14, "fontweight":"600", "color":"white"},
                    linewidths=.5, linecolor=BG,
                    xticklabels=["Légitime", "Malware"],
                    yticklabels=["Légitime", "Malware"])
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        th(ax, "Matrice de confusion")
        st.pyplot(fig, use_container_width=True)
        plt.close()
        
        st.markdown(f"""
        <div class='insight'>
        <b>Interprétation :</b><br>
        • <b>Vrais Négatifs (VN)</b> : {cm[0,0]} légitimes correctement identifiés.<br>
        • <b>Vrais Positifs (VP)</b> : {cm[1,1]} malwares correctement détectés.<br>
        • <b>Faux Positifs (FP)</b> : {cm[0,1]} légitimes classés comme malwares (fausses alertes).<br>
        • <b>Faux Négatifs (FN)</b> : {cm[1,0]} malwares non détectés (risque majeur).
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        sec("Courbe ROC (Receiver Operating Characteristic)")
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = nfig(6, 4.5)
            ax.plot(fpr, tpr, color=PAL[0], lw=2.5, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0,1], [0,1], color=GRID, lw=1, ls="--", label="Aléatoire")
            ax.fill_between(fpr, tpr, alpha=.12, color=PAL[0])
            ax.legend(fontsize=10, loc="lower right")
            th(ax, "Courbe ROC", xl="Taux de faux positifs (1 - Spécificité)", yl="Taux de vrais positifs (Sensibilité)")
            ax.set_xlim([0,1])
            ax.set_ylim([0,1.02])
            st.pyplot(fig, use_container_width=True)
            plt.close()
            
            st.markdown(f"""
            <div class='insight good'>
            <b>Aire sous la courbe (AUC) = {roc_auc:.3f}</b><br>
            Un score proche de 1 indique une excellente capacité du modèle à distinguer les malwares des fichiers légitimes.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Ce modèle ne fournit pas de probabilités (utilisez Random Forest ou SVM avec probability=True).")
    
    with tab3:
        sec("Rapport de classification détaillé")
        report = classification_report(y_test, y_pred, target_names=["Légitime", "Malware"], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)
        
        # Validation croisée
        sec("Validation croisée (Cross-Validation)")
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='f1')
        st.markdown(f"""
        <div class='insight'>
        • Scores F1 par fold : {', '.join([f"{s:.3f}" for s in cv_scores])}<br>
        • **F1 moyen (CV-5) : {cv_scores.mean():.4f}** (± {cv_scores.std():.4f})
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Importance (Random Forest uniquement)
        if algorithm == "Random Forest" and hasattr(model, "feature_importances_"):
            sec("Importance des caractéristiques")
            imp_df = pd.DataFrame({
                'feature': selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = nfig(8, 5)
            ax.barh(imp_df['feature'].head(10), imp_df['importance'].head(10), 
                    color=PAL[0], edgecolor=BG, alpha=.85)
            th(ax, "Top 10 des caractéristiques les plus importantes", xl="Importance")
            ax.grid(axis='y', visible=False)
            st.pyplot(fig, use_container_width=True)
            plt.close()
    
    with tab4:
        sec("Prédiction en temps réel")
        st.markdown("Entrez les caractéristiques d'un fichier PE pour prédire s'il est malveillant :")
        
        # Création des champs de saisie pour chaque feature
        cols = st.columns(3)
        input_data = {}
        for i, feat in enumerate(selected_features):
            col = cols[i % 3]
            # Valeur médiane du dataset comme valeur par défaut
            default_val = float(df[feat].median())
            input_data[feat] = col.number_input(feat, value=default_val, format="%.2f")
        
        if st.button("🔍 Analyser le fichier", use_container_width=True, type="primary"):
            # Création du DataFrame de prédiction
            input_df = pd.DataFrame([input_data])
            # Normalisation avec le scaler entraîné
            input_scaled = scaler.transform(input_df[selected_features])
            # Prédiction
            pred = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else [0, 0]
            
            if pred == 1:
                st.error(f"🚨 **MALWARE DÉTECTÉ !** (Probabilité : {proba[1]*100:.2f}%)")
                st.markdown("<div class='insight alert'>Ce fichier présente des caractéristiques malveillantes. Une analyse approfondie est recommandée.</div>", unsafe_allow_html=True)
            else:
                st.success(f"✅ **Fichier LÉGITIME** (Probabilité : {proba[0]*100:.2f}%)")
                st.markdown("<div class='insight good'>Aucune signature malveillante détectée.</div>", unsafe_allow_html=True)
    
    # Export du modèle
    st.markdown("---")
    sec("Export")
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.download_button(
            "📥 Télécharger le modèle (best_model.pkl)",
            data=open("best_model.pkl", "rb").read(),
            file_name="malware_classifier.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )
    with col_exp2:
        st.download_button(
            "📥 Télécharger le scaler (scaler.pkl)",
            data=open("scaler.pkl", "rb").read(),
            file_name="scaler.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

elif df is None:
    st.info("👈 Chargez votre fichier CSV dans la barre latérale pour commencer.")
else:
    st.warning("Veuillez sélectionner au moins une variable et lancer l'entraînement.")
