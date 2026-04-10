import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io
import gc
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import joblib

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MalwareShield — ML Classifier",
    page_icon="🛡️",
    layout="wide",
)

# ──────────────────────────────────────────────
# INIT SESSION STATE
# ──────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "df" not in st.session_state:
    st.session_state.df = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ MalwareShield")
    up = st.file_uploader("Charger un CSV", type=["csv"])
    
    st.markdown("---")
    test_size = st.slider("Taille du test set", 0.1, 0.4, 0.2, 0.05)
    random_seed = st.number_input("Random seed", 0, 999, 42)
    
    st.markdown("---")
    use_svm = st.checkbox("SVM", True)
    use_rf = st.checkbox("Random Forest", True)
    use_knn = st.checkbox("KNN", True)
    
    st.markdown("---")
    train_button = st.button("🚀 Lancer l'entraînement", use_container_width=True, type="primary")

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("🛡️ MalwareShield — Classification de Fichiers PE")
st.caption("Analyse statique · SVM · Random Forest · KNN")

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
if up is None:
    st.info("📂 Chargez un fichier CSV dans la sidebar pour commencer")
    st.stop()

@st.cache_data
def load_data(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

df = load_data(up.read())
st.session_state.df = df

if "legitimate" not in df.columns:
    st.error("❌ Colonne 'legitimate' introuvable")
    st.stop()

# Nettoyage
df = df.drop_duplicates().dropna().reset_index(drop=True)

feature_cols = [c for c in df.columns if c != "legitimate"]
X = df[feature_cols]
y = df["legitimate"]
st.session_state.feature_cols = feature_cols

# Affichage stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total fichiers", len(df))
col2.metric("Légitimes", int(y.sum()), f"{y.mean()*100:.1f}%")
col3.metric("Malveillants", int((1-y).sum()), f"{(1-y.mean())*100:.1f}%")
col4.metric("Features", len(feature_cols))

# ──────────────────────────────────────────────
# TRAINING FUNCTION
# ──────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, use_svm, use_rf, use_knn, random_seed):
    results = {}
    
    # Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    st.session_state.scaler = scaler
    
    if use_svm:
        with st.spinner("Entraînement SVM..."):
            svm = SVC(kernel='rbf', probability=True, random_state=random_seed)
            svm.fit(X_train_sc, y_train)
            y_pred = svm.predict(X_test_sc)
            y_proba = svm.predict_proba(X_test_sc)[:, 1]
            results['SVM'] = {
                'model': svm,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'needs_scale': True
            }
    
    if use_rf:
        with st.spinner("Entraînement Random Forest..."):
            rf = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:, 1]
            results['Random Forest'] = {
                'model': rf,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'needs_scale': False
            }
    
    if use_knn:
        with st.spinner("Entraînement KNN..."):
            knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
            knn.fit(X_train_sc, y_train)
            y_pred = knn.predict(X_test_sc)
            y_proba = knn.predict_proba(X_test_sc)[:, 1]
            results['KNN'] = {
                'model': knn,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'needs_scale': True
            }
    
    return results, scaler, X_train_sc, X_test_sc

# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
if train_button:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    with st.spinner("Entraînement en cours..."):
        results, scaler, X_train_sc, X_test_sc = train_models(
            X_train.values, X_test.values, y_train.values, y_test.values,
            use_svm, use_rf, use_knn, random_seed
        )
    
    if results:
        st.session_state.results = results
        st.session_state.X_train = X_train.values
        st.session_state.X_test = X_test.values
        st.session_state.y_train = y_train.values
        st.session_state.y_test = y_test.values
        st.session_state.X_train_sc = X_train_sc
        st.session_state.X_test_sc = X_test_sc
        st.session_state.trained = True
        st.success("✅ Entraînement terminé !")
        st.rerun()

# ──────────────────────────────────────────────
# DISPLAY RESULTS
# ──────────────────────────────────────────────
if not st.session_state.trained:
    st.info("👈 Configurez les paramètres et cliquez sur 'Lancer l'entraînement'")
    st.stop()

results = st.session_state.results
y_test = st.session_state.y_test

# DataFrame des métriques
metrics_df = pd.DataFrame({
    name: {
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1-Score': r['f1']
    }
    for name, r in results.items()
}).T.round(4)

champion = metrics_df['F1-Score'].idxmax()

# ──────────────────────────────────────────────
# TABS (simplifiés)
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Résultats", "📈 Courbes", "🎯 Prédiction"])

# TAB 1 - RÉSULTATS
with tab1:
    st.subheader(f"🏆 Modèle champion : {champion}")
    st.dataframe(metrics_df.style.highlight_max(subset=['F1-Score'], color='lightgreen'))
    
    st.subheader("Détails par modèle")
    for name, res in results.items():
        with st.expander(f"{'🏆 ' if name == champion else ''}{name} - F1: {res['f1']:.4f}"):
            st.text(classification_report(y_test, res['y_pred'], target_names=['Malveillant', 'Légitime']))

# TAB 2 - COURBES
with tab2:
    st.subheader("Matrices de confusion")
    cols = st.columns(len(results))
    for col, (name, res) in zip(cols, results.items()):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y_test, res['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(name)
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Réel')
            st.pyplot(fig)
            plt.close(fig)
    
    st.subheader("Courbes ROC")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('Taux Faux Positifs')
    ax.set_ylabel('Taux Vrais Positifs')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# TAB 3 - PRÉDICTION
with tab3:
    st.subheader("Prédiction en temps réel")
    
    champion_model = results[champion]['model']
    needs_scale = results[champion]['needs_scale']
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols
    
    cols = st.columns(3)
    input_values = {}
    for i, feat in enumerate(feature_cols[:9]):  # Limité à 9 features pour l'UI
        col = cols[i % 3]
        input_values[feat] = col.number_input(feat, value=float(df[feat].median()))
    
    if st.button("Prédire", type="primary"):
        X_input = np.array([[input_values[f] for f in feature_cols[:9]]])
        if needs_scale:
            X_input = scaler.transform(X_input)
        
        pred = champion_model.predict(X_input)[0]
        proba = champion_model.predict_proba(X_input)[0]
        
        if pred == 0:
            st.error(f"🚨 **MALVEILLANT** (confiance: {proba[0]*100:.1f}%)")
        else:
            st.success(f"✅ **LÉGITIME** (confiance: {proba[1]*100:.1f}%)")

gc.collect()
