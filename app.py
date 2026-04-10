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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
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
if "grid_result" not in st.session_state:
    st.session_state.grid_result = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "df" not in st.session_state:
    st.session_state.df = None
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None
if "champion" not in st.session_state:
    st.session_state.champion = None

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ MalwareShield")
    up = st.file_uploader("Charger un CSV", type=["csv"])
    
    st.markdown("---")
    test_size = st.slider("Taille du test set", 0.1, 0.4, 0.2, 0.05)
    random_seed = st.number_input("Random seed", 0, 999, 42)
    cv_folds = st.slider("Folds Cross-Validation", 3, 10, 5)
    
    st.markdown("---")
    use_svm = st.checkbox("SVM", True)
    use_rf = st.checkbox("Random Forest", True)
    use_knn = st.checkbox("KNN", True)
    
    st.markdown("---")
    use_gridsearch = st.checkbox("🔧 GridSearchCV (optimisation)", True)
    
    st.markdown("---")
    train_button = st.button("🚀 Lancer l'entraînement", use_container_width=True, type="primary")

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("🛡️ MalwareShield — Classification de Fichiers PE")
st.caption("Analyse statique · SVM · Random Forest · KNN · GridSearchCV")

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
# TRAINING FUNCTIONS
# ──────────────────────────────────────────────
def train_base_models(X_train, X_test, y_train, y_test, use_svm, use_rf, use_knn, random_seed):
    results = {}
    
    # Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    st.session_state.scaler = scaler
    
    if use_svm:
        svm = SVC(kernel='rbf', probability=True, random_state=random_seed)
        svm.fit(X_train_sc, y_train)
        y_pred = svm.predict(X_test_sc)
        y_proba = svm.predict_proba(X_test_sc)[:, 1]
        cv_score = cross_val_score(svm, X_train_sc, y_train, cv=StratifiedKFold(5), scoring='f1').mean()
        results['SVM'] = {
            'model': svm, 'y_pred': y_pred, 'y_proba': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cv_f1': cv_score,
            'needs_scale': True
        }
    
    if use_rf:
        rf = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_proba = rf.predict_proba(X_test)[:, 1]
        cv_score = cross_val_score(rf, X_train, y_train, cv=StratifiedKFold(5), scoring='f1').mean()
        results['Random Forest'] = {
            'model': rf, 'y_pred': y_pred, 'y_proba': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cv_f1': cv_score,
            'needs_scale': False
        }
    
    if use_knn:
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(X_train_sc, y_train)
        y_pred = knn.predict(X_test_sc)
        y_proba = knn.predict_proba(X_test_sc)[:, 1]
        cv_score = cross_val_score(knn, X_train_sc, y_train, cv=StratifiedKFold(5), scoring='f1').mean()
        results['KNN'] = {
            'model': knn, 'y_pred': y_pred, 'y_proba': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'cv_f1': cv_score,
            'needs_scale': True
        }
    
    return results, scaler, X_train_sc, X_test_sc


def run_gridsearch(model_name, X_train, X_test, y_train, y_test, needs_scale, random_seed, cv_folds):
    """Exécute GridSearchCV sur le modèle champion"""
    
    if needs_scale:
        X_tr = st.session_state.X_train_sc
        X_te = st.session_state.X_test_sc
    else:
        X_tr = X_train
        X_te = X_test
    
    param_grids = {
        'SVM': {
            'model': SVC(probability=True, random_state=random_seed),
            'params': {
                'C': [0.1, 1, 10, 50],
                'gamma': ['scale', 'auto', 0.01],
                'kernel': ['rbf']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=random_seed, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(n_jobs=-1),
            'params': {
                'n_neighbors': [3, 5, 7, 11],
                'weights': ['uniform', 'distance']
            }
        }
    }
    
    if model_name not in param_grids:
        return None
    
    cfg = param_grids[model_name]
    
    with st.spinner(f"🔧 GridSearchCV sur {model_name}..."):
        gs = GridSearchCV(
            cfg['model'], cfg['params'],
            cv=StratifiedKFold(cv_folds), scoring='f1',
            n_jobs=-1, verbose=0
        )
        gs.fit(X_tr, y_train)
        
        best_model = gs.best_estimator_
        y_pred = best_model.predict(X_te)
        y_proba = best_model.predict_proba(X_te)[:, 1]
        
        return {
            'best_model': best_model,
            'best_params': gs.best_params_,
            'best_score': gs.best_score_,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
if train_button:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    with st.spinner("Entraînement des modèles de base..."):
        results, scaler, X_train_sc, X_test_sc = train_base_models(
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
        
        # Identifier le champion
        metrics_df = pd.DataFrame({
            name: {'f1': r['f1']} for name, r in results.items()
        }).T
        champion = metrics_df['f1'].idxmax()
        st.session_state.champion = champion
        
        # GridSearch sur le champion
        if use_gridsearch:
            grid_result = run_gridsearch(
                champion,
                st.session_state.X_train, st.session_state.X_test,
                st.session_state.y_train, st.session_state.y_test,
                results[champion]['needs_scale'],
                random_seed, cv_folds
            )
            st.session_state.grid_result = grid_result
        
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
champion = st.session_state.champion
grid_result = st.session_state.grid_result

# DataFrame des métriques
metrics_df = pd.DataFrame({
    name: {
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1-Score': r['f1'],
        'CV F1': r['cv_f1']
    }
    for name, r in results.items()
}).T.round(4)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Résultats", "📈 Courbes", "🔧 GridSearchCV", "🎯 Prédiction"])

# TAB 1 - RÉSULTATS
with tab1:
    st.subheader(f"🏆 Modèle champion : {champion}")
    st.dataframe(metrics_df.style.highlight_max(subset=['F1-Score'], color='lightgreen'))
    
    # Comparaison graphique
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    
    for i, (name, res) in enumerate(results.items()):
        values = [res['accuracy'], res['precision'], res['recall'], res['f1']]
        offset = (i - len(results)/2) * width + width/2
        bars = ax.bar(x + offset, values, width, label=name, alpha=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_ylabel('Score')
    ax.set_title('Comparaison des modèles')
    st.pyplot(fig)
    plt.close(fig)
    
    # Classification reports
    st.subheader("Rapports détaillés")
    for name, res in results.items():
        with st.expander(f"{'🏆 ' if name == champion else ''}{name} - F1: {res['f1']:.4f}"):
            st.text(classification_report(y_test, res['y_pred'], target_names=['Malveillant', 'Légitime']))

# TAB 2 - COURBES
with tab2:
    st.subheader("Matrices de confusion")
    cols = st.columns(min(len(results), 3))
    for idx, (name, res) in enumerate(results.items()):
        with cols[idx % len(cols)]:
            fig, ax = plt.subplots(figsize=(4, 3))
            cm = confusion_matrix(y_test, res['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{name}\nF1: {res['f1']:.3f}")
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Réeel')
            st.pyplot(fig)
            plt.close(fig)
    
    st.subheader("Courbes ROC")
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1], [0,1], 'k--', lw=1, label='Aléatoire')
    ax.set_xlabel('Taux Faux Positifs')
    ax.set_ylabel('Taux Vrais Positifs')
    ax.set_title('Courbes ROC')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# TAB 3 - GRIDSEARCHCV
with tab3:
    if grid_result is None:
        st.info("🔧 Activez GridSearchCV dans la sidebar pour optimiser le modèle champion")
    else:
        st.subheader(f"✨ Optimisation de {champion} avec GridSearchCV")
        
        # Métriques avant/après
        before_f1 = results[champion]['f1']
        after_f1 = grid_result['f1']
        improvement = (after_f1 - before_f1) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("F1 avant", f"{before_f1:.4f}")
        col2.metric("F1 après", f"{after_f1:.4f}", f"{improvement:+.2f}%")
        col3.metric("Best CV Score", f"{grid_result['best_score']:.4f}")
        col4.metric("Accuracy", f"{grid_result['accuracy']:.4f}")
        
        st.subheader("🏆 Meilleurs hyperparamètres")
        st.json(grid_result['best_params'])
        
        # Comparaison visuelle
        st.subheader("Comparaison avant/après")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Matrice avant
        cm_before = confusion_matrix(y_test, results[champion]['y_pred'])
        sns.heatmap(cm_before, annot=True, fmt='d', cmap='Oranges', ax=axes[0])
        axes[0].set_title(f"Avant optimisation\nF1: {before_f1:.4f}")
        axes[0].set_xlabel('Prédit')
        axes[0].set_ylabel('Réel')
        
        # Matrice après
        cm_after = confusion_matrix(y_test, grid_result['y_pred'])
        sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title(f"Après optimisation\nF1: {after_f1:.4f}")
        axes[1].set_xlabel('Prédit')
        axes[1].set_ylabel('Réel')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # ROC comparée
        st.subheader("Courbes ROC comparées")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        fpr_b, tpr_b, _ = roc_curve(y_test, results[champion]['y_proba'])
        auc_b = auc(fpr_b, tpr_b)
        ax.plot(fpr_b, tpr_b, '--', lw=2, label=f"Avant (AUC={auc_b:.3f})", color='orange')
        
        fpr_a, tpr_a, _ = roc_curve(y_test, grid_result['y_proba'])
        auc_a = auc(fpr_a, tpr_a)
        ax.plot(fpr_a, tpr_a, '-', lw=2, label=f"Après (AUC={auc_a:.3f})", color='green')
        
        ax.plot([0,1], [0,1], 'k--', lw=1, label='Aléatoire')
        ax.set_xlabel('Taux Faux Positifs')
        ax.set_ylabel('Taux Vrais Positifs')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
        
        # Rapport après optimisation
        st.subheader("Rapport du modèle optimisé")
        st.text(classification_report(y_test, grid_result['y_pred'], target_names=['Malveillant', 'Légitime']))
        
        # Téléchargement
        st.subheader("💾 Télécharger le modèle optimisé")
        model_bytes = io.BytesIO()
        joblib.dump(grid_result['best_model'], model_bytes)
        model_bytes.seek(0)
        
        scaler_bytes = io.BytesIO()
        joblib.dump(st.session_state.scaler, scaler_bytes)
        scaler_bytes.seek(0)
        
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("📥 Modèle (.joblib)", model_bytes.getvalue(), 
                              "malware_model_optimized.joblib", "application/octet-stream")
        with col_dl2:
            st.download_button("📥 Scaler (.joblib)", scaler_bytes.getvalue(),
                              "scaler.joblib", "application/octet-stream")

# TAB 4 - PRÉDICTION
with tab4:
    st.subheader("🎯 Prédiction en temps réel")
    
    # Utiliser le modèle optimisé si dispo, sinon le champion
    if grid_result is not None:
        active_model = grid_result['best_model']
        needs_scale = results[champion]['needs_scale']
        st.info(f"📌 Utilisation du modèle **{champion} optimisé** (F1: {grid_result['f1']:.4f})")
    else:
        active_model = results[champion]['model']
        needs_scale = results[champion]['needs_scale']
        st.info(f"📌 Utilisation du modèle **{champion}** (F1: {results[champion]['f1']:.4f})")
    
    scaler = st.session_state.scaler
    feature_cols = st.session_state.feature_cols
    
    # Limiter à 9 features pour l'UI
    display_cols = feature_cols[:min(9, len(feature_cols))]
    cols = st.columns(3)
    input_values = {}
    
    for i, feat in enumerate(display_cols):
        col = cols[i % 3]
        input_values[feat] = col.number_input(
            feat, 
            value=float(df[feat].median()),
            format="%.6f"
        )
    
    if st.button("🔍 Analyser", type="primary", use_container_width=True):
        # Construire le vecteur d'entrée
        X_input = np.zeros((1, len(feature_cols)))
        for i, feat in enumerate(feature_cols):
            if feat in input_values:
                X_input[0, i] = input_values[feat]
            else:
                X_input[0, i] = df[feat].median()
        
        if needs_scale:
            X_input = scaler.transform(X_input)
        
        pred = active_model.predict(X_input)[0]
        proba = active_model.predict_proba(X_input)[0]
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if pred == 0:
                st.error(f"## 🚨 MALVEILLANT\n\n**Confiance : {proba[0]*100:.1f}%**")
            else:
                st.success(f"## ✅ LÉGITIME\n\n**Confiance : {proba[1]*100:.1f}%**")
        
        with col_res2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(['Malveillant', 'Légitime'], [proba[0]*100, proba[1]*100], 
                   color=['#ff4444', '#44ff44'], alpha=0.7)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probabilité (%)')
            ax.axvline(50, color='gray', linestyle='--')
            for i, (bar, val) in enumerate(zip(ax.containers[0], [proba[0]*100, proba[1]*100])):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
                       va='center', fontweight='bold')
            st.pyplot(fig)
            plt.close(fig)

gc.collect()
