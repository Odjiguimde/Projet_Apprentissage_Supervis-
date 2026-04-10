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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
    page_title="MalwareShield",
    page_icon="🛡️",
    layout="centered",  # Plus simple que "wide"
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
if "scaler" not in st.session_state:
    st.session_state.scaler = None

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("🛡️ MalwareShield")
    up = st.file_uploader("CSV", type=["csv"])
    
    st.divider()
    test_size = st.slider("Test size", 0.1, 0.4, 0.2)
    random_seed = st.number_input("Seed", 0, 999, 42)
    
    st.divider()
    use_gridsearch = st.checkbox("GridSearchCV", value=True)
    
    st.divider()
    train_btn = st.button("🚀 TRAIN", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
st.title("🛡️ MalwareShield")
st.caption("PE File Classifier | SVM · RF · KNN")

if up is None:
    st.info("📂 Upload CSV in sidebar")
    st.stop()

# Load data
@st.cache_data
def load_data(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df.drop_duplicates().dropna().reset_index(drop=True)

df = load_data(up.read())

if "legitimate" not in df.columns:
    st.error("Missing 'legitimate' column")
    st.stop()

# Prepare data
feature_cols = [c for c in df.columns if c != "legitimate"]
X = df[feature_cols]
y = df["legitimate"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
st.session_state.scaler = scaler

# Stats
c1, c2, c3 = st.columns(3)
c1.metric("Samples", len(df))
c2.metric("Features", len(feature_cols))
c3.metric("Malware %", f"{(1-y.mean())*100:.1f}%")

# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
def train_all():
    results = {}
    
    # SVM
    svm = SVC(kernel='rbf', probability=True, random_state=random_seed)
    svm.fit(X_train_sc, y_train)
    y_pred = svm.predict(X_test_sc)
    results['SVM'] = {
        'model': svm,
        'y_pred': y_pred,
        'y_proba': svm.predict_proba(X_test_sc)[:, 1],
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'scale': True
    }
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'model': rf,
        'y_pred': y_pred,
        'y_proba': rf.predict_proba(X_test)[:, 1],
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'scale': False
    }
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_sc, y_train)
    y_pred = knn.predict(X_test_sc)
    results['KNN'] = {
        'model': knn,
        'y_pred': y_pred,
        'y_proba': knn.predict_proba(X_test_sc)[:, 1],
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'scale': True
    }
    
    return results

def run_gridsearch(model_name, needs_scale):
    if needs_scale:
        X_tr, X_te = X_train_sc, X_test_sc
    else:
        X_tr, X_te = X_train.values, X_test.values
    
    grids = {
        'SVM': {
            'model': SVC(probability=True, random_state=random_seed),
            'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=random_seed),
            'params': {'n_estimators': [100, 200], 'max_depth': [None, 10]}
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
        }
    }
    
    cfg = grids[model_name]
    gs = GridSearchCV(cfg['model'], cfg['params'], cv=3, scoring='f1', n_jobs=-1)
    gs.fit(X_tr, y_train)
    
    best = gs.best_estimator_
    y_pred = best.predict(X_te)
    
    return {
        'model': best,
        'params': gs.best_params_,
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_proba': best.predict_proba(X_te)[:, 1]
    }

if train_btn:
    with st.spinner("Training..."):
        results = train_all()
        st.session_state.results = results
        
        # Find champion
        champion = max(results.keys(), key=lambda x: results[x]['f1'])
        st.session_state.champion = champion
        
        # GridSearch
        if use_gridsearch:
            grid_result = run_gridsearch(champion, results[champion]['scale'])
            st.session_state.grid_result = grid_result
        
        st.session_state.trained = True
        st.success("✅ Done!")
        st.rerun()

# ──────────────────────────────────────────────
# DISPLAY
# ──────────────────────────────────────────────
if not st.session_state.trained:
    st.info("👈 Configure & click TRAIN")
    st.stop()

results = st.session_state.results
champion = st.session_state.champion
grid_result = st.session_state.grid_result

# Simple table
st.subheader("📊 Results")
df_results = pd.DataFrame({
    name: {
        'F1': r['f1'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall']
    }
    for name, r in results.items()
}).T.round(4)

st.dataframe(df_results.style.highlight_max(color='lightgreen'))

# Champion
st.success(f"🏆 **Champion: {champion}** (F1: {results[champion]['f1']:.4f})")

# GridSearch results
if grid_result:
    st.subheader("🔧 GridSearchCV Results")
    
    improvement = (grid_result['f1'] - results[champion]['f1']) * 100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Before F1", f"{results[champion]['f1']:.4f}")
    c2.metric("After F1", f"{grid_result['f1']:.4f}", f"{improvement:+.2f}%")
    c3.metric("Best Params", str(grid_result['params']))
    
    # Confusion matrices side by side
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(3, 3))
        cm = confusion_matrix(y_test, results[champion]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax)
        ax.set_title(f"Before (F1={results[champion]['f1']:.3f})")
        st.pyplot(fig)
        plt.close(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(3, 3))
        cm = confusion_matrix(y_test, grid_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
        ax.set_title(f"After (F1={grid_result['f1']:.3f})")
        st.pyplot(fig)
        plt.close(fig)
    
    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 4))
    fpr1, tpr1, _ = roc_curve(y_test, results[champion]['y_proba'])
    fpr2, tpr2, _ = roc_curve(y_test, grid_result['y_proba'])
    ax.plot(fpr1, tpr1, '--', label=f"Before (AUC={auc(fpr1, tpr1):.3f})")
    ax.plot(fpr2, tpr2, '-', label=f"After (AUC={auc(fpr2, tpr2):.3f})")
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    
    # Download
    model_bytes = io.BytesIO()
    joblib.dump(grid_result['model'], model_bytes)
    model_bytes.seek(0)
    st.download_button("💾 Download Model", model_bytes.getvalue(), "model.joblib")

# Classification report
with st.expander("📄 Classification Reports"):
    for name, res in results.items():
        st.text(f"=== {name} ===")
        st.text(classification_report(y_test, res['y_pred'], target_names=['Malware', 'Legit']))

# Prediction
st.divider()
st.subheader("🎯 Predict")

# Simple prediction form
with st.form("predict_form"):
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(feature_cols[:6]):  # Limit to 6 for simplicity
        col = cols[i % 3]
        inputs[feat] = col.number_input(feat, value=float(df[feat].median()))
    
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    # Use best model
    if grid_result:
        model = grid_result['model']
        needs_scale = results[champion]['scale']
    else:
        model = results[champion]['model']
        needs_scale = results[champion]['scale']
    
    # Build input
    X_input = np.array([[inputs.get(f, df[f].median()) for f in feature_cols]])
    
    if needs_scale:
        X_input = scaler.transform(X_input)
    
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0]
    
    if pred == 0:
        st.error(f"### 🚨 MALWARE\nConfidence: {proba[0]*100:.1f}%")
    else:
        st.success(f"### ✅ LEGITIMATE\nConfidence: {proba[1]*100:.1f}%")

gc.collect()
