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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
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
if "scaler" not in st.session_state:
    st.session_state.scaler = None
if "X_train_sc" not in st.session_state:
    st.session_state.X_train_sc = None
if "X_test_sc" not in st.session_state:
    st.session_state.X_test_sc = None

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("🛡️ MalwareShield")
    up = st.file_uploader("📂 CSV File", type=["csv"])
    
    st.divider()
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_seed = st.number_input("Random Seed", 0, 999, 42)
    cv_folds = st.slider("CV Folds", 3, 10, 5)
    
    st.divider()
    use_gridsearch = st.checkbox("🔧 GridSearchCV", value=True)
    
    st.divider()
    train_btn = st.button("🚀 START TRAINING", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
st.title("🛡️ MalwareShield - PE File Classifier")
st.caption("Machine Learning | SVM · Random Forest · KNN | GridSearchCV Optimization")

if up is None:
    st.info("📂 Upload a CSV file in the sidebar to begin")
    with st.expander("ℹ️ Dataset Requirements"):
        st.markdown("""
        - Column **'legitimate'** as target (1 = legitimate, 0 = malware)
        - Features: PE file static analysis attributes
        - No missing values (automatically dropped)
        """)
    st.stop()

# ──────────────────────────────────────────────
# LOAD & PREPARE DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_and_clean(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    return df

df = load_and_clean(up.read())

if "legitimate" not in df.columns:
    st.error("❌ Column 'legitimate' not found in the dataset")
    st.stop()

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
st.session_state.X_train_sc = X_train_sc
st.session_state.X_test_sc = X_test_sc

# Dataset Stats
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Samples", len(df))
col2.metric("Features", len(feature_cols))
col3.metric("Legitimate", int(y.sum()), f"{y.mean()*100:.1f}%")
col4.metric("Malware", int((1-y).sum()), f"{(1-y.mean())*100:.1f}%")
col5.metric("Train/Test", f"{len(X_train)}/{len(X_test)}")

# ──────────────────────────────────────────────
# TRAINING FUNCTIONS
# ──────────────────────────────────────────────
def train_base_models():
    results = {}
    
    # SVM
    with st.spinner("Training SVM..."):
        svm = SVC(kernel='rbf', probability=True, random_state=random_seed)
        svm.fit(X_train_sc, y_train)
        y_pred = svm.predict(X_test_sc)
        cv_scores = cross_val_score(svm, X_train_sc, y_train, cv=StratifiedKFold(cv_folds), scoring='f1')
        results['SVM'] = {
            'model': svm,
            'y_pred': y_pred,
            'y_proba': svm.predict_proba(X_test_sc)[:, 1],
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'needs_scale': True
        }
    
    # Random Forest
    with st.spinner("Training Random Forest..."):
        rf = RandomForestClassifier(n_estimators=100, random_state=random_seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        cv_scores = cross_val_score(rf, X_train, y_train, cv=StratifiedKFold(cv_folds), scoring='f1')
        results['Random Forest'] = {
            'model': rf,
            'y_pred': y_pred,
            'y_proba': rf.predict_proba(X_test)[:, 1],
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'needs_scale': False
        }
    
    # KNN
    with st.spinner("Training KNN..."):
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(X_train_sc, y_train)
        y_pred = knn.predict(X_test_sc)
        cv_scores = cross_val_score(knn, X_train_sc, y_train, cv=StratifiedKFold(cv_folds), scoring='f1')
        results['KNN'] = {
            'model': knn,
            'y_pred': y_pred,
            'y_proba': knn.predict_proba(X_test_sc)[:, 1],
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'needs_scale': True
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
            'params': {'C': [0.1, 1, 10, 50], 'gamma': ['scale', 'auto', 0.01], 'kernel': ['rbf']}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=random_seed, n_jobs=-1),
            'params': {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}
        },
        'KNN': {
            'model': KNeighborsClassifier(n_jobs=-1),
            'params': {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
        }
    }
    
    cfg = grids[model_name]
    
    with st.spinner(f"🔧 GridSearchCV on {model_name}..."):
        gs = GridSearchCV(cfg['model'], cfg['params'], cv=min(3, cv_folds), scoring='f1', n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_train)
        
        best = gs.best_estimator_
        y_pred = best.predict(X_te)
        y_proba = best.predict_proba(X_te)[:, 1]
        
        return {
            'model': best,
            'params': gs.best_params_,
            'score': gs.best_score_,
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba
        }

# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
if train_btn:
    results = train_base_models()
    st.session_state.results = results
    
    # Find champion
    champion = max(results.keys(), key=lambda x: results[x]['f1'])
    st.session_state.champion = champion
    
    # GridSearch
    if use_gridsearch:
        grid_result = run_gridsearch(champion, results[champion]['needs_scale'])
        st.session_state.grid_result = grid_result
    
    st.session_state.trained = True
    st.success("✅ Training completed!")
    st.rerun()

if not st.session_state.trained:
    st.info("👈 Configure parameters and click START TRAINING")
    st.stop()

# ──────────────────────────────────────────────
# DISPLAY RESULTS
# ──────────────────────────────────────────────
results = st.session_state.results
champion = st.session_state.champion
grid_result = st.session_state.grid_result
y_test = y_test.values if isinstance(y_test, pd.Series) else y_test

# ==================== TAB 1: MODEL COMPARISON ====================
st.header("📊 Model Performance")

# Metrics DataFrame
df_metrics = pd.DataFrame({
    name: {
        'F1-Score': r['f1'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'CV F1 (mean)': r['cv_f1_mean'],
        'CV F1 (std)': r['cv_f1_std']
    }
    for name, r in results.items()
}).T.round(4)

st.dataframe(df_metrics.style.highlight_max(subset=['F1-Score'], color='lightgreen', axis=0))

# Champion banner
st.success(f"🏆 **Champion Model: {champion}** (F1-Score = {results[champion]['f1']:.4f})")

# Bar chart comparison
fig, ax = plt.subplots(figsize=(12, 5))
metrics = ['F1-Score', 'Accuracy', 'Precision', 'Recall']
x = np.arange(len(metrics))
width = 0.8 / len(results)

for i, (name, r) in enumerate(results.items()):
    values = [r['f1'], r['accuracy'], r['precision'], r['recall']]
    offset = (i - len(results)/2) * width + width/2
    bars = ax.bar(x + offset, values, width, label=name, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}', 
               ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score', fontsize=11)
ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='y')
st.pyplot(fig)
plt.close(fig)

# ==================== TAB 2: CONFUSION MATRICES ====================
st.header("📈 Confusion Matrices")
cols = st.columns(len(results))
for idx, (name, r) in enumerate(results.items()):
    with cols[idx]:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm = confusion_matrix(y_test, r['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f"{name}\nF1: {r['f1']:.4f}", fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close(fig)

# ==================== TAB 3: ROC & PR CURVES ====================
st.header("📉 ROC & Precision-Recall Curves")

col_roc, col_pr = st.columns(2)

with col_roc:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1], [0,1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

with col_pr:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_test, r['y_proba'])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=2, label=f"{name} (AUC={pr_auc:.3f})")
    ax.axhline(y=y_test.mean(), color='red', linestyle='--', lw=1, label='Baseline')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# ==================== TAB 4: CROSS-VALIDATION ====================
st.header("🔄 Cross-Validation Results")
fig, ax = plt.subplots(figsize=(10, 5))
cv_data = [(name, r['cv_f1_mean'], r['cv_f1_std']) for name, r in results.items()]
names = [d[0] for d in cv_data]
means = [d[1] for d in cv_data]
stds = [d[2] for d in cv_data]

bars = ax.bar(names, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'], 
              alpha=0.7, edgecolor='black', linewidth=1)
ax.set_ylim(0, 1)
ax.set_ylabel('F1-Score')
ax.set_title(f'{cv_folds}-Fold Cross-Validation F1 Scores', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)

st.pyplot(fig)
plt.close(fig)

# ==================== TAB 5: GRIDSEARCH OPTIMIZATION ====================
if grid_result:
    st.header("🔧 GridSearchCV Optimization")
    
    improvement = (grid_result['f1'] - results[champion]['f1']) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Before F1", f"{results[champion]['f1']:.4f}")
    col2.metric("After F1", f"{grid_result['f1']:.4f}", f"{improvement:+.2f}%")
    col3.metric("Best CV Score", f"{grid_result['score']:.4f}")
    col4.metric("Accuracy", f"{grid_result['accuracy']:.4f}")
    
    st.subheader("🏆 Best Hyperparameters")
    st.json(grid_result['params'])
    
    # Comparison matrices
    st.subheader("Before vs After Optimization")
    col_cm1, col_cm2 = st.columns(2)
    
    with col_cm1:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm_before = confusion_matrix(y_test, results[champion]['y_pred'])
        sns.heatmap(cm_before, annot=True, fmt='d', cmap='Oranges', ax=ax, cbar=False)
        ax.set_title(f"Before Optimization\nF1: {results[champion]['f1']:.4f}", fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close(fig)
    
    with col_cm2:
        fig, ax = plt.subplots(figsize=(4, 3.5))
        cm_after = confusion_matrix(y_test, grid_result['y_pred'])
        sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
        ax.set_title(f"After Optimization\nF1: {grid_result['f1']:.4f}", fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)
        plt.close(fig)
    
    # ROC Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    fpr_b, tpr_b, _ = roc_curve(y_test, results[champion]['y_proba'])
    fpr_a, tpr_a, _ = roc_curve(y_test, grid_result['y_proba'])
    auc_b = auc(fpr_b, tpr_b)
    auc_a = auc(fpr_a, tpr_a)
    ax.plot(fpr_b, tpr_b, '--', lw=2, label=f"Before (AUC={auc_b:.3f})", color='orange')
    ax.plot(fpr_a, tpr_a, '-', lw=2, label=f"After (AUC={auc_a:.3f})", color='green')
    ax.plot([0,1], [0,1], 'k--', lw=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: Before vs After GridSearchCV', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    # Download optimized model
    st.subheader("💾 Download Optimized Model")
    model_bytes = io.BytesIO()
    joblib.dump(grid_result['model'], model_bytes)
    model_bytes.seek(0)
    
    scaler_bytes = io.BytesIO()
    joblib.dump(scaler, scaler_bytes)
    scaler_bytes.seek(0)
    
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button("📥 Download Model (.joblib)", model_bytes.getvalue(), 
                          "malware_model_optimized.joblib", "application/octet-stream", use_container_width=True)
    with col_dl2:
        st.download_button("📥 Download Scaler (.joblib)", scaler_bytes.getvalue(),
                          "scaler.joblib", "application/octet-stream", use_container_width=True)

# ==================== TAB 6: CLASSIFICATION REPORTS ====================
st.header("📋 Detailed Classification Reports")

for name, r in results.items():
    with st.expander(f"📄 {name} - F1: {r['f1']:.4f}"):
        st.text(classification_report(y_test, r['y_pred'], target_names=['Malware (0)', 'Legitimate (1)']))

# ==================== TAB 7: FEATURE IMPORTANCE ====================
if 'Random Forest' in results:
    st.header("🌲 Feature Importance (Random Forest)")
    
    rf_model = results['Random Forest']['model']
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(20, len(importances))
    top_features = importances.head(top_n)
    
    bars = ax.barh(range(top_n), top_features.values, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features.index)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Most Important Features', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, top_features.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=8)
    
    st.pyplot(fig)
    plt.close(fig)

# ==================== TAB 8: LIVE PREDICTION ====================
st.header("🎯 Live Prediction")

# Use optimized model if available
if grid_result:
    active_model = grid_result['model']
    needs_scale = results[champion]['needs_scale']
    st.info(f"✨ Using **optimized {champion}** model (F1: {grid_result['f1']:.4f})")
else:
    active_model = results[champion]['model']
    needs_scale = results[champion]['needs_scale']
    st.info(f"📌 Using **{champion}** model (F1: {results[champion]['f1']:.4f})")

with st.form("prediction_form"):
    st.subheader("Enter PE File Features")
    
    # Create 3 columns for inputs
    cols = st.columns(3)
    input_values = {}
    
    for i, feat in enumerate(feature_cols[:12]):  # Limit to 12 for better UI
        col = cols[i % 3]
        default_val = float(df[feat].median())
        input_values[feat] = col.number_input(feat, value=default_val, format="%.6f", key=f"input_{feat}")
    
    predict_btn = st.form_submit_button("🔍 Predict Malware", type="primary", use_container_width=True)

if predict_btn:
    # Build input vector
    X_input = np.zeros((1, len(feature_cols)))
    for i, feat in enumerate(feature_cols):
        X_input[0, i] = input_values.get(feat, df[feat].median())
    
    if needs_scale:
        X_input = scaler.transform(X_input)
    
    prediction = active_model.predict(X_input)[0]
    probabilities = active_model.predict_proba(X_input)[0]
    
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        if prediction == 0:
            st.error(f"""
            ### 🚨 **MALWARE DETECTED**
            
            **Confidence: {probabilities[0]*100:.2f}%**
            
            This file exhibits malicious characteristics.
            """)
        else:
            st.success(f"""
            ### ✅ **LEGITIMATE FILE**
            
            **Confidence: {probabilities[1]*100:.2f}%**
            
            This file appears to be legitimate.
            """)
    
    with col_res2:
        fig, ax = plt.subplots(figsize=(5, 3))
        bars = ax.bar(['Malware', 'Legitimate'], [probabilities[0]*100, probabilities[1]*100],
                     color=['#ff4444', '#44ff44'], alpha=0.7, edgecolor='black')
        ax.set_ylim(0, 100)
        ax.set_ylabel('Probability (%)')
        ax.set_title('Prediction Probabilities', fontweight='bold')
        ax.axhline(50, color='gray', linestyle='--', alpha=0.7)
        
        for bar, prob in zip(bars, [probabilities[0]*100, probabilities[1]*100]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        plt.close(fig)

# Cleanup
gc.collect()
