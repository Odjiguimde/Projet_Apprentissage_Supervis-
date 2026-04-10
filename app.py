import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
            'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=random_seed, n_jobs=-1),
            'params': {'n_estimators': [100, 200], 'max_depth': [None, 10]}
        },
        'KNN': {
            'model': KNeighborsClassifier(n_jobs=-1),
            'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
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
# DISPLAY RESULTS WITH PLOTLY
# ──────────────────────────────────────────────
results = st.session_state.results
champion = st.session_state.champion
grid_result = st.session_state.grid_result
y_test = y_test.values if isinstance(y_test, pd.Series) else y_test

# ==================== MODEL COMPARISON ====================
st.header("📊 Model Performance")

# Metrics DataFrame
df_metrics = pd.DataFrame({
    name: {
        'F1-Score': r['f1'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'CV F1': r['cv_f1_mean']
    }
    for name, r in results.items()
}).T.round(4)

st.dataframe(df_metrics.style.highlight_max(subset=['F1-Score'], color='lightgreen', axis=0))

# Champion banner
st.success(f"🏆 **Champion Model: {champion}** (F1-Score = {results[champion]['f1']:.4f})")

# Bar chart with Plotly
fig = go.Figure()
metrics = ['F1-Score', 'Accuracy', 'Precision', 'Recall']

for name, r in results.items():
    values = [r['f1'], r['accuracy'], r['precision'], r['recall']]
    fig.add_trace(go.Bar(
        name=name,
        x=metrics,
        y=values,
        text=[f'{v:.3f}' for v in values],
        textposition='outside'
    ))

fig.update_layout(
    title="Model Performance Comparison",
    xaxis_title="Metric",
    yaxis_title="Score",
    yaxis_range=[0, 1],
    barmode='group',
    height=500,
    template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True)

# ==================== CONFUSION MATRICES ====================
st.header("📈 Confusion Matrices")
cols = st.columns(len(results))
for idx, (name, r) in enumerate(results.items()):
    with cols[idx]:
        cm = confusion_matrix(y_test, r['y_pred'])
        fig = px.imshow(
            cm, 
            text_auto=True, 
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Malware', 'Legitimate'],
            y=['Malware', 'Legitimate'],
            title=f"{name}<br>F1: {r['f1']:.4f}",
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=450, width=400)
        st.plotly_chart(fig, use_container_width=True)

# ==================== ROC & PR CURVES ====================
st.header("📉 ROC & Precision-Recall Curves")

# ROC Curves
fig = go.Figure()
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    roc_auc = auc(fpr, tpr)
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f"{name} (AUC={roc_auc:.3f})",
        line=dict(width=2)
    ))

fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode='lines',
    name='Random', line=dict(dash='dash', color='gray')
))

fig.update_layout(
    title="ROC Curves",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    height=500,
    template='plotly_white',
    legend=dict(x=0.7, y=0.05)
)
st.plotly_chart(fig, use_container_width=True)

# Precision-Recall Curves
fig = go.Figure()
for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_test, r['y_proba'])
    pr_auc = auc(rec, prec)
    fig.add_trace(go.Scatter(
        x=rec, y=prec, mode='lines',
        name=f"{name} (AUC={pr_auc:.3f})",
        line=dict(width=2)
    ))

fig.add_hline(y=y_test.mean(), line_dash="dash", line_color="red", annotation_text="Baseline")

fig.update_layout(
    title="Precision-Recall Curves",
    xaxis_title="Recall",
    yaxis_title="Precision",
    height=500,
    template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True)

# ==================== CROSS-VALIDATION ====================
st.header("🔄 Cross-Validation Results")
fig = go.Figure()
names = list(results.keys())
means = [results[n]['cv_f1_mean'] for n in names]
stds = [results[n]['cv_f1_std'] for n in names]

fig.add_trace(go.Bar(
    x=names,
    y=means,
    error_y=dict(type='data', array=stds, visible=True),
    text=[f'{m:.3f}±{s:.3f}' for m, s in zip(means, stds)],
    textposition='outside',
    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
))

fig.update_layout(
    title=f"{cv_folds}-Fold Cross-Validation F1 Scores",
    xaxis_title="Model",
    yaxis_title="F1-Score",
    yaxis_range=[0, 1],
    height=500,
    template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True)

# ==================== GRIDSEARCH OPTIMIZATION ====================
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
    
    # Confusion matrices comparison
    col_cm1, col_cm2 = st.columns(2)
    
    with col_cm1:
        cm_before = confusion_matrix(y_test, results[champion]['y_pred'])
        fig = px.imshow(
            cm_before, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=['Malware', 'Legitimate'],
            y=['Malware', 'Legitimate'],
            title=f"Before Optimization<br>F1: {results[champion]['f1']:.4f}",
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_cm2:
        cm_after = confusion_matrix(y_test, grid_result['y_pred'])
        fig = px.imshow(
            cm_after, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=['Malware', 'Legitimate'],
            y=['Malware', 'Legitimate'],
            title=f"After Optimization<br>F1: {grid_result['f1']:.4f}",
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC Comparison
    fig = go.Figure()
    
    fpr_b, tpr_b, _ = roc_curve(y_test, results[champion]['y_proba'])
    fpr_a, tpr_a, _ = roc_curve(y_test, grid_result['y_proba'])
    auc_b = auc(fpr_b, tpr_b)
    auc_a = auc(fpr_a, tpr_a)
    
    fig.add_trace(go.Scatter(
        x=fpr_b, y=tpr_b, mode='lines',
        name=f"Before (AUC={auc_b:.3f})",
        line=dict(dash='dash', color='orange', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=fpr_a, y=tpr_a, mode='lines',
        name=f"After (AUC={auc_a:.3f})",
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random', line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title="ROC Curve: Before vs After GridSearchCV",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Download buttons
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

# ==================== CLASSIFICATION REPORTS ====================
st.header("📋 Detailed Classification Reports")

for name, r in results.items():
    with st.expander(f"📄 {name} - F1: {r['f1']:.4f}"):
        st.text(classification_report(y_test, r['y_pred'], target_names=['Malware (0)', 'Legitimate (1)']))

# ==================== FEATURE IMPORTANCE ====================
if 'Random Forest' in results:
    st.header("🌲 Feature Importance (Random Forest)")
    
    rf_model = results['Random Forest']['model']
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    
    top_n = min(20, len(importances))
    top_features = importances.tail(top_n)
    
    fig = go.Figure(go.Bar(
        x=top_features.values,
        y=top_features.index,
        orientation='h',
        marker_color='steelblue',
        text=[f'{v:.4f}' for v in top_features.values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Most Important Features",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== LIVE PREDICTION ====================
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
    
    cols = st.columns(3)
    input_values = {}
    
    for i, feat in enumerate(feature_cols[:9]):
        col = cols[i % 3]
        default_val = float(df[feat].median())
        input_values[feat] = col.number_input(feat, value=default_val, format="%.6f", key=f"input_{feat}")
    
    predict_btn = st.form_submit_button("🔍 Predict Malware", type="primary", use_container_width=True)

if predict_btn:
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
            """)
        else:
            st.success(f"""
            ### ✅ **LEGITIMATE FILE**
            
            **Confidence: {probabilities[1]*100:.2f}%**
            """)
    
    with col_res2:
        fig = go.Figure(go.Bar(
            x=['Malware', 'Legitimate'],
            y=[probabilities[0]*100, probabilities[1]*100],
            text=[f'{probabilities[0]*100:.1f}%', f'{probabilities[1]*100:.1f}%'],
            textposition='outside',
            marker_color=['#ff4444', '#44ff44']
        ))
        fig.update_layout(
            title="Prediction Probabilities",
            xaxis_title="Class",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

gc.collect()
