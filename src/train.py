"""
train.py
--------
Entraînement, comparaison multimodèle et optimisation par GridSearchCV.
Modèles : SVM, Random Forest, KNN
Métrique champion : F1-Score (macro)
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from preprocessing import load_dataset, preprocess, DATA_PATH

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
MODELS_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH  = os.path.join(MODELS_DIR, "scaler.pkl")
META_PATH    = os.path.join(MODELS_DIR, "meta.pkl")


# ─────────────────────────────────────────────
#  Définition des modèles & hyperparamètres
# ─────────────────────────────────────────────
MODELS = {
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_jobs=-1),
}

PARAM_GRIDS = {
    "SVM": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
}


# ─────────────────────────────────────────────
#  Comparaison initiale (validation croisée)
# ─────────────────────────────────────────────
def compare_models(X_train, y_train, cv: int = 5) -> pd.DataFrame:
    """Évalue chaque modèle par CV 5-fold et retourne un DataFrame de résultats."""
    print("\n══ Comparaison des Modèles (Cross-Validation) ══════════════")
    results = []
    for name, model in MODELS.items():
        start = time.time()
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="f1_macro", n_jobs=-1)
        elapsed = time.time() - start
        results.append({
            "Modèle": name,
            "F1 Moyen": scores.mean().round(4),
            "F1 Std": scores.std().round(4),
            "Temps (s)": round(elapsed, 2),
        })
        print(f"  {name:20s}  F1={scores.mean():.4f} ± {scores.std():.4f}  ({elapsed:.1f}s)")

    df_results = pd.DataFrame(results).sort_values("F1 Moyen", ascending=False)
    print("────────────────────────────────────────────────────────────")
    print(df_results.to_string(index=False))
    return df_results


# ─────────────────────────────────────────────
#  Optimisation GridSearchCV du champion
# ─────────────────────────────────────────────
def optimize_champion(champion_name: str, X_train, y_train) -> object:
    """Lance GridSearchCV sur le modèle champion."""
    print(f"\n══ Optimisation GridSearchCV : {champion_name} ══════════════")
    base_model  = MODELS[champion_name]
    param_grid  = PARAM_GRIDS[champion_name]

    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = param_grid,
        scoring    = "f1_macro",
        cv         = 5,
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Meilleurs paramètres : {grid_search.best_params_}")
    print(f"  Meilleur F1 (CV)     : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# ─────────────────────────────────────────────
#  Évaluation finale sur le jeu de test
# ─────────────────────────────────────────────
def evaluate_final(model, X_test, y_test, champion_name: str) -> dict:
    """Évalue le modèle optimisé sur le jeu de test et retourne les métriques."""
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }

    print(f"\n══ Résultats Finaux — {champion_name} ══════════════════════════")
    for k, v in metrics.items():
        print(f"  {k:12s} : {v}")
    print("\n── Rapport de Classification ────────────────────────────────")
    print(classification_report(y_test, y_pred,
                                 target_names=["Bénin", "Malware"]))

    # Matrice de confusion
    _plot_confusion_matrix(y_test, y_pred, champion_name)
    return metrics


def _plot_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bénin", "Malware"],
                yticklabels=["Bénin", "Malware"], ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(f"Matrice de Confusion — {model_name}")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Matrice sauvegardée → {path}")


# ─────────────────────────────────────────────
#  Sauvegarde
# ─────────────────────────────────────────────
def save_artifacts(model, scaler, feature_names: list, champion_name: str,
                   metrics: dict):
    """Sauvegarde le modèle, le scaler et les métadonnées."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({
        "champion": champion_name,
        "feature_names": feature_names,
        "metrics": metrics,
    }, META_PATH)
    print(f"\n  [✓] Modèle   → {MODEL_PATH}")
    print(f"  [✓] Scaler   → {SCALER_PATH}")
    print(f"  [✓] Metadata → {META_PATH}")


# ─────────────────────────────────────────────
#  Pipeline principal
# ─────────────────────────────────────────────
def main():
    # 1. Chargement et prétraitement
    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    # 2. Comparaison des modèles
    df_results = compare_models(X_train, y_train)

    # 3. Identification du champion
    champion_name = df_results.iloc[0]["Modèle"]
    print(f"\n  🏆 Modèle champion : {champion_name}")

    # 4. Optimisation
    best_model = optimize_champion(champion_name, X_train, y_train)

    # 5. Évaluation finale
    metrics = evaluate_final(best_model, X_test, y_test, champion_name)

    # 6. Visualisation de la comparaison
    _plot_comparison(df_results)

    # 7. Sauvegarde
    save_artifacts(best_model, scaler, feature_names, champion_name, metrics)
    print("\n✅ Entraînement terminé avec succès !")


def _plot_comparison(df_results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2196F3" if i == 0 else "#B0BEC5"
              for i in range(len(df_results))]
    bars = ax.barh(df_results["Modèle"], df_results["F1 Moyen"],
                   color=colors, xerr=df_results["F1 Std"],
                   error_kw={"elinewidth": 1.5, "capsize": 4})
    ax.set_xlabel("F1-Score Macro (CV 5-fold)")
    ax.set_title("Comparaison des Modèles")
    ax.set_xlim(0, 1)
    for bar, val in zip(bars, df_results["F1 Moyen"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(os.path.dirname(__file__), "..", "results", "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Comparaison sauvegardée → {path}")


if __name__ == "__main__":
    main()
