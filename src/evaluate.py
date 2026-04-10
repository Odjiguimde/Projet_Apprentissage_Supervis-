"""
evaluate.py
-----------
Évaluation approfondie du modèle sauvegardé :
  - Métriques détaillées
  - Courbe ROC
  - Importance des features (si Random Forest)
  - Rapport HTML
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    ConfusionMatrixDisplay
)

from preprocessing import load_dataset, preprocess

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_artifacts():
    """Charge le modèle, le scaler et les métadonnées."""
    model  = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    meta   = joblib.load(os.path.join(MODELS_DIR, "meta.pkl"))
    return model, scaler, meta


def plot_roc_curve(model, X_test, y_test):
    """Trace et sauvegarde la courbe ROC."""
    if not hasattr(model, "predict_proba"):
        print("  [!] Ce modèle ne supporte pas predict_proba — ROC ignorée.")
        return

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_xlabel("Taux de Faux Positifs (FPR)")
    ax.set_ylabel("Taux de Vrais Positifs (TPR)")
    ax.set_title("Courbe ROC")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Courbe ROC sauvegardée → {path} (AUC={roc_auc:.4f})")


def plot_precision_recall(model, X_test, y_test):
    """Trace la courbe Précision-Rappel."""
    if not hasattr(model, "predict_proba"):
        return

    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#E91E63", lw=2,
            label=f"AUC = {pr_auc:.4f}")
    ax.fill_between(recall, precision, alpha=0.1, color="#E91E63")
    ax.set_xlabel("Rappel")
    ax.set_ylabel("Précision")
    ax.set_title("Courbe Précision-Rappel")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "precision_recall_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Courbe P-R sauvegardée → {path}")


def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    """Trace l'importance des features (Random Forest uniquement)."""
    if not hasattr(model, "feature_importances_"):
        print("  [!] Pas d'importance des features pour ce modèle.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    names       = [feature_names[i] for i in indices]
    values      = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Blues_r", len(names))
    ax.barh(names[::-1], values[::-1], color=palette)
    ax.set_xlabel("Importance (Gini)")
    ax.set_title(f"Top {top_n} Features les Plus Importantes")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [✓] Importance des features → {path}")


def generate_html_report(meta: dict, metrics_test: dict):
    """Génère un rapport HTML récapitulatif."""
    champion = meta["champion"]
    cv_metrics = meta["metrics"]

    rows_cv   = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                        for k, v in cv_metrics.items())
    rows_test = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                        for k, v in metrics_test.items())

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <title>Rapport — Classification Malwares</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; color: #333; }}
    h1   {{ color: #1565C0; }}
    h2   {{ color: #0D47A1; border-bottom: 2px solid #90CAF9; padding-bottom: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #1565C0; color: white; }}
    tr:nth-child(even) {{ background: #f5f5f5; }}
    .badge {{ background: #4CAF50; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 6px; margin: 8px 0; }}
  </style>
</head>
<body>
  <h1>🛡️ Rapport de Classification de Malwares</h1>
  <p>Modèle champion : <span class="badge">{champion}</span></p>

  <h2>Métriques sur le Jeu de Test</h2>
  <table>
    <tr><th>Métrique</th><th>Valeur</th></tr>
    {rows_test}
  </table>

  <h2>Métriques de l'Optimisation (GridSearchCV)</h2>
  <table>
    <tr><th>Métrique</th><th>Valeur</th></tr>
    {rows_cv}
  </table>

  <h2>Visualisations</h2>
  <img src="confusion_matrix.png"        alt="Matrice de confusion">
  <img src="roc_curve.png"               alt="Courbe ROC">
  <img src="precision_recall_curve.png"  alt="Courbe Précision-Rappel">
  <img src="feature_importance.png"      alt="Importance des features">

  <p style="color:#aaa; font-size:12px; margin-top:40px;">
    Généré automatiquement — Projet ML 4A Cycle Ingénieur
  </p>
</body>
</html>"""

    path = os.path.join(RESULTS_DIR, "rapport.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [✓] Rapport HTML → {path}")


def main():
    print("══ Évaluation du Modèle Sauvegardé ══════════════════════════")

    # Chargement
    model, scaler, meta = load_artifacts()
    champion = meta["champion"]
    feature_names = meta["feature_names"]
    print(f"  Modèle chargé : {champion}")

    # Re-prétraitement pour récupérer X_test / y_test
    from preprocessing import load_dataset, preprocess, DATA_PATH
    df = load_dataset(DATA_PATH)
    _, X_test, _, y_test, _, _ = preprocess(df)

    # Métriques
    from sklearn.metrics import (accuracy_score, f1_score,
                                  precision_score, recall_score)
    y_pred = model.predict(X_test)
    metrics_test = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
    }

    print("\n── Métriques Finales ─────────────────────────────────────────")
    for k, v in metrics_test.items():
        print(f"  {k:12s} : {v}")

    # Graphiques
    plot_roc_curve(model, X_test, y_test)
    plot_precision_recall(model, X_test, y_test)
    plot_feature_importance(model, feature_names)

    # Rapport
    generate_html_report(meta, metrics_test)
    print("\n✅ Évaluation terminée !")


if __name__ == "__main__":
    main()
