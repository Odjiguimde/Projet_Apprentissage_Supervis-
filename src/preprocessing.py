"""
preprocessing.py
----------------
Chargement, nettoyage et prétraitement du dataset de malwares.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os


# ─────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.xlsx")
TARGET_COLUMN = "label"          # ← adapter si votre colonne cible a un autre nom
TEST_SIZE     = 0.2
RANDOM_STATE  = 42


# ─────────────────────────────────────────────
#  Chargement
# ─────────────────────────────────────────────
def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Charge le dataset depuis un fichier Excel ou CSV."""
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Format non supporté : {ext}")
    print(f"[✓] Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────
#  Analyse Exploratoire Rapide
# ─────────────────────────────────────────────
def quick_eda(df: pd.DataFrame) -> None:
    """Affiche les informations essentielles sur le dataset."""
    print("\n── Aperçu ──────────────────────────────────")
    print(df.head(3))
    print("\n── Types & Valeurs manquantes ──────────────")
    missing = df.isnull().sum()
    print(pd.DataFrame({
        "dtype": df.dtypes,
        "missing": missing,
        "missing_%": (missing / len(df) * 100).round(2)
    }))
    if TARGET_COLUMN in df.columns:
        print(f"\n── Distribution de '{TARGET_COLUMN}' ─────────")
        print(df[TARGET_COLUMN].value_counts())
    print("─" * 46)


# ─────────────────────────────────────────────
#  Prétraitement
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame, target_col: str = TARGET_COLUMN):
    """
    Pipeline complet de prétraitement :
      1. Suppression des doublons
      2. Encodage de la cible
      3. Gestion des valeurs manquantes
      4. Séparation features / target
      5. Train/test split
      6. Normalisation (StandardScaler)

    Retourne :
        X_train, X_test, y_train, y_test, scaler, feature_names
    """

    # 1. Doublons
    before = len(df)
    df = df.drop_duplicates()
    print(f"[✓] Doublons supprimés : {before - len(df)}")

    # 2. Encodage de la cible
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    print(f"[✓] Classes encodées : {dict(enumerate(le.classes_))}")

    # 3. Séparation X / y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. Garder seulement les colonnes numériques
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    print(f"[✓] Features numériques retenues : {len(feature_names)}")

    # 5. Imputation des NaN par la médiane
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=feature_names)

    # 6. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 7. Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"[✓] Train : {X_train_scaled.shape} | Test : {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, feature_names


# ─────────────────────────────────────────────
#  Point d'entrée standalone
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_dataset()
    quick_eda(df)
    preprocess(df)
