"""
tests/test_pipeline.py
----------------------
Tests unitaires pour les modules du projet.
Lancer avec : python -m pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Crée un DataFrame de test synthétique."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "FileSize":             np.random.randint(1000, 1_000_000, n),
        "NumberOfSections":     np.random.randint(1, 10, n),
        "SizeOfCode":           np.random.randint(0, 500_000, n),
        "SizeOfInitializedData": np.random.randint(0, 300_000, n),
        "AddressOfEntryPoint":  np.random.randint(0, 100_000, n),
        "ImageBase":            np.random.choice([0x400000, 0x10000000], n),
        "SectionsNb":           np.random.randint(1, 8, n),
        "SectionsMeanEntropy":  np.random.uniform(0, 8, n),
        "SectionsMaxEntropy":   np.random.uniform(4, 8, n),
        "ImportsNbDLL":         np.random.randint(0, 20, n),
        "ImportsNb":            np.random.randint(0, 200, n),
        "ExportNb":             np.random.randint(0, 50, n),
        "FileEntropy":          np.random.uniform(0, 8, n),
        "label":                np.random.choice(["benign", "malware"], n),
    })
    return df


# ─────────────────────────────────────────────
#  Tests preprocessing
# ─────────────────────────────────────────────
class TestPreprocessing:

    def test_preprocess_shapes(self, sample_df):
        from preprocessing import preprocess
        X_tr, X_te, y_tr, y_te, scaler, feat_names = preprocess(
            sample_df, target_col="label"
        )
        n = len(sample_df)
        assert len(X_tr) + len(X_te) == n
        assert len(y_tr) == len(X_tr)
        assert len(y_te) == len(X_te)

    def test_no_nan_after_preprocess(self, sample_df):
        from preprocessing import preprocess
        X_tr, X_te, _, _, _, _ = preprocess(sample_df, target_col="label")
        assert not np.isnan(X_tr).any(), "NaN dans X_train"
        assert not np.isnan(X_te).any(),  "NaN dans X_test"

    def test_scaler_is_fitted(self, sample_df):
        from preprocessing import preprocess
        from sklearn.preprocessing import StandardScaler
        _, _, _, _, scaler, _ = preprocess(sample_df, target_col="label")
        assert hasattr(scaler, "mean_"), "Scaler non entraîné"

    def test_feature_names_match_columns(self, sample_df):
        from preprocessing import preprocess
        X_tr, _, _, _, _, feat_names = preprocess(sample_df, target_col="label")
        assert len(feat_names) == X_tr.shape[1]

    def test_duplicate_removal(self):
        from preprocessing import preprocess
        df = pd.DataFrame({
            "a": [1, 1, 2],
            "label": ["benign", "benign", "malware"]
        })
        # On s'attend à ce que le doublon soit supprimé sans lever d'exception
        X_tr, X_te, y_tr, y_te, _, _ = preprocess(df, target_col="label")
        assert len(X_tr) + len(X_te) == 2


# ─────────────────────────────────────────────
#  Tests feature extractor (sans pefile réel)
# ─────────────────────────────────────────────
class TestFeatureExtractor:

    def test_entropy_zero_for_empty(self):
        from feature_extractor import _entropy
        assert _entropy(b"") == 0.0

    def test_entropy_max_for_uniform(self):
        from feature_extractor import _entropy
        data = bytes(range(256)) * 4
        ent = _entropy(data)
        assert 7.9 < ent <= 8.0, f"Entropie inattendue : {ent}"

    def test_entropy_zero_for_constant(self):
        from feature_extractor import _entropy
        data = b"\x00" * 1000
        assert _entropy(data) == 0.0

    def test_align_features_fills_missing(self):
        from feature_extractor import align_features
        raw = {"FileSize": 12345, "NumberOfSections": 5}
        model_features = ["FileSize", "NumberOfSections", "ExportNb"]
        vector = align_features(raw, model_features)
        assert len(vector) == 3
        assert vector[2] == 0  # ExportNb manquant → 0

    def test_align_features_order(self):
        from feature_extractor import align_features
        raw = {"A": 1, "B": 2, "C": 3}
        vector = align_features(raw, ["C", "A", "B"])
        assert vector == [3, 1, 2]

    def test_file_not_found(self):
        import pytest
        from feature_extractor import extract_features
        with pytest.raises(FileNotFoundError):
            extract_features("/chemin/inexistant/fichier.exe")


# ─────────────────────────────────────────────
#  Tests train (smoke tests)
# ─────────────────────────────────────────────
class TestModels:

    def test_svm_trains_and_predicts(self, sample_df):
        from preprocessing import preprocess
        from sklearn.svm import SVC
        X_tr, X_te, y_tr, y_te, _, _ = preprocess(sample_df, target_col="label")
        clf = SVC(kernel="rbf", probability=True, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert len(preds) == len(y_te)
        assert set(preds).issubset({0, 1})

    def test_random_forest_trains_and_predicts(self, sample_df):
        from preprocessing import preprocess
        from sklearn.ensemble import RandomForestClassifier
        X_tr, X_te, y_tr, y_te, _, _ = preprocess(sample_df, target_col="label")
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert len(preds) == len(y_te)

    def test_knn_trains_and_predicts(self, sample_df):
        from preprocessing import preprocess
        from sklearn.neighbors import KNeighborsClassifier
        X_tr, X_te, y_tr, y_te, _, _ = preprocess(sample_df, target_col="label")
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        assert len(preds) == len(y_te)

    def test_f1_score_is_valid(self, sample_df):
        from preprocessing import preprocess
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        X_tr, X_te, y_tr, y_te, _, _ = preprocess(sample_df, target_col="label")
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        f1 = f1_score(y_te, y_pred, average="macro")
        assert 0.0 <= f1 <= 1.0


# ─────────────────────────────────────────────
#  Point d'entrée
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
