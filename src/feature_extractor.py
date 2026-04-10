"""
feature_extractor.py
--------------------
Extraction de features statiques depuis un fichier PE (.exe, .dll)
via la bibliothèque pefile.

Ces features doivent correspondre aux colonnes du dataset d'entraînement.
"""

import os
import math
import hashlib
from typing import Optional

try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False
    print("[!] pefile non installé — pip install pefile")


# ─────────────────────────────────────────────
#  Utilitaires
# ─────────────────────────────────────────────
def _entropy(data: bytes) -> float:
    """Calcule l'entropie de Shannon d'une séquence d'octets."""
    if not data:
        return 0.0
    freq = [0] * 256
    for byte in data:
        freq[byte] += 1
    n = len(data)
    entropy = 0.0
    for f in freq:
        if f > 0:
            p = f / n
            entropy -= p * math.log2(p)
    return round(entropy, 6)


def _safe_get(pe_attr, default=0):
    """Retourne la valeur d'un attribut PE ou une valeur par défaut."""
    try:
        return pe_attr
    except Exception:
        return default


# ─────────────────────────────────────────────
#  Extraction principale
# ─────────────────────────────────────────────
def extract_features(file_path: str) -> Optional[dict]:
    """
    Extrait les features statiques d'un fichier PE.

    Retourne un dictionnaire feature_name → valeur,
    ou None si le fichier n'est pas un PE valide.
    """
    if not PEFILE_AVAILABLE:
        raise RuntimeError("pefile n'est pas installé.")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    try:
        pe = pefile.PE(file_path)
    except pefile.PEFormatError as e:
        print(f"[!] Fichier PE invalide : {e}")
        return None

    features = {}

    # ── Taille du fichier ───────────────────────────────────────
    features["FileSize"]         = os.path.getsize(file_path)

    # ── En-tête DOS ─────────────────────────────────────────────
    features["e_magic"]          = _safe_get(pe.DOS_HEADER.e_magic)
    features["e_lfanew"]         = _safe_get(pe.DOS_HEADER.e_lfanew)

    # ── FILE_HEADER ─────────────────────────────────────────────
    features["Machine"]          = _safe_get(pe.FILE_HEADER.Machine)
    features["NumberOfSections"] = _safe_get(pe.FILE_HEADER.NumberOfSections)
    features["TimeDateStamp"]    = _safe_get(pe.FILE_HEADER.TimeDateStamp)
    features["Characteristics"]  = _safe_get(pe.FILE_HEADER.Characteristics)

    # ── OPTIONAL_HEADER ─────────────────────────────────────────
    oh = pe.OPTIONAL_HEADER
    features["MajorLinkerVersion"]      = _safe_get(oh.MajorLinkerVersion)
    features["MinorLinkerVersion"]      = _safe_get(oh.MinorLinkerVersion)
    features["SizeOfCode"]              = _safe_get(oh.SizeOfCode)
    features["SizeOfInitializedData"]   = _safe_get(oh.SizeOfInitializedData)
    features["SizeOfUninitializedData"] = _safe_get(oh.SizeOfUninitializedData)
    features["AddressOfEntryPoint"]     = _safe_get(oh.AddressOfEntryPoint)
    features["BaseOfCode"]              = _safe_get(oh.BaseOfCode)
    features["ImageBase"]               = _safe_get(oh.ImageBase)
    features["SectionAlignment"]        = _safe_get(oh.SectionAlignment)
    features["FileAlignment"]           = _safe_get(oh.FileAlignment)
    features["MajorOperatingSystemVersion"] = _safe_get(oh.MajorOperatingSystemVersion)
    features["MajorImageVersion"]       = _safe_get(oh.MajorImageVersion)
    features["MajorSubsystemVersion"]   = _safe_get(oh.MajorSubsystemVersion)
    features["SizeOfImage"]             = _safe_get(oh.SizeOfImage)
    features["SizeOfHeaders"]           = _safe_get(oh.SizeOfHeaders)
    features["CheckSum"]                = _safe_get(oh.CheckSum)
    features["Subsystem"]               = _safe_get(oh.Subsystem)
    features["DllCharacteristics"]      = _safe_get(oh.DllCharacteristics)
    features["SizeOfStackReserve"]      = _safe_get(oh.SizeOfStackReserve)
    features["SizeOfStackCommit"]       = _safe_get(oh.SizeOfStackCommit)
    features["SizeOfHeapReserve"]       = _safe_get(oh.SizeOfHeapReserve)
    features["SizeOfHeapCommit"]        = _safe_get(oh.SizeOfHeapCommit)
    features["LoaderFlags"]             = _safe_get(oh.LoaderFlags)
    features["NumberOfRvaAndSizes"]     = _safe_get(oh.NumberOfRvaAndSizes)

    # ── Sections ─────────────────────────────────────────────────
    section_entropies = []
    section_sizes     = []
    for section in pe.sections:
        try:
            data = section.get_data()
            section_entropies.append(_entropy(data))
            section_sizes.append(len(data))
        except Exception:
            pass

    features["SectionsNb"]            = len(pe.sections)
    features["SectionsMeanEntropy"]   = round(sum(section_entropies) / len(section_entropies), 6) \
                                         if section_entropies else 0.0
    features["SectionsMaxEntropy"]    = round(max(section_entropies), 6) \
                                         if section_entropies else 0.0
    features["SectionsMinEntropy"]    = round(min(section_entropies), 6) \
                                         if section_entropies else 0.0
    features["SectionsMeanRawsize"]   = int(sum(section_sizes) / len(section_sizes)) \
                                         if section_sizes else 0
    features["SectionsMaxRawsize"]    = max(section_sizes) if section_sizes else 0

    # ── Imports ───────────────────────────────────────────────────
    try:
        import_count = sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT)
        dll_count    = len(pe.DIRECTORY_ENTRY_IMPORT)
    except AttributeError:
        import_count = 0
        dll_count    = 0
    features["ImportsNbDLL"]  = dll_count
    features["ImportsNb"]     = import_count

    # ── Exports ───────────────────────────────────────────────────
    try:
        exports_nb = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
    except AttributeError:
        exports_nb = 0
    features["ExportNb"] = exports_nb

    # ── Entropie globale du fichier ───────────────────────────────
    with open(file_path, "rb") as f:
        raw = f.read()
    features["FileEntropy"] = _entropy(raw)

    # ── MD5 (non utilisé pour l'inférence mais utile au logging) ─
    features["MD5"] = hashlib.md5(raw).hexdigest()

    pe.close()
    return features


# ─────────────────────────────────────────────
#  Alignement sur les features du modèle
# ─────────────────────────────────────────────
def align_features(raw_features: dict, model_feature_names: list) -> list:
    """
    Réordonne et complète le vecteur de features pour correspondre
    exactement aux colonnes attendues par le modèle.
    """
    vector = []
    for col in model_feature_names:
        vector.append(raw_features.get(col, 0))
    return vector


# ─────────────────────────────────────────────
#  Inférence directe
# ─────────────────────────────────────────────
def predict_file(file_path: str,
                 model,
                 scaler,
                 feature_names: list) -> dict:
    """
    Pipeline complet : fichier PE → prédiction.

    Retourne :
        {"prediction": "Malware"|"Bénin",
         "probability": float,
         "features": dict}
    """
    import numpy as np

    raw = extract_features(file_path)
    if raw is None:
        return {"error": "Fichier PE invalide ou non parseable."}

    vector   = align_features(raw, feature_names)
    X        = np.array(vector).reshape(1, -1)
    X_scaled = scaler.transform(X)

    label_map = {0: "Bénin", 1: "Malware"}
    pred      = model.predict(X_scaled)[0]
    label     = label_map.get(int(pred), str(pred))

    proba = None
    if hasattr(model, "predict_proba"):
        proba = round(float(model.predict_proba(X_scaled)[0][int(pred)]), 4)

    return {
        "prediction":  label,
        "probability": proba,
        "features":    raw,
    }


# ─────────────────────────────────────────────
#  Test en ligne de commande
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage : python feature_extractor.py <chemin_vers_fichier.exe>")
        sys.exit(1)

    feats = extract_features(sys.argv[1])
    if feats:
        import json
        print(json.dumps({k: v for k, v in feats.items() if k != "MD5"},
                         indent=2))
