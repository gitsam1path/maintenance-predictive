"""
Données NASA C-MAPSS (jeu FD001) : 100 moteurs d'avion simulés, suivis cycle après cycle jusqu'à la panne.

Chaque ligne = un moteur à un cycle donné : 3 réglages de vol et 21 capteurs.
- Entraînement : chaque moteur est suivi jusqu'à sa panne -> on connaît sa durée de vie restante (RUL) à chaque cycle.
- Test : chaque moteur est arrêté avant la panne ; la vraie RUL au dernier cycle est fournie à part.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

COLUMNS = ["moteur", "cycle", "reglage_1", "reglage_2", "reglage_3"] + [f"s{i}" for i in range(1, 22)]

# Capteurs (Saxena et al., 2008). Ceux qui restent constants sur FD001 sont écartés.
SENSOR_LABELS = {
    "s2": "Température sortie compresseur BP (T24)",
    "s3": "Température sortie compresseur HP (T30)",
    "s4": "Température sortie turbine BP (T50)",
    "s7": "Pression sortie compresseur HP (P30)",
    "s8": "Vitesse du fan (Nf)",
    "s9": "Vitesse du cœur (Nc)",
    "s11": "Pression statique compresseur HP (Ps30)",
    "s12": "Rapport carburant / Ps30 (phi)",
    "s13": "Vitesse corrigée du fan (NRf)",
    "s14": "Vitesse corrigée du cœur (NRc)",
    "s15": "Taux de dilution (BPR)",
    "s17": "Enthalpie de prélèvement (htBleed)",
    "s20": "Débit de refroidissement turbine HP (W31)",
    "s21": "Débit de refroidissement turbine BP (W32)",
}
SENSORS = list(SENSOR_LABELS)
RUL_CAP = 125  # au début de sa vie, un moteur ne montre aucune usure : on plafonne la cible
HORIZON_PANNE = 30  # classification : « panne dans les 30 prochains cycles ? »


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / name, sep=r"\s+", header=None, names=COLUMNS)


def load_train() -> pd.DataFrame:
    df = _read("train_FD001.txt")
    df["rul"] = df.groupby("moteur")["cycle"].transform("max") - df["cycle"]
    return df


def load_test() -> pd.DataFrame:
    """Jeu de test : la RUL vraie n'est connue qu'au dernier cycle observé ; on la propage aux cycles précédents."""
    df = _read("test_FD001.txt")
    final = pd.read_csv(RAW / "RUL_FD001.txt", header=None, names=["rul_finale"])
    final["moteur"] = np.arange(1, len(final) + 1)
    df = df.merge(final, on="moteur")
    df["rul"] = df["rul_finale"] + df.groupby("moteur")["cycle"].transform("max") - df["cycle"]
    return df.drop(columns="rul_finale")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Variables construites moteur par moteur, en n'utilisant QUE le passé (fenêtres glissantes) :
    à un cycle donné, on ne connaît pas encore les cycles suivants.
    """
    df = df.sort_values(["moteur", "cycle"]).copy()
    g = df.groupby("moteur")
    out = [df[["moteur", "cycle", "rul"] + SENSORS]]
    feats = {}
    for s in SENSORS:
        feats[f"{s}_moy10"] = g[s].transform(lambda x: x.rolling(10, min_periods=1).mean())
        feats[f"{s}_moy30"] = g[s].transform(lambda x: x.rolling(30, min_periods=1).mean())
        feats[f"{s}_ecart10"] = g[s].transform(lambda x: x.rolling(10, min_periods=2).std()).fillna(0)
        # pente sur les 20 derniers cycles : la vitesse d'usure
        feats[f"{s}_pente20"] = g[s].transform(_rolling_slope)
    out.append(pd.DataFrame(feats, index=df.index))
    return pd.concat(out, axis=1)


def _rolling_slope(x: pd.Series, window: int = 20) -> pd.Series:
    t = pd.Series(np.arange(len(x)), index=x.index, dtype=float)
    cov = (x * t).rolling(window, min_periods=5).mean() - x.rolling(window, min_periods=5).mean() * t.rolling(window, min_periods=5).mean()
    var = t.rolling(window, min_periods=5).var(ddof=0)
    return (cov / var).fillna(0)


def feature_columns(df: pd.DataFrame, temporal: bool = True) -> list[str]:
    cols = ["cycle"] + SENSORS
    if temporal:
        cols += [c for c in df.columns if c.startswith("s") and "_" in c]
    return cols


def last_cycles(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("cycle").groupby("moteur").tail(1).sort_values("moteur")
