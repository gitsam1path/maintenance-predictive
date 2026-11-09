import json
from pathlib import Path

import numpy as np
import pytest

from maintenance.data import RUL_CAP, SENSORS, add_features, last_cycles, load_test, load_train
from maintenance.train import _cost, nasa_score

ROOT = Path(__file__).resolve().parents[1]


def test_donnees():
    train, test = load_train(), load_test()
    assert train["moteur"].nunique() == 100 and test["moteur"].nunique() == 100
    # la RUL vaut 0 au dernier cycle de chaque moteur d'entraînement (le moteur tombe en panne)
    assert (last_cycles(train)["rul"] == 0).all()
    # au test, la RUL décroît de 1 à chaque cycle
    m1 = test[test["moteur"] == 1].sort_values("cycle")
    assert (np.diff(m1["rul"]) == -1).all()


def test_variables_sans_fuite_du_futur():
    """Les variables d'un cycle ne doivent dépendre que des cycles passés."""
    train = load_train()
    full = add_features(train)
    cut = add_features(train[train["cycle"] <= 50])  # on coupe la vie des moteurs au cycle 50
    a = full[full["cycle"] == 50].set_index("moteur").sort_index()
    b = cut[cut["cycle"] == 50].set_index("moteur").sort_index()
    cols = [c for c in a.columns if c.startswith("s")]
    np.testing.assert_allclose(a[cols].to_numpy(), b[cols].to_numpy(), rtol=1e-9, atol=1e-9)


def test_capteurs_constants_ecartes():
    train = load_train()
    assert all(train[s].std() > 1e-3 for s in SENSORS)
    assert "s1" not in SENSORS and "s18" not in SENSORS


def test_score_nasa_asymetrique():
    # prévoir une panne 10 cycles trop TARD (optimiste) coûte plus que 10 cycles trop TÔT
    assert nasa_score([50], [60]) > nasa_score([50], [40])
    assert nasa_score([50], [50]) == 0


def test_cout_maintenance():
    y = np.array([1, 1, 0, 0])
    assert _cost(y, np.array([False] * 4)) == 200_000          # 2 pannes non anticipées
    assert _cost(y, np.array([True] * 4)) == 40_000            # 4 révisions préventives
    assert _cost(y, np.array([True, True, False, False])) == 20_000


def test_rapport():
    path = ROOT / "reports" / "metrics.json"
    if not path.exists():
        pytest.skip("lancer d'abord : cd src && python -m maintenance.train")
    r = json.loads(path.read_text(encoding="utf-8"))
    res = {m["modele"]: m for m in r["regression"]["resultats"]}
    # le meilleur modèle doit largement battre la référence
    best = res[r["regression"]["modele_retenu"]]
    assert best["test_rmse"] < 0.5 * res["Référence (RUL moyenne)"]["test_rmse"]
    assert r["donnees"]["rul_plafond"] == RUL_CAP
    c = r["classification"]["couts_test"]
    assert c["Alerte du modèle"] < min(c["Aucune alerte (on attend la panne)"], c["Révision de tous les moteurs"])
