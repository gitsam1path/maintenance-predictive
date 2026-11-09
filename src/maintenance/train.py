"""
Entraînement et évaluation.

1. Régression : prévoir la durée de vie restante (RUL, en cycles) de chaque moteur.
   5 modèles comparés par validation croisée GROUPÉE PAR MOTEUR, puis évaluation sur le jeu de test officiel.
2. Classification : « ce moteur tombera-t-il en panne dans les 30 prochains cycles ? »
   Deux approches comparées, et seuil d'alerte choisi pour minimiser le coût de maintenance.

Lancer depuis src/ :  python -m maintenance.train
"""

from __future__ import annotations

import json
import warnings

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (HORIZON_PANNE, ROOT, RUL_CAP, SENSOR_LABELS, SENSORS, add_features, feature_columns,
                   last_cycles, load_test, load_train)

warnings.filterwarnings("ignore")
REPORTS, FIG, MODELS = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "models"
SEED = 42

# Coûts de maintenance (hypothèses illustratives, modifiables)
COUT_MAINTENANCE_PREVENTIVE = 10_000   # immobiliser l'avion pour une révision planifiée
COUT_PANNE = 100_000                   # panne non anticipée : avion cloué au sol, réparation lourde, retards

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})
BLUE, ORANGE, GREEN, RED, GREY = "#1a73e8", "#e8710a", "#188038", "#d93025", "#9aa0a6"


def nasa_score(y_true, y_pred) -> float:
    """Score officiel du challenge PHM08 : une prévision TROP OPTIMISTE (panne plus tôt que prévu) coûte plus cher."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def regression_models() -> dict:
    lgb_params = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, verbose=-1)
    return {
        "Référence (RUL moyenne)": ("aucune", None),
        "Régression linéaire (capteurs bruts)": ("brutes", make_pipeline(StandardScaler(), LinearRegression())),
        "Ridge (+ variables temporelles)": ("temporelles", make_pipeline(StandardScaler(), Ridge(alpha=10))),
        "Forêt aléatoire": ("temporelles", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.3,
                                                                n_jobs=-1, random_state=SEED)),
        "LightGBM": ("temporelles", lgb.LGBMRegressor(**lgb_params)),
    }


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    for d in (FIG, MODELS):
        d.mkdir(parents=True, exist_ok=True)
    train = add_features(load_train())
    test = add_features(load_test())
    test_last = last_cycles(test)
    y = train["rul"].clip(upper=RUL_CAP)
    groups = train["moteur"]
    cols_raw, cols_all = feature_columns(train, temporal=False), feature_columns(train, temporal=True)
    print(f"Entraînement : {train['moteur'].nunique()} moteurs, {len(train)} cycles · test : {test_last.shape[0]} moteurs")
    print(f"Variables : {len(cols_raw)} brutes, {len(cols_all)} avec les variables temporelles")

    # ───────────── 1. Régression : validation croisée groupée par moteur ─────────────
    gkf = GroupKFold(n_splits=5)
    results, oof = [], {}
    for name, (kind, model) in regression_models().items():
        cols = cols_raw if kind == "brutes" else cols_all
        pred = np.zeros(len(train))
        for tr, va in gkf.split(train, y, groups):
            if model is None:
                pred[va] = y.iloc[tr].mean()
            else:
                m = _clone(model)
                m.fit(train.iloc[tr][cols], y.iloc[tr])
                pred[va] = np.clip(m.predict(train.iloc[va][cols]), 0, None)
        oof[name] = pred
        # sur le test officiel : modèle réentraîné sur tous les moteurs, prévision au dernier cycle observé
        if model is None:
            test_pred = np.full(len(test_last), y.mean())
        else:
            final = _clone(model).fit(train[cols], y)
            test_pred = np.clip(final.predict(test_last[cols]), 0, None)
        true_test = test_last["rul"].to_numpy()
        results.append({
            "modele": name,
            "cv_rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "cv_mae": float(mean_absolute_error(y, pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(true_test, test_pred))),
            "test_mae": float(mean_absolute_error(true_test, test_pred)),
            "test_score_nasa": nasa_score(true_test, test_pred),
            "test_pred": test_pred,
        })
        print(f"  {name:38s} CV RMSE {results[-1]['cv_rmse']:6.2f} · test RMSE {results[-1]['test_rmse']:6.2f} "
              f"· score NASA {results[-1]['test_score_nasa']:8.0f}")

    table = pd.DataFrame(results).sort_values("cv_rmse").reset_index(drop=True)
    best = table.iloc[0]["modele"]
    best_kind, best_model = regression_models()[best]
    best_cols = cols_raw if best_kind == "brutes" else cols_all
    final_reg = _clone(best_model).fit(train[best_cols], y)
    test_pred_best = table.iloc[0]["test_pred"]
    print(f"Modèle retenu (meilleure validation croisée) : {best}")

    # importance des capteurs (par permutation, regroupée par capteur d'origine)
    sample = test.sample(4000, random_state=SEED)
    imp = permutation_importance(final_reg, sample[best_cols], sample["rul"].clip(upper=RUL_CAP),
                                 scoring="neg_root_mean_squared_error", n_repeats=5, random_state=SEED, n_jobs=-1)
    imp = pd.Series(imp.importances_mean, index=best_cols)
    by_sensor = imp.groupby(lambda c: c.split("_")[0]).sum().drop("cycle", errors="ignore").sort_values(ascending=False)

    # ───────────── 2. Classification : panne dans les 30 prochains cycles ? ─────────────
    yc = (train["rul"] <= HORIZON_PANNE).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=SEED, verbose=-1)
    approaches = {
        "Classifieur dédié (LightGBM)": clf,
        "Régression logistique": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
    }
    clf_results, proba_oof = [], {}
    for name, model in approaches.items():
        p = np.zeros(len(train))
        for tr, va in gkf.split(train, yc, groups):
            p[va] = _clone(model).fit(train.iloc[tr][cols_all], yc.iloc[tr]).predict_proba(train.iloc[va][cols_all])[:, 1]
        proba_oof[name] = p
    # 3e approche : utiliser la régression (« alerte si RUL prévue ≤ seuil ») — score = −RUL prévue
    proba_oof[f"Régression ({best}) seuillée"] = -oof[best]

    yct = (test_last["rul"] <= HORIZON_PANNE).astype(int).to_numpy()
    for name, s in proba_oof.items():
        clf_results.append({"approche": name, "cv_roc_auc": float(roc_auc_score(yc, s)),
                            "cv_pr_auc": float(average_precision_score(yc, s))})
    clf_table = pd.DataFrame(clf_results).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
    best_clf_name = clf_table.iloc[0]["approche"]
    print(clf_table.round(3).to_string(index=False))

    # seuil d'alerte : celui qui minimise le coût de maintenance (choisi sur la validation croisée)
    score_oof = proba_oof[best_clf_name]
    thresholds = np.unique(np.quantile(score_oof, np.linspace(0.3, 0.995, 300)))
    # coût moyen par décision (une décision = contrôler un moteur à un cycle donné : alerter ou non)
    costs = [_cost(yc.to_numpy(), score_oof >= t) / len(train) for t in thresholds]
    t_best = float(thresholds[int(np.argmin(costs))])

    if best_clf_name.startswith("Régression ("):
        test_score = -test_pred_best
        final_clf = None
    else:
        final_clf = _clone(approaches[best_clf_name]).fit(train[cols_all], yc)
        test_score = final_clf.predict_proba(test_last[cols_all])[:, 1]
    alert = test_score >= t_best
    tp, fn = int((alert & (yct == 1)).sum()), int((~alert & (yct == 1)).sum())
    fp, tn = int((alert & (yct == 0)).sum()), int((~alert & (yct == 0)).sum())
    test_cost = _cost(yct, alert)
    policies = {
        "Aucune alerte (on attend la panne)": _cost(yct, np.zeros_like(alert)),
        "Révision de tous les moteurs": _cost(yct, np.ones_like(alert)),
        "Alerte du modèle": test_cost,
    }
    print(f"Classification retenue : {best_clf_name} · test : {tp} pannes anticipées, {fn} manquées, {fp} fausses alertes")

    # ───────────── Figures ─────────────
    fig_degradation(train)
    fig_comparaison(table)
    fig_test_scatter(test_last["rul"].to_numpy(), test_pred_best, best)
    fig_trajectoire(train, oof[best], best)
    fig_importance(by_sensor)
    fig_cout(thresholds, costs, t_best, best_clf_name)

    # ───────────── Sauvegardes ─────────────
    joblib.dump({"regression": final_reg, "colonnes": best_cols, "classification": final_clf,
                 "seuil_alerte": t_best, "approche_classification": best_clf_name, "rul_plafond": RUL_CAP},
                MODELS / "model.joblib")
    report = {
        "donnees": {"moteurs_entrainement": int(train["moteur"].nunique()), "cycles_entrainement": int(len(train)),
                    "moteurs_test": int(len(test_last)), "capteurs_utilises": SENSORS, "rul_plafond": RUL_CAP},
        "regression": {
            "modele_retenu": best,
            "resultats": json.loads(table.drop(columns="test_pred").round(3).to_json(orient="records", force_ascii=False)),
            "importance_capteurs": {f"{k} – {SENSOR_LABELS.get(k, k)}": round(float(v), 3) for k, v in by_sensor.head(8).items()},
        },
        "classification": {
            "horizon_cycles": HORIZON_PANNE,
            "approche_retenue": best_clf_name,
            "resultats_cv": json.loads(clf_table.round(4).to_json(orient="records", force_ascii=False)),
            "seuil": round(t_best, 4),
            "test": {"pannes_reelles": int(yct.sum()), "anticipees": tp, "manquees": fn, "fausses_alertes": fp,
                     "moteurs_sains_laisses": tn,
                     "roc_auc": float(roc_auc_score(yct, test_score)), "pr_auc": float(average_precision_score(yct, test_score))},
            "couts_test": {k: int(v) for k, v in policies.items()},
            "hypotheses": {"maintenance_preventive": COUT_MAINTENANCE_PREVENTIVE, "panne": COUT_PANNE},
        },
    }
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nCoûts sur les 100 moteurs de test :")
    for k, v in policies.items():
        print(f"  {k:38s} {v:>10,.0f} €".replace(",", " "))


def _cost(y_true, alert) -> float:
    """Chaque alerte = une maintenance préventive ; chaque panne non anticipée = le coût d'une panne."""
    alert = np.asarray(alert, dtype=bool)
    missed = (~alert) & (np.asarray(y_true) == 1)
    return float(alert.sum() * COUT_MAINTENANCE_PREVENTIVE + missed.sum() * COUT_PANNE)


def _clone(model):
    from sklearn.base import clone
    return clone(model)


# ─────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────
def fig_degradation(train):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, s in zip(axes.ravel(), ["s4", "s11", "s12", "s7"]):
        for i, (_, d) in enumerate(train[train["moteur"].isin([1, 20, 45, 69, 92])].groupby("moteur")):
            ax.plot(-d["rul"], d[s], lw=0.9, alpha=0.85)
        ax.set_title(f"{s} – {SENSOR_LABELS[s]}", fontsize=9.5)
    for ax in axes[1]:
        ax.set_xlabel("Cycles avant la panne (0 = panne)")
    fig.suptitle("L'usure se lit dans les capteurs : 5 moteurs jusqu'à leur panne", fontsize=11)
    _save(fig, "degradation_capteurs")


def fig_comparaison(table):
    t = table.sort_values("test_rmse", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["cv_rmse"], height=0.38, color=BLUE, label="Validation croisée (100 moteurs)")
    ax.barh(y - 0.2, t["test_rmse"], height=0.38, color=ORANGE, label="Test officiel (100 autres moteurs)")
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(r["test_rmse"] + 0.5, yi - 0.2, f"{r['test_rmse']:.1f}", va="center", fontsize=8.5)
        ax.text(r["cv_rmse"] + 0.5, yi + 0.2, f"{r['cv_rmse']:.1f}", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(t["modele"])
    ax.set_xlabel("Erreur RMSE en cycles (plus c'est court, mieux c'est)")
    ax.set_title("Prévoir la durée de vie restante : comparaison des modèles")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, "comparaison_modeles")


def fig_test_scatter(true, pred, name):
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.scatter(true, pred, s=22, color=BLUE, alpha=0.8)
    lim = max(true.max(), pred.max()) + 5
    ax.plot([0, lim], [0, lim], color="black", lw=1, ls="--", label="Prévision parfaite")
    ax.axhline(RUL_CAP, color=GREY, lw=1, ls=":")
    ax.text(lim * 0.98, RUL_CAP + 2, f"plafond {RUL_CAP}", ha="right", fontsize=8, color=GREY)
    ax.set_xlabel("RUL réelle (cycles)")
    ax.set_ylabel("RUL prévue (cycles)")
    ax.set_title(f"Test officiel : 100 moteurs ({name})", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    _save(fig, "test_reel_vs_prevu")


def fig_trajectoire(train, pred, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, m in zip(axes, [24, 81]):
        d = train["moteur"] == m
        sub = train[d]
        ax.plot(sub["cycle"], sub["rul"], color="black", lw=1.5, label="RUL réelle")
        ax.plot(sub["cycle"], sub["rul"].clip(upper=RUL_CAP), color=GREY, lw=1.2, ls="--", label="Cible plafonnée")
        ax.plot(sub["cycle"], pred[d.to_numpy()], color=GREEN, lw=1.6, label="Prévision")
        ax.axhspan(0, HORIZON_PANNE, color=RED, alpha=0.08)
        ax.set_title(f"Moteur {m} (jamais vu à l'entraînement)", fontsize=10)
        ax.set_xlabel("Cycle")
    axes[0].set_ylabel("Durée de vie restante (cycles)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Suivi d'un moteur sur toute sa vie ({name}) ; zone rouge : panne dans moins de 30 cycles", fontsize=10.5)
    _save(fig, "trajectoire_moteur")


def fig_importance(by_sensor):
    s = by_sensor.head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh([f"{k} – {SENSOR_LABELS.get(k, k)}" for k in s.index], s.values, color=BLUE)
    ax.set_xlabel("Hausse de l'erreur (cycles) si l'on brouille ce capteur")
    ax.set_title("Quels capteurs annoncent la panne ?")
    ax.grid(axis="y", visible=False)
    _save(fig, "importance_capteurs")


def fig_cout(thresholds, costs, t_best, name):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = -thresholds if name.startswith("Régression (") else thresholds
    ax.plot(x, costs, color=BLUE, lw=2)
    xb = -t_best if name.startswith("Régression (") else t_best
    ax.axvline(xb, color=RED, ls="--", lw=1)
    ax.set_xlabel("Alerte si RUL prévue ≤ … cycles" if name.startswith("Régression (") else "Seuil de probabilité d'alerte")
    ax.set_ylabel("Coût moyen par décision (€)")
    ax.set_ylim(0, min(max(costs), 4 * min(costs)))
    ax.set_title(f"Choisir le seuil d'alerte qui minimise le coût ({name})", fontsize=10)
    _save(fig, "cout_seuil")


if __name__ == "__main__":
    main()
