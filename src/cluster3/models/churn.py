"""Modelo de propensión en dos etapas para el Cluster 3.

Etapa 1 · Intención de cancelación (BAN_INTENCION_CANCELACION, 19,9 %):
    alerta temprana, antes de que el cliente llame. Solo variables conocidas antes
    de la intención (se excluyen reincidencias, conteo de intenciones y reacciones
    de retención).
Etapa 2 · Churn (BAN_CHURN, 0,52 %):
    quién se va de verdad. Las señales de intención sí se permiten (ocurren antes
    de la baja); se excluyen el estado de la cuenta y las variables posteriores.

Decisiones:
- Desbalance: pesos de clase (scale_pos_weight). SMOTE se descarta: con 104 positivos
  genera clientes sintéticos casi idénticos y el modelo aprende ruido.
- Validación: CV estratificada repetida (3 × 5) → AUC con desviación estándar.
- Calibración: isotónica en intención, sigmoide en churn (pocos positivos).
- Umbral: por valor esperado de negocio, no 0,5.
- Importancia: SHAP por fold (estabilidad) y agregada por categoría del diccionario.
"""
from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from cluster3 import config
from cluster3.data.clean import model_features

warnings.filterwarnings("ignore", category=UserWarning)

# Supuestos económicos para el umbral de negocio (se documentan en el reporte).
MESES_VIDA_EXTRA = 12
TASA_EXITO_RETENCION = 0.30
COSTO_CONTACTO_COP = 15_000


@dataclass
class StageSpec:
    name: str
    target: str
    features: list[str]
    params: dict
    calibration: str
    notes: list[str] = field(default_factory=list)


def lgb_params(pos_rate: float, small: bool = False) -> dict:
    return dict(
        n_estimators=300 if small else 400,
        learning_rate=0.03,
        num_leaves=7 if small else 15,
        min_child_samples=20 if small else 40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        scale_pos_weight=(1 - pos_rate) / pos_rate,
        random_state=config.SEED,
        verbose=-1,
    )


def build_specs(d: pd.DataFrame) -> dict[str, StageSpec]:
    base = model_features(d)
    intent_signals = [c for c in config.LEAK_INTENCION if c in d.columns] + [config.TARGET_INTENCION]
    return {
        "intencion": StageSpec(
            "intencion", config.TARGET_INTENCION, base, lgb_params(d[config.TARGET_INTENCION].mean()), "isotonic",
            ["Sin reincidencias, conteo de intenciones ni reacciones de retención"],
        ),
        "churn": StageSpec(
            "churn", config.TARGET_CHURN, [c for c in base if c not in config.POST_EVENTO_CHURN] + intent_signals, lgb_params(d[config.TARGET_CHURN].mean(), small=True),
            "sigmoid", ["Incluye señales de intención (anteceden a la baja); excluye ESTADO_FUENTE_*, planes de TV con sufijo I, "
                       "VAL_SALDO_ACTUAL y post-evento"],
        ),
    }


# ----------------------------------------------------------------- métricas --
def lift_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    order = np.argsort(-p)
    ys = y[order]
    bins = np.array_split(np.arange(len(y)), n_bins)
    base = y.mean()
    rows, cum_pos = [], 0
    for i, idx in enumerate(bins, start=1):
        pos = ys[idx].sum()
        cum_pos += pos
        rows.append(
            {
                "decil": i,
                "clientes": len(idx),
                "positivos": int(pos),
                "tasa": pos / len(idx),
                "lift": (pos / len(idx)) / base,
                "captura_acumulada": cum_pos / y.sum(),
                "lift_acumulado": (cum_pos / sum(len(b) for b in bins[:i])) / base,
            }
        )
    return pd.DataFrame(rows)


def lift_at(y: np.ndarray, p: np.ndarray, q: float) -> float:
    n = max(1, int(len(y) * q))
    idx = np.argsort(-p)[:n]
    return float(y[idx].mean() / y.mean())


def business_threshold(y: np.ndarray, p: np.ndarray, arpu: np.ndarray) -> dict:
    """Umbral que maximiza el valor esperado de contactar: salvar renta vs costo de contacto."""
    best = {"umbral": 0.5, "valor": -np.inf}
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.995, 60))):
        sel = p >= t
        valor = (y[sel] * arpu[sel] * MESES_VIDA_EXTRA * TASA_EXITO_RETENCION).sum() - sel.sum() * COSTO_CONTACTO_COP
        if valor > best["valor"]:
            best = {"umbral": float(t), "valor": float(valor), "contactados": int(sel.sum())}
    return best


def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    return {
        "umbral": threshold,
        "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ------------------------------------------------------------ entrenamiento --
def cross_validate(d: pd.DataFrame, spec: StageSpec, repeats: int = 3, with_shap: bool = True) -> dict:
    X = d[spec.features]
    y = d[spec.target].values
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=config.SEED)
    aucs, aps = [], []
    oof = np.zeros(len(y))
    shap_folds = []
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        m = lgb.LGBMClassifier(**spec.params).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
        if i < 5:  # primera repetición = OOF completo + SHAP por fold
            oof[te] = p
            if with_shap:
                sv = shap.TreeExplainer(m).shap_values(X.iloc[te])
                sv = sv[1] if isinstance(sv, list) else sv
                shap_folds.append(pd.Series(np.abs(sv).mean(axis=0), index=spec.features, name=f"fold{i+1}"))
                if i == 0:
                    direction = pd.Series([_corr(X.iloc[te][c].fillna(X[c].median()).values, sv[:, j])
                                           for j, c in enumerate(spec.features)], index=spec.features)
    res = {
        "auc_media": float(np.mean(aucs)),
        "auc_sd": float(np.std(aucs)),
        "pr_auc_media": float(np.mean(aps)),
        "base_rate": float(y.mean()),
        "oof": oof,
        "lift_5": lift_at(y, oof, 0.05),
        "lift_10": lift_at(y, oof, 0.10),
        "lift_20": lift_at(y, oof, 0.20),
        "lift_tabla": lift_table(y, oof),
    }
    if with_shap:
        sf = pd.concat(shap_folds, axis=1)
        ranks = sf.rank(ascending=False)
        res["shap"] = pd.DataFrame(
            {
                "shap_medio": sf.mean(axis=1),
                "shap_sd": sf.std(axis=1),
                "folds_en_top15": (ranks <= 15).sum(axis=1),
                "direccion": np.sign(direction).map({1.0: "sube el riesgo", -1.0: "baja el riesgo", 0.0: "mixta"}),
            }
        ).sort_values("shap_medio", ascending=False)
    return res


def leakage_comparison(d: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Métricas con y sin variables con fuga, para mostrar por qué se excluyen."""
    rows = []
    specs = build_specs(d)
    leak_all = (config.LEAK_ESTADO + config.LEAK_ESTADO_TV + config.LEAK_INTENCION + config.POST_EVENTO
                + config.POST_EVENTO_CHURN)
    for key, spec in specs.items():
        with_leak = list(dict.fromkeys(spec.features + [c for c in leak_all if c in raw.columns and c != spec.target]))
        dd = d.copy()
        for c in with_leak:
            if c not in dd.columns:
                dd[c] = raw[c].values
        for label, feats in [("con fuga", with_leak), ("sin fuga (modelo final)", spec.features)]:
            s = StageSpec(spec.name, spec.target, feats, spec.params, spec.calibration)
            r = cross_validate(dd, s, repeats=1, with_shap=False)
            rows.append({"modelo": key, "variables": label, "n_variables": len(feats), "auc": round(r["auc_media"], 3),
                         "pr_auc": round(r["pr_auc_media"], 3), "lift_10": round(r["lift_10"], 2)})
    return pd.DataFrame(rows)


def sensitivity_without(d: pd.DataFrame, spec: StageSpec, drop: list[str]) -> dict:
    s = StageSpec(spec.name, spec.target, [c for c in spec.features if c not in drop], spec.params, spec.calibration)
    r = cross_validate(d, s, repeats=1, with_shap=False)
    return {"sin": drop, "auc": r["auc_media"], "pr_auc": r["pr_auc_media"], "lift_10": r["lift_10"]}


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n: int = 500) -> tuple[float, float]:
    rng = np.random.default_rng(config.SEED)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fit_final(d: pd.DataFrame, spec: StageSpec):
    """Modelo final calibrado, entrenado con todo el dataset."""
    base = lgb.LGBMClassifier(**spec.params)
    model = CalibratedClassifierCV(base, method=spec.calibration, cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED))
    model.fit(d[spec.features], d[spec.target])
    return model


def category_importance(shap_df: pd.DataFrame, dic_rec: pd.DataFrame) -> pd.DataFrame:
    cat = dic_rec.set_index("variable")["categoria"].to_dict()
    cat["SCORE_CREDITICIO_FIX"] = "Facturación & Cartera"
    t = shap_df[["shap_medio"]].copy()
    t["categoria"] = [cat.get(v, "Sin categoría") for v in t.index]
    g = t.groupby("categoria")["shap_medio"].sum().sort_values(ascending=False)
    return (g / g.sum() * 100).round(1).rename("pct_importancia").reset_index()


def compare_targets(shap_int: pd.DataFrame, shap_churn: pd.DataFrame, top: int = 20) -> pd.DataFrame:
    """Balance de importancia entre targets: palancas estructurales, de negociación o de salida."""
    r1 = shap_int["shap_medio"].rank(ascending=False)
    r2 = shap_churn["shap_medio"].rank(ascending=False)
    t = pd.DataFrame({"rank_intencion": r1, "rank_churn": r2}).dropna(how="all")
    t = t[(t["rank_intencion"] <= top) | (t["rank_churn"] <= top)].copy()

    def _tipo(row):
        a, b = row["rank_intencion"] <= top, row["rank_churn"] <= top
        if a and b:
            return "Estructural (ambos)"
        return "Negociación (solo intención)" if a else "Salida (solo churn)"

    t["tipo_palanca"] = t.apply(_tipo, axis=1)
    return t.sort_values(["tipo_palanca", "rank_intencion"]).reset_index(names="variable")


def plot_model(res: dict, name: str, shap_top: int = 15):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cluster3.eda.profile import BLUE, GRID, INK, INK2, _style

    lt = res["lift_tabla"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.r_[0, lt["decil"] * 10], np.r_[0, lt["captura_acumulada"] * 100], color=BLUE, lw=2, marker="o", ms=5)
    ax.plot([0, 100], [0, 100], color=GRID, lw=1.5, ls="--")
    ax.set_xlabel("% de clientes contactados (ordenados por riesgo)", color=INK2)
    ax.set_ylabel("% de positivos capturados", color=INK2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID)
    ax.set_title(f"Curva de ganancia · {name}  (AUC {res['auc_media']:.3f}, LIFT@10 % {res['lift_10']:.1f})", loc="left",
                 fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / f"ganancia_{name}.png", dpi=150)
    plt.close(fig)

    s = res["shap"].head(shap_top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(s) + 1.2))
    ax.barh(s.index, s["shap_medio"], xerr=s["shap_sd"], color=BLUE, height=0.55, error_kw={"ecolor": INK2, "lw": 1})
    for yi, (v, dirn) in enumerate(zip(s["shap_medio"], s["direccion"])):
        ax.text(v, yi, f"   {dirn}", va="center", fontsize=8, color=INK2)
    _style(ax, f"Importancia SHAP (media ± sd entre folds) · {name}")
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / f"shap_{name}.png", dpi=150)
    plt.close(fig)


def run_models(d: pd.DataFrame, raw: pd.DataFrame, dic_rec: pd.DataFrame) -> dict:
    config.ensure_dirs()
    specs = build_specs(d)
    out: dict = {"supuestos_umbral": {"meses_vida_extra": MESES_VIDA_EXTRA, "tasa_exito": TASA_EXITO_RETENCION,
                                      "costo_contacto_cop": COSTO_CONTACTO_COP}}
    scores = pd.DataFrame(index=d.index)
    arpu = d[config.ARPU_COL].values
    shap_tables = {}
    for key, spec in specs.items():
        res = cross_validate(d, spec)
        y = d[spec.target].values
        lo, hi = bootstrap_auc_ci(y, res["oof"])
        thr = business_threshold(y, res["oof"], arpu)
        top10 = float(np.quantile(res["oof"], 0.90))
        final = fit_final(d, spec)
        p_final = final.predict_proba(d[spec.features])[:, 1]
        with open(config.MODELS_DIR / f"modelo_{key}.pkl", "wb") as f:
            pickle.dump({"model": final, "features": spec.features, "target": spec.target}, f)
        scores[f"p_{key}"] = p_final  # modelo final calibrado (para la app y el scoring)
        scores[f"p_{key}_oof"] = res["oof"]  # fuera de muestra (para evaluar y dimensionar accionables)
        res["lift_tabla"].to_csv(config.OUT_TAB / f"lift_{key}.csv", index=False)
        res["shap"].to_csv(config.OUT_TAB / f"shap_{key}.csv", index_label="variable")
        category_importance(res["shap"], dic_rec).to_csv(config.OUT_TAB / f"importancia_categoria_{key}.csv", index=False)
        plot_model(res, key)
        shap_tables[key] = res["shap"]
        out[key] = {
            "target": spec.target,
            "n_variables": len(spec.features),
            "positivos": int(y.sum()),
            "tasa_base": res["base_rate"],
            "auc_cv_media": res["auc_media"],
            "auc_cv_sd": res["auc_sd"],
            "auc_oof_ic95": [lo, hi],
            "pr_auc": res["pr_auc_media"],
            "lift_5": res["lift_5"],
            "lift_10": res["lift_10"],
            "lift_20": res["lift_20"],
            "brier_calibrado": float(brier_score_loss(y, p_final)),
            "matriz_top10": confusion(y, res["oof"], top10),
            "matriz_umbral_negocio": confusion(y, res["oof"], thr["umbral"]) | {"valor_esperado_cop": thr["valor"]},
            "top_variables": res["shap"].head(10).index.tolist(),
            "notas": spec.notes,
        }
    # Equipos adicionales y UltraWiFi son más altos en cuentas no activas: se mide cuánto depende el modelo de ellos
    out["sensibilidad_churn_sin_equipos"] = sensitivity_without(d, specs["churn"], ["VAL_UW", "VAL_EQUIP_ADIC"])
    comp = compare_targets(shap_tables["intencion"], shap_tables["churn"])
    comp.to_csv(config.OUT_TAB / "balance_importancia_targets.csv", index=False)
    leak = leakage_comparison(d, raw)
    leak.to_csv(config.OUT_TAB / "comparacion_fuga.csv", index=False)
    out["comparacion_fuga"] = leak.to_dict(orient="records")

    scores["riesgo_renta_cop"] = scores["p_churn"] * d[config.ARPU_COL].values
    # Deciles con probabilidades fuera de muestra: evita que el decil 1 "capture" todo por sobreajuste.
    for key in ["intencion", "churn"]:
        scores[f"decil_{key}"] = pd.qcut(scores[f"p_{key}_oof"].rank(method="first", ascending=False), 10,
                                         labels=range(1, 11)).astype(int)
    full = pd.concat([d.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    full.insert(0, "cliente_id", np.arange(1, len(full) + 1))
    full.to_parquet(config.T_SCORES, index=False)
    (config.OUT_JSON / "metricas_modelo.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    _log_mlflow(out)
    return out


def _log_mlflow(out: dict) -> None:
    """Registra métricas en MLflow si está instalado (local: ./mlruns; Databricks: tracking del workspace)."""
    try:
        import mlflow
    except ImportError:
        return
    import os

    uris = [os.getenv("MLFLOW_TRACKING_URI")] if os.getenv("MLFLOW_TRACKING_URI") else [
        f"sqlite:///{config.ROOT / 'mlflow.db'}",
        f"file:{config.ROOT / 'mlruns'}",
    ]
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    for uri in uris:
        try:  # el tracking nunca debe romper el pipeline
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "cluster3_churn"))
            _log_runs(mlflow, out)
            print(f"[mlflow] métricas registradas en {uri}")
            return
        except Exception as e:
            print(f"[mlflow] {uri} no disponible: {str(e)[:120]}")


def _log_runs(mlflow, out: dict) -> None:
    for key in ["intencion", "churn"]:
        with mlflow.start_run(run_name=f"lgbm_{key}"):
            m = out[key]
            mlflow.log_params({"target": m["target"], "n_variables": m["n_variables"], **out["supuestos_umbral"]})
            mlflow.log_metrics({k: m[k] for k in ["auc_cv_media", "auc_cv_sd", "pr_auc", "lift_5", "lift_10", "lift_20",
                                                  "brier_calibrado"]})
            mlflow.log_artifact(str(config.OUT_FIG / f"shap_{key}.png"))
            mlflow.log_artifact(str(config.MODELS_DIR / f"modelo_{key}.pkl"))
