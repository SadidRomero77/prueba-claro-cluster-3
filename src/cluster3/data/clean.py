"""Limpieza del dataset y registro de decisiones por variable.

Criterio de aceptación: el 100 % de las variables tiene una decisión documentada
(usar, transformar, excluir del modelo o eliminar) con su motivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cluster3 import config
from cluster3.data.quality import fix_score_crediticio

WINSOR = ["VAL_DIAS_PAGO", "VAL_SALDO_ACTUAL"]


def clean_clientes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (dataset limpio, registro de decisiones)."""
    d = df.copy()
    log: dict[str, tuple[str, str]] = {}

    # Estrato: 'NR' → nulo
    d["ESTRATO"] = pd.to_numeric(d["ESTRATO"], errors="coerce")
    log["ESTRATO"] = ("transformar", "'NR' (6 registros) a nulo; numérico")

    # Score crediticio con escala rota
    d["SCORE_CREDITICIO_FIX"] = fix_score_crediticio(d["VAL_SCORE_CREDITICIO"])
    log["VAL_SCORE_CREDITICIO"] = ("eliminar", "Escala rota; reemplazada por SCORE_CREDITICIO_FIX (0–1000)")
    log["SCORE_CREDITICIO_FIX"] = ("usar", "Score reescalado a 0–1000")

    # Downtime negativo es imposible
    n_neg = int((d["VAL_DOWNTIME"] < 0).sum())
    d.loc[d["VAL_DOWNTIME"] < 0, "VAL_DOWNTIME"] = np.nan
    log["VAL_DOWNTIME"] = ("transformar", f"{n_neg} valores negativos a nulo")

    for c in WINSOR:
        lo, hi = d[c].quantile([0.01, 0.99])
        d[c] = d[c].clip(lo, hi)
        log[c] = ("transformar", f"Recorte p1–p99 ({lo:,.0f} a {hi:,.0f})".replace(",", "."))

    # Variables sin información
    const = [c for c in d.columns if d[c].nunique(dropna=True) <= 1 and c not in config.ID_COLS]
    for c in const:
        motivo = "100 % nula" if d[c].isna().all() else "Constante en los 20.000 registros"
        log[c] = ("eliminar", motivo)
    d = d.drop(columns=const + ["VAL_SCORE_CREDITICIO"])

    for c in config.ID_COLS:
        log[c] = ("excluir_modelo", "Identificador o periodo único")
    for c in [config.TARGET_CHURN, config.TARGET_INTENCION]:
        log[c] = ("target", "Variable objetivo")
    for c in config.LEAK_ESTADO:
        if c in d.columns:
            log[c] = ("excluir_modelo", "Fuga: estado de la cuenta después de la baja (ESTADO_FUENTE_C = BAN_CHURN)")
    for c in config.LEAK_ESTADO_TV:
        if c in d.columns:
            log[c] = ("excluir_modelo", "Fuga: el sufijo I del plan de TV solo aparece en cuentas no activas "
                                        "(ESTADO_FUENTE_A = 0 en el 100 %)")
    for c in config.LEAK_INTENCION:
        log[c] = ("excluir_modelo", "Fuga: derivada de la intención de cancelación")
    for c in config.POST_EVENTO:
        if c in d.columns:
            log[c] = ("excluir_modelo", "Posterior al evento: reacción de retención a la intención")

    log["VAL_SALDO_ACTUAL"] = ("transformar", log["VAL_SALDO_ACTUAL"][1] + "; excluida del modelo de churn: "
                               "vale 0 en los 104 clientes que se fueron (cuenta liquidada)")
    for c in df.columns.tolist() + ["SCORE_CREDITICIO_FIX"]:
        log.setdefault(c, ("usar", "Sin hallazgos de calidad"))

    registro = pd.DataFrame(
        [{"variable": k, "decision": v[0], "motivo": v[1]} for k, v in log.items()]
    ).sort_values(["decision", "variable"])
    return d, registro


def model_features(df: pd.DataFrame) -> list[str]:
    """Variables candidatas para el modelo: sin fuga, sin post-evento, sin IDs ni targets."""
    excl = set(config.ID_COLS + [config.TARGET_CHURN, config.TARGET_INTENCION])
    excl |= set(config.LEAK_ESTADO + config.LEAK_ESTADO_TV + config.LEAK_INTENCION + config.POST_EVENTO)
    return [c for c in df.columns if c not in excl and pd.api.types.is_numeric_dtype(df[c])]


def run_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.ensure_dirs()
    d, registro = clean_clientes(df)
    d.to_parquet(config.T_CLIENTES_LIMPIO, index=False)
    registro.to_csv(config.OUT_TAB / "registro_decisiones_variables.csv", index=False)
    return d, registro
