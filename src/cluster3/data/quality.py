"""Calidad, consistencia y representatividad de los datos.

Produce evidencia (tablas) y hallazgos que alimentan:
- docs/calidad_datos.md (reporte)
- la limpieza (clean.py)
- la selección de variables del modelo (evita fuga de información)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cluster3 import config



def _miles(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")

def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil por variable: tipo, únicos, nulos, rango y valores negativos."""
    rows = []
    for c in df.columns:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        is_num = num.notna().sum() >= 0.95 * s.notna().sum() and s.notna().sum() > 0
        rows.append(
            {
                "variable": c,
                "dtype": str(s.dtype),
                "n_unicos": int(s.nunique(dropna=True)),
                "pct_nulos": round(float(s.isna().mean() * 100), 2),
                "constante": bool(s.nunique(dropna=True) <= 1),
                "min": float(num.min()) if is_num else np.nan,
                "p01": float(num.quantile(0.01)) if is_num else np.nan,
                "mediana": float(num.median()) if is_num else np.nan,
                "p99": float(num.quantile(0.99)) if is_num else np.nan,
                "max": float(num.max()) if is_num else np.nan,
                "n_negativos": int((num < 0).sum()) if is_num else 0,
                "pct_ceros": round(float((num == 0).mean() * 100), 2) if is_num else np.nan,
            }
        )
    return pd.DataFrame(rows)


def univariate_auc(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """AUC univariado de cada variable contra un target (max(AUC, 1-AUC)).

    Un valor cercano a 1 en una sola variable es señal de fuga de información.
    """
    y = df[target]
    rows = []
    for c in df.columns:
        if c == target:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.nunique(dropna=True) < 2:
            continue
        x = x.fillna(x.min() - 1)
        auc = roc_auc_score(y, x)
        rows.append(
            {
                "variable": c,
                "auc_univariado": round(max(auc, 1 - auc), 4),
                "media_positivos": float(x[y == 1].mean()),
                "media_negativos": float(x[y == 0].mean()),
                "identica_al_target": bool((x == y).all()),
            }
        )
    return pd.DataFrame(rows).sort_values("auc_univariado", ascending=False).reset_index(drop=True)


def fix_score_crediticio(s: pd.Series) -> pd.Series:
    """El score perdió el separador decimal (8763653 = 876,3653). Se lleva a escala 0–1000."""

    def _fix(v):
        if pd.isna(v):
            return np.nan
        x = float(v)
        while x >= 1000:
            x /= 10
        return x

    return s.map(_fix)


def reconcile_dictionary(df: pd.DataFrame, dic: pd.DataFrame) -> pd.DataFrame:
    """Cruce dataset ↔ diccionario: qué está en ambos, solo en datos o solo en diccionario."""
    in_data = set(df.columns)
    in_dic = set(dic["variable"])
    cat = dic.drop_duplicates("variable").set_index("variable")["categoria"].to_dict()
    defs = dic.drop_duplicates("variable").set_index("variable")["definicion"].to_dict()
    rows = []
    for v in sorted(in_data | in_dic):
        estado = "ambos" if v in in_data and v in in_dic else ("solo_dataset" if v in in_data else "solo_diccionario")
        rows.append({"variable": v, "estado": estado, "categoria": cat.get(v, _infer_category(v)), "definicion": defs.get(v, "")})
    return pd.DataFrame(rows)


def _infer_category(v: str) -> str:
    """Categoría para variables que no están en el diccionario (dummies y derivadas)."""
    if v.startswith("ESTRATO_"):
        return "Segmentación"
    if v.startswith(("TIPO_RED_", "TIPO_TV_")):
        return "Segmentación"
    if v.startswith("ESTADO_FUENTE_"):
        return "Intención de Cancelación & Churn"
    if v.startswith("MOTIVO_LLAM_"):
        return "Reclamos, Llamadas & Soporte"
    if v.startswith("CLASIF_OKK_"):
        return "Zona, Nodo & Calidad de Red"
    if v.startswith("VAL_NUM_LINEAS_") or v.startswith("VAL_SUM_"):
        return "Servicio Móvil"
    if v == "BAN_CAZA_OFERTA":
        return "Campañas & Retención (NBO/NBA)"
    return "Sin categoría"


def _ct(t: pd.DataFrame) -> dict:
    """Crosstab a dict JSON-serializable (claves como texto)."""
    return {str(i): {str(c): int(v) for c, v in row.items()} for i, row in t.iterrows()}


def consistency_findings(df: pd.DataFrame) -> list[dict]:
    """Chequeos de consistencia entre variables relacionadas."""
    f = []
    ct = pd.crosstab(df["ESTADO_FUENTE_C"], df[config.TARGET_CHURN])
    f.append(
        {
            "id": "fuga_estado_fuente",
            "hallazgo": "ESTADO_FUENTE_C es idéntica a BAN_CHURN (estado de la cuenta después de la baja)",
            "evidencia": _ct(ct),
            "decision": "Excluir ESTADO_FUENTE_* de los modelos",
        }
    )
    r30 = df[df["BAN_REINCIDENTE_30"] == 1][config.TARGET_INTENCION].mean()
    f.append(
        {
            "id": "fuga_reincidencia",
            "hallazgo": f"El {r30:.0%} de los reincidentes a 30 días tiene intención = 1: son un subconjunto del target",
            "evidencia": _ct(pd.crosstab([df["BAN_REINCIDENTE_30"], df["BAN_REINCIDENTE_60"]], df[config.TARGET_INTENCION])),
            "decision": "No usar reincidencias ni CANTIDAD_INTENCIONES para predecir intención",
        }
    )
    m = pd.crosstab(df["MOTIVO_LLAM_CANCELA"], df[config.TARGET_INTENCION])
    f.append(
        {
            "id": "incoherencia_motivo_cancela",
            "hallazgo": f"MOTIVO_LLAM_CANCELA = 1 en {int(df['MOTIVO_LLAM_CANCELA'].sum())} clientes, "
            f"pero solo {int(m.loc[1, 1])} tienen intención = 1",
            "evidencia": _ct(m),
            "decision": "Documentar; se trata como variable de contacto, no como etiqueta",
        }
    )
    n_hist = int((df["CANTIDAD_INTENCIONES"] > 0).sum())
    f.append(
        {
            "id": "definicion_cantidad_intenciones",
            "hallazgo": f"CANTIDAD_INTENCIONES > 0 en {n_hist} clientes vs {int(df[config.TARGET_INTENCION].sum())} con intención en el mes",
            "evidencia": {},
            "decision": "Supuesto: BAN_INTENCION es del mes; CANTIDAD_INTENCIONES es histórica",
        }
    )
    sc = df["VAL_SCORE_CREDITICIO"]
    f.append(
        {
            "id": "escala_score",
            "hallazgo": f"VAL_SCORE_CREDITICIO va de {_miles(sc.min())} a {_miles(sc.max())}: el separador decimal se perdió",
            "evidencia": {"n_mayor_1e6": int((sc >= 1e6).sum()), "n_menor_1000": int((sc <= 1000).sum())},
            "decision": "Reescalar a 0–1000 (SCORE_CREDITICIO_FIX)",
        }
    )
    for c, regla in [
        ("VAL_DOWNTIME", "negativos a nulo"),
        ("VAL_DIAS_PAGO", "recorte p1–p99"),
        ("VAL_SALDO_ACTUAL", "recorte p1–p99"),
    ]:
        n = int((df[c] < 0).sum())
        f.append(
            {
                "id": f"rango_{c.lower()}",
                "hallazgo": f"{c} tiene {_miles(n)} valores negativos (mínimo {_miles(df[c].min())})",
                "evidencia": {},
                "decision": regla,
            }
        )
    saldo = df.loc[df[config.TARGET_CHURN] == 1, "VAL_SALDO_ACTUAL"]
    f.append(
        {
            "id": "fuga_saldo_churn",
            "hallazgo": f"VAL_SALDO_ACTUAL = 0 en {int((saldo == 0).sum())} de {len(saldo)} clientes con churn, "
            f"vs {(df.loc[df[config.TARGET_CHURN] == 0, 'VAL_SALDO_ACTUAL'] != 0).mean():.0%} con saldo en el resto",
            "evidencia": {},
            "decision": "La cuenta se liquida al cancelar: se excluye del modelo de churn",
        }
    )
    tv_i = df[config.LEAK_ESTADO_TV].max(axis=1) == 1
    inactiva = df["ESTADO_FUENTE_A"] == 0
    f.append(
        {
            "id": "fuga_plan_tv_i",
            "hallazgo": f"TIPO_TV_DIGITAL_PI/BI = 1 en {int(tv_i.sum())} clientes, todos con cuenta no activa "
            f"(ESTADO_FUENTE_A = 0 en {int((tv_i & inactiva).sum())} de {int(tv_i.sum())}); son "
            f"{int((tv_i & inactiva).sum())} de las {int(inactiva.sum())} cuentas no activas y "
            f"{int(df.loc[tv_i, config.TARGET_CHURN].sum())} de las {int(df[config.TARGET_CHURN].sum())} bajas",
            "evidencia": {},
            "decision": "El sufijo I codifica el estado de la cuenta: se excluye de ambos modelos (validar con Claro)",
        }
    )
    f.append(
        {
            "id": "periodo_unico",
            "hallazgo": f"PERIODO único = {df['PERIODO'].unique().tolist()}",
            "evidencia": {},
            "decision": "Sin validación temporal posible; se reporta como limitación",
        }
    )
    return f


def call_findings(calls: pd.DataFrame) -> list[dict]:
    """Calidad de las transcripciones."""
    t = calls["transcription"]
    swapped = t.str.contains(
        r"CLIENT:[^\n]{0,200}(?:con qui[ée]n tengo el gusto|[áa]rea de cancelaci|en qu[ée] le puedo ayudar)",
        case=False,
        regex=True,
    )
    name_leak = t.str.contains(r"(?:mi nombre es|hablas con|habla con) [a-záéíóúñ]{3,}", case=False, regex=True)
    return [
        {"id": "llamadas_por_cluster", "hallazgo": {str(k): int(v) for k, v in calls["cluster"].value_counts().sort_index().items()}},
        {"id": "largo_caracteres", "hallazgo": t.str.len().describe().round(0).to_dict()},
        {"id": "roles_invertidos", "hallazgo": f"{int(swapped.sum())} llamadas con guion del agente en turnos CLIENT"},
        {"id": "pii_residual", "hallazgo": f"{int(name_leak.sum())} llamadas con posibles nombres sin anonimizar"},
        {"id": "llamadas_cortas", "hallazgo": f"{int((t.str.len() < 1000).sum())} llamadas con menos de 1.000 caracteres"},
        {"id": "sin_llave", "hallazgo": "Las llamadas no tienen CUENTA ni DOCUMENTO: no se pueden unir individualmente al dataset"},
    ]


def run_quality(df: pd.DataFrame, calls: pd.DataFrame, dic: pd.DataFrame) -> dict:
    """Ejecuta todos los chequeos y guarda las tablas de evidencia."""
    config.ensure_dirs()
    perfil = profile_columns(df)
    perfil.to_csv(config.OUT_TAB / "perfil_variables.csv", index=False)
    auc_churn = univariate_auc(df, config.TARGET_CHURN)
    auc_churn.to_csv(config.OUT_TAB / "auc_univariado_churn.csv", index=False)
    auc_int = univariate_auc(df, config.TARGET_INTENCION)
    auc_int.to_csv(config.OUT_TAB / "auc_univariado_intencion.csv", index=False)
    dic_rec = reconcile_dictionary(df, dic)
    dic_rec.to_csv(config.OUT_TAB / "diccionario_reconciliado.csv", index=False)
    result = {
        "n_filas": int(len(df)),
        "n_columnas": int(df.shape[1]),
        "duplicados": int(df.duplicated().sum()),
        "constantes": perfil.loc[perfil["constante"], "variable"].tolist(),
        "vacias_100": perfil.loc[perfil["pct_nulos"] >= 100, "variable"].tolist(),
        "diccionario": dic_rec["estado"].value_counts().to_dict(),
        "sospecha_fuga_churn": auc_churn.loc[auc_churn["auc_univariado"] >= 0.95, "variable"].tolist(),
        "sospecha_fuga_intencion": auc_int.loc[auc_int["auc_univariado"] >= 0.95, "variable"].tolist(),
        "consistencia": consistency_findings(df),
        "llamadas": call_findings(calls),
    }
    (config.OUT_JSON / "calidad_datos.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return result
