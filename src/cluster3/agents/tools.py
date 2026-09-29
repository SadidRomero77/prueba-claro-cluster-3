"""Herramientas (function calling) de los agentes.

Principio: el LLM nunca calcula ni inventa cifras; toda cifra sale de una herramienta.
Las herramientas son funciones puras de Python con tipos y docstring, así sirven igual
para el LLM (tool calling), para el modo sin LLM y para las pruebas.

Controles:
- consultar_sql: solo lectura (SELECT/WITH), una sentencia, máximo 200 filas.
- generar_lista_contacto: sensible → requiere aprobación humana (human-in-the-loop).
"""
from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache

import duckdb
import numpy as np
import pandas as pd

from cluster3 import config

SENSIBLES = {"generar_lista_contacto"}
MAX_ROWS = 200

TABLAS = {
    "clientes": "Clientes del Cluster 3 limpios + probabilidades (p_intencion, p_churn, deciles, riesgo_renta_cop). 1 fila por cliente.",
    "llamadas": "Análisis por llamada: motivo, submotivo, sentimiento, urgencia, evidencia, cluster.",
    "accionables": "Accionables priorizados con evidencia e impacto anual por escenario (COP).",
    "segmentos": "Churn, intención y ARPU por segmento del Cluster 3.",
    "shap_intencion": "Importancia SHAP de variables del modelo de intención.",
    "shap_churn": "Importancia SHAP de variables del modelo de churn.",
    "lift_intencion": "Tabla de lift por decil del modelo de intención.",
    "lift_churn": "Tabla de lift por decil del modelo de churn.",
    "puente": "Cruce agregado motivo de llamadas ↔ variable del dataset.",
}


def build_duckdb() -> None:
    """Materializa las salidas del pipeline en una base DuckDB de solo lectura para el agente."""
    src = {
        "clientes": config.T_SCORES,
        "llamadas": config.T_LLAMADAS_ANALISIS,
        "accionables": config.OUT_TAB / "accionables.csv",
        "segmentos": config.OUT_TAB / "segmentos_cluster3.csv",
        "shap_intencion": config.OUT_TAB / "shap_intencion.csv",
        "shap_churn": config.OUT_TAB / "shap_churn.csv",
        "lift_intencion": config.OUT_TAB / "lift_intencion.csv",
        "lift_churn": config.OUT_TAB / "lift_churn.csv",
        "puente": config.OUT_TAB / "puente_llamadas_dataset.csv",
    }
    if config.DUCKDB_PATH.exists():
        config.DUCKDB_PATH.unlink()
    con = duckdb.connect(str(config.DUCKDB_PATH))
    for name, path in src.items():
        if not path.exists():
            continue
        reader = "read_parquet" if path.suffix == ".parquet" else "read_csv_auto"
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM {reader}('{path.as_posix()}')")
    con.close()


@lru_cache(maxsize=1)
def _con():
    if not config.DUCKDB_PATH.exists():
        build_duckdb()
    return duckdb.connect(str(config.DUCKDB_PATH), read_only=True)


def _fmt(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, float_format=lambda x: f"{x:.4g}")


# ------------------------------------------------------------------ herramientas --
def describir_tablas() -> str:
    """Lista las tablas disponibles para consultar_sql, con sus columnas."""
    out = []
    for t, desc in TABLAS.items():
        try:
            cols = _con().execute(f"SELECT * FROM {t} LIMIT 0").df().columns.tolist()
        except Exception:
            continue
        out.append(f"{t}: {desc}\n  columnas: {', '.join(cols[:60])}{' …' if len(cols) > 60 else ''}")
    return "\n".join(out)


def consultar_sql(query: str) -> str:
    """Ejecuta una consulta SQL de SOLO LECTURA (DuckDB) sobre las tablas del Cluster 3.

    Usa describir_tablas() para ver columnas. Una sola sentencia SELECT o WITH. Máximo 200 filas.
    Ejemplo: SELECT ESTRATO, AVG(BAN_CHURN) FROM clientes GROUP BY 1 ORDER BY 1
    """
    q = query.strip().rstrip(";")
    if ";" in q or not re.match(r"(?is)^\s*(select|with)\b", q):
        return "ERROR: solo se permite una consulta SELECT o WITH."
    if re.search(r"(?i)\b(insert|update|delete|drop|create|alter|attach|copy|export|pragma|install|load)\b", q):
        return "ERROR: operación no permitida (solo lectura)."
    try:
        df = _con().execute(q).df()
    except Exception as e:
        return f"ERROR SQL: {str(e)[:300]}"
    extra = f"\n(se muestran {MAX_ROWS} de {len(df)} filas)" if len(df) > MAX_ROWS else ""
    return _fmt(df.head(MAX_ROWS)) + extra


def kpis_cluster() -> str:
    """KPIs del Cluster 3: clientes, churn, intención, ARPU, renta en riesgo."""
    return (config.OUT_JSON / "kpis_cluster3.json").read_text(encoding="utf-8")


def metricas_modelo(modelo: str = "intencion") -> str:
    """Métricas del modelo 'intencion' o 'churn': AUC, PR-AUC, LIFT, matriz de confusión y top variables."""
    m = json.loads((config.OUT_JSON / "metricas_modelo.json").read_text(encoding="utf-8"))
    if modelo not in ("intencion", "churn"):
        return "ERROR: modelo debe ser 'intencion' o 'churn'"
    return json.dumps(m[modelo], ensure_ascii=False, default=float)


def importancia_variables(modelo: str = "intencion", top: int = 10) -> str:
    """Variables más importantes (SHAP medio, estabilidad entre folds y dirección) del modelo 'intencion' o 'churn'."""
    df = pd.read_csv(config.OUT_TAB / f"shap_{modelo}.csv").head(int(top))
    return _fmt(df)


def riesgo_segmento(condicion: str) -> str:
    """Resume el riesgo de un segmento definido por una condición SQL sobre la tabla clientes.

    Ejemplo de condicion: "ESTRATO = 2 AND TIPO_RED_FTT = 1". Devuelve clientes, churn real,
    intención real, probabilidad media de cada modelo y renta mensual en riesgo (COP).
    """
    if re.search(r"(?i)\b(select|insert|update|delete|drop|;)\b", condicion):
        return "ERROR: la condición debe ser una expresión WHERE simple."
    q = f"""SELECT COUNT(*) AS clientes, AVG(BAN_CHURN) AS churn_real, AVG(BAN_INTENCION_CANCELACION) AS intencion_real,
        AVG(p_churn) AS p_churn_media, AVG(p_intencion) AS p_intencion_media,
        SUM(riesgo_renta_cop) AS renta_mensual_en_riesgo_cop, MEDIAN(VAL_RENTA_ACTUAL) AS arpu_mediano
        FROM clientes WHERE {condicion}"""
    return consultar_sql(q)


@lru_cache(maxsize=2)
def _model(modelo: str):
    with open(config.MODELS_DIR / f"modelo_{modelo}.pkl", "rb") as f:
        return pickle.load(f)


def explicar_cliente(cliente_id: int, modelo: str = "churn") -> str:
    """Probabilidad de un cliente y las 5 variables que más empujan su riesgo (SHAP)."""
    import shap

    art = _model(modelo)
    df = _con().execute(f"SELECT * FROM clientes WHERE cliente_id = {int(cliente_id)}").df()
    if df.empty:
        return f"No existe el cliente {cliente_id}"
    X = df[art["features"]]
    base = art["model"].calibrated_classifiers_[0].estimator
    sv = shap.TreeExplainer(base).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    contrib = pd.Series(sv[0], index=art["features"]).sort_values(key=np.abs, ascending=False).head(5)
    rows = [{"variable": k, "valor": X.iloc[0][k], "efecto": "sube" if v > 0 else "baja", "shap": round(float(v), 4)}
            for k, v in contrib.items()]
    return json.dumps({"cliente_id": int(cliente_id), "modelo": modelo,
                       "probabilidad": float(df[f"p_{modelo}"].iloc[0]), "razones": rows}, ensure_ascii=False, default=float)


def resumen_llamadas(cluster: int = 3) -> str:
    """Motivos, urgencia, sentimiento y resultado de las llamadas de un cluster (por defecto el 3)."""
    q = f"""SELECT motivo, COUNT(*) AS llamadas, ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (),1) AS pct,
        ROUND(AVG(sent_inicio),2) AS sent_inicio, ROUND(AVG(sent_fin),2) AS sent_fin,
        ROUND(100.0*AVG(CASE WHEN urgencia='alta' THEN 1 ELSE 0 END),1) AS urgencia_alta_pct
        FROM llamadas WHERE cluster = {int(cluster)} AND calidad_transcripcion <> 'sin_contenido'
        GROUP BY 1 ORDER BY llamadas DESC"""
    return consultar_sql(q)


@lru_cache(maxsize=1)
def _call_index():
    from sklearn.feature_extraction.text import TfidfVectorizer

    pre = pd.read_parquet(config.T_LLAMADAS_LIMPIAS)[["id_llamada", "texto_cliente"]]
    an = pd.read_parquet(config.T_LLAMADAS_ANALISIS)[["id_llamada", "cluster", "motivo", "submotivo", "evidencia", "urgencia"]]
    df = an.merge(pre, on="id_llamada")
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000)
    return df, vec, vec.fit_transform(df["texto_cliente"])


def buscar_llamadas(texto: str, k: int = 3, motivo: str | None = None) -> str:
    """Busca las llamadas más parecidas a un texto (búsqueda semántica liviana). Devuelve id, motivo y evidencia."""
    from sklearn.metrics.pairwise import cosine_similarity

    df, vec, mat = _call_index()
    sims = cosine_similarity(vec.transform([texto]), mat).ravel()
    res = df.assign(similitud=sims)
    if motivo:
        res = res[res["motivo"] == motivo]
    res = res.sort_values("similitud", ascending=False).head(int(k))
    return _fmt(res[["id_llamada", "cluster", "motivo", "submotivo", "urgencia", "evidencia", "similitud"]])


def listar_accionables(tipo: str | None = None) -> str:
    """Accionables priorizados con evidencia, métrica e impacto anual (COP) por escenario. tipo: Proactivo | Reactivo."""
    df = pd.read_csv(config.OUT_TAB / "accionables.csv")
    if tipo:
        df = df[df["tipo"].str.lower() == tipo.lower()]
    cols = ["id", "accionable", "tipo", "prioridad", "evidencia", "metrica", "impacto_anual_conservador_cop",
            "impacto_anual_base_cop", "impacto_anual_optimista_cop"]
    nombres = {"impacto_anual_conservador_cop": "anual_conservador_cop", "impacto_anual_base_cop": "anual_base_cop",
               "impacto_anual_optimista_cop": "anual_optimista_cop"}
    return _fmt(df[cols].rename(columns=nombres))


def calcular_impacto(n_clientes: int, prob_irse: float, tasa_exito: float = 0.30, arpu_cop: float | None = None,
                     meses: int = 12, costo_contacto_cop: float = 15_000) -> str:
    """Impacto económico de retener: n × prob_irse × tasa_exito × ARPU × meses − n × costo de contacto (COP)."""
    if arpu_cop is None:
        arpu_cop = float(json.loads(kpis_cluster())["arpu_mediano"])
    bruto = n_clientes * prob_irse * tasa_exito * arpu_cop * meses
    costo = n_clientes * costo_contacto_cop
    return json.dumps({"clientes_salvados": round(n_clientes * prob_irse * tasa_exito, 1), "renta_salvada_cop": round(bruto),
                       "costo_cop": round(costo), "impacto_neto_cop": round(bruto - costo), "supuestos":
                       {"tasa_exito": tasa_exito, "arpu_cop": arpu_cop, "meses": meses, "costo_contacto_cop": costo_contacto_cop}})


def generar_lista_contacto(decil_max: int = 1, limite: int = 500) -> str:
    """[SENSIBLE · requiere aprobación humana] Exporta la lista de clientes de los deciles de mayor riesgo de churn."""
    q = f"""SELECT cliente_id, p_churn, p_intencion, decil_churn, VAL_RENTA_ACTUAL, riesgo_renta_cop FROM clientes
            WHERE decil_churn <= {int(decil_max)} ORDER BY p_churn DESC LIMIT {int(limite)}"""
    df = _con().execute(q).df()
    out = config.OUT / "listas"
    out.mkdir(exist_ok=True)
    fn = out / f"lista_contacto_decil{decil_max}.csv"
    df.to_csv(fn, index=False)
    return f"Lista generada: {fn.relative_to(config.ROOT)} ({len(df)} clientes)"


TOOLS_POR_AGENTE = {
    "perfilado": [describir_tablas, consultar_sql, kpis_cluster, metricas_modelo, importancia_variables,
                  riesgo_segmento, explicar_cliente],
    "voz_cliente": [resumen_llamadas, buscar_llamadas, consultar_sql],
    "estrategia": [listar_accionables, calcular_impacto, kpis_cluster, generar_lista_contacto],
}
ALL_TOOLS = {f.__name__: f for fs in TOOLS_POR_AGENTE.values() for f in fs}
