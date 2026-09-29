"""Rutas, constantes y reglas de negocio compartidas por todo el proyecto.

Todo lo que es una decisión (variables excluidas, umbrales, supuestos) vive aquí
para que quede en un solo lugar y sea fácil de auditar.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(os.getenv("CLUSTER3_ROOT", Path(__file__).resolve().parents[2]))
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_LABELS = ROOT / "data" / "labels"
MODELS_DIR = ROOT / "models"
OUT = ROOT / "outputs"
OUT_FIG = OUT / "figures"
OUT_TAB = OUT / "tables"
OUT_JSON = OUT / "json"
OUT_REP = OUT / "reports"
DOCS = ROOT / "docs"

FILE_CLIENTES = DATA_RAW / "Clientes_Cluster_3.xlsx"
FILE_LLAMADAS = DATA_RAW / "Llamadas.xlsx"
FILE_DICCIONARIO = DATA_RAW / "Diccionario_Datos_Cluster3.xlsx"

# Tablas procesadas (Parquet). En Databricks son tablas Delta con el mismo nombre.
T_CLIENTES_LIMPIO = DATA_PROCESSED / "clientes_c3_limpio.parquet"
T_SCORES = DATA_PROCESSED / "clientes_c3_scores.parquet"
T_LLAMADAS_LIMPIAS = DATA_PROCESSED / "llamadas_limpias.parquet"
T_LLAMADAS_ANALISIS = DATA_PROCESSED / "llamadas_analisis.parquet"
# Clasificación híbrida (Jev + LLM) de las 500 llamadas: se reutiliza en corridas sin API para no volver a reglas.
T_LLAMADAS_HIBRIDO = DATA_PROCESSED / "llamadas_hibrido.parquet"
DUCKDB_PATH = DATA_PROCESSED / "cluster3.duckdb"

SEED = 42
CLUSTER_CRITICO = 3

# --- Targets -------------------------------------------------------------------
TARGET_CHURN = "BAN_CHURN"
TARGET_INTENCION = "BAN_INTENCION_CANCELACION"

# --- Fuga de información (decisión documentada en docs/supuestos_y_decisiones.md) ---
# Estado de la cuenta después de la baja: ESTADO_FUENTE_C == BAN_CHURN en el 100 % de los registros.
LEAK_ESTADO = ["ESTADO_FUENTE_A", "ESTADO_FUENTE_C", "ESTADO_FUENTE_MAS", "ESTADO_FUENTE_MENOS"]
# Planes de TV con sufijo "I" (PI, BI): el 100 % son cuentas no activas (ESTADO_FUENTE_A = 0) y suman 103 de las
# 159 cuentas no activas. El sufijo codifica el estado de la cuenta, no el plan: misma fuga que ESTADO_FUENTE_*.
LEAK_ESTADO_TV = ["TIPO_TV_DIGITAL_PI", "TIPO_TV_DIGITAL_BI"]
# Derivadas de la intención: los reincidentes a 30 días son un subconjunto exacto de BAN_INTENCION.
LEAK_INTENCION = ["CANTIDAD_INTENCIONES", "BAN_REINCIDENTE_30", "BAN_REINCIDENTE_60", "MOTIVO_LLAM_CANCELA"]
# Reacción de la compañía a la intención (posteriores al evento).
# BAN_CAMPANA_VENTA describe el tipo de la misma campaña del mes que BAN_CAMPANA_ACTIVA/RETENCION/CORRECTIVA:
# con un solo corte no se sabe si la campaña fue antes o después de la llamada (intención 46 % vs 15 %).
POST_EVENTO = [
    "BAN_RETENCION_ACTIVA",
    "BAN_CAMPANA_RETENCION",
    "BAN_CAMPANA_VENTA",
    "VAL_CAMPANA_RETENCION_36M",
    "BAN_CAMPANA_ACTIVA",
    "BAN_CAMPANA_CORRECTIVA",
    "VAL_CAMPANA",
    "VAL_OFER_ANTERIORES",
]
# Solo para churn: el saldo es 0 en los 104 clientes que se fueron (vs 33 % con saldo en el resto):
# la cuenta se liquida al cancelar. Es un efecto de la baja, no una causa.
POST_EVENTO_CHURN = ["VAL_SALDO_ACTUAL"]
ID_COLS = ["CLUSTER_ID", "PERIODO"]

# --- Supuestos de negocio -------------------------------------------------------------
MONEDA = "COP"
ARPU_COL = "VAL_RENTA_ACTUAL"  # renta mensual como aproximación al ARPU

# --- LLM / Jev --------------------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai | openrouter | databricks
LLM_MODEL = os.getenv("LLM_MODEL", "")  # vacío = el default del proveedor
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")


def ensure_dirs() -> None:
    for p in [DATA_INTERIM, DATA_PROCESSED, DATA_LABELS, MODELS_DIR, OUT_FIG, OUT_TAB, OUT_JSON, OUT_REP, DOCS]:
        p.mkdir(parents=True, exist_ok=True)
