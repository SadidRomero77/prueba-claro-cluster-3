"""Carga de los tres insumos del caso."""
from __future__ import annotations

import pandas as pd

from cluster3 import config


def load_clientes(path=config.FILE_CLIENTES) -> pd.DataFrame:
    """Dataset de clusterización del Cluster 3 (20.000 filas × 129 variables)."""
    return pd.read_excel(path, sheet_name=0)


def load_llamadas(path=config.FILE_LLAMADAS) -> pd.DataFrame:
    """500 transcripciones con intención de cancelación (ID, transcription, cluster)."""
    df = pd.read_excel(path, sheet_name=0)
    df["transcription"] = df["transcription"].fillna("").astype(str)
    return df


def load_diccionario(path=config.FILE_DICCIONARIO) -> pd.DataFrame:
    """Diccionario de datos. Una fila por variable; los alias ('A / B') se separan en filas."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Variable")][0]
    dic = raw.iloc[header_row + 1 :, :4].copy()
    dic.columns = ["variable", "definicion", "tipo", "categoria"]
    dic = dic.dropna(subset=["variable"])
    dic["variable"] = dic["variable"].astype(str).str.split("/")
    dic = dic.explode("variable")
    dic["variable"] = dic["variable"].str.strip()
    return dic.reset_index(drop=True)
