"""Evaluación de los clasificadores contra la muestra etiquetada a mano.

Métricas: exactitud y F1 macro por campo, kappa de Cohen (acuerdo más allá del azar),
latencia y costo por llamada. Es la base para elegir entre baseline, LLM y Jev.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

from cluster3 import config
from cluster3.nlp.rag import LABELS_FILE

CAMPOS = {"motivo": "motivo_humano", "urgencia": "urgencia_humana"}


def load_labels() -> pd.DataFrame:
    if not LABELS_FILE.exists():
        return pd.DataFrame()
    lab = pd.read_csv(LABELS_FILE)
    return lab[lab["motivo_humano"].notna() & (lab["motivo_humano"].astype(str).str.strip() != "")]


def evaluate(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """preds: {metodo: DataFrame con id_llamada, motivo, urgencia}."""
    lab = load_labels()
    if lab.empty:
        return pd.DataFrame([{"estado": "Sin etiquetas humanas todavía: completa data/labels/muestra_etiquetada.csv"}])
    rows = []
    for metodo, p in preds.items():
        m = lab.merge(p, on="id_llamada", how="inner")
        for campo, col_h in CAMPOS.items():
            if campo not in m or col_h not in m:
                continue
            mm = m.dropna(subset=[campo, col_h])
            if mm.empty:
                continue
            y, yhat = mm[col_h].astype(str), mm[campo].astype(str)
            rows.append(
                {
                    "metodo": metodo,
                    "campo": campo,
                    "n": len(mm),
                    "exactitud": round(accuracy_score(y, yhat), 3),
                    "f1_macro": round(f1_score(y, yhat, average="macro", zero_division=0), 3),
                    "kappa": round(cohen_kappa_score(y, yhat), 3),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(config.OUT_TAB / "nlp_benchmark.csv", index=False)
    return out
