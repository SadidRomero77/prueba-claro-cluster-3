"""Recuperación de ejemplos para el clasificador LLM (RAG liviano).

No se busca en documentos: se recuperan las llamadas ya etiquetadas por una persona
que más se parecen a la llamada actual, y se pasan como ejemplos (few-shot dinámico).
Esto da consistencia entre llamadas parecidas y permite ampliar la taxonomía sin
reescribir el prompt. En producción el índice vive en Databricks Vector Search.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cluster3 import config

LABELS_FILE = config.DATA_LABELS / "muestra_etiquetada.csv"


class ExampleRetriever:
    def __init__(self, pre: pd.DataFrame, labels_file=LABELS_FILE):
        self.examples = pd.DataFrame()
        if labels_file.exists():
            lab = pd.read_csv(labels_file)
            lab = lab[lab["motivo_humano"].notna() & (lab["motivo_humano"].astype(str).str.strip() != "")]
            if len(lab):
                self.examples = lab.merge(pre[["id_llamada", "texto_cliente"]], on="id_llamada")
        self.vec = None
        if len(self.examples):
            self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
            self.matrix = self.vec.fit_transform(self.examples["texto_cliente"])

    def retrieve(self, text: str, k: int = 3, exclude_id: int | None = None) -> list[dict]:
        if self.vec is None:
            return []
        sims = cosine_similarity(self.vec.transform([text]), self.matrix).ravel()
        ex = self.examples.assign(sim=sims)
        if exclude_id is not None:
            ex = ex[ex["id_llamada"] != exclude_id]
        out = []
        for _, r in ex.sort_values("sim", ascending=False).head(k).iterrows():
            out.append(
                {
                    "fragmento_cliente": r["texto_cliente"][:600],
                    "motivo": r["motivo_humano"],
                    "submotivo": r.get("submotivo_humano", ""),
                    "urgencia": r.get("urgencia_humana", ""),
                }
            )
        return out
