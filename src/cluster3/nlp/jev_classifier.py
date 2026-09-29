"""Clasificación con Jev (TypeSafe AI), modelo "System One".

Jev no genera texto: responde preguntas tipadas (Choice, Score, Noul) con probabilidades
calibradas, en milisegundos y a muy bajo costo. Aquí responde motivo, urgencia,
sentimiento y dos preguntas sí/no. La evidencia textual y el submotivo quedan para el LLM.

SDK: pip install typesafe-sdk · variable TYPESAFE_API_KEY.
Verifica la firma exacta de Choice/Score/Noul en https://docs.typesafe.ai si el SDK cambia.
"""
from __future__ import annotations

import time

import pandas as pd

from cluster3 import config
from cluster3.nlp import taxonomy as tx

MAX_CHARS = 60_000  # el estado admite hasta 64k tokens; las llamadas caben completas


def _get(resp, key):
    """Acceso tolerante a la respuesta (objeto o dict)."""
    for getter in (lambda: resp[key], lambda: getattr(resp, key), lambda: resp.answers[key]):
        try:
            return getter()
        except Exception:
            continue
    raise KeyError(key)


def _field(ans, *names, default=None):
    for n in names:
        v = getattr(ans, n, None)
        if v is None and isinstance(ans, dict):
            v = ans.get(n)
        if v is not None:
            return v
    return default


class JevClassifier:
    def __init__(self):
        from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

        if not config.TYPESAFE_API_KEY:
            raise RuntimeError("Falta TYPESAFE_API_KEY en el entorno (.env)")
        self.client = TypeSafeClient()
        self.questions = {
            "motivo": Choice(
                instructions="¿Cuál es el motivo principal por el que el cliente quiere cancelar el servicio?",
                criteria={m: v["definicion"] for m, v in tx.MOTIVOS.items()},
            ),
            "urgencia": Score(
                instructions="¿Qué tan firme y urgente es la intención de cancelar del cliente?",
                # Rúbrica ordenada de menor a mayor: el score es la posición (0 = baja, 2 = alta)
                criteria=[tx.URGENCIA["baja"], tx.URGENCIA["media"], tx.URGENCIA["alta"]],
            ),
            "sentimiento": Score(
                instructions="¿Cómo se siente el cliente durante la llamada?",
                criteria=["Enojado o frustrado", "Molesto", "Neutral", "Conforme", "Satisfecho"],
            ),
            "competidor": Noul(instructions="¿El cliente dice que se va o tiene oferta de otro operador?"),
            "reincidencia": Noul(instructions="¿El cliente dice que ya había reclamado antes por lo mismo?"),
        }

    def classify(self, row: pd.Series) -> dict:
        t0 = time.time()
        resp = self.client.system_one(
            state={"llamada_cancelacion_cliente": row["texto_cliente"][:MAX_CHARS]},
            questions=self.questions,
            model=config.JEV_MODEL,
        )
        m, u, s = _get(resp, "motivo"), _get(resp, "urgencia"), _get(resp, "sentimiento")
        levels_s = ["muy_negativo", "negativo", "neutral", "positivo", "muy_positivo"]
        score_s = _field(s, "score", default=2)
        return {
            "id_llamada": int(row["id_llamada"]),
            "jev_motivo": _field(m, "choice"),
            "jev_motivo_prob": _field(m, "confidence"),
            "jev_motivo_probs": _field(m, "probabilities"),
            "jev_urgencia": ["baja", "media", "alta"][min(2, max(0, round(float(_field(u, "score", default=1)))))],
            "jev_urgencia_score": _field(u, "score"),
            "jev_sentimiento": round((float(score_s) / (len(levels_s) - 1)) * 2 - 1, 3),
            "jev_competidor_prob": _field(_get(resp, "competidor"), "noul"),
            "jev_reincidencia_prob": _field(_get(resp, "reincidencia"), "noul"),
            "jev_latencia_s": round(time.time() - t0, 3),
            "jev_modelo": config.JEV_MODEL,
        }
