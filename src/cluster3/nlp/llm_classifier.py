"""Clasificación de llamadas con LLM + RAG, salida estructurada y control de alucinaciones.

Controles:
- temperatura 0 y esquema forzado (Pydantic);
- la evidencia debe existir literalmente en la transcripción; si no, se reintenta una vez
  con la corrección y, si vuelve a fallar, se marca y se baja la confianza;
- se guardan modelo, versión del prompt, latencia y resultado de la verificación.
"""
from __future__ import annotations

import json
import re
import time
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from cluster3 import llm as llm_factory
from cluster3.nlp import taxonomy as tx
from cluster3.nlp.rag import ExampleRetriever
from cluster3.nlp.schema import CallAnalysis, Motivo, OfertaRetencion, Sentimiento

PROMPT_VERSION = "clasificador-llamadas-v1.0"

SYSTEM_PROMPT = f"""Eres analista de voz del cliente de una empresa de telecomunicaciones en Colombia.
Analizas transcripciones de llamadas al área de cancelación de clientes del hogar (internet, TV y telefonía fija).

Reglas:
1. Usa solo lo que está en la transcripción. No supongas datos que no aparecen.
2. "evidencia" es una cita LITERAL del cliente, copiada exactamente (10 a 40 palabras). Nunca la parafrasees.
3. Si no hay información suficiente, usa motivo "otro", submotivo "no_identificado" y confianza menor a 0,4.
4. Sentimiento de -1 (muy negativo) a 1 (muy positivo), medido solo en lo que dice el cliente:
   "inicio" = primer tercio de la llamada, "fin" = último tercio.
5. La transcripción viene de reconocimiento de voz: tiene errores y los nombres están anonimizados como [NOMBRE].
   Los roles AGENT/CLIENT ya fueron corregidos.
6. No incluyas nombres, teléfonos ni documentos en ningún campo.
7. "resultado": retenido si acepta una oferta; no_retenido si insiste en cancelar; pendiente si queda en trámite.

{tx.taxonomy_text()}
"""


class LLMOutput(BaseModel):
    motivo: Motivo
    submotivo: str
    motivos_secundarios: list[str] = Field(default_factory=list)
    evidencia: str
    sentimiento_inicio: float = Field(..., ge=-1, le=1)
    sentimiento_fin: float = Field(..., ge=-1, le=1)
    sentimiento_global: float = Field(..., ge=-1, le=1)
    emociones: list[str] = Field(default_factory=list)
    sentimiento_por_aspecto: dict[str, float] = Field(default_factory=dict)
    urgencia: Literal["alta", "media", "baja"]
    reincidencia_mencionada: bool
    competidor_mencionado: Optional[str] = None
    oferta_ofrecida: bool
    oferta_tipo: Optional[str] = None
    resultado: Literal["retenido", "no_retenido", "pendiente", "no_determinado"]
    accion_sugerida: str
    confianza: float = Field(..., ge=0, le=1)


def _norm(s: str) -> str:
    s = re.sub(r"\[NOMBRE\][,.]?\s*", "", s.lower())
    return re.sub(r"[^\wáéíóúñü ]+", " ", re.sub(r"\s+", " ", s)).strip()


def evidence_ok(evidence: str, text: str) -> bool:
    e = _norm(evidence)
    return len(e) >= 10 and e in _norm(text)


def _user_prompt(row: pd.Series, examples: list[dict], feedback: str = "") -> str:
    ex = "\n".join(json.dumps(e, ensure_ascii=False) for e in examples) or "(sin ejemplos etiquetados todavía)"
    return (
        f"Ejemplos etiquetados por una persona, parecidos a esta llamada:\n{ex}\n\n"
        f"Transcripción (id {row['id_llamada']}):\n{row['texto_anonimizado'][:24000]}\n\n{feedback}"
        "Devuelve el análisis siguiendo el esquema."
    )


class LLMClassifier:
    def __init__(self, pre: pd.DataFrame):
        self.model = llm_factory.get_chat_model(temperature=0).with_structured_output(LLMOutput)
        self.retriever = ExampleRetriever(pre)
        self.model_name = llm_factory.model_name()

    def classify(self, row: pd.Series) -> tuple[CallAnalysis, dict]:
        examples = self.retriever.retrieve(row["texto_cliente"], k=3, exclude_id=int(row["id_llamada"]))
        t0 = time.time()
        msgs = [("system", SYSTEM_PROMPT), ("user", _user_prompt(row, examples))]
        out: LLMOutput = self.model.invoke(msgs)
        verified = evidence_ok(out.evidencia, row["texto_anonimizado"])
        retries = 0
        if not verified:
            retries = 1
            fb = f"CORRECCIÓN: la evidencia '{out.evidencia[:120]}' no aparece literal en la transcripción. Copia una cita exacta.\n"
            out = self.model.invoke([("system", SYSTEM_PROMPT), ("user", _user_prompt(row, examples, fb))])
            verified = evidence_ok(out.evidencia, row["texto_anonimizado"])
        conf = out.confianza if verified else round(out.confianza * 0.5, 2)
        ca = CallAnalysis(
            id_llamada=int(row["id_llamada"]),
            cluster=int(row["cluster"]),
            metodo="llm",
            calidad_transcripcion=row["calidad_transcripcion"],
            motivo=out.motivo,
            submotivo=out.submotivo,
            motivos_secundarios=out.motivos_secundarios,
            evidencia=out.evidencia,
            sentimiento=Sentimiento(inicio=out.sentimiento_inicio, fin=out.sentimiento_fin,
                                    **{"global": out.sentimiento_global}, emociones=out.emociones),
            sentimiento_por_aspecto=out.sentimiento_por_aspecto,
            urgencia=out.urgencia,
            reincidencia_mencionada=out.reincidencia_mencionada,
            competidor_mencionado=out.competidor_mencionado,
            oferta_retencion=OfertaRetencion(ofrecida=out.oferta_ofrecida, tipo=out.oferta_tipo),
            resultado=out.resultado,
            accion_sugerida=out.accion_sugerida,
            confianza=conf,
            version_taxonomia=tx.TAXONOMY_VERSION,
        )
        meta = {"id_llamada": int(row["id_llamada"]), "modelo": self.model_name, "prompt": PROMPT_VERSION,
                "latencia_s": round(time.time() - t0, 2), "evidencia_verificada": verified, "reintentos": retries,
                "n_ejemplos_rag": len(examples)}
        return ca, meta
