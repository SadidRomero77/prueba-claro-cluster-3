"""Evaluación del sistema multiagente con preguntas doradas.

Por pregunta se mide: ruta del orquestador, cifras respaldadas (control del crítico),
presencia de datos esperados y latencia. Con LLM disponible se agrega un juez (G-Eval).

    python -m cluster3.agents.evals            # modo automático (llm si hay clave, si no offline)
    python -m cluster3.agents.evals --offline
"""
from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from cluster3 import config
from cluster3 import llm as llm_factory
from cluster3.agents import prompts as P
from cluster3.agents.graph import ask, default_mode, resume

GOLDEN = [
    {"pregunta": "Hola, ¿quién eres y en qué me puedes ayudar?", "ruta": ["conversacion"], "contiene": ["perfilado"]},
    {"pregunta": "¿Ya tienen el modelo de machine learning? ¿Cómo funciona?", "ruta": ["perfilado"], "contiene": ["LightGBM"]},
    {"pregunta": "¿Qué insights de negocio identificaron y qué se puede mejorar?", "ruta": ["estrategia"], "contiene": ["precio_facturacion"]},
    {"pregunta": "¿Cuál es la tasa de churn y de intención de cancelación del Cluster 3?", "ruta": ["perfilado"], "contiene": ["0.0052", "0.199"]},
    {"pregunta": "¿Qué variables explican la intención de cancelación?", "ruta": ["perfilado", "voz_cliente"], "contiene": ["VAL_VAR_RENTA"]},
    {"pregunta": "¿Qué tan bueno es el modelo de churn? Dame AUC y lift", "ruta": ["perfilado"], "contiene": ["auc"]},
    {"pregunta": "¿Cuáles son los principales motivos de cancelación en las llamadas del Cluster 3?", "ruta": ["voz_cliente"], "contiene": ["precio_facturacion"]},
    {"pregunta": "Dame ejemplos de lo que dicen los clientes sobre cobros que no pidieron", "ruta": ["voz_cliente"], "contiene": ["evidencia"]},
    {"pregunta": "¿Qué accionables recomiendas y cuál es su impacto económico?", "ruta": ["estrategia"], "contiene": ["A1"]},
    {"pregunta": "¿Qué acciones proactivas priorizarías para retener clientes?", "ruta": ["estrategia"], "contiene": ["Proactivo"]},
    {"pregunta": "Explica el riesgo de churn del cliente 15", "ruta": ["perfilado"], "contiene": ["probabilidad"]},
    {"pregunta": "Genera la lista de contacto del decil de mayor riesgo", "ruta": ["estrategia"], "contiene": [], "hitl": True},
    {"pregunta": "Analiza esta llamada:\nAsesor: Buenas tardes, área de cancelaciones.\nCliente: Quiero cancelar. Me "
                 "cobraron un paquete que nunca pedí y ya es la tercera vez que llamo por lo mismo.\nAsesor: Permítame "
                 "revisar.\nCliente: Siempre me dicen que lo quitan y vuelve a aparecer.",
     "ruta": ["voz_cliente"], "contiene": ["intencion_cancelar_prob", "oferta_sugerida"]},
]


def _judge(q: str, r: dict) -> dict:
    try:
        out = llm_factory.get_chat_model().invoke([
            ("system", P.RUBRICA_JUEZ),
            ("user", f"Pregunta: {q}\nRespuesta: {r.get('respuesta')}\nEvidencia: {' '.join(r.get('evidencia') or [])[:10000]}")])
        txt = out.content
        return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    except Exception as e:
        return {"error": str(e)[:120]}


def run(modo: str | None = None) -> pd.DataFrame:
    modo = modo or default_mode()
    rows = []
    for g in GOLDEN:
        t0 = time.time()
        r = ask(g["pregunta"], modo=modo, usar_cache=False)
        hitl = "interrupt" in r
        if hitl:
            r = resume(r["thread_id"], aprobado=False)  # en evaluación nunca se exportan datos
        texto = (r.get("respuesta") or "") + " ".join(r.get("evidencia") or [])
        row = {
            "pregunta": g["pregunta"],
            "modo": modo,
            "ruta": ",".join(r.get("agentes") or []),
            "ruta_ok": all(a in (r.get("agentes") or []) for a in g["ruta"][:1]),
            "cifras_respaldadas": bool((r.get("critica") or {}).get("aprobado")),
            "contenido_ok": all(c.lower() in texto.lower() for c in g["contiene"]),
            "hitl_ok": (hitl if g.get("hitl") else not hitl),
            "latencia_s": round(time.time() - t0, 2),
        }
        if modo == "llm":
            row.update({f"juez_{k}": v for k, v in _judge(g["pregunta"], r).items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(config.OUT_TAB / "agent_evals.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    d = run("offline" if a.offline else None)
    print(d.to_string())
    print("\nResumen:", d[["ruta_ok", "cifras_respaldadas", "contenido_ok", "hitl_ok"]].mean().round(2).to_dict())
