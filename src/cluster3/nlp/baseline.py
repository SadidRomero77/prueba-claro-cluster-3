"""Clasificador base por reglas (sin API).

Sirve para: (1) tener el JSON de las 500 llamadas aunque no haya claves de API,
(2) fijar un piso de comparación contra el LLM y Jev, y (3) sugerir etiquetas
iniciales para la muestra que se etiqueta a mano (siempre revisadas por una persona).
"""
from __future__ import annotations

import re

import pandas as pd

from cluster3.nlp import taxonomy as tx
from cluster3.nlp.schema import CallAnalysis, OfertaRetencion, Sentimiento

OFERTA_RE = {
    "descuento": r"(?:descuento|rebaja|precio especial|tarifa preferencial)",
    "cambio_plan": r"(?:cambio de plan|otro plan|plan m[áa]s econ[óo]mico|bajar (?:el )?plan|solo internet)",
    "beneficio": r"(?:beneficio|meses gratis|bono|upgrade|m[áa]s megas|sin costo)",
    "visita_tecnica": r"(?:agendar (?:una )?visita|enviar un t[ée]cnico|programar (?:la )?visita)",
}
RETENIDO_RE = r"(?:acepto|me quedo|d[ée]jelo as[íi]|est[áa] bien,? (?:lo )?acepto|listo,? (?:hag[áa]moslo|aplíquelo))"
NO_RETENIDO_RE = r"(?:igual (?:quiero|voy a) cancelar|no me interesa|proceda con la cancelaci|ya tom[ée] la decisi|no,? gracias)"


def _sentiment(text: str) -> float:
    t = text.lower()
    neg = sum(t.count(w) for w in tx.LEXICO_NEGATIVO)
    pos = sum(t.count(w) for w in tx.LEXICO_POSITIVO)
    if neg + pos == 0:
        return 0.0
    return round(max(-1.0, min(1.0, (pos - neg) / (pos + neg + 2))), 3)


def _evidence(text: str, pattern: str, width: int = 160) -> str | None:
    m = re.search(pattern, text, re.I)
    if not m:
        return None
    a, b = max(0, m.start() - width // 2), min(len(text), m.end() + width // 2)
    return text[a:b].strip()


def classify_row(r: pd.Series) -> CallAnalysis:
    client = r["texto_cliente"]
    full = r["texto_anonimizado"]
    scores: dict[str, tuple[int, str, str]] = {}
    for motivo in tx.PRIORIDAD:
        best = None
        total = 0
        for sub, pat in tx.MOTIVOS[motivo]["submotivos"].items():
            n = len(re.findall(pat, client, re.I))
            total += n
            if n and (best is None or n > best[1]):
                best = (sub, n, pat)
        if best:
            scores[motivo] = (total, best[0], best[2])
    if r["calidad_transcripcion"] == "sin_contenido" or not scores:
        motivo, sub, ev, conf = "otro", "no_identificado", client[:160] or full[:160], 0.2
        secundarios = []
    else:
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1][0], tx.PRIORIDAD.index(kv[0])))
        motivo, (n, sub, pat) = ranked[0]
        ev = _evidence(client, pat) or client[:160]
        secundarios = [m for m, _ in ranked[1:3]]
        total = sum(v[0] for v in scores.values())
        conf = round(min(0.9, 0.35 + 0.5 * n / total), 2)

    s_ini, s_fin, s_glob = _sentiment(r["cliente_inicio"]), _sentiment(r["cliente_fin"]), _sentiment(client)
    emociones = [e for e, p in tx.EMOCIONES.items() if re.search(p, client, re.I)]
    aspectos = {}
    for asp, m in [("facturacion", "precio_facturacion"), ("tecnico", "falla_tecnica"), ("atencion", "atencion_servicio")]:
        frases = [s for s in re.split(r"[.?!]", client) if any(re.search(p, s, re.I) for p in tx.MOTIVOS[m]["submotivos"].values())]
        if frases:
            aspectos[asp] = _sentiment(" ".join(frases))

    comp = re.search(tx.COMPETIDORES, client, re.I)
    reinc = bool(re.search(tx.MOTIVOS["atencion_servicio"]["submotivos"]["reclamo_repetido"], client, re.I))
    if re.search(tx.PATRONES_URGENCIA_ALTA, client, re.I) or reinc:
        urg = "alta"
    elif re.search(tx.PATRONES_URGENCIA_BAJA, client, re.I):
        urg = "baja"
    else:
        urg = "media"

    ofertas = [k for k, p in OFERTA_RE.items() if re.search(p, full, re.I)]
    fin = r["cliente_fin"]
    if re.search(RETENIDO_RE, fin, re.I):
        res = "retenido"
    elif re.search(NO_RETENIDO_RE, fin, re.I):
        res = "no_retenido"
    else:
        res = "no_determinado"

    return CallAnalysis(
        id_llamada=int(r["id_llamada"]),
        cluster=int(r["cluster"]),
        metodo="baseline_reglas",
        calidad_transcripcion=r["calidad_transcripcion"],
        motivo=motivo,
        submotivo=sub,
        motivos_secundarios=secundarios,
        evidencia=ev,
        sentimiento=Sentimiento(inicio=s_ini, fin=s_fin, **{"global": s_glob}, emociones=emociones),
        sentimiento_por_aspecto=aspectos,
        urgencia=urg,
        reincidencia_mencionada=reinc,
        competidor_mencionado=comp.group(1).lower() if comp else None,
        oferta_retencion=OfertaRetencion(ofrecida=bool(ofertas), tipo=ofertas[0] if ofertas else None),
        resultado=res,
        accion_sugerida=ACCION_POR_MOTIVO[motivo],
        confianza=conf,
        version_taxonomia=tx.TAXONOMY_VERSION,
    )


ACCION_POR_MOTIVO = {
    "precio_facturacion": "Revisar la factura, reversar cobros no reconocidos y ofrecer plan ajustado antes de descuento",
    "servicios_no_usados": "Ajustar el plan a lo que usa (retirar TV, decodificadores o adicionales) en lugar de descuento",
    "falla_tecnica": "Escalar a soporte con prioridad y agendar visita con seguimiento de cierre",
    "competencia": "Contraoferta comparable solo si el valor del cliente lo justifica",
    "traslado_cobertura": "Validar cobertura en la nueva dirección y ofrecer traslado o alternativa móvil",
    "atencion_servicio": "Asignar caso con dueño único y cierre en primer contacto",
    "situacion_economica": "Ofrecer plan de menor valor o acuerdo de pago",
    "otro": "Revisión manual",
}


def classify_all(pre: pd.DataFrame) -> list[CallAnalysis]:
    return [classify_row(r) for _, r in pre.iterrows()]
