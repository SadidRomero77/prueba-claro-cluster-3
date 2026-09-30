"""Analizador de una llamada y copiloto del asesor (por texto, en tiempo real).

- analizar_llamada(texto): una transcripción completa → intención de cancelar, motivo, submotivo, urgencia,
  sentimiento (inicio → fin), cita textual, acción sugerida y la oferta del plan de accionables, con el contexto
  del Cluster 3. Usa el mismo método híbrido validado con 50 llamadas etiquetadas a mano:
  Jev (motivo, sentimiento, intención) + LLM con RAG (urgencia, submotivo, cita); sin claves responde con reglas.
- copiloto(turnos): acompaña al asesor mientras transcurre la llamada. Con cada turno actualiza el motivo probable,
  la urgencia y el sentimiento; si el motivo no está claro le propone al asesor una pregunta para identificarlo;
  sugiere la oferta y el guion, y alerta cuando hay que respetar la decisión de cancelar.

Decide una persona: el copiloto sugiere, no ejecuta ni ofrece nada por su cuenta.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import pandas as pd

from cluster3 import config
from cluster3 import llm as llm_factory
from cluster3.nlp import taxonomy as tx
from cluster3.nlp.baseline import classify_row
from cluster3.nlp.preprocess import preprocess_calls

PROMPT_VERSION = "copiloto-v1.1"
MIN_CARACTERES = 40
ROL_RE = re.compile(r"^\s*(cliente|client|usuario|asesor|agente|agent)\s*:\s*", re.I | re.M)
INTENCION_RE = r"(cancelar|dar de baja|retirar(?:me)? el servicio|terminar el contrato|dar por terminado|no quiero (?:seguir|continuar))"
INSISTE_RE = r"(igual (?:quiero|voy a) cancelar|no me interesa|proceda con la cancelaci|ya tom[ée] la decisi|quiero la baja ya|no,? gracias,? solo quiero cancelar)"

# Oferta sugerida por motivo, alineada con el plan de accionables (docs/07). Es una propuesta: el catálogo
# y la política de retención reales los define Claro.
OFERTAS = {
    "precio_facturacion": ("Revisar la factura con el cliente: si subió por fin de promoción, ajustar el plan a lo que "
                           "usa antes que descontar; sin descuento extra a cazadores de ofertas.", "Ajustar el plan"),
    "servicios_no_usados": ("Retirar la TV, los decodificadores o los adicionales que no usa y bajar la factura.",
                            "Ajustar el plan"),
    "falla_tecnica": ("Agendar visita técnica prioritaria con seguimiento; si ya hubo visitas fallidas, escalar a "
                      "segundo nivel.", None),
    "atencion_servicio": ("Tomar el caso de punta a punta en esta llamada; si hay un reclamo repetido, resolverlo antes "
                          "de hablar de ofertas.", "Resolver cobros no reconocidos"),
    "competencia": ("Comparar el valor total (velocidad, TV, móvil convergente) y ajustar el plan; no igualar precio "
                    "a ciegas.", "Oferta de retención por motivo"),
    "traslado_cobertura": ("Verificar cobertura en la nueva dirección; si no hay, gestionar la baja sin fricción.", None),
    "situacion_economica": ("Ofrecer un plan de menor valor que conserve lo que más usa (opciones a validar con Claro).",
                            "Ajustar el plan"),
    "otro": ("Hacer la pregunta sugerida para identificar el motivo antes de ofrecer algo.", None),
}
OFERTA_SUBMOTIVO = {
    "cobro_no_reconocido": ("Reversar el cobro no reconocido en esta misma llamada.", "Resolver cobros no reconocidos"),
    "adicionales_no_solicitados": ("Retirar el adicional que no pidió y reversar lo cobrado.", "Ajustar el plan"),
    "fin_promocion_descuento": ("Explicar el fin de la promoción y ajustar el plan a lo que usa, en lugar de otro "
                                "descuento.", "Ajustar el plan"),
}
# Pregunta para confirmar un motivo, y preguntas para distinguir motivos que se confunden.
PREGUNTAS = {
    "precio_facturacion": "¿Lo que le preocupa es un aumento reciente en la factura o un cobro que no reconoce?",
    "servicios_no_usados": "¿Hay servicios o equipos (TV, decodificadores, adicionales) que no esté usando?",
    "falla_tecnica": "¿El servicio se cae o está lento? ¿Desde cuándo y ya tuvo visita técnica?",
    "atencion_servicio": "¿Ya había llamado antes por esto? ¿Qué le dijeron?",
    "competencia": "¿Tiene una oferta de otro operador? ¿Qué le ofrecen?",
    "traslado_cobertura": "¿Se va a mudar? ¿Tiene la nueva dirección para revisar cobertura?",
    "situacion_economica": "¿Necesita bajar el valor que paga por su situación actual?",
    "otro": "¿Me cuenta qué lo lleva a querer cancelar el servicio?",
}
DISTINGUIR = {
    frozenset({"precio_facturacion", "servicios_no_usados"}):
        "¿El problema es que la factura subió o que está pagando por cosas que no usa?",
    frozenset({"precio_facturacion", "situacion_economica"}):
        "¿Siente que el servicio está caro para lo que recibe, o necesita reducir gastos en general?",
    frozenset({"falla_tecnica", "atencion_servicio"}):
        "¿Lo que más le molesta es la falla del servicio o cómo lo han atendido cuando reclamó?",
    frozenset({"precio_facturacion", "competencia"}):
        "¿Le hicieron una oferta en otro operador o es el precio actual lo que no le cuadra?",
}
UMBRAL_CONFIANZA = 0.55  # por debajo, el copiloto pide una pregunta para confirmar el motivo


# --------------------------------------------------------------------------------------------- utilidades
def normalizar(texto: str) -> str:
    """Convierte 'Cliente:' / 'Asesor:' en CLIENT/AGENT. Sin marcas, todo el texto se toma como del cliente."""
    t = texto.strip()
    if not ROL_RE.search(t) and not re.search(r"^(AGENT|CLIENT):", t, re.M):
        return f"CLIENT: {t}"
    return ROL_RE.sub(lambda m: "CLIENT: " if m.group(1).lower() in ("cliente", "client", "usuario") else "AGENT: ", t)


def preparar(texto: str) -> pd.Series:
    row = preprocess_calls(pd.DataFrame([{"ID": 0, "cluster": config.CLUSTER_CRITICO,
                                          "transcription": normalizar(texto)}])).iloc[0].copy()
    # El filtro de "sin contenido" (menos de 1.000 caracteres) es para el lote; aquí se analiza lo que haya.
    if row["calidad_transcripcion"] == "sin_contenido" and len(row["texto_cliente"]) >= MIN_CARACTERES:
        row["calidad_transcripcion"] = "baja"
    return row


@lru_cache(maxsize=1)
def _contexto_c3() -> dict:
    """Peso del motivo, retención y urgencia en las llamadas del Cluster 3 (para dar contexto al asesor)."""
    out = {}
    f = config.OUT_TAB / "nlp_sentimiento_por_motivo_c3.csv"
    if f.exists():
        t = pd.read_csv(f)
        tot = t["llamadas"].sum()
        for _, r in t.iterrows():
            out[r["motivo"]] = {"pct_llamadas_c3": round(float(100 * r["llamadas"] / tot), 1),
                                "retenido_pct_c3": round(float(r["retenido_pct"]), 1),
                                "urgencia_alta_pct_c3": round(float(r["urgencia_alta_pct"]), 1)}
    return out


@lru_cache(maxsize=1)
def _ids_accionables() -> dict:
    f = config.OUT_TAB / "accionables.csv"
    return dict(zip(pd.read_csv(f)["accionable"], pd.read_csv(f)["id"])) if f.exists() else {}


def _accionable(clave: str | None) -> str | None:
    if not clave:
        return None
    for nombre, i in _ids_accionables().items():
        if nombre.startswith(clave):
            return f"{i} · {nombre}"
    return None


@lru_cache(maxsize=1)
def _jev():
    from cluster3.nlp.jev_classifier import JevClassifier

    return JevClassifier()


@lru_cache(maxsize=1)
def _llm():
    from cluster3.nlp.llm_classifier import LLMClassifier

    pre = pd.read_parquet(config.T_LLAMADAS_LIMPIAS) if config.T_LLAMADAS_LIMPIAS.exists() else pd.DataFrame(
        columns=["id_llamada", "texto_cliente"])
    return LLMClassifier(pre)  # el RAG usa las llamadas etiquetadas a mano como ejemplos


def jev_disponible() -> bool:
    return bool(config.TYPESAFE_API_KEY)


# --------------------------------------------------------------------------------------------- análisis
def analizar_llamada(texto: str, usar_llm: bool | None = None, usar_jev: bool | None = None) -> dict:
    """Analiza una transcripción. usar_llm/usar_jev en None = se usan si hay clave."""
    if len((texto or "").strip()) < MIN_CARACTERES:
        raise ValueError(f"La llamada es muy corta: se necesitan al menos {MIN_CARACTERES} caracteres.")
    usar_llm = llm_factory.llm_available() if usar_llm is None else usar_llm
    usar_jev = jev_disponible() if usar_jev is None else usar_jev
    row = preparar(texto)
    base = classify_row(row).model_dump(by_alias=True)
    s = base["sentimiento"]
    r = {
        "motivo": base["motivo"], "submotivo": base["submotivo"], "urgencia": base["urgencia"],
        "sentimiento_inicio": s["inicio"], "sentimiento_fin": s["fin"], "sentimiento_global": s["global"],
        "emociones": s["emociones"], "evidencia": base["evidencia"], "reincidencia_mencionada": base["reincidencia_mencionada"],
        "competidor_mencionado": base["competidor_mencionado"], "accion_sugerida": base["accion_sugerida"],
        "resultado": base["resultado"], "motivo_probabilidades": None,
        "intencion_cancelar_prob": 0.9 if re.search(INTENCION_RE, row["texto_cliente"], re.I) else 0.2,
        "fuentes": {"motivo": "reglas", "urgencia": "reglas", "sentimiento": "reglas (léxico)", "intencion": "reglas"},
        "evidencia_verificada": None, "errores": [],
    }
    if usar_jev:
        try:
            j = _jev().classify(row)
            r.update({"motivo": j["jev_motivo"], "motivo_probabilidades": j["jev_motivo_probs"],
                      "motivo_confianza": j["jev_motivo_prob"], "sentimiento_global": j["jev_sentimiento"]})
            if j.get("jev_intencion_prob") is not None:
                r["intencion_cancelar_prob"] = float(j["jev_intencion_prob"])
                r["fuentes"]["intencion"] = "jev"
            r["fuentes"].update({"motivo": "jev", "sentimiento": "jev (global)"})
            if r["submotivo"] not in tx.MOTIVOS.get(r["motivo"], {}).get("submotivos", {}):
                r["submotivo"] = "no_identificado"  # el submotivo de reglas era de otro motivo
        except Exception as e:  # la API caída no tumba el análisis: sigue con lo que haya
            r["errores"].append(f"jev: {str(e)[:120]}")
    if usar_llm:
        try:
            ca, meta = _llm().classify(row)
            d = ca.model_dump(by_alias=True)
            submotivo = d["submotivo"] if d["submotivo"] in tx.MOTIVOS.get(r["motivo"], {}).get("submotivos", {}) \
                else "no_identificado"
            if not usar_jev or r["fuentes"]["motivo"] != "jev":
                r["motivo"] = d["motivo"]
                submotivo = d["submotivo"]
            r.update({"urgencia": d["urgencia"], "submotivo": submotivo, "evidencia": d["evidencia"],
                      "sentimiento_inicio": d["sentimiento"]["inicio"], "sentimiento_fin": d["sentimiento"]["fin"],
                      "emociones": d["sentimiento"]["emociones"], "accion_sugerida": d["accion_sugerida"],
                      "resultado": d["resultado"], "reincidencia_mencionada": d["reincidencia_mencionada"],
                      "evidencia_verificada": meta["evidencia_verificada"]})
            r["fuentes"].update({"urgencia": "llm", "submotivo": "llm", "evidencia": "llm (cita verificada)"})
        except Exception as e:
            r["errores"].append(f"llm: {str(e)[:120]}")
    return _enriquecer(r, row)


def _enriquecer(r: dict, row: pd.Series) -> dict:
    """Oferta sugerida, contexto del Cluster 3, pregunta para confirmar el motivo y alertas."""
    oferta, clave = OFERTA_SUBMOTIVO.get(r["submotivo"]) or OFERTAS.get(r["motivo"], OFERTAS["otro"])
    r["oferta_sugerida"] = oferta
    r["accionable_relacionado"] = _accionable(clave)
    r["contexto_c3"] = _contexto_c3().get(r["motivo"])
    r["pregunta_sugerida"] = _pregunta(r)
    alertas = []
    if re.search(INSISTE_RE, row["texto_cliente"], re.I):
        alertas.append("El cliente insiste en cancelar: respetar su decisión y gestionar la baja (es su derecho).")
    if r["urgencia"] == "alta":
        alertas.append("Urgencia alta: resolver en esta llamada, sin transferir.")
    if r.get("reincidencia_mencionada"):
        alertas.append("Ya había reclamado antes: no prometer lo mismo; dar una solución concreta y con fecha.")
    r["alertas"] = alertas
    r["texto_cliente_analizado"] = row["texto_cliente"][:2000]
    r["version_prompt"] = PROMPT_VERSION
    return r


def _pregunta(r: dict) -> str | None:
    probs = r.get("motivo_probabilidades") or {}
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:2]
    if r["motivo"] == "otro":
        return PREGUNTAS["otro"]
    if len(top) == 2 and top[0][1] - top[1][1] < 0.2:
        return DISTINGUIR.get(frozenset({top[0][0], top[1][0]}), PREGUNTAS.get(top[0][0]))
    confianza = r.get("motivo_confianza", top[0][1] if top else 0.5)
    return PREGUNTAS.get(r["motivo"]) if confianza < UMBRAL_CONFIANZA else None


# --------------------------------------------------------------------------------------------- copiloto
GUION_PROMPT = f"""Eres el copiloto de un asesor de retención de Claro Colombia (clientes del hogar, Cluster 3).
Con el análisis de la llamada en curso, escribe un guion breve (2 o 3 frases, en segunda persona hacia el cliente,
tono cálido y concreto, tratando al cliente de usted) que el asesor puede decir ahora. Devuelve solo el texto del
guion, sin títulos, comillas ni viñetas. Reglas: no prometas descuentos ni beneficios que no estén en
la oferta sugerida; si el cliente insiste en cancelar, el guion confirma que se gestionará la baja; no inventes datos
del cliente. Versión {PROMPT_VERSION}."""


def _limpiar_guion(t: str) -> str:
    """Quita títulos, negritas y comillas que el modelo a veces agrega alrededor del guion."""
    lineas = [x for x in t.strip().splitlines() if x.strip() and not re.match(r"^\s*(\*\*|#)", x)]
    return re.sub(r'^[\s"“”]+|[\s"“”]+$', "", " ".join(lineas)).replace("**", "")


def _guion_reglas(r: dict) -> str:
    if any("insiste en cancelar" in a for a in r["alertas"]):
        return "Entiendo su decisión y la respeto. Ya mismo le gestiono la cancelación y le confirmo los pasos."
    if r.get("pregunta_sugerida"):
        return f"Quiero entender bien su caso para ayudarle. {r['pregunta_sugerida']}"
    return f"Entiendo su molestia y quiero resolverlo hoy. {r['oferta_sugerida']}"


def copiloto(turnos: list[dict], usar_llm: bool | None = None, usar_jev: bool | None = None,
             modelo: str | None = None) -> dict:
    """turnos: [{"rol": "cliente" | "asesor", "texto": ...}] en orden. Devuelve el análisis y la guía para el asesor.

    Para responder rápido no usa el clasificador LLM completo: Jev (≈0,2 s) + reglas para el análisis y, si hay
    clave, un modelo rápido solo para el guion.
    """
    texto = "\n".join(f"{'CLIENT' if t['rol'] == 'cliente' else 'AGENT'}: {t['texto']}" for t in turnos if t.get("texto"))
    r = analizar_llamada(texto, usar_llm=False, usar_jev=usar_jev)
    usar_llm = llm_factory.llm_available() if usar_llm is None else usar_llm
    r["guion"], r["fuentes"]["guion"] = _guion_reglas(r), "reglas"
    insiste = any("insiste en cancelar" in a for a in r["alertas"])
    if insiste:
        # Guardarraíl: si el cliente insiste, no se intenta retener; el guion confirma la baja (plantilla fija).
        r["oferta_sugerida"] = "Gestionar la cancelación sin más ofertas. Si el cliente lo pide, recordarle las opciones."
        r["fuentes"]["guion"] = "reglas (el cliente insiste en cancelar)"
    if usar_llm and not insiste:
        try:
            resumen = {k: r[k] for k in ["motivo", "submotivo", "urgencia", "intencion_cancelar_prob", "oferta_sugerida",
                                         "pregunta_sugerida", "alertas"]}
            msg = f"Análisis: {json.dumps(resumen, ensure_ascii=False)}\n\nÚltimos turnos:\n{texto[-2500:]}"
            modelo_guion = modelo or llm_factory.model_name(rapido=True)
            guion = llm_factory.get_chat_model(modelo=modelo_guion, max_tokens=300).invoke(
                [("system", GUION_PROMPT), ("user", msg)]).content
            r["guion"] = _limpiar_guion(guion)
            r["fuentes"]["guion"] = f"llm ({modelo_guion})"
        except Exception as e:
            r["errores"].append(f"guion: {str(e)[:120]}")
    r["turnos"] = len(turnos)
    return r


def turnos_de_llamada(texto_anonimizado: str) -> list[dict]:
    """Parte una transcripción AGENT/CLIENT en turnos para simular el copiloto con una llamada real."""
    out = []
    for linea in texto_anonimizado.splitlines():
        m = re.match(r"^(AGENT|CLIENT):\s?(.*)$", linea)
        if m and m.group(2).strip():
            out.append({"rol": "cliente" if m.group(1) == "CLIENT" else "asesor", "texto": m.group(2).strip()})
    return out
