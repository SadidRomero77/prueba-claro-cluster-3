"""Sistema multiagente en LangGraph.

    pregunta → orquestador ─(saludo o charla)→ conversación → fin
                         └→ especialistas (perfilado · voz_cliente · estrategia)
             → aprobación humana (si se pidió una herramienta sensible)
             → síntesis → crítico ──(falla y quedan intentos)──→ síntesis
                                  └─(aprueba)──→ fin

- Modo "llm": los especialistas son agentes ReAct con tool calling.
- Modo "offline": reglas deterministas; sirve para pruebas y como respaldo de la demo sin API.
- Autoevaluación: el crítico verifica que cada cifra de la respuesta exista en la evidencia
  de las herramientas (control determinista) y, en modo llm, pide un juicio adicional al LLM.
- Human-in-the-loop: generar_lista_contacto se detiene con interrupt() hasta que una persona apruebe.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from cluster3 import llm as llm_factory
from cluster3.agents import prompts as P
from cluster3.agents import tools as T

AGENTES = ["perfilado", "voz_cliente", "estrategia"]
SYSTEM = {"perfilado": P.PERFILADO, "voz_cliente": P.VOZ_CLIENTE, "estrategia": P.ESTRATEGIA}
MAX_INTENTOS = 2
PIDE_LISTA = r"lista de contacto|exporta|genera(?:r)? (?:la |una )?lista"


class State(TypedDict, total=False):
    pregunta: str
    pregunta_autonoma: str
    modelo: str | None
    historial: list[dict]
    modo: str
    agentes: list[str]
    motivo_ruta: str
    resultados: dict
    evidencia: list[str]
    pendiente: dict | None
    respuesta: str
    critica: dict
    intentos: int
    traza: list[dict]


class Ruta(BaseModel):
    agentes: list[Literal["conversacion", "perfilado", "voz_cliente", "estrategia"]] = Field(..., min_length=1)
    motivo: str
    pregunta_autonoma: str = Field("", description="La pregunta reescrita para que se entienda sin el historial")


class Veredicto(BaseModel):
    aprobado: bool
    observacion: str


# --------------------------------------------------------------- orquestador --
SALUDO = (r"^\s*(hola|buen[oa]s|hey|saludos|qu[ée] tal|gracias|muchas gracias|qui[ée]n eres|"
          r"qu[ée] (puedes|sabes) hacer|c[óo]mo (funcionas|me ayudas)|pres[ée]ntate|ayuda)\b")
RULES = {
    "voz_cliente": r"llamad|motivo|dicen|sentim|urgenc|queja|voz|transcrip|ejemplo|reclam|competid|\btigo\b",
    "estrategia": r"accion|estrateg|impacto|priori|recomiend|recomenda|qu[ée] hac|retener|retenci|lista|contact|plan de|ahorro|insight|hallazg|mejorar|oportunidad|identific|conclusi|aprendi",
    "perfilado": r"churn|intenci|modelo|variable|segment|cliente|tasa|arpu|riesgo|estrato|score|cu[áa]nt|shap|lift|auc|kpi|perfil|\bml\b|machine|predictiv|entren",
}


def route_rules(q: str) -> tuple[list[str], str]:
    ag = [a for a in AGENTES if re.search(RULES[a], q, re.I)]
    if not ag and re.search(SALUDO, q, re.I):
        return ["conversacion"], "saludo o charla (reglas)"
    return (ag or ["perfilado"]), "reglas por palabras clave"


def _historial_txt(state: State, n: int = 6) -> str:
    h = (state.get("historial") or [])[-n:]
    return "\n".join(f"{t['rol']}: {str(t['texto'])[:600]}" for t in h)


def orquestador(state: State) -> dict:
    t0 = time.time()
    q = state["pregunta"]
    autonoma = q
    if state.get("modo") == "llm":
        hist = _historial_txt(state)
        user = f"Historial reciente:\n{hist}\n\nPregunta actual: {q}" if hist else q
        r: Ruta = llm_factory.get_chat_model(rapido=True).with_structured_output(Ruta).invoke(
            [("system", P.ORQUESTADOR), ("user", user)])
        ag, motivo = list(dict.fromkeys(r.agentes)), r.motivo
        autonoma = r.pregunta_autonoma.strip() or q
        if len(ag) > 1:  # si además de charlar pide datos, responden los especialistas
            ag = [a for a in ag if a != "conversacion"]
    else:
        ag, motivo = route_rules(q)
    return {"agentes": ag, "motivo_ruta": motivo, "pregunta_autonoma": autonoma, "resultados": {}, "evidencia": [],
            "intentos": 0,
            "traza": state.get("traza", []) + [{"nodo": "orquestador", "agentes": ag, "ms": _ms(t0)}]}


# -------------------------------------------------------------- especialistas --
def _offline_calls(agente: str, q: str) -> list[tuple[str, dict]]:
    ql = q.lower()
    modelo = "churn" if "churn" in ql and "intenci" not in ql else "intencion"
    if agente == "perfilado":
        m = re.search(r"cliente\s*(?:id\s*)?(\d+)", ql)
        if m:
            return [("explicar_cliente", {"cliente_id": int(m.group(1)), "modelo": modelo})]
        if re.search(r"variable|importan|shap|explica|por qu[ée]|driver|factor", ql):
            return [("importancia_variables", {"modelo": modelo, "top": 8})]
        if re.search(r"auc|lift|m[ée]trica|desempe|matriz|precisi", ql):
            return [("metricas_modelo", {"modelo": modelo})]
        if re.search(r"modelo|\bml\b|machine|predictiv|entren", ql):
            return [("resumen_modelo", {})]
        return [("kpis_cluster", {})]
    if agente == "voz_cliente":
        calls = [("resumen_llamadas", {"cluster": 3})]
        if re.search(r"ejemplo|dicen|cita|busca", ql):
            calls.append(("buscar_llamadas", {"texto": q, "k": 3}))
        return calls
    if re.search(r"insight|hallazg|mejorar|oportunidad|identific|conclusi|aprendi", ql):
        calls = [("hallazgos_negocio", {})]
    else:
        calls = [("listar_accionables", {})]
    if re.search(PIDE_LISTA, ql):
        calls.append(("generar_lista_contacto", {"decil_max": 1, "limite": 500}))
    return calls


def _run_tool(name: str, args: dict) -> str:
    try:
        return str(T.ALL_TOOLS[name](**args))
    except Exception as e:
        return f"ERROR en {name}: {e}"


def especialistas(state: State) -> dict:
    q = state.get("pregunta_autonoma") or state["pregunta"]
    resultados, evidencia, traza = dict(state.get("resultados") or {}), list(state.get("evidencia") or []), list(state.get("traza") or [])
    pendiente = None

    def _uno(ag: str):
        t0 = time.time()
        if state.get("modo") == "llm":
            res, ev, tools_used, pend = _run_react(ag, q, state.get("modelo"))
        else:
            res, ev, tools_used, pend = [], [], [], None
            for name, args in _offline_calls(ag, q):
                if name in T.SENSIBLES:
                    pend = {"agente": ag, "herramienta": name, "args": args}
                    continue
                out = _run_tool(name, args)
                ev.append(f"[{name}] {out}")
                tools_used.append(name)
            res = "\n\n".join(ev)
        return ag, res, ev, tools_used, pend, _ms(t0)

    # Los especialistas son independientes: en modo llm trabajan en paralelo (la latencia es la del más lento).
    agentes = state["agentes"]
    if state.get("modo") == "llm" and len(agentes) > 1:
        with ThreadPoolExecutor(max_workers=len(agentes)) as ex:
            salidas = list(ex.map(_uno, agentes))
    else:
        salidas = [_uno(a) for a in agentes]
    for ag, res, ev, tools_used, pend, ms in salidas:
        resultados[ag] = {"respuesta": res if isinstance(res, str) else str(res), "herramientas": tools_used}
        evidencia += ev
        pendiente = pendiente or pend
        traza.append({"nodo": ag, "herramientas": tools_used, "ms": ms})
    # La aprobación humana no depende de que el LLM elija la herramienta: si se pide una lista, siempre pasa por HITL.
    if not pendiente and re.search(PIDE_LISTA, q, re.I):
        pendiente = {"agente": "estrategia", "herramienta": "generar_lista_contacto", "args": {"decil_max": 1, "limite": 500}}
    return {"resultados": resultados, "evidencia": evidencia, "pendiente": pendiente, "traza": traza}


def _run_react(agente: str, q: str, modelo: str | None = None):
    """Agente ReAct con tool calling. Las herramientas sensibles se interceptan para aprobación."""
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import create_react_agent

    pend: list[dict] = []

    def _wrap(fn):
        if fn.__name__ in T.SENSIBLES:
            def guarded(**kwargs):
                pend.append({"agente": agente, "herramienta": fn.__name__, "args": kwargs})
                return "PENDIENTE: esta acción requiere aprobación humana; informa al usuario que quedó en espera."
            return StructuredTool.from_function(guarded, name=fn.__name__, description=fn.__doc__,
                                                args_schema=StructuredTool.from_function(fn).args_schema)
        return StructuredTool.from_function(fn)

    tools = [_wrap(f) for f in T.TOOLS_POR_AGENTE[agente]]
    agent = create_react_agent(llm_factory.get_chat_model(modelo=modelo), tools, prompt=SYSTEM[agente])
    out = agent.invoke({"messages": [("user", q)]}, {"recursion_limit": 12})
    msgs = out["messages"]
    ev = [f"[{m.name}] {m.content}" for m in msgs if isinstance(m, ToolMessage)]
    used = [m.name for m in msgs if isinstance(m, ToolMessage)]
    return msgs[-1].content, ev, used, (pend[0] if pend else None)


# ---------------------------------------------------------------- conversación --
def conversacion(state: State) -> dict:
    """Saludo y presentación del equipo de agentes. No usa herramientas ni da cifras."""
    t0 = time.time()
    if state.get("modo") == "llm":
        hist = _historial_txt(state)
        user = f"Historial reciente:\n{hist}\n\nMensaje: {state['pregunta']}" if hist else state["pregunta"]
        resp = llm_factory.get_chat_model(modelo=state.get("modelo")).invoke([("system", P.CONVERSACION), ("user", user)]).content
    else:
        equipo = "\n".join(f"- **{n}**: {d}" for k, (n, d) in P.AGENTES_INFO.items() if k != "orquestador")
        resp = ("¡Hola! Soy el asistente del Cluster 3 de Claro Colombia. Coordino un equipo de agentes que analiza "
                "la intención de cancelación y el churn de este clúster:\n\n" + equipo +
                "\n\nPuedes preguntarme, por ejemplo: «¿Ya tienen el modelo de ML y qué tan bueno es?», "
                "«¿Qué insights de negocio identificaron?» o «¿Por qué llaman a cancelar los clientes?».")
    return {"respuesta": resp, "critica": {"aprobado": True, "observacion": "", "control": "sin cifras (conversación)"},
            "traza": (state.get("traza") or []) + [{"nodo": "conversacion", "ms": _ms(t0)}]}


def _after_orquestador(state: State) -> str:
    return "conversacion" if state.get("agentes") == ["conversacion"] else "especialistas"


# ------------------------------------------------------------------ aprobación --
def aprobacion(state: State) -> dict:
    p = state.get("pendiente")
    if not p:
        return {}
    decision = interrupt({"tipo": "aprobacion", "mensaje": f"El agente {p['agente']} quiere ejecutar {p['herramienta']}",
                          "herramienta": p["herramienta"], "args": p["args"]})
    # Se descarta el aviso provisional "PENDIENTE" del especialista: la evidencia refleja la decisión humana.
    ev = [e for e in (state.get("evidencia") or []) if not e.startswith(f"[{p['herramienta']}] PENDIENTE")]
    if decision is True or (isinstance(decision, dict) and decision.get("aprobado")):
        out = _run_tool(p["herramienta"], p["args"])
        ev.append(f"[{p['herramienta']}] {out}")
    else:
        ev.append(f"[{p['herramienta']}] RECHAZADO por la persona que revisa: la lista NO se generó ni se exportó.")
    return {"evidencia": ev, "pendiente": None,
            "traza": (state.get("traza") or []) + [{"nodo": "aprobacion", "aprobado": bool(decision)}]}


def _after_especialistas(state: State) -> str:
    if state.get("pendiente"):
        return "aprobacion"
    # Con un solo especialista en modo llm su respuesta ya es la final: se ahorra una llamada al modelo.
    if state.get("modo") == "llm" and len(state.get("agentes") or []) == 1:
        return "directo"
    return "sintesis"


def directo(state: State) -> dict:
    ag = state["agentes"][0]
    return {"respuesta": state["resultados"][ag]["respuesta"], "intentos": 1,
            "traza": (state.get("traza") or []) + [{"nodo": "respuesta_directa", "agente": ag}]}


# -------------------------------------------------------------------- síntesis --
def sintesis(state: State) -> dict:
    t0 = time.time()
    q = state["pregunta"]
    feedback = (state.get("critica") or {}).get("observacion", "") if state.get("intentos", 0) else ""
    if state.get("modo") == "llm":
        partes = "\n\n".join(f"### {a}\n{r['respuesta']}" for a, r in state["resultados"].items())
        ev = "\n".join(e[:900] for e in state.get("evidencia", [])[-6:])  # menos texto: síntesis más rápida
        hist = _historial_txt(state, 4)
        msg = (f"Historial reciente:\n{hist}\n\n" if hist else "") + \
            f"Pregunta: {q}\n\nRespuestas de especialistas:\n{partes}\n\nEvidencia de herramientas:\n{ev}"
        if feedback:
            msg += f"\n\nCORRECCIÓN DEL CRÍTICO: {feedback}"
        resp = llm_factory.get_chat_model(modelo=state.get("modelo")).invoke([("system", P.SINTESIS), ("user", msg)]).content
    else:
        resp = _offline_answer(state)
    return {"respuesta": resp, "intentos": state.get("intentos", 0) + 1,
            "traza": (state.get("traza") or []) + [{"nodo": "sintesis", "ms": _ms(t0)}]}


def _offline_answer(state: State) -> str:
    lines = [f"Respuesta generada sin LLM (modo offline) para: «{state['pregunta']}»", ""]
    for e in state.get("evidencia", []):
        name = e[1:e.index("]")] if e.startswith("[") else "fuente"
        body = e[e.index("]") + 1:].strip() if "]" in e else e
        lines.append(f"**Fuente [{name}]**")
        lines.append(_pretty(body))
        lines.append("")
    return "\n".join(lines).strip()


def _pretty(body: str, max_lines: int = 14) -> str:
    import io

    import pandas as pd

    try:
        obj = json.loads(body)
        if isinstance(obj, dict):
            return "\n".join(f"- {k}: {_num(v)}" for k, v in list(obj.items())[:max_lines])
    except Exception:
        pass
    rows = body.strip().splitlines()
    if len(rows) > 1 and "," in rows[0]:
        try:
            df = pd.read_csv(io.StringIO(body.split("\n(se muestran")[0])).head(max_lines)
            if df.shape[1] > 6:  # tablas anchas: conserva la primera columna de texto largo y quita las demás
                largas = [c for c in df.columns
                          if not pd.api.types.is_numeric_dtype(df[c]) and df[c].astype(str).str.len().mean() > 40]
                df = df.drop(columns=largas[1:])
            head = [str(c) for c in df.columns]
            md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            md += ["| " + " | ".join(_num(v) if isinstance(v, float) else _cut(str(v)) for v in r) + " |"
                   for r in df.itertuples(index=False)]
            return "\n".join(md)
        except Exception:
            pass
    return body[:1500]


def _num(v):
    if isinstance(v, float):
        return f"{v:,.0f}".replace(",", ".") if abs(v) >= 1000 else f"{v:.4g}"
    return v


def _cut(txt: str, n: int = 90) -> str:
    """Recorta texto largo en un límite de palabra (nunca parte una cifra)."""
    return txt if len(txt) <= n else txt[: txt.rfind(" ", 0, n)] + " …"


# --------------------------------------------------------------------- crítico --
NUM_TOKEN = re.compile(r"-?\d[\d.,]*(?:e[+-]?\d+)?", re.I)


def _interpretations(tok: str, csv: bool = True) -> set[float]:
    """Lecturas posibles de un texto numérico: 1.234,5 · 1,234.5 · 46,2 · 9.48e+04 · CSV '139,46.2'."""
    tok = tok.rstrip(".,")
    vals: set[float] = set()

    def add(x: str):
        try:
            vals.add(float(x))
        except ValueError:
            pass

    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", tok):
        add(tok.replace(".", "").replace(",", "."))
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", tok):
        add(tok.replace(",", ""))
    if re.fullmatch(r"-?\d+,\d+", tok):
        add(tok.replace(",", "."))
    if re.fullmatch(r"-?\d+(\.\d+)?(e[+-]?\d+)?", tok, re.I):
        add(tok)
    if csv and "," in tok:  # solo en la evidencia (salidas CSV de herramientas)
        for part in tok.split(","):
            if re.fullmatch(r"-?\d+(\.\d+)?(e[+-]?\d+)?", part, re.I):
                add(part)
    return vals


def numbers_supported(answer: str, evidence: list[str]) -> tuple[bool, list[str]]:
    """Cada cifra relevante de la respuesta debe estar en la evidencia (tolera %, redondeo, formatos y millones)."""
    ev: set[float] = set()
    for e in evidence:
        for tok in NUM_TOKEN.findall(e):
            for v in _interpretations(tok):
                ev |= {v, v * 100, v / 100, v / 1e6, v / 1e3}
    ev_list = [x for x in ev if x]
    faltan = []
    body = re.sub(r"\[[a-z_]+\]", "", answer)
    for tok in NUM_TOKEN.findall(body):
        todas = _interpretations(tok, csv=False)
        grandes = [v for v in todas if abs(v) > 10 and not (2020 <= v <= 2030) and v != 202508]
        if not grandes:  # cifras pequeñas (conteos cortos, porcentajes < 10) no se verifican
            continue
        # basta con que una lectura coincida: "0.016" puede ser 0,016 o 16 (miles con punto)
        if not any(abs(v - x) <= max(0.051 * abs(x), 0.051) for v in todas for x in ev_list):
            faltan.append(tok)
    return (not faltan), faltan


def critico(state: State) -> dict:
    t0 = time.time()
    ok, faltan = numbers_supported(state.get("respuesta", ""), state.get("evidencia", []))
    obs = "" if ok else f"Estas cifras no aparecen en la evidencia: {', '.join(faltan[:8])}. Quítalas o usa las de las herramientas."
    veredicto = {"aprobado": ok, "observacion": obs, "control": "cifras_vs_evidencia"}
    if ok and state.get("modo") == "llm":
        try:
            v: Veredicto = llm_factory.get_chat_model(rapido=True).with_structured_output(Veredicto).invoke([
                ("system", P.CRITICO),
                ("user", f"Pregunta: {state['pregunta']}\n\nRespuesta: {state['respuesta']}\n\n"
                         f"Evidencia: {' '.join(state.get('evidencia', []))[:12000]}")])
            veredicto = {"aprobado": v.aprobado, "observacion": v.observacion, "control": "cifras + juez LLM"}
        except Exception as e:
            veredicto["observacion"] = f"Juez LLM no disponible: {e}"
    return {"critica": veredicto,
            "traza": (state.get("traza") or []) + [{"nodo": "critico", **veredicto, "ms": _ms(t0)}]}


def _after_critico(state: State) -> str:
    if state["critica"]["aprobado"] or state.get("intentos", 0) >= MAX_INTENTOS:
        return END
    return "sintesis"


def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


# ------------------------------------------------------------------------ grafo --
def build_graph():
    g = StateGraph(State)
    g.add_node("orquestador", orquestador)
    g.add_node("conversacion", conversacion)
    g.add_node("especialistas", especialistas)
    g.add_node("aprobacion", aprobacion)
    g.add_node("sintesis", sintesis)
    g.add_node("directo", directo)
    g.add_node("critico", critico)
    g.add_edge(START, "orquestador")
    g.add_conditional_edges("orquestador", _after_orquestador, ["conversacion", "especialistas"])
    g.add_edge("conversacion", END)
    g.add_conditional_edges("especialistas", _after_especialistas, ["aprobacion", "sintesis", "directo"])
    g.add_edge("directo", "critico")
    g.add_edge("aprobacion", "sintesis")
    g.add_edge("sintesis", "critico")
    g.add_conditional_edges("critico", _after_critico, ["sintesis", END])
    return g.compile(checkpointer=MemorySaver())


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def default_mode() -> str:
    return "llm" if llm_factory.llm_available() else "offline"


_CACHE: dict[tuple[str, str, str], dict] = {}
NODOS_QUE_REDACTAN = {"sintesis", "conversacion"}


def _texto(chunk) -> str:
    c = chunk.content
    if isinstance(c, list):  # bloques de contenido (formato Anthropic)
        return "".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c or ""


def ask(pregunta: str, thread_id: str | None = None, modo: str | None = None,
        historial: list[dict] | None = None, usar_cache: bool = True,
        on_token=None, on_evento=None, modelo: str | None = None) -> dict:
    """Ejecuta el grafo. Si se detiene por aprobación, devuelve {'interrupt': ..., 'thread_id': ...}.

    modelo: LLM de los especialistas, la síntesis y la conversación (None = LLM_MODEL o el default del proveedor).

    historial: turnos previos [{"rol": "user" | "assistant", "texto": ...}] para preguntas de seguimiento.
    usar_cache: una pregunta repetida sin historial se responde desde memoria (nunca las que piden aprobación).
    on_token(texto): recibe la respuesta final parcial mientras se escribe (streaming). Es una vista previa:
        la respuesta que vale es la que devuelve la función, ya revisada por el crítico.
    on_evento(nodo, datos): avisa cuando termina cada nodo del grafo (para mostrar el progreso).
    """
    t0 = time.time()
    modo = modo or default_mode()
    clave = (re.sub(r"\s+", " ", pregunta.strip().lower()), modo, modelo or "")
    if usar_cache and not historial and clave in _CACHE:
        return {**_CACHE[clave], "thread_id": thread_id or str(uuid.uuid4()), "desde_cache": True,
                "latencia_ms": _ms(t0)}
    thread_id = thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    entrada = {"pregunta": pregunta, "modo": modo, "modelo": modelo, "traza": [], "historial": historial or []}
    agentes: list[str] = []
    msg_id, parcial = None, ""
    for ns, tipo, data in get_graph().stream(entrada, cfg, stream_mode=["messages", "updates"], subgraphs=True):
        if tipo == "updates":
            if not ns:
                for nodo, valores in data.items():
                    if nodo == "orquestador" and isinstance(valores, dict):
                        agentes = valores.get("agentes") or []
                    if on_evento and not nodo.startswith("__"):
                        on_evento(nodo, valores)
            continue
        if on_token is None:
            continue
        chunk, meta = data
        nodo = meta.get("langgraph_node")
        if getattr(chunk, "tool_call_chunks", None):
            continue
        # Redactan la respuesta final: la síntesis, la conversación o el único especialista (su agente ReAct).
        es_final = (not ns and nodo in NODOS_QUE_REDACTAN) or \
            (ns and ns[0].startswith("especialistas") and nodo == "agent" and len(agentes) == 1)
        if not es_final:
            continue
        txt = _texto(chunk)
        if not txt:
            continue
        if chunk.id != msg_id:  # mensaje nuevo (por ejemplo, tras una herramienta o un reintento): se reinicia
            msg_id, parcial = chunk.id, ""
        parcial += txt
        on_token(parcial)
    snap = get_graph().get_state(cfg)
    if snap.interrupts:
        return {"thread_id": thread_id, "interrupt": snap.interrupts[0].value, "latencia_ms": _ms(t0)}
    r = _package(snap.values, thread_id) | {"latencia_ms": _ms(t0)}
    if usar_cache and not historial and (r.get("critica") or {}).get("aprobado"):
        _CACHE[clave] = r
    return r


def resume(thread_id: str, aprobado: bool) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    out = get_graph().invoke(Command(resume=aprobado), cfg)
    return _package(out, thread_id)


def _package(out: dict, thread_id: str) -> dict:
    if "__interrupt__" in out:
        return {"thread_id": thread_id, "interrupt": out["__interrupt__"][0].value}
    return {"thread_id": thread_id, "respuesta": out.get("respuesta"), "agentes": out.get("agentes"),
            "critica": out.get("critica"), "traza": out.get("traza"), "evidencia": out.get("evidencia"),
            "modelo": out.get("modelo")}


if __name__ == "__main__":
    import sys

    r = ask(" ".join(sys.argv[1:]) or "¿Cuál es la tasa de churn e intención del Cluster 3?")
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str)[:4000])
