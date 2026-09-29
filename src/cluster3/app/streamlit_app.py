"""App de sustentación: dashboard del Cluster 3 + chat con el sistema multiagente + evaluación.

    uv run streamlit run src/cluster3/app/streamlit_app.py

Lee solo los artefactos que genera el pipeline (outputs/, data/processed/, models/).
Funciona sin clave de LLM (modo offline por reglas); con clave usa el modo llm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Permite ejecutar la app sin instalar el paquete (Databricks Apps, contenedores simples).
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402


@st.cache_resource(show_spinner="Descargando artefactos del Volume...")
def _pull_volume(volume: str, root: str) -> int:
    from cluster3.deploy.volume_sync import pull

    return pull(volume, root)


# Databricks Apps: los artefactos viven en un Volume de Unity Catalog; se descargan una vez por proceso.
if os.getenv("CLUSTER3_VOLUME"):
    os.environ.setdefault("CLUSTER3_ROOT", "/tmp/cluster3")
    _pull_volume(os.environ["CLUSTER3_VOLUME"], os.environ["CLUSTER3_ROOT"])

from cluster3 import config  # noqa: E402
from cluster3 import llm as llm_factory  # noqa: E402
from cluster3.eda.profile import _slug  # noqa: E402

st.set_page_config(page_title="Cluster 3 · Churn Claro", page_icon="📉", layout="wide")

BLUE, ORANGE = "#2a78d6", "#eb6834"


# --------------------------------------------------------------------------- carga
@st.cache_data(show_spinner=False)
def _json(name: str):
    p = config.OUT_JSON / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def _csv(name: str) -> pd.DataFrame | None:
    p = config.OUT_TAB / name
    return pd.read_csv(p) if p.exists() else None


def _fig(name: str, caption: str | None = None) -> None:
    p = config.OUT_FIG / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:,.{d}f} %".replace(",", "X").replace(".", ",").replace("X", ".")


def _cop(x: float) -> str:
    if abs(x) >= 1e6:
        return f"$ {x / 1e6:,.1f} M".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"$ {x:,.0f}".replace(",", ".")


def _d(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}".replace(".", ",")


def _n(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


kpis = _json("kpis_cluster3.json")
if kpis is None:
    st.error("No hay resultados todavía. Ejecuta primero el pipeline: `uv run cluster3`")
    st.stop()

metricas = _json("metricas_modelo.json") or {}
nlp = _json("nlp_insights.json") or {}

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Cluster 3 · Churn")
    st.caption("Prueba técnica Claro Colombia · periodo 202508")
    hay_llm = llm_factory.llm_available()
    opciones = ["llm", "offline"] if hay_llm else ["offline"]
    modo = st.radio("Modo del agente", opciones, index=0,
                    help="llm: agentes ReAct con herramientas. offline: enrutamiento y herramientas por reglas, "
                         "sin clave de API.")
    st.caption(f"Proveedor: `{config.LLM_PROVIDER}` · modelo: `{llm_factory.model_name()}`" if hay_llm
               else "Sin clave de LLM: el agente responde en modo offline.")
    st.divider()
    st.caption("Datos internos de Claro sin datos personales. Se eliminan al terminar el proceso.")

st.title("Cluster Crítico (Cluster 3) — churn e intención de cancelación")

tabs = st.tabs(["Resumen", "Segmentos", "Modelos", "Voz del cliente", "Accionables", "Agente", "Evaluación"])

# --------------------------------------------------------------------------- 1. Resumen
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Clientes", _n(kpis["clientes"]))
    c[1].metric(f"Churn del mes · {_n(kpis['churn_n'])} bajas", _pct(kpis["churn_tasa"], 2))
    c[2].metric(f"Intención de cancelar · {_n(kpis['intencion_n'])}", _pct(kpis["intencion_tasa"]))
    c[3].metric("Renta mensual total", _cop(kpis["renta_mensual_total"]))
    c[4].metric("Renta de quienes quieren irse", _cop(kpis["renta_mensual_intencion"]))

    c = st.columns(5)
    c[0].metric("ARPU mediano", _cop(kpis["arpu_mediano"]))
    c[1].metric("Churn si llamó a cancelar", _pct(kpis["churn_si_intencion"], 2))
    c[2].metric("Churn si no llamó", _pct(kpis["churn_no_intencion"], 2))
    c[3].metric("Convergentes fijo + móvil", _pct(kpis["pct_convergente"]))
    c[4].metric("Estratos 2 y 3", _pct(kpis["pct_estrato_2_3"]))

    st.subheader("Lo que hay que saber")
    a1 = _csv("accionables.csv")
    ints = metricas.get("intencion", {})
    ch = metricas.get("churn", {})
    bullets = [
        f"El **{_pct(kpis['intencion_tasa'])}** de los clientes llamó a cancelar, pero solo el "
        f"**{_pct(kpis['churn_tasa'], 2)}** se fue: la retención funciona, el problema es el volumen de llamadas.",
        f"Precio y facturación explican el **{_pct(nlp.get('motivos_c3_pct', {}).get('precio_facturacion', 0) / 100)}** "
        "de las llamadas de cancelación del Cluster 3.",
    ]
    fuga = _csv("comparacion_fuga.csv")
    if fuga is not None and len(fuga) >= 4:
        bullets.append("Las variables que ya contienen el resultado (estado de la cuenta, plan de TV con sufijo I, "
                       "reincidencias, reacciones de retención) se excluyeron: con ellas ambos modelos darían AUC 1,0 "
                       "sin servir para anticipar nada.")
    if a1 is not None and len(a1):
        top = a1.sort_values("impacto_anual_base_cop", ascending=False).iloc[0]
        bullets.append(f"Accionable de mayor impacto: **{top['accionable']}** "
                       f"({_cop(top['impacto_anual_base_cop'])} al año en el escenario base).")
    if ints:
        bullets.append(f"El modelo de intención ordena bien el riesgo (AUC **{_d(ints['auc_cv_media'], 3)}**); "
                       f"el 10 % de mayor riesgo concentra **{_d(ints['lift_10'], 1)}×** la tasa base.")
    if ch:
        bullets.append(f"El modelo de churn (AUC **{_d(ch['auc_cv_media'], 3)}**) pone en el primer decil casi todas "
                       "las bajas, validado fuera de muestra.")
    for b in bullets:
        st.markdown(f"- {b}")

# --------------------------------------------------------------------------- 2. Segmentos
with tabs[1]:
    seg = _csv("segmentos_cluster3.csv")
    if seg is not None:
        nombres = list(dict.fromkeys(seg["segmento"]))
        elegido = st.selectbox("Segmento", nombres,
                               index=nombres.index("Variación de renta vs 6 meses")
                               if "Variación de renta vs 6 meses" in nombres else 0)
        d = seg[seg["segmento"] == elegido].drop(columns="segmento")
        col1, col2 = st.columns([3, 2])
        with col1:
            _fig(f"segmento_{_slug(elegido)}.png")
        with col2:
            st.dataframe(d.rename(columns={"nivel": "Nivel", "clientes": "Clientes", "churn_pct": "Churn %",
                                           "intencion_pct": "Intención %", "arpu_mediano": "ARPU mediano"}),
                         hide_index=True, width="stretch")
        with st.expander("Todos los segmentos (tabla)"):
            st.dataframe(seg, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 3. Modelos
with tabs[2]:
    if metricas:
        m1, m2 = st.columns(2)
        for col, key, titulo in [(m1, "intencion", "Intención de cancelar"), (m2, "churn", "Churn")]:
            m = metricas.get(key)
            if not m:
                continue
            with col:
                st.subheader(titulo)
                c = st.columns(4)
                c[0].metric("AUC (CV 3×5)", _d(m['auc_cv_media']),
                            help=f"Desviación estándar entre folds: {_d(m['auc_cv_sd'], 3)}")
                c[1].metric("PR-AUC", _d(m['pr_auc']), help=f"Tasa base (PR-AUC de un modelo al azar): {_d(m['tasa_base'])}")
                c[2].metric("Lift decil 1", f"{_d(m['lift_10'], 1)}×")
                c[3].metric("Variables", m["n_variables"])
                ic = m.get("auc_oof_ic95")
                if ic:
                    st.caption(f"IC 95 % del AUC (bootstrap): {_d(ic[0], 3)} – {_d(ic[1], 3)}")
                _fig(f"shap_{key}.png")
                _fig(f"ganancia_{key}.png")
                for nota in m.get("notas", []):
                    st.caption(f"· {nota}")

        st.subheader("Por qué se excluyeron variables (fuga de información)")
        fuga = _csv("comparacion_fuga.csv")
        if fuga is not None:
            st.dataframe(fuga, hide_index=True, width="stretch")
        sens = metricas.get("sensibilidad_churn_sin_equipos")
        if sens:
            st.caption(f"Sensibilidad: churn sin equipos adicionales ni UltraWiFi → AUC {_d(sens['auc'], 3)}, "
                       f"lift decil 1 {_d(sens['lift_10'], 1)}×.")

        st.subheader("Balance de importancia entre los dos objetivos")
        bal = _csv("balance_importancia_targets.csv")
        if bal is not None:
            st.dataframe(bal, hide_index=True, width="stretch")
        cat1, cat2 = _csv("importancia_categoria_intencion.csv"), _csv("importancia_categoria_churn.csv")
        if cat1 is not None and cat2 is not None:
            comp = cat1.merge(cat2, on="categoria", how="outer", suffixes=(" intención", " churn")).fillna(0)
            comp = comp.set_index("categoria").sort_values("pct_importancia intención", ascending=False)
            st.caption("Importancia SHAP agregada por categoría del diccionario (%)")
            st.bar_chart(comp, horizontal=True, color=[BLUE, ORANGE], stack=False)

        with st.expander("Tablas de lift por decil"):
            for key in ["intencion", "churn"]:
                lt = _csv(f"lift_{key}.csv")
                if lt is not None:
                    st.markdown(f"**{key}**")
                    st.dataframe(lt, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 4. Voz del cliente
with tabs[3]:
    if nlp:
        c = st.columns(4)
        c[0].metric("Llamadas del Cluster 3", _n(nlp.get("llamadas_c3", 0)))
        c[1].metric("Urgencia alta", _pct(nlp.get('urgencia_c3_pct', {}).get('alta', 0) / 100))
        c[2].metric("Sentimiento empeora en la llamada", _pct(nlp.get('sentimiento_c3', {}).get('empeora_pct', 0) / 100))
        c[3].metric("Menciones a Tigo", _n(nlp.get("competidores_c3", {}).get("tigo", 0)))
        st.caption(f"Método: {nlp.get('metodo_final', 'baseline_reglas')} · "
                   f"preprocesamiento: {json.dumps(nlp.get('preprocesamiento', {}), ensure_ascii=False)}")
        col1, col2 = st.columns([3, 2])
        with col1:
            _fig("nlp_motivos_c3_vs_otros.png")
        with col2:
            mot = _csv("nlp_motivos_c3_vs_otros.csv")
            if mot is not None:
                st.dataframe(mot, hide_index=True, width="stretch")
        sub = _csv("nlp_submotivos_c3.csv")
        if sub is not None:
            st.subheader("Submotivos")
            st.dataframe(sub, hide_index=True, width="stretch")
        puente = _csv("puente_llamadas_dataset.csv")
        if puente is not None:
            st.subheader("Puente llamadas → dataset")
            st.caption("Relación agregada entre lo que dicen las llamadas y las variables del dataset "
                       "(no hay llave común a nivel cliente).")
            st.dataframe(puente, hide_index=True, width="stretch")
        if config.T_LLAMADAS_ANALISIS.exists():
            with st.expander("Explorar llamadas analizadas"):
                ll = pd.read_parquet(config.T_LLAMADAS_ANALISIS)
                cols = [c for c in ["id_llamada", "cluster", "motivo", "submotivo", "urgencia", "sent_inicio", "sent_fin",
                                    "competidor_mencionado", "resultado", "evidencia", "calidad_transcripcion"]
                        if c in ll.columns]
                motivo = st.selectbox("Motivo", ["(todos)"] + sorted(ll["motivo"].dropna().unique().tolist()))
                if motivo != "(todos)":
                    ll = ll[ll["motivo"] == motivo]
                st.dataframe(ll[cols], hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 5. Accionables
with tabs[4]:
    acc = _csv("accionables.csv")
    if acc is not None:
        col1, col2 = st.columns([2, 3])
        with col1:
            _fig("matriz_impacto_esfuerzo.png")
        with col2:
            tot = acc["impacto_anual_base_cop"].sum()
            st.metric("Impacto anual escenario base (suma de accionables)", _cop(tot))
            st.caption("Escenarios de éxito de retención: conservador 15 %, base 30 %, optimista 45 %; "
                       "horizonte 12 meses; costo de contacto $15.000.")
            vista = acc[["id", "prioridad", "accionable", "tipo", "n_objetivo", "impacto_anual_conservador_cop",
                         "impacto_anual_base_cop", "impacto_anual_optimista_cop"]].copy()
            for c_ in ["impacto_anual_conservador_cop", "impacto_anual_base_cop", "impacto_anual_optimista_cop"]:
                vista[c_] = vista[c_].map(_cop)
            st.dataframe(vista, hide_index=True, width="stretch")
        for _, r in acc.iterrows():
            with st.expander(f"{r['id']} · {r['accionable']}"):
                st.markdown(f"**Tipo:** {r['tipo']} · **Prioridad:** {r['prioridad']}")
                st.markdown(f"**Evidencia:** {r['evidencia']}")
                st.markdown(f"**Objetivo:** {r['objetivo']}")
                st.markdown(f"**Métrica de seguimiento:** {r['metrica']}")

# --------------------------------------------------------------------------- 6. Agente
with tabs[5]:
    from cluster3.agents.graph import ask, resume  # import tardío: carga el grafo solo si se usa

    st.caption("Pregunta en lenguaje natural. El orquestador elige especialistas (perfilado, voz del cliente, "
               "estrategia), un crítico verifica que cada cifra venga de una herramienta y las acciones que "
               "exportan datos piden aprobación humana.")
    ss = st.session_state
    ss.setdefault("chat", [])
    ss.setdefault("thread", None)
    ss.setdefault("pendiente", None)

    ejemplos = [
        "¿Cuál es la tasa de churn y de intención de cancelación del Cluster 3?",
        "¿Qué variables explican la intención de cancelación?",
        "¿Cuáles son los principales motivos de cancelación en las llamadas del Cluster 3?",
        "¿Qué accionables recomiendas y cuál es su impacto económico?",
        "Explica el riesgo de churn del cliente 15",
        "Genera la lista de contacto del decil de mayor riesgo",
    ]
    cols = st.columns(3)
    elegido = None
    for i, e in enumerate(ejemplos):
        if cols[i % 3].button(e, key=f"ej{i}", width="stretch"):
            elegido = e

    def _mostrar(r: dict, k: int = 0) -> None:
        st.markdown(r.get("respuesta") or "")
        texto = (r.get("respuesta") or "") + " ".join(r.get("evidencia") or [])
        if "Lista generada:" in texto:
            for f in sorted((config.OUT / "listas").glob("lista_contacto_*.csv")):
                st.download_button(f"Descargar {f.name}", f.read_bytes(), file_name=f.name, mime="text/csv",
                                   key=f"dl{k}{f.name}")
        crit = r.get("critica") or {}
        meta = f"Agentes: {', '.join(r.get('agentes') or [])}"
        if crit:
            meta += f" · Crítico: {'aprobado' if crit.get('aprobado') else 'con observaciones'}"
        st.caption(meta)
        with st.expander("Traza y evidencia"):
            if crit:
                st.json(crit)
            st.json(r.get("traza") or [])
            for ev in (r.get("evidencia") or [])[:8]:
                st.code(ev[:1500])

    for i, turno in enumerate(ss.chat):
        with st.chat_message(turno["rol"]):
            if turno["rol"] == "user":
                st.markdown(turno["texto"])
            else:
                _mostrar(turno["r"], i)

    if ss.pendiente:
        with st.chat_message("assistant"):
            st.warning("Esta acción exporta datos de clientes y necesita aprobación humana.")
            st.json(ss.pendiente)
            a, b = st.columns(2)
            if a.button("Aprobar", type="primary", width="stretch"):
                with st.spinner("Generando..."):
                    r = resume(ss.thread, aprobado=True)
                ss.pendiente = None
                ss.chat.append({"rol": "assistant", "r": r})
                st.rerun()
            if b.button("Rechazar", width="stretch"):
                with st.spinner("Cerrando..."):
                    r = resume(ss.thread, aprobado=False)
                ss.pendiente = None
                ss.chat.append({"rol": "assistant", "r": r})
                st.rerun()

    pregunta = st.chat_input("Pregunta sobre el Cluster 3") or elegido
    if pregunta and not ss.pendiente:
        ss.chat.append({"rol": "user", "texto": pregunta})
        with st.spinner("Consultando agentes..."):
            r = ask(pregunta, thread_id=None, modo=modo)
        ss.thread = r["thread_id"]
        if "interrupt" in r:
            ss.pendiente = r["interrupt"]
        else:
            ss.chat.append({"rol": "assistant", "r": r})
        st.rerun()

    if ss.chat and st.button("Nueva conversación"):
        ss.chat, ss.thread, ss.pendiente = [], None, None
        st.rerun()

# --------------------------------------------------------------------------- 7. Evaluación
with tabs[6]:
    st.subheader("Sistema multiagente — preguntas doradas")
    ev = _csv("agent_evals.csv")
    if ev is not None:
        res = ev[["ruta_ok", "cifras_respaldadas", "contenido_ok", "hitl_ok"]].mean()
        c = st.columns(4)
        for i, (k, v) in enumerate(res.items()):
            c[i].metric(k.replace("_", " ").capitalize(), _pct(v, 0))
        st.dataframe(ev, hide_index=True, width="stretch")
        st.caption("Regenerar: `uv run python -m cluster3.agents.evals` (añade juez LLM si hay clave).")
    st.subheader("Clasificación de llamadas — benchmark")
    bench = _csv("nlp_benchmark.csv")
    if bench is not None:
        st.dataframe(bench, hide_index=True, width="stretch")
    else:
        st.info("Pendiente: etiquetar data/labels/muestra_etiquetada.csv (50 llamadas) y correr "
                "`uv run python -m cluster3.nlp.run --llm --jev`.")
    st.subheader("Calidad de datos")
    cal = _json("calidad_datos.json")
    if cal:
        c = st.columns(4)
        c[0].metric("Filas", _n(cal.get("n_filas", 0)))
        c[1].metric("Columnas", _n(cal.get("n_columnas", 0)))
        c[2].metric("Duplicados", _n(cal.get("duplicados", 0) if not isinstance(cal.get("duplicados"), dict)
                                     else sum(v for v in cal["duplicados"].values() if isinstance(v, (int, float)))))
        c[3].metric("Constantes", _n(len(cal.get("constantes", []))))
        with st.expander("Sospechas de fuga y hallazgos de consistencia"):
            st.json({k: cal.get(k) for k in ["sospecha_fuga_churn", "sospecha_fuga_intencion", "consistencia"]},
                    expanded=False)
    reg = _csv("registro_decisiones_variables.csv")
    if reg is not None:
        with st.expander("Registro de decisiones por variable"):
            st.dataframe(reg, hide_index=True, width="stretch")
