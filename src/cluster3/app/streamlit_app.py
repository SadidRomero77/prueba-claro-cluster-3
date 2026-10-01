"""App de sustentación: dashboard del Cluster 3 + chat con el sistema multiagente + evaluación.

    uv run streamlit run src/cluster3/app/streamlit_app.py

Lee solo los artefactos que genera el pipeline (outputs/, data/processed/, models/).
Funciona sin clave de LLM (modo offline por reglas); con clave usa el modo llm.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

# Permite ejecutar la app sin instalar el paquete (Databricks Apps, contenedores simples).
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import os  # noqa: E402

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
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
from cluster3.agents import prompts as P  # noqa: E402
from cluster3.eda.profile import BLUE, GRID, INK, INK2, ORANGE  # noqa: E402

st.set_page_config(page_title="Cluster 3 · Churn Claro", page_icon=":material/insights:", layout="wide")

# --------------------------------------------------------------------------- estilo
# Tokens de la paleta del proyecto (docs/CLAUDE.md). El texto usa tinta neutra; el color identifica series.
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=block');
:root {{
  --ink: #1b2430; --ink2: #4a5563; --muted: #6b7685; --grid: #e3e8ef; --surface: #ffffff; --bg: #f4f6f9;
  --brand: #1f5fbf; --brand-dark: #174a96; --brand-soft: #eaf1fb; --blue: {BLUE};
  --orange: {ORANGE}; --orange-dark: #a84113; --orange-soft: #fdf0ea; --good: #1f7a4d; --good-soft: #e7f4ec;
  --warn: #8a5a00; --warn-soft: #fbf2df;
}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp td, .stApp th,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp span:not([data-testid*="Icon"]):not(.msr) {{ font-family: 'IBM Plex Sans', sans-serif; }}
.stApp {{ background: var(--bg); color: var(--ink); }}
.block-container {{ padding-top: 3rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; color: var(--ink); }}
.msr {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal; line-height: 1;
  letter-spacing: normal; text-transform: none; white-space: nowrap; direction: ltr; -webkit-font-smoothing: antialiased;
  font-size: 22px; vertical-align: middle; }}

.hero {{ background: var(--surface); border: 1px solid var(--grid); border-left: 6px solid var(--brand);
  border-radius: 12px; padding: 22px 28px 18px; margin-bottom: 18px; }}
.hero .eyebrow {{ font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--brand); font-weight: 600; }}
.hero h1 {{ color: var(--ink); font-size: 28px; font-weight: 600; margin: 6px 0 6px; line-height: 1.2; }}
.hero p {{ color: var(--ink2); font-size: 15px; margin: 0 0 12px; max-width: 900px; }}
.chip {{ display: inline-block; font-size: 12.5px; font-weight: 500; padding: 4px 10px; border-radius: 6px;
  margin: 0 6px 6px 0; background: var(--brand-soft); color: var(--brand-dark); border: 1px solid #d3e2f5; }}

.kpi {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 10px; padding: 14px 16px 12px;
  height: 100%; min-height: 118px; position: relative; overflow: hidden; }}
.kpi::before {{ content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3px; background: var(--grid); }}
.kpi.blue::before {{ background: var(--brand); }} .kpi.orange::before {{ background: var(--orange); }}
.kpi .lbl {{ font-size: 12.5px; color: var(--muted); font-weight: 500; text-transform: none; }}
.kpi .val {{ font-size: 25px; font-weight: 600; color: var(--ink); margin-top: 6px; line-height: 1.1; }}
.kpi .sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}

.sec {{ margin: 24px 0 10px; }}
.sec .t {{ font-size: 18px; font-weight: 600; color: var(--ink); }}
.sec .s {{ font-size: 13.5px; color: var(--muted); margin-top: 2px; }}

.card {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 10px; padding: 16px 18px; height: 100%; }}
.card .ic {{ color: var(--brand); }} .card .ic .msr {{ font-size: 22px; }}
.card .ct {{ font-weight: 600; font-size: 14.5px; margin: 6px 0 4px; color: var(--ink); }}
.card .cx {{ font-size: 13.5px; color: var(--ink2); line-height: 1.5; }}

.agent {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 10px; padding: 14px 15px;
  height: 100%; min-height: 200px; }}
.agent .ic {{ color: var(--brand); }}
.agent .an {{ font-weight: 600; font-size: 14px; color: var(--ink); margin: 6px 0 3px; }}
.agent .ad {{ font-size: 12.5px; color: var(--ink2); line-height: 1.45; }}
.agent .at {{ font-size: 11.5px; color: var(--muted); margin-top: 8px; }}

.pill {{ display: inline-block; font-size: 12px; font-weight: 500; padding: 3px 9px; border-radius: 6px;
  margin: 0 5px 4px 0; border: 1px solid transparent; }}
.pill.blue {{ background: var(--brand-soft); color: var(--brand-dark); border-color: #d3e2f5; }}
.pill.orange {{ background: var(--orange-soft); color: var(--orange-dark); border-color: #f5d3c3; }}
.pill.good {{ background: var(--good-soft); color: var(--good); border-color: #c9e6d4; }}
.pill.warn {{ background: var(--warn-soft); color: var(--warn); border-color: #efdcb0; }}
.pill.gray {{ background: #eef1f5; color: var(--ink2); border-color: var(--grid); }}

.stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid var(--grid); }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; font-weight: 500; color: var(--ink2); }}
.stTabs [aria-selected="true"] {{ color: var(--brand); }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--brand); }}
[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--grid); }}
[data-testid="stChatMessage"] {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 10px;
  padding: 10px 14px; margin-bottom: 8px; }}
[data-testid="stMetricValue"] {{ font-weight: 600; }}
div[data-testid="stExpander"] details {{ border-radius: 10px; border-color: var(--grid); background: var(--surface); }}
.stButton button {{ border-radius: 8px; border-color: var(--grid); }}

.guide {{ background: var(--surface); border: 1px solid var(--grid); border-left: 4px solid var(--brand);
  border-radius: 10px; padding: 12px 16px; margin: 4px 0 14px; }}
.guide .gt {{ font-weight: 600; font-size: 13.5px; color: var(--brand-dark); margin-bottom: 4px; }}
.guide .gt .msr {{ font-size: 19px; margin-right: 4px; }}
.guide .gx {{ font-size: 13.5px; color: var(--ink); line-height: 1.55; }}
.guide .gl {{ font-size: 12.5px; color: var(--ink2); margin-top: 6px; }}
.guide.alerta {{ border-left-color: var(--orange); }} .guide.alerta .gt {{ color: var(--orange-dark); }}
.lectura {{ font-size: 14px; color: var(--ink); background: var(--surface); border: 1px solid var(--grid);
  border-left: 4px solid var(--orange); border-radius: 10px; padding: 10px 14px; margin: 6px 0 10px; }}
.quote {{ border-left: 3px solid var(--brand); background: var(--surface); padding: 8px 12px; margin: 6px 0;
  border-radius: 0 8px 8px 0; font-size: 13.5px; color: var(--ink); font-style: italic; }}
.quote .qm {{ font-style: normal; font-size: 11.5px; color: var(--muted); margin-top: 4px; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Íconos de línea (Material Symbols) en lugar de emojis: en HTML con ic(), en avatares con :material/…:
ICONO = {"conversacion": "forum", "perfilado": "query_stats", "voz_cliente": "record_voice_over", "estrategia": "flag",
         "critico": "fact_check", "orquestador": "hub"}
AVATAR = {k: f":material/{v}:" for k, v in ICONO.items()}
TONO_AGENTE = {"perfilado": "blue", "voz_cliente": "orange", "estrategia": "good", "conversacion": "gray"}


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


# --------------------------------------------------------------------------- componentes
def _e(x) -> str:
    return html.escape(str(x))


def ic(nombre: str) -> str:
    """Ícono de línea (Material Symbols) para usar dentro de HTML."""
    return f'<span class="msr">{nombre}</span>'


def section(titulo: str, sub: str | None = None) -> None:
    s = f'<div class="s">{_e(sub)}</div>' if sub else ""
    st.markdown(f'<div class="sec"><div class="t">{_e(titulo)}</div>{s}</div>', unsafe_allow_html=True)


def kpis_row(items: list[tuple[str, str, str, str]]) -> None:
    """items: (etiqueta, valor, subtítulo, tono blue|orange|'')"""
    cols = st.columns(len(items))
    for col, (lbl, val, sub, tono) in zip(cols, items):
        col.markdown(f'<div class="kpi {tono}"><div class="lbl">{_e(lbl)}</div><div class="val">{_e(val)}</div>'
                     f'<div class="sub">{_e(sub)}</div></div>', unsafe_allow_html=True)


def cards(items: list[tuple[str, str, str]], por_fila: int = 3) -> None:
    """items: (nombre de ícono Material, título, texto con **negritas**)"""
    for i in range(0, len(items), por_fila):
        cols = st.columns(por_fila)
        for col, (icono, t, x) in zip(cols, items[i:i + por_fila]):
            x = _e(x)
            while "**" in x:  # **texto** → <b>texto</b>
                x = x.replace("**", "<b>", 1).replace("**", "</b>", 1)
            col.markdown(f'<div class="card"><div class="ic">{ic(icono)}</div><div class="ct">{_e(t)}</div>'
                         f'<div class="cx">{x}</div></div>', unsafe_allow_html=True)
        st.write("")


def _md(x: str) -> str:
    """Escapa HTML y convierte **texto** en negrita."""
    x = _e(x)
    while "**" in x:
        x = x.replace("**", "<b>", 1).replace("**", "</b>", 1)
    return x


def guia(que: str, como: str | None = None) -> None:
    """Caja "Qué estás viendo" al inicio de cada pestaña: qué muestra y cómo leerlo."""
    extra = f'<div class="gl"><b>Cómo leerlo:</b> {_md(como)}</div>' if como else ""
    st.markdown(f'<div class="guide"><div class="gt">{ic("info")}Qué estás viendo</div><div class="gx">{_md(que)}</div>{extra}'
                "</div>", unsafe_allow_html=True)


def lectura(texto: str) -> None:
    """Conclusión en una frase, calculada con los datos que se están mostrando."""
    st.markdown(f'<div class="lectura">{_md(texto)}</div>', unsafe_allow_html=True)


def barras_segmento(d: pd.DataFrame, col: str, color: str, base: float, titulo: str) -> alt.Chart:
    """Barras por nivel del segmento con una línea en el promedio del Cluster 3 (niveles < 200 clientes, tenues)."""
    orden = d["nivel"].astype(str).tolist()
    datos = d.assign(nivel=d["nivel"].astype(str), confiable=d["clientes"] >= 200)
    barras = alt.Chart(datos).mark_bar(cornerRadiusEnd=4, color=color).encode(
        y=alt.Y("nivel:N", sort=orden, title=None,
                axis=alt.Axis(labelColor=INK2, ticks=False, domain=False, labelOverlap=False, labelLimit=160)),
        x=alt.X(f"{col}:Q", title=titulo, axis=alt.Axis(gridColor=GRID, labelColor=INK2, titleColor=INK2,
                                                        domain=False, ticks=False)),
        opacity=alt.condition("datum.confiable", alt.value(1), alt.value(0.35)),
        tooltip=[alt.Tooltip("nivel:N", title="Nivel"), alt.Tooltip(f"{col}:Q", title=titulo, format=".2f"),
                 alt.Tooltip("clientes:Q", title="Clientes", format=",")],
    )
    regla = alt.Chart(pd.DataFrame({"v": [base]})).mark_rule(color=INK2, strokeDash=[4, 3]).encode(
        x="v:Q", tooltip=[alt.Tooltip("v:Q", title="Promedio del Cluster 3", format=".2f")])
    return (barras + regla).properties(height=max(180, 44 * len(d))).configure_view(stroke=None) \
        .configure(background="transparent", font="IBM Plex Sans")


def curva_captura(lt: pd.DataFrame, decil: int, color: str, etiqueta: str) -> alt.Chart:
    """Captura acumulada por % de clientes contactados, frente a contactar al azar; resalta el punto elegido."""
    d = lt.assign(pct_contactado=lt["decil"] * 10, captura=lt["captura_acumulada"] * 100,
                  azar=lt["decil"] * 10, elegido=lt["decil"] == decil)
    ejes = dict(gridColor=GRID, labelColor=INK2, titleColor=INK2, domain=False, ticks=False)
    x = alt.X("pct_contactado:Q", title="% de clientes contactados (de mayor a menor riesgo)",
              scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(**ejes))
    y = alt.Y("captura:Q", title=f"% de {etiqueta} capturadas", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(**ejes))
    tt = [alt.Tooltip("pct_contactado:Q", title="% contactado"), alt.Tooltip("captura:Q", title="% capturado", format=".1f")]
    modelo = alt.Chart(d).mark_line(color=color, strokeWidth=2, point=alt.OverlayMarkDef(size=60, color=color)) \
        .encode(x=x, y=y, tooltip=tt)
    azar = alt.Chart(d).mark_line(color=INK2, strokeDash=[4, 3], strokeWidth=1.5).encode(x=x, y=alt.Y("azar:Q"))
    punto = alt.Chart(d[d["elegido"]]).mark_point(size=260, color=color, filled=False, strokeWidth=3) \
        .encode(x=x, y=y, tooltip=tt)
    return (azar + modelo + punto).properties(height=300).configure_view(stroke=None) \
        .configure(background="transparent", font="IBM Plex Sans")


def pill(texto: str, tono: str = "gray") -> str:
    return f'<span class="pill {tono}">{_e(texto)}</span>'


def barras_agrupadas(df: pd.DataFrame, cat: str, series: list[str], colores: list[str], titulo_x: str,
                     alto: int = 320) -> alt.Chart:
    """Barras horizontales agrupadas, una serie por color fijo, con tooltip."""
    largo = df.melt(id_vars=cat, value_vars=series, var_name="serie", value_name="valor")
    orden = df.sort_values(series[0], ascending=False)[cat].tolist()
    base = alt.Chart(largo).encode(
        y=alt.Y(f"{cat}:N", sort=orden, title=None, axis=alt.Axis(labelLimit=260, labelColor=INK2, ticks=False,
                                                                  domain=False)),
        yOffset=alt.YOffset("serie:N", sort=series),
        x=alt.X("valor:Q", title=titulo_x, axis=alt.Axis(gridColor=GRID, labelColor=INK2, titleColor=INK2,
                                                         domain=False, ticks=False)),
        color=alt.Color("serie:N", sort=series, scale=alt.Scale(domain=series, range=colores),
                        legend=alt.Legend(orient="top", title=None, labelColor=INK2)),
        tooltip=[alt.Tooltip(f"{cat}:N", title=cat), alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("valor:Q", title=titulo_x, format=".1f")],
    )
    return (base.mark_bar(cornerRadiusEnd=4, height={"band": 0.9})
            .properties(height=alto).configure_view(stroke=None).configure(background="transparent", font="IBM Plex Sans"))


def barras_simples(df: pd.DataFrame, x: str, y: str, color: str, titulo_x: str, titulo_y: str,
                   fmt: str = ".1f", alto: int = 260) -> alt.Chart:
    return (alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color)
            .encode(x=alt.X(f"{x}:O", title=titulo_x, axis=alt.Axis(labelAngle=0, labelColor=INK2, titleColor=INK2,
                                                                    ticks=False, domain=False)),
                    y=alt.Y(f"{y}:Q", title=titulo_y, axis=alt.Axis(gridColor=GRID, labelColor=INK2,
                                                                    titleColor=INK2, domain=False, ticks=False)),
                    tooltip=[alt.Tooltip(f"{x}:O", title=titulo_x), alt.Tooltip(f"{y}:Q", title=titulo_y, format=fmt)])
            .properties(height=alto).configure_view(stroke=None).configure(background="transparent", font="IBM Plex Sans"))


# --------------------------------------------------------------------------- datos
kpis = _json("kpis_cluster3.json")
if kpis is None:
    st.error("No hay resultados todavía. Ejecuta primero el pipeline: `uv run cluster3`")
    st.stop()

metricas = _json("metricas_modelo.json") or {}
nlp = _json("nlp_insights.json") or {}
ints, ch = metricas.get("intencion", {}), metricas.get("churn", {})
hay_llm = llm_factory.llm_available()
hay_jev = bool(config.TYPESAFE_API_KEY)

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Cluster 3 · Churn")
    st.caption("Prueba técnica Claro Colombia · periodo 202508")
    st.markdown(pill("LLM conectado" if hay_llm else "LLM sin clave", "good" if hay_llm else "warn")
                + pill("Jev conectado" if hay_jev else "Jev sin clave", "good" if hay_jev else "warn"),
                unsafe_allow_html=True)
    opciones = ["llm", "offline"] if hay_llm else ["offline"]
    modo = st.radio("Modo del agente", opciones, index=0, horizontal=True,
                    help="llm: agentes ReAct con herramientas. offline: enrutamiento y herramientas por reglas, "
                         "sin clave de API.")
    modelo_elegido = None
    if hay_llm and config.LLM_PROVIDER == "openrouter" and modo == "llm":
        opciones_m = list(llm_factory.MODELOS_OPENROUTER)
        por_defecto = llm_factory.model_name()
        modelo_elegido = st.selectbox(
            "Modelo de lenguaje", opciones_m,
            index=opciones_m.index(por_defecto) if por_defecto in opciones_m else 0,
            format_func=lambda m: llm_factory.MODELOS_OPENROUTER[m],
            help="Modelo de los especialistas, la síntesis y la conversación. El orquestador y el juez usan "
                 f"siempre el modelo rápido ({llm_factory.model_name(rapido=True)}).")
        st.caption(f"Proveedor `openrouter` · `{modelo_elegido}`")
    elif hay_llm:
        st.caption(f"Proveedor `{config.LLM_PROVIDER}` · modelo `{llm_factory.model_name()}`")
    st.caption(f"Prompts `{P.PROMPT_VERSION}`")
    st.divider()
    st.caption("Datos internos de Claro sin datos personales. Se eliminan al terminar el proceso.")

# --------------------------------------------------------------------------- portada
chips = [f"{_n(kpis['clientes'])} clientes", f"{_n(nlp.get('llamadas_c3', 0))} llamadas del Cluster 3"]
if ints:
    chips.append(f"Modelo de intención · AUC {_d(ints['auc_cv_media'])}")
if ch:
    chips.append(f"Modelo de churn · AUC {_d(ch['auc_cv_media'])}")
chips.append("Agentes con LLM" if hay_llm else "Agentes en modo offline")
st.markdown(
    '<div class="hero"><div class="eyebrow">Claro Colombia · Analítica avanzada</div>'
    "<h1>Cluster Crítico: intención de cancelación y churn</h1>"
    f"<p>El {_e(_pct(kpis['intencion_tasa']))} de los clientes pidió cancelar este mes y el "
    f"{_e(_pct(kpis['churn_tasa'], 2))} se fue. Qué los mueve, a quién contactar primero y cuánto vale hacerlo.</p>"
    + "".join(f'<span class="chip">{_e(c)}</span>' for c in chips) + "</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs([":material/dashboard: Resumen", ":material/groups: Segmentos", ":material/model_training: Modelo ML",
                ":material/record_voice_over: Voz del cliente", ":material/task_alt: Accionables",
                ":material/forum: Agente", ":material/support_agent: Copiloto", ":material/verified: Evaluación"])

# --------------------------------------------------------------------------- 1. Resumen
with tabs[0]:
    veces = kpis["churn_si_intencion"] / kpis["churn_no_intencion"] if kpis["churn_no_intencion"] else 0
    guia("La foto del Cluster 3 en el corte 202508: cuántos clientes piden cancelar, cuántos se van de verdad y cuánta "
         "renta está en juego. Abajo están los hallazgos clave y la ruta para recorrer el panel.",
         "**azul** = intención de cancelar (el cliente llamó a pedir la baja: alerta temprana). "
         "**naranja** = churn (el cliente efectivamente se fue).")
    kpis_row([
        ("Clientes del Cluster 3", _n(kpis["clientes"]), "periodo 202508", ""),
        ("Intención de cancelar", _pct(kpis["intencion_tasa"]), f"{_n(kpis['intencion_n'])} clientes", "blue"),
        ("Churn del mes", _pct(kpis["churn_tasa"], 2), f"{_n(kpis['churn_n'])} bajas", "orange"),
        ("Renta de quienes quieren irse", _cop(kpis["renta_mensual_intencion"]), "renta mensual", "blue"),
        ("ARPU mediano", _cop(kpis["arpu_mediano"]), "renta mensual por cliente", ""),
    ])
    st.write("")
    kpis_row([
        ("Churn si llamó a cancelar", _pct(kpis["churn_si_intencion"], 2),
         f"{_d(veces, 1)}× más que si no llamó ({_pct(kpis['churn_no_intencion'], 2)})", "orange"),
        ("Renta de los que se fueron", _cop(kpis["renta_mensual_churn"]), "renta mensual perdida", "orange"),
        ("Antigüedad mediana", f"{_n(kpis['antiguedad_mediana_meses'])} meses", "", ""),
        ("Convergentes fijo + móvil", _pct(kpis["pct_convergente"]), "", ""),
        ("Estratos 2 y 3", _pct(kpis["pct_estrato_2_3"]), "", ""),
    ])

    section("Lo que hay que saber", "Hallazgos principales, calculados desde los datos")
    a1 = _csv("accionables.csv")
    fuga = _csv("comparacion_fuga.csv")
    mc3 = nlp.get("motivos_c3_pct", {})
    hallazgos = [
        ("call", "Muchos amenazan, pocos se van",
         f"El **{_pct(kpis['intencion_tasa'])}** llamó a cancelar y solo el **{_pct(kpis['churn_tasa'], 2)}** se fue. "
         f"Quien llama tiene **{_d(veces, 1)}×** más riesgo de irse."),
        ("payments", "El precio es el motivo número uno",
         f"Precio y facturación explica el **{_pct(mc3.get('precio_facturacion', 0) / 100)}** de las llamadas del "
         f"Cluster 3; en el **{_pct(mc3.get('otro', 0) / 100)}** el cliente no deja claro el motivo."),
    ]
    if ch:
        top10 = ch.get("matriz_top10", {})
        hallazgos.append(("track_changes", "El modelo encuentra a los que se van",
                          f"El 10 % de mayor riesgo concentra **{_n(top10.get('VP', 0))} de {_n(ch['positivos'])}** bajas "
                          f"(lift **{_d(ch['lift_10'], 1)}×**), validado fuera de muestra."))
    if fuga is not None and len(fuga):
        auc_fuga = fuga.loc[fuga["variables"] == "con fuga", "auc"].max()
        hallazgos.append(("shield", "Se evitó la trampa de la fuga",
                          f"Con variables que ya contienen el resultado, el modelo daba AUC **{_d(auc_fuga, 2)}**. "
                          "Se excluyeron: estado de la cuenta, planes de TV con sufijo I, reincidencias y campañas."))
    if nlp.get("urgencia_c3_pct"):
        hallazgos.append(("replay", "Llaman con urgencia y ya habían reclamado",
                          f"El **{_pct(nlp['urgencia_c3_pct'].get('alta', 0) / 100)}** de las llamadas del Cluster 3 "
                          f"tiene urgencia alta y el **{_pct(nlp.get('reincidencia_c3_pct', 0) / 100)}** menciona un "
                          f"reclamo previo; se retiene al **{_pct(nlp.get('resultado_c3_pct', {}).get('retenido', 0) / 100)}**."))
    if a1 is not None and len(a1):
        top = a1.sort_values("impacto_anual_base_cop", ascending=False).iloc[0]
        hallazgos.append(("rocket_launch", "Primer paso recomendado",
                          f"**{top['id']} · {top['accionable']}**: {_cop(top['impacto_anual_base_cop'])} al año "
                          "en el escenario base."))
    cards(hallazgos)

    section("Cómo recorrer el panel", "Cada pestaña responde una pregunta de negocio")
    cards([
        ("groups", "Segmentos · ¿quiénes?", "Qué grupos de clientes tienen más intención y más churn."),
        ("model_training", "Modelo ML · ¿a quién llamar?", "Cuántas bajas se capturan contactando a pocos clientes."),
        ("record_voice_over", "Voz del cliente · ¿por qué?", "Motivos, urgencia y frases reales de las llamadas."),
        ("task_alt", "Accionables · ¿qué hacer?", "Acciones priorizadas y su impacto en pesos por escenario."),
        ("forum", "Agente · pregúntale", "Conversa con los datos; cada cifra viene de una herramienta."),
        ("support_agent", "Copiloto · en la llamada", "Analiza una llamada o conversa con el asesor: motivo, qué preguntar y qué ofrecer."),
        ("verified", "Evaluación · ¿es confiable?", "Cómo se validaron el NLP y el sistema de agentes."),
    ])

# --------------------------------------------------------------------------- 2. Segmentos
with tabs[1]:
    seg = _csv("segmentos_cluster3.csv")
    if seg is not None:
        guia("Cómo cambian la intención de cancelar y el churn según una característica del cliente (score crediticio, "
             "antigüedad, reclamos, red…). Sirve para saber **a quién** apuntar cada acción.",
             "cada barra es un grupo; la línea punteada es el promedio del Cluster 3. Si la barra la supera, ese grupo "
             "está en mayor riesgo. Las barras tenues tienen menos de 200 clientes: son poco confiables. "
             "Pasa el mouse para ver el detalle.")
        nombres = list(dict.fromkeys(seg["segmento"]))
        elegido = st.selectbox("Elige una característica", nombres,
                               index=nombres.index("Variación de renta vs 6 meses")
                               if "Variación de renta vs 6 meses" in nombres else 0)
        d = seg[seg["segmento"] == elegido].drop(columns="segmento")
        base_i, base_c = kpis["intencion_tasa"] * 100, kpis["churn_tasa"] * 100
        grandes = d[d["clientes"] >= 200]
        if len(grandes):
            top_i, top_c = grandes.loc[grandes["intencion_pct"].idxmax()], grandes.loc[grandes["churn_pct"].idxmax()]
            lectura(f"En **{elegido}**, el grupo **{top_i['nivel']}** tiene la mayor intención "
                    f"(**{_d(top_i['intencion_pct'], 1)} %** frente a {_d(base_i, 1)} % del Cluster 3) y el grupo "
                    f"**{top_c['nivel']}** el mayor churn (**{_d(top_c['churn_pct'], 2)} %** frente a "
                    f"{_d(base_c, 2)} %).")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Intención de cancelar (%)**")
            st.altair_chart(barras_segmento(d, "intencion_pct", BLUE, base_i, "Intención %"), width="stretch")
        with c2:
            st.markdown("**Churn (%)**")
            st.altair_chart(barras_segmento(d, "churn_pct", ORANGE, base_c, "Churn %"), width="stretch")
        with st.expander("Ver la tabla de este segmento"):
            st.dataframe(d.rename(columns={"nivel": "Nivel", "clientes": "Clientes", "churn_pct": "Churn %",
                                           "intencion_pct": "Intención %", "arpu_mediano": "ARPU mediano"}),
                         hide_index=True, width="stretch")
        with st.expander("Todos los segmentos (tabla)"):
            st.dataframe(seg, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 3. Modelo ML
with tabs[2]:
    if metricas:
        guia("Dos modelos de machine learning (LightGBM) que ordenan a los 20.000 clientes por riesgo: uno para la "
             "**intención de cancelar** (alerta temprana) y otro para el **churn** (baja efectiva). Se validaron con "
             "datos que el modelo no vio y sin variables que ya contienen la respuesta (fuga).",
             "**AUC** = qué tan bien ordena el riesgo (0,5 es azar, 1 es perfecto). **Lift** = cuántas veces más "
             "eventos hay en el grupo de mayor riesgo que al azar. Usa el simulador para ver cuántos casos capturas "
             "según a cuántos clientes contactes.")
        m1, m2 = st.columns(2)
        for col, key, titulo, tono in [(m1, "intencion", "Intención de cancelar · alerta temprana", "blue"),
                                       (m2, "churn", "Churn · baja efectiva", "orange")]:
            m = metricas.get(key)
            if not m:
                continue
            with col:
                st.markdown(f"**{titulo}**")
                ic95 = m.get("auc_oof_ic95")
                kpis_row([
                    ("AUC", _d(m["auc_cv_media"]), f"IC 95 %: {_d(ic95[0])}–{_d(ic95[1])}" if ic95 else "", tono),
                    ("PR-AUC", _d(m["pr_auc"]), f"azar: {_d(m['tasa_base'])}", tono),
                    ("Lift decil 1", f"{_d(m['lift_10'], 1)}×", f"{_n(m['positivos'])} casos reales", tono),
                ])

        section("Simulador: ¿a cuántos clientes contacto?",
                "Se contacta primero a quien el modelo ve más riesgoso; la línea punteada es contactar al azar")
        s1, s2 = st.columns([1, 2])
        with s1:
            cual = st.radio("Modelo", ["Churn", "Intención de cancelar"], horizontal=True)
            key = "churn" if cual == "Churn" else "intencion"
            pct = st.slider("% de clientes a contactar", 10, 100, 10, step=10)
        lt = _csv(f"lift_{key}.csv")
        if lt is not None:
            dec = pct // 10
            fila = lt[lt["decil"] == dec].iloc[0]
            contactados = int(lt.loc[lt["decil"] <= dec, "clientes"].sum())
            capturados = int(lt.loc[lt["decil"] <= dec, "positivos"].sum())
            total = int(lt["positivos"].sum())
            etiqueta = "bajas" if key == "churn" else "intenciones"
            with s1:
                kpis_row([("Clientes a contactar", _n(contactados), f"{pct} % del Cluster 3", "")])
                st.write("")
                kpis_row([(f"{etiqueta.capitalize()} capturadas", f"{_n(capturados)} de {_n(total)}",
                           f"{_pct(fila['captura_acumulada'], 0)} del total", "orange" if key == "churn" else "blue")])
            with s2:
                st.altair_chart(curva_captura(lt, dec, ORANGE if key == "churn" else BLUE, etiqueta), width="stretch")
            lectura(f"Contactando al **{pct} %** de mayor riesgo ({_n(contactados)} clientes) se llega al "
                    f"**{_pct(fila['captura_acumulada'], 0)}** de las {etiqueta}: **{_d(fila['lift_acumulado'], 1)}×** "
                    "más que contactar al azar.")

        section("Qué variables explican el riesgo", "SHAP: cuánto empuja cada variable el riesgo, promedio entre folds")
        st.caption("Barra más larga = variable más influyente. La etiqueta dice si un valor alto sube o baja el riesgo.")
        g1, g2 = st.columns(2)
        for col, key in [(g1, "intencion"), (g2, "churn")]:
            with col:
                _fig(f"shap_{key}.png")

        cat1, cat2 = _csv("importancia_categoria_intencion.csv"), _csv("importancia_categoria_churn.csv")
        if cat1 is not None and cat2 is not None:
            section("Qué explica el riesgo, por categoría", "Importancia SHAP agregada por categoría del diccionario (%)")
            comp = cat1.merge(cat2, on="categoria", how="outer", suffixes=(" intención", " churn")).fillna(0)
            comp = comp.rename(columns={"pct_importancia intención": "Intención", "pct_importancia churn": "Churn"})
            st.altair_chart(barras_agrupadas(comp, "categoria", ["Intención", "Churn"], [BLUE, ORANGE],
                                             "% de la importancia", alto=380), width="stretch")

        section("Por qué se excluyeron variables", "Con fuga de información las métricas son perfectas e inútiles")
        fuga = _csv("comparacion_fuga.csv")
        if fuga is not None:
            fila_f = fuga[fuga["variables"] == "con fuga"]
            if len(fila_f):
                lectura(f"Con las variables que ya contienen el resultado, el AUC llegaba a **{_d(fila_f['auc'].max(), 2)}**: "
                        "el modelo no anticipaba nada, solo copiaba la respuesta. Por eso se excluyeron.")
            st.dataframe(fuga, hide_index=True, width="stretch")
        with st.expander("Curvas de ganancia, notas y balance de importancia"):
            for key in ["intencion", "churn"]:
                _fig(f"ganancia_{key}.png")
                for nota in metricas.get(key, {}).get("notas", []):
                    st.caption(f"· {nota}")
            sens = metricas.get("sensibilidad_churn_sin_equipos")
            if sens:
                st.caption(f"Sensibilidad: churn sin equipos adicionales ni UltraWiFi → AUC {_d(sens['auc'], 3)}, "
                           f"lift decil 1 {_d(sens['lift_10'], 1)}×.")
            bal = _csv("balance_importancia_targets.csv")
            if bal is not None:
                st.dataframe(bal, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 4. Voz del cliente
with tabs[3]:
    if nlp:
        bench = _csv("nlp_benchmark.csv")
        precision = ""
        if bench is not None and "hibrido" in set(bench["metodo"]):
            b = bench[bench["metodo"] == "hibrido"].set_index("campo")
            precision = (f" El método se validó contra {_n(b['n'].max())} llamadas etiquetadas a mano: acierta el "
                         f"motivo en el **{_pct(b.loc['motivo', 'exactitud'], 0)}** y la urgencia en el "
                         f"**{_pct(b.loc['urgencia', 'exactitud'], 0)}** de los casos.")
        guia("Qué dicen los clientes cuando llaman a cancelar. Cada una de las 500 llamadas se clasificó por motivo, "
             "urgencia y sentimiento con un método híbrido: Jev para motivo y sentimiento, y un LLM para urgencia, "
             "submotivo y la cita textual." + precision,
             "empieza por el gráfico de motivos y luego elige un motivo en el explorador para ver su urgencia, cómo "
             "termina la llamada y frases reales de clientes.")
        mc3 = nlp.get("motivos_c3_pct", {})
        kpis_row([
            ("Llamadas del Cluster 3", _n(nlp.get("llamadas_c3", 0)), f"de {_n(nlp.get('llamadas_validas', 0))} válidas", ""),
            ("Precio y facturación", _pct(mc3.get("precio_facturacion", 0) / 100), "motivo identificable n.º 1", "blue"),
            ("Urgencia alta", _pct(nlp.get("urgencia_c3_pct", {}).get("alta", 0) / 100), "pide la baja ya", "orange"),
            ("Reclamo previo mencionado", _pct(nlp.get("reincidencia_c3_pct", 0) / 100), "ya había reclamado", ""),
            ("Retenidos en la llamada", _pct(nlp.get("resultado_c3_pct", {}).get("retenido", 0) / 100),
             "aceptan una oferta", ""),
        ])
        mot = _csv("nlp_motivos_c3_vs_otros.csv")
        if mot is not None:
            section("¿Por qué llaman a cancelar?", "% de llamadas por motivo · Cluster 3 frente a los demás clústeres")
            st.altair_chart(barras_agrupadas(mot, "motivo", ["Cluster 3", "Otros clústeres"], [BLUE, ORANGE],
                                             "% de llamadas"), width="stretch")
            idx = mot.set_index("motivo")
            dif = idx.drop(index="otro", errors="ignore")["diferencia_pp"].idxmax()
            lectura(f"Lo más característico del Cluster 3 frente al resto es **{dif}** "
                    f"({_d(idx.loc[dif, 'Cluster 3'], 1)} % vs {_d(idx.loc[dif, 'Otros clústeres'], 1)} %).")

        if config.T_LLAMADAS_ANALISIS.exists():
            ll = pd.read_parquet(config.T_LLAMADAS_ANALISIS)
            c3 = ll[(ll["cluster"] == config.CLUSTER_CRITICO) & (ll["calidad_transcripcion"] != "sin_contenido")]
            section("Explorador por motivo", "Elige un motivo para ver cómo son esas llamadas en el Cluster 3")
            opciones_mot = c3["motivo"].value_counts().index.tolist()
            motivo = st.selectbox("Motivo", opciones_mot,
                                  index=opciones_mot.index("precio_facturacion") if "precio_facturacion" in opciones_mot else 0)
            x = c3[c3["motivo"] == motivo]
            kpis_row([
                ("Llamadas", _n(len(x)), f"{_pct(len(x) / len(c3))} del Cluster 3", "blue"),
                ("Urgencia alta", _pct((x["urgencia"] == "alta").mean()), "", "orange"),
                ("Retenidos", _pct((x["resultado"] == "retenido").mean()), "", ""),
                ("Sentimiento inicio → fin", f"{_d(x['sent_inicio'].mean(), 2)} → {_d(x['sent_fin'].mean(), 2)}",
                 "de −1 (muy negativo) a 1", ""),
            ])
            e1, e2 = st.columns([1, 1])
            with e1:
                subm = x["submotivo"].value_counts().rename_axis("submotivo").reset_index(name="llamadas")
                if len(subm):
                    st.markdown("**Submotivos**")
                    st.altair_chart(barras_simples(subm, "submotivo", "llamadas", BLUE, "Submotivo", "Llamadas", ".0f",
                                                   alto=240), width="stretch")
            with e2:
                st.markdown("**Lo que dicen los clientes**")
                citas = x[(x["confianza"] >= 0.6) & (x["evidencia"].str.len() >= 60)] \
                    .sort_values("confianza", ascending=False).head(3)
                if len(citas):
                    for _, r in citas.iterrows():
                        st.markdown(f'<div class="quote">“{_e(r["evidencia"][:280])}”<div class="qm">llamada '
                                    f'{r["id_llamada"]} · urgencia {_e(r["urgencia"])} · {_e(r["resultado"])}</div></div>',
                                    unsafe_allow_html=True)
                else:
                    st.caption("No hay citas con confianza suficiente para este motivo.")
            with st.expander("Ver todas las llamadas analizadas"):
                cols = [c for c in ["id_llamada", "cluster", "motivo", "submotivo", "urgencia", "sent_inicio",
                                    "sent_fin", "resultado", "evidencia", "confianza", "calidad_transcripcion"]
                        if c in ll.columns]
                st.dataframe(ll[cols], hide_index=True, width="stretch")

        puente = _csv("puente_llamadas_dataset.csv")
        if puente is not None:
            with st.expander("Puente llamadas → dataset (cruce agregado: no hay llave común entre llamadas y clientes)"):
                st.dataframe(puente, hide_index=True, width="stretch")
        st.caption(f"Método: {nlp.get('metodo_final', 'baseline_reglas')} · "
                   f"preprocesamiento: {json.dumps(nlp.get('preprocesamiento', {}), ensure_ascii=False)}")

# --------------------------------------------------------------------------- 5. Accionables
with tabs[4]:
    acc = _csv("accionables.csv")
    if acc is not None:
        sup = (_json("accionables.json") or {}).get("supuestos", {})
        esc_tasas = sup.get("escenarios_tasa_exito", {})
        guia("Acciones concretas para el Cluster 3, priorizadas por impacto y esfuerzo. **Proactivas**: se hacen antes "
             "de que el cliente llame (por ejemplo, con la lista del modelo). **Reactivas**: se hacen cuando el cliente "
             "ya llamó.",
             f"el impacto es renta anual salvada neta del costo de contacto. Depende de qué tanto funcione la retención: "
             f"elige un escenario ({', '.join(f'{k} {_pct(v, 0)}' for k, v in esc_tasas.items())} de éxito). "
             "Un valor negativo significa que en ese escenario la acción cuesta más de lo que salva.")
        nombres_esc = {"conservador": "Conservador", "base": "Base", "optimista": "Optimista"}
        esc = st.radio("Escenario de éxito de retención", list(nombres_esc), index=1, horizontal=True,
                       format_func=lambda k: nombres_esc[k])
        colv = f"impacto_anual_{esc}_cop"
        kpis_row([
            (f"Impacto anual · escenario {esc}", _cop(acc[colv].sum()), "suma de accionables", "blue"),
            ("Accionables con impacto positivo", f"{_n((acc[colv] > 0).sum())} de {_n(acc[colv].notna().sum())}",
             "con impacto cuantificado", ""),
            ("Proactivos / reactivos", f"{_n((acc['tipo'] == 'Proactivo').sum())} / {_n((acc['tipo'] == 'Reactivo').sum())}",
             "", ""),
            ("Supuestos", f"{sup.get('meses', '')} meses", f"contacto ${_n(sup.get('costo_contacto_cop', 0))}", ""),
        ])
        imp = acc[acc[colv].notna()].assign(millones=lambda t: t[colv] / 1e6,
                                             nombre=lambda t: t["id"] + " · " + t["accionable"].str[:55])
        if len(imp):
            section("Impacto anual por accionable (millones de COP)")
            graf = alt.Chart(imp).mark_bar(cornerRadiusEnd=4).encode(
                y=alt.Y("nombre:N", sort="-x", title=None, axis=alt.Axis(labelLimit=380, labelColor=INK2, ticks=False,
                                                                        domain=False)),
                x=alt.X("millones:Q", title="Millones de COP al año",
                        axis=alt.Axis(gridColor=GRID, labelColor=INK2, titleColor=INK2, domain=False, ticks=False)),
                color=alt.Color("tipo:N", scale=alt.Scale(domain=["Proactivo", "Reactivo"], range=[BLUE, ORANGE]),
                                legend=alt.Legend(orient="top", title=None, labelColor=INK2)),
                tooltip=[alt.Tooltip("id:N"), alt.Tooltip("accionable:N"), alt.Tooltip("tipo:N"),
                         alt.Tooltip("millones:Q", title="Millones COP", format=",.1f"),
                         alt.Tooltip("n_objetivo:Q", title="Clientes objetivo", format=",.0f")],
            ).properties(height=48 * len(imp)).configure_view(stroke=None).configure(background="transparent", font="IBM Plex Sans")
            st.altair_chart(graf, width="stretch")
            mejor = imp.sort_values(colv, ascending=False).iloc[0]
            lectura(f"En el escenario **{esc}**, la acción de mayor impacto es **{mejor['id']} · {mejor['accionable']}** "
                    f"con **{_cop(mejor[colv])}** al año.")

        section("Detalle de cada accionable")
        col1, col2 = st.columns([2, 3])
        with col1:
            _fig("matriz_impacto_esfuerzo.png")
            st.caption("Arriba a la izquierda: alto impacto y bajo esfuerzo (quick wins).")
        with col2:
            for _, r in acc.iterrows():
                tono = "blue" if r["tipo"] == "Proactivo" else "orange"
                with st.container(border=True):
                    st.markdown(f"{pill(r['id'], 'gray')}{pill(r['tipo'], tono)}{pill(r['prioridad'], 'good')}"
                                f"<br><b>{_e(r['accionable'])}</b>", unsafe_allow_html=True)
                    valor = _cop(r[colv]) if pd.notna(r[colv]) else "no cuantificado"
                    st.caption(f"Impacto anual ({esc}): {valor} · objetivo: {r['objetivo']}")
                    with st.expander("Evidencia y métrica"):
                        st.markdown(f"**Evidencia:** {r['evidencia']}")
                        st.markdown(f"**Métrica de seguimiento:** {r['metrica']}")
        with st.expander("Tabla completa"):
            vista = acc[["id", "prioridad", "accionable", "tipo", "n_objetivo", "impacto_anual_conservador_cop",
                         "impacto_anual_base_cop", "impacto_anual_optimista_cop"]].copy()
            for c_ in ["impacto_anual_conservador_cop", "impacto_anual_base_cop", "impacto_anual_optimista_cop"]:
                vista[c_] = vista[c_].map(lambda v: _cop(v) if pd.notna(v) else "")
            st.dataframe(vista, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 6. Agente
with tabs[5]:
    from cluster3.agents.graph import ask, resume  # import tardío: carga el grafo solo si se usa
    from cluster3.agents.tools import TOOLS_POR_AGENTE

    guia("Un equipo de agentes de IA con el que puedes conversar sobre el Cluster 3. El **orquestador** decide qué "
         "especialista responde, cada especialista consulta sus **herramientas** (datos, modelo, llamadas) y el "
         "**crítico** revisa que cada cifra venga de una herramienta. Exportar una lista de clientes siempre pide tu "
         "**aprobación**.",
         "escribe como le hablarías a un analista (puedes saludar y hacer preguntas de seguimiento). Arriba de cada "
         "respuesta verás qué agente respondió, si las cifras quedaron verificadas, el modelo usado y el tiempo; en "
         "*Traza y evidencia* está lo que devolvió cada herramienta.")
    section("Tu equipo de agentes", "El orquestador elige quién responde; el crítico verifica cada cifra; "
                                    "exportar datos siempre pide aprobación humana")
    equipo = ["orquestador", "perfilado", "voz_cliente", "estrategia", "critico"]
    cols = st.columns(len(equipo))
    for col, k in zip(cols, equipo):
        nombre, desc = P.AGENTES_INFO[k]
        tools_k = [f.__name__ for f in TOOLS_POR_AGENTE.get(k, [])]
        extra = (f'<div class="at" title="{_e(", ".join(tools_k))}">{len(tools_k)} herramientas</div>'
                 if tools_k else "")
        col.markdown(f'<div class="agent"><div class="ic">{ic(ICONO[k])}</div><div class="an">{_e(nombre)}</div>'
                     f'<div class="ad">{_e(desc)}</div>{extra}</div>', unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("chat", [])
    ss.setdefault("thread", None)
    ss.setdefault("pendiente", None)

    section("Conversa con los datos")
    ejemplos = [
        "Hola, ¿quién eres y en qué me ayudas?",
        "¿Ya tienen el modelo de ML? ¿Qué tan bueno es?",
        "¿Qué insights de negocio identificaron y qué se puede mejorar?",
        "¿Cuáles son los principales motivos de cancelación en las llamadas?",
        "Explica el riesgo de churn del cliente 15",
        "Genera la lista de contacto del decil de mayor riesgo",
    ]
    cols = st.columns(3)
    elegido = None
    for i, e in enumerate(ejemplos):
        if cols[i % 3].button(e, key=f"ej{i}", width="stretch"):
            elegido = e

    def _avatar(r: dict) -> str:
        ag = r.get("agentes") or []
        return AVATAR.get(ag[0], AVATAR["orquestador"]) if len(ag) == 1 else AVATAR["orquestador"]

    def _mostrar(r: dict, k: int = 0) -> None:
        ag = r.get("agentes") or []
        crit = r.get("critica") or {}
        badges = "".join(pill(P.AGENTES_INFO.get(a, (a,))[0], TONO_AGENTE.get(a, "gray"))
                         for a in ag if a != "conversacion")
        if crit and ag != ["conversacion"]:
            badges += pill("Cifras verificadas" if crit.get("aprobado") else "Crítico con observaciones",
                           "good" if crit.get("aprobado") else "warn")
        if r.get("desde_cache"):
            badges += pill("En caché", "gray")
        elif r.get("latencia_ms"):
            badges += pill(f"{_d(r['latencia_ms'] / 1000, 1)} s", "gray")
        if r.get("modelo"):
            badges += pill(f"{llm_factory.MODELOS_OPENROUTER.get(r['modelo'], r['modelo']).split(' · ')[0]}", "gray")
        if badges:
            st.markdown(badges, unsafe_allow_html=True)
        st.markdown(r.get("respuesta") or "")
        texto = (r.get("respuesta") or "") + " ".join(r.get("evidencia") or [])
        if "Lista generada:" in texto:
            for f in sorted((config.OUT / "listas").glob("lista_contacto_*.csv")):
                st.download_button(f"Descargar {f.name}", f.read_bytes(), file_name=f.name, mime="text/csv",
                                   key=f"dl{k}{f.name}")
        if r.get("traza") and ag != ["conversacion"]:
            with st.expander("Traza y evidencia"):
                if crit:
                    st.json(crit)
                st.json(r.get("traza") or [])
                for ev in (r.get("evidencia") or [])[:8]:
                    st.code(ev[:1500])

    if not ss.chat and not ss.pendiente:
        with st.chat_message("assistant", avatar=AVATAR["conversacion"]):
            st.markdown("¡Hola! Soy el asistente del Cluster 3. Pregúntame por los clientes, el modelo de ML, "
                        "lo que dicen las llamadas o qué acciones tomar. Puedes empezar con uno de los ejemplos.")

    for i, turno in enumerate(ss.chat):
        if turno["rol"] == "user":
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(turno["texto"])
        else:
            with st.chat_message("assistant", avatar=_avatar(turno["r"])):
                _mostrar(turno["r"], i)

    if ss.pendiente:
        with st.chat_message("assistant", avatar=":material/pan_tool:"):
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

    pregunta = st.chat_input("Escribe tu pregunta sobre el Cluster 3") or elegido
    if pregunta and not ss.pendiente:
        historial = [{"rol": t["rol"], "texto": t["texto"] if t["rol"] == "user" else (t["r"].get("respuesta") or "")}
                     for t in ss.chat][-6:]
        ss.chat.append({"rol": "user", "texto": pregunta})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(pregunta)
        with st.chat_message("assistant", avatar=":material/support_agent:"):
            estado = st.empty()
            vista = st.empty()
            estado.markdown(pill("El orquestador está eligiendo quién responde…", "gray"), unsafe_allow_html=True)

            def _evento(nodo: str, datos) -> None:
                if nodo == "orquestador" and isinstance(datos, dict):
                    nombres = [P.AGENTES_INFO.get(a, (a,))[0] for a in datos.get("agentes") or []
                               if a != "conversacion"]
                    if nombres:
                        estado.markdown(pill("Trabajando: " + " · ".join(nombres), "blue"), unsafe_allow_html=True)
                elif nodo in ("sintesis", "directo"):
                    estado.markdown(pill("El crítico está verificando las cifras…", "gray"), unsafe_allow_html=True)

            # Vista previa en vivo; al terminar se reemplaza por la respuesta verificada por el crítico.
            r = ask(pregunta, thread_id=None, modo=modo, historial=historial,
                    on_token=lambda t: vista.markdown(t + " ▌"), on_evento=_evento, modelo=modelo_elegido)
        ss.thread = r["thread_id"]
        if "interrupt" in r:
            ss.pendiente = r["interrupt"]
        else:
            ss.chat.append({"rol": "assistant", "r": r})
        st.rerun()

    if ss.chat and st.button("Nueva conversación"):
        ss.chat, ss.thread, ss.pendiente = [], None, None
        st.rerun()

# --------------------------------------------------------------------------- 7. Copiloto
with tabs[6]:
    from cluster3.nlp import analizador as AN

    guia("Dos herramientas para el área de cancelaciones. **Analizar una llamada**: pegas una transcripción y obtienes "
         "la intención de cancelar, el motivo, la urgencia, el sentimiento, la cita que lo prueba y la oferta sugerida. "
         "**Copiloto (chat)**: el asesor le cuenta el caso con sus palabras y el copiloto conversa con él: le dice qué "
         "está pasando, qué preguntarle al cliente y qué ofrecerle, con una frase lista para decir.",
         "el copiloto **sugiere** y el asesor **decide**. Si el cliente insiste en cancelar, la alerta pide respetar su "
         "decisión. Usa el método híbrido validado con 50 llamadas etiquetadas (Jev + LLM); sin claves responde con reglas.")

    @st.cache_data(show_spinner=False)
    def _ejemplos_llamadas() -> dict[str, str]:
        """Una llamada real del Cluster 3 por motivo, para probar sin pegar texto."""
        if not (config.T_LLAMADAS_ANALISIS.exists() and config.T_LLAMADAS_LIMPIAS.exists()):
            return {}
        an = pd.read_parquet(config.T_LLAMADAS_ANALISIS)
        an = an[(an["cluster"] == config.CLUSTER_CRITICO) & (an["calidad_transcripcion"].isin(["alta", "media"]))]
        textos = pd.read_parquet(config.T_LLAMADAS_LIMPIAS).set_index("id_llamada")["texto_anonimizado"]
        elegidas = an.sort_values("confianza", ascending=False).groupby("motivo").head(1)
        return {f"Llamada {int(r.id_llamada)} · {r.motivo}": textos[r.id_llamada] for r in elegidas.itertuples()}

    def _ficha(r: dict, clave: str) -> None:
        kpis_row([
            ("Intención de cancelar", _pct(r["intencion_cancelar_prob"], 0), f"fuente: {r['fuentes'].get('intencion')}",
             "orange" if r["intencion_cancelar_prob"] >= 0.5 else ""),
            ("Motivo", r["motivo"], r["submotivo"], "blue"),
            ("Urgencia", r["urgencia"], "alta = resolver hoy", "orange" if r["urgencia"] == "alta" else ""),
            ("Sentimiento inicio → fin", f"{_d(r['sentimiento_inicio'], 2)} → {_d(r['sentimiento_fin'], 2)}",
             f"global {_d(r['sentimiento_global'], 2)} (−1 a 1)", ""),
        ])
        st.write("")
        if r.get("pregunta_sugerida"):
            st.markdown(f'<div class="guide alerta"><div class="gt">{ic("help")}Pregúntale al cliente</div>'
                        f'<div class="gx">{_e(r["pregunta_sugerida"])}</div></div>', unsafe_allow_html=True)
        for a in r.get("alertas") or []:
            st.warning(a, icon=":material/warning:")
        c1, c2 = st.columns([3, 2])
        with c1:
            ctx = r.get("contexto_c3")
            extra = (f"<div class='cx' style='margin-top:6px'>En el Cluster 3 este motivo es el "
                     f"<b>{_d(ctx['pct_llamadas_c3'], 1)} %</b> de las llamadas y se retiene al "
                     f"<b>{_d(ctx['retenido_pct_c3'], 1)} %</b>.</div>") if ctx else ""
            acc_txt = f"<div class='cx' style='margin-top:6px'>Accionable: {_e(r['accionable_relacionado'])}</div>" \
                if r.get("accionable_relacionado") else ""
            st.markdown(f'<div class="card"><div class="ic">{ic("lightbulb")}</div><div class="ct">Oferta sugerida</div>'
                        f'<div class="cx">{_e(r["oferta_sugerida"])}</div>{acc_txt}{extra}</div>', unsafe_allow_html=True)
            if r.get("guion"):
                st.markdown(f'<div class="card" style="margin-top:12px"><div class="ic">{ic("record_voice_over")}</div><div class="ct">Guion '
                            f'sugerido</div><div class="cx">{_e(r["guion"])}</div></div>', unsafe_allow_html=True)
            if r.get("evidencia"):
                ver = {True: " · cita verificada", False: " · cita no verificada"}.get(r.get("evidencia_verificada"), "")
                st.markdown(f'<div class="quote">“{_e(str(r["evidencia"])[:320])}”<div class="qm">evidencia del motivo'
                            f'{ver}</div></div>', unsafe_allow_html=True)
        with c2:
            probs = r.get("motivo_probabilidades") or {}
            if probs:
                dp = pd.DataFrame({"motivo": list(probs), "prob": [float(v) for v in probs.values()]})
                dp = dp[dp["prob"] > 0].sort_values("prob", ascending=False)
                st.markdown("**Probabilidad por motivo (Jev)**")
                st.altair_chart(
                    alt.Chart(dp).mark_bar(cornerRadiusEnd=4, color=BLUE).encode(
                        y=alt.Y("motivo:N", sort="-x", title=None, axis=alt.Axis(labelColor=INK2, ticks=False, domain=False)),
                        x=alt.X("prob:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                                axis=alt.Axis(format="%", gridColor=GRID, labelColor=INK2, domain=False, ticks=False)),
                        tooltip=["motivo:N", alt.Tooltip("prob:Q", format=".0%")],
                    ).properties(height=max(120, 34 * len(dp))).configure_view(stroke=None)
                    .configure(background="transparent", font="IBM Plex Sans"), width="stretch")
            if r.get("emociones"):
                st.markdown("**Emociones:** " + ", ".join(r["emociones"]))
            st.caption("Fuentes: " + " · ".join(f"{k}: {v}" for k, v in r["fuentes"].items()))
            if r.get("errores"):
                st.caption("Avisos: " + "; ".join(r["errores"]))
            st.download_button("Descargar JSON", json.dumps(r, ensure_ascii=False, indent=2, default=str),
                               file_name="analisis_llamada.json", mime="application/json", key=f"dl_{clave}")

    ejemplos_ll = _ejemplos_llamadas()
    modo_c = st.radio("Modo", ["Analizar una llamada", "Copiloto (chat)"], horizontal=True, key="modo_copiloto")
    ss = st.session_state

    if modo_c == "Analizar una llamada":
        section("Analizar una llamada", "Pega la transcripción (con o sin 'Cliente:' / 'Asesor:'), sube un .txt o usa un ejemplo")
        e1, e2 = st.columns([2, 1])
        with e1:
            elegido_ej = st.selectbox("Usar una llamada real de ejemplo", ["(ninguna)"] + list(ejemplos_ll))
        with e2:
            archivo = st.file_uploader("o sube un .txt", type=["txt"])
        texto_def = ejemplos_ll.get(elegido_ej, "")
        if archivo is not None:
            texto_def = archivo.read().decode("utf-8", errors="ignore")
        texto_ll = st.text_area("Transcripción", value=texto_def, height=220,
                                placeholder="Cliente: Quiero cancelar, me llegó un cobro que no reconozco…")
        if st.button("Analizar llamada", type="primary"):
            try:
                with st.spinner("Analizando con Jev y el LLM…"):
                    ss.analisis_llamada = AN.analizar_llamada(texto_ll)
            except ValueError as e:
                st.error(str(e))
        if ss.get("analisis_llamada"):
            section("Resultado")
            _ficha(ss.analisis_llamada, "analisis")

    else:
        ss.setdefault("cop_chat", [])
        section("Copiloto del asesor", "Cuéntale el caso con tus palabras, como a un colega; recuerda lo que ya le dijiste")

        def _chips(a: dict) -> str:
            if not a:
                return ""
            tono_int = "orange" if a["intencion_cancelar_prob"] >= 0.5 else "gray"
            chips = [pill(f"Intención de cancelar {_pct(a['intencion_cancelar_prob'], 0)}", tono_int),
                     pill(f"Motivo: {a['motivo']}", "blue"),
                     pill(f"Urgencia {a['urgencia']}", "orange" if a["urgencia"] == "alta" else "gray")]
            if a.get("accionable_relacionado"):
                chips.append(pill(a["accionable_relacionado"].split(" · ")[0], "good"))
            return "".join(chips)

        if not ss.cop_chat:
            with st.chat_message("assistant", avatar=":material/support_agent:"):
                st.markdown(AN.SALUDO_CHAT)
            ejemplos_cop = ["El cliente dice que quiere otro plan porque no usa los datos",
                            "Tengo un cliente que llama porque el internet se le cae todas las noches",
                            "La cliente dice que la factura le subió y no sabe por qué"]
            cols_ej = st.columns(3)
            for i, e in enumerate(ejemplos_cop):
                if cols_ej[i].button(e, key=f"cop_ej{i}", width="stretch"):
                    ss.cop_pendiente = e
        for i, m in enumerate(ss.cop_chat):
            if m["rol"] == "asesor":
                with st.chat_message("user", avatar=":material/headset_mic:"):
                    st.markdown(m["texto"])
            else:
                with st.chat_message("assistant", avatar=":material/support_agent:"):
                    if m.get("analisis"):
                        st.markdown(_chips(m["analisis"]), unsafe_allow_html=True)
                    st.markdown(m["texto"])
                    if m.get("analisis"):
                        with st.expander("Ficha del caso"):
                            _ficha(m["analisis"], f"cop{i}")
                    st.caption(f"Fuente: {m.get('fuente', '')}")

        nuevo = st.chat_input("Cuéntale al copiloto qué pasa con el cliente…", key="cop_input") or ss.pop("cop_pendiente", None)
        if nuevo:
            ss.cop_chat.append({"rol": "asesor", "texto": nuevo})
            with st.chat_message("user", avatar=":material/headset_mic:"):
                st.markdown(nuevo)
            with st.chat_message("assistant", avatar=":material/support_agent:"):
                vista_cop = st.empty()
                vista_cop.markdown(pill("Analizando el caso…", "gray"), unsafe_allow_html=True)
                out = AN.copiloto_chat(ss.cop_chat, usar_llm=hay_llm and modo == "llm", modelo=modelo_elegido,
                                       on_token=lambda t: vista_cop.markdown(t + " ▌"))
            ss.cop_chat.append({"rol": "copiloto", "texto": out["respuesta"], "analisis": out["analisis"],
                                "fuente": out["fuente"]})
            st.rerun()
        if ss.cop_chat and st.button("Nuevo caso"):
            ss.cop_chat = []
            st.rerun()

# --------------------------------------------------------------------------- 8. Evaluación
with tabs[7]:
    guia("Cómo se comprobó que lo que muestra el panel es confiable: (1) la clasificación de llamadas se midió contra "
         "llamadas etiquetadas a mano, (2) el sistema de agentes se prueba con preguntas doradas de respuesta conocida "
         "y (3) los datos pasaron por controles de calidad y de fuga de información.",
         "**exactitud** = % de aciertos; **kappa** = acuerdo descontando el azar (0,4–0,6 es moderado, más de 0,6 "
         "bueno); **correlación** = qué tanto el puntaje de sentimiento sigue al criterio humano.")
    section("Sistema multiagente · preguntas doradas", "Ruta, cifras respaldadas, contenido esperado y aprobación humana")
    ev = _csv("agent_evals.csv")
    if ev is not None:
        res = ev[["ruta_ok", "cifras_respaldadas", "contenido_ok", "hitl_ok"]].mean()
        nombres = {"ruta_ok": "Ruta correcta", "cifras_respaldadas": "Cifras respaldadas",
                   "contenido_ok": "Contenido esperado", "hitl_ok": "Aprobación humana"}
        kpis_row([(nombres[k], _pct(v, 0), f"{_n(len(ev))} preguntas · modo {ev['modo'].iloc[0]}", "blue")
                  for k, v in res.items()])
        st.write("")
        st.dataframe(ev, hide_index=True, width="stretch")
        st.caption("Regenerar: `uv run python -m cluster3.agents.evals` (añade juez LLM si hay clave).")
    section("Clasificación de llamadas · benchmark", "Reglas vs LLM vs Jev contra la muestra etiquetada a mano")
    bench = _csv("nlp_benchmark.csv")
    if bench is not None and "metodo" in bench:
        nombres_m = {"baseline": "Reglas", "llm": "LLM + RAG", "jev": "Jev", "hibrido": "Híbrido (final)"}
        metodos = [m for m in nombres_m if m in set(bench["metodo"])]
        b = bench.assign(Método=bench["metodo"].map(nombres_m),
                         valor=np.where(bench["campo"] == "sentimiento", bench.get("correlacion_spearman"),
                                        bench["exactitud"]))
        b["métrica"] = np.where(b["campo"] == "sentimiento", "correlación", "exactitud")
        paleta = {"Reglas": "#b9b7b0", "LLM + RAG": ORANGE, "Jev": "#7a5bd6", "Híbrido (final)": BLUE}
        graf = alt.Chart(b).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Método:N", sort=[nombres_m[m] for m in metodos], title=None,
                    axis=alt.Axis(labelAngle=0, labelColor=INK2, ticks=False, domain=False)),
            y=alt.Y("valor:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(gridColor=GRID, labelColor=INK2, domain=False, ticks=False, format="%")),
            color=alt.Color("Método:N", scale=alt.Scale(domain=list(paleta), range=list(paleta.values())),
                            legend=alt.Legend(orient="top", title=None, labelColor=INK2)),
            column=alt.Column("campo:N", title=None, sort=["motivo", "urgencia", "sentimiento"],
                              header=alt.Header(labelColor=INK, labelFontSize=13)),
            tooltip=["Método:N", "campo:N", "métrica:N", alt.Tooltip("valor:Q", format=".2f"),
                     alt.Tooltip("kappa:Q", format=".2f"), alt.Tooltip("n:Q", title="llamadas")],
        ).properties(width=210, height=240).configure_view(stroke=None).configure(background="transparent", font="IBM Plex Sans")
        st.altair_chart(graf)
        st.caption("Motivo y urgencia: exactitud. Sentimiento: correlación con la etiqueta humana (−1 / 0 / 1).")
        with st.expander("Tabla del benchmark"):
            st.dataframe(bench, hide_index=True, width="stretch")
    else:
        st.info("Pendiente: etiquetar data/labels/muestra_etiquetada.csv (50 llamadas) y correr "
                "`uv run python -m cluster3.nlp.run --llm --jev`.")
    section("Calidad de datos")
    cal = _json("calidad_datos.json")
    if cal:
        dup = cal.get("duplicados", 0)
        if isinstance(dup, dict):
            dup = sum(v for v in dup.values() if isinstance(v, (int, float)))
        kpis_row([
            ("Filas", _n(cal.get("n_filas", 0)), "", ""),
            ("Columnas", _n(cal.get("n_columnas", 0)), "", ""),
            ("Duplicados", _n(dup), "", ""),
            ("Variables constantes", _n(len(cal.get("constantes", []))), "eliminadas", "orange"),
        ])
        st.write("")
        with st.expander("Sospechas de fuga y hallazgos de consistencia"):
            st.json({k: cal.get(k) for k in ["sospecha_fuga_churn", "sospecha_fuga_intencion", "consistencia"]},
                    expanded=False)
    reg = _csv("registro_decisiones_variables.csv")
    if reg is not None:
        with st.expander("Registro de decisiones por variable"):
            st.dataframe(reg, hide_index=True, width="stretch")
