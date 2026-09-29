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
from cluster3.eda.profile import BLUE, GRID, INK, INK2, ORANGE, _slug  # noqa: E402

st.set_page_config(page_title="Cluster 3 · Churn Claro", page_icon="📉", layout="wide")

# --------------------------------------------------------------------------- estilo
# Tokens de la paleta del proyecto (docs/CLAUDE.md). El texto usa tinta neutra; el color identifica series.
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{
  --ink: {INK}; --ink2: {INK2}; --muted: #8a8983; --grid: {GRID};
  --surface: #ffffff; --bg: #f7f6f3; --blue: {BLUE}; --blue-soft: #e9f1fb;
  --orange: {ORANGE}; --orange-soft: #fdeee7; --good: #1f7a4d; --good-soft: #e6f4ec;
  --warn: #9a6200; --warn-soft: #fcf1dc;
}}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

.hero {{ background: linear-gradient(120deg, #0b0b0b 0%, #1b2a3f 55%, #2a78d6 140%); color: #fff;
  border-radius: 18px; padding: 26px 30px 22px; margin-bottom: 18px; }}
.hero .eyebrow {{ font-size: 12px; letter-spacing: .12em; text-transform: uppercase; opacity: .7; }}
.hero h1 {{ color: #fff; font-size: 30px; font-weight: 700; margin: 6px 0 6px; line-height: 1.15; }}
.hero p {{ color: #d9dee6; font-size: 15px; margin: 0 0 14px; max-width: 900px; }}
.chip {{ display: inline-block; font-size: 12.5px; font-weight: 500; padding: 5px 11px; border-radius: 999px;
  margin: 0 6px 6px 0; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18); color: #fff; }}

.kpi {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 14px; padding: 15px 16px 13px;
  height: 100%; min-height: 124px; position: relative; overflow: hidden; }}
.kpi::before {{ content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--grid); }}
.kpi.blue::before {{ background: var(--blue); }} .kpi.orange::before {{ background: var(--orange); }}
.kpi .lbl {{ font-size: 12.5px; color: var(--ink2); font-weight: 500; }}
.kpi .val {{ font-size: 26px; font-weight: 700; color: var(--ink); margin-top: 4px; line-height: 1.1; }}
.kpi .sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}

.sec {{ margin: 22px 0 10px; }}
.sec .t {{ font-size: 19px; font-weight: 650; color: var(--ink); }}
.sec .s {{ font-size: 13.5px; color: var(--ink2); margin-top: 2px; }}

.card {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 14px; padding: 16px 18px;
  height: 100%; }}
.card .ic {{ font-size: 20px; }} .card .ct {{ font-weight: 650; font-size: 14.5px; margin: 6px 0 4px; color: var(--ink); }}
.card .cx {{ font-size: 13.5px; color: var(--ink2); line-height: 1.45; }}

.agent {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 14px; padding: 14px 15px;
  height: 100%; min-height: 210px; }}
.agent .an {{ font-weight: 650; font-size: 14px; color: var(--ink); margin: 4px 0 3px; }}
.agent .ad {{ font-size: 12.5px; color: var(--ink2); line-height: 1.4; }}
.agent .at {{ font-size: 11.5px; color: var(--muted); margin-top: 8px; }}

.pill {{ display: inline-block; font-size: 12px; font-weight: 600; padding: 3px 9px; border-radius: 999px;
  margin: 0 5px 4px 0; border: 1px solid transparent; }}
.pill.blue {{ background: var(--blue-soft); color: #1d5aa3; }}
.pill.orange {{ background: var(--orange-soft); color: #a84113; }}
.pill.good {{ background: var(--good-soft); color: var(--good); }}
.pill.warn {{ background: var(--warn-soft); color: var(--warn); }}
.pill.gray {{ background: #efeeea; color: var(--ink2); }}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--grid); }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; border-radius: 10px 10px 0 0; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ background: var(--surface); }}
[data-testid="stSidebar"] {{ background: #fbfaf8; border-right: 1px solid var(--grid); }}
[data-testid="stChatMessage"] {{ background: var(--surface); border: 1px solid var(--grid); border-radius: 14px;
  padding: 10px 14px; margin-bottom: 8px; }}
[data-testid="stMetricValue"] {{ font-weight: 700; }}
div[data-testid="stExpander"] details {{ border-radius: 12px; border-color: var(--grid); background: var(--surface); }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

AVATAR = {"conversacion": "👋", "perfilado": "📊", "voz_cliente": "🎧", "estrategia": "🎯", "critico": "✅",
          "orquestador": "🧭"}
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
    """items: (ícono, título, texto con **negritas**)"""
    for i in range(0, len(items), por_fila):
        cols = st.columns(por_fila)
        for col, (ic, t, x) in zip(cols, items[i:i + por_fila]):
            x = _e(x)
            while "**" in x:  # **texto** → <b>texto</b>
                x = x.replace("**", "<b>", 1).replace("**", "</b>", 1)
            col.markdown(f'<div class="card"><div class="ic">{ic}</div><div class="ct">{_e(t)}</div>'
                         f'<div class="cx">{x}</div></div>', unsafe_allow_html=True)
        st.write("")


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
            .properties(height=alto).configure_view(stroke=None).configure(background="transparent"))


def barras_simples(df: pd.DataFrame, x: str, y: str, color: str, titulo_x: str, titulo_y: str,
                   fmt: str = ".1f", alto: int = 260) -> alt.Chart:
    return (alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color)
            .encode(x=alt.X(f"{x}:O", title=titulo_x, axis=alt.Axis(labelAngle=0, labelColor=INK2, titleColor=INK2,
                                                                    ticks=False, domain=False)),
                    y=alt.Y(f"{y}:Q", title=titulo_y, axis=alt.Axis(gridColor=GRID, labelColor=INK2,
                                                                    titleColor=INK2, domain=False, ticks=False)),
                    tooltip=[alt.Tooltip(f"{x}:O", title=titulo_x), alt.Tooltip(f"{y}:Q", title=titulo_y, format=fmt)])
            .properties(height=alto).configure_view(stroke=None).configure(background="transparent"))


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
    st.markdown("### 📉 Cluster 3 · Churn")
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

tabs = st.tabs(["🏠 Resumen", "🧩 Segmentos", "🤖 Modelo ML", "🎧 Voz del cliente", "🎯 Accionables",
                "💬 Agente", "🧪 Evaluación"])

# --------------------------------------------------------------------------- 1. Resumen
with tabs[0]:
    veces = kpis["churn_si_intencion"] / kpis["churn_no_intencion"] if kpis["churn_no_intencion"] else 0
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
    hallazgos = [
        ("📞", "Muchos amenazan, pocos se van",
         f"El **{_pct(kpis['intencion_tasa'])}** llamó a cancelar y solo el **{_pct(kpis['churn_tasa'], 2)}** se fue. "
         f"Quien llama tiene **{_d(veces, 1)}×** más riesgo de irse."),
        ("💸", "El precio manda",
         f"Precio y facturación explican el **{_pct(nlp.get('motivos_c3_pct', {}).get('precio_facturacion', 0) / 100)}** "
         f"de las llamadas del Cluster 3 (vs **{_pct(nlp.get('motivos_otros_pct', {}).get('precio_facturacion', 0) / 100)}** "
         "en otros clústeres)."),
    ]
    if ch:
        top10 = ch.get("matriz_top10", {})
        hallazgos.append(("🎯", "El modelo encuentra a los que se van",
                          f"El 10 % de mayor riesgo concentra **{_n(top10.get('VP', 0))} de {_n(ch['positivos'])}** bajas "
                          f"(lift **{_d(ch['lift_10'], 1)}×**), validado fuera de muestra."))
    if fuga is not None and len(fuga):
        auc_fuga = fuga.loc[fuga["variables"] == "con fuga", "auc"].max()
        hallazgos.append(("🧯", "Se evitó la trampa de la fuga",
                          f"Con variables que ya contienen el resultado, el modelo daba AUC **{_d(auc_fuga, 2)}**. "
                          "Se excluyeron: estado de la cuenta, planes de TV con sufijo I, reincidencias y retención."))
    if nlp.get("urgencia_c3_pct"):
        hallazgos.append(("🔁", "Llaman con urgencia y ya habían reclamado",
                          f"El **{_pct(nlp['urgencia_c3_pct'].get('alta', 0) / 100)}** de las llamadas del Cluster 3 "
                          f"tiene urgencia alta y el **{_pct(nlp.get('reincidencia_c3_pct', 0) / 100)}** menciona un "
                          f"reclamo previo; se retiene al **{_pct(nlp.get('resultado_c3_pct', {}).get('retenido', 0) / 100)}**."))
    if a1 is not None and len(a1):
        top = a1.sort_values("impacto_anual_base_cop", ascending=False).iloc[0]
        hallazgos.append(("🚀", "Primer paso recomendado",
                          f"**{top['id']} · {top['accionable']}**: {_cop(top['impacto_anual_base_cop'])} al año "
                          "en el escenario base."))
    cards(hallazgos)

    mot = _csv("nlp_motivos_c3_vs_otros.csv")
    if mot is not None:
        section("Por qué llaman a cancelar", "% de llamadas por motivo · Cluster 3 frente a los demás clústeres")
        st.altair_chart(barras_agrupadas(mot, "motivo", ["Cluster 3", "Otros clústeres"], [BLUE, ORANGE],
                                         "% de llamadas"), width="stretch")

# --------------------------------------------------------------------------- 2. Segmentos
with tabs[1]:
    seg = _csv("segmentos_cluster3.csv")
    if seg is not None:
        section("Churn e intención por segmento", "Elige una dimensión; cada gráfico lee juntas las dos variables objetivo")
        nombres = list(dict.fromkeys(seg["segmento"]))
        elegido = st.selectbox("Segmento", nombres,
                               index=nombres.index("Variación de renta vs 6 meses")
                               if "Variación de renta vs 6 meses" in nombres else 0)
        d = seg[seg["segmento"] == elegido].drop(columns="segmento")
        grandes = d[d["clientes"] >= 200]
        if len(grandes):
            top_i, top_c = grandes.loc[grandes["intencion_pct"].idxmax()], grandes.loc[grandes["churn_pct"].idxmax()]
            kpis_row([
                ("Mayor intención", str(top_i["nivel"]), f"{_d(top_i['intencion_pct'], 1)} % · {_n(top_i['clientes'])} clientes", "blue"),
                ("Mayor churn", str(top_c["nivel"]), f"{_d(top_c['churn_pct'], 2)} % · {_n(top_c['clientes'])} clientes", "orange"),
                ("Niveles", _n(len(d)), "con 200 o más clientes: " + _n(len(grandes)), ""),
            ])
            st.write("")
        col1, col2 = st.columns([3, 2])
        with col1:
            _fig(f"segmento_{_slug(elegido)}.png")
        with col2:
            st.dataframe(d.rename(columns={"nivel": "Nivel", "clientes": "Clientes", "churn_pct": "Churn %",
                                           "intencion_pct": "Intención %", "arpu_mediano": "ARPU mediano"}),
                         hide_index=True, width="stretch")
        with st.expander("Todos los segmentos (tabla)"):
            st.dataframe(seg, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 3. Modelo ML
with tabs[2]:
    if metricas:
        section("Modelo predictivo en dos etapas",
                "LightGBM · validación cruzada estratificada 3×5 · probabilidades calibradas y fuera de muestra")
        m1, m2 = st.columns(2)
        for col, key, titulo, tono in [(m1, "intencion", "Intención de cancelar · alerta temprana", "blue"),
                                       (m2, "churn", "Churn · baja efectiva", "orange")]:
            m = metricas.get(key)
            if not m:
                continue
            with col:
                st.markdown(f"**{titulo}**")
                ic = m.get("auc_oof_ic95")
                kpis_row([
                    ("AUC", _d(m["auc_cv_media"]), f"IC 95 %: {_d(ic[0])}–{_d(ic[1])}" if ic else "", tono),
                    ("PR-AUC", _d(m["pr_auc"]), f"base {_d(m['tasa_base'])}", tono),
                    ("Lift decil 1", f"{_d(m['lift_10'], 1)}×", f"{_n(m['positivos'])} positivos", tono),
                ])
                st.write("")
                _fig(f"shap_{key}.png")
                with st.expander("Curva de ganancia y notas"):
                    _fig(f"ganancia_{key}.png")
                    for nota in m.get("notas", []):
                        st.caption(f"· {nota}")

        section("Lift por decil", "Qué tanto se concentra el evento en cada decil de riesgo (1 = mayor riesgo)")
        cl1, cl2 = st.columns(2)
        for col, key, color, titulo in [(cl1, "intencion", BLUE, "Intención"), (cl2, "churn", ORANGE, "Churn")]:
            lt = _csv(f"lift_{key}.csv")
            if lt is not None:
                with col:
                    st.markdown(f"**{titulo}**")
                    st.altair_chart(barras_simples(lt, "decil", "lift", color, "Decil", "Lift (× tasa base)", ".2f"),
                                    width="stretch")

        section("Por qué se excluyeron variables", "Con fuga de información las métricas son perfectas e inútiles")
        fuga = _csv("comparacion_fuga.csv")
        if fuga is not None:
            st.dataframe(fuga, hide_index=True, width="stretch")
        sens = metricas.get("sensibilidad_churn_sin_equipos")
        if sens:
            st.caption(f"Sensibilidad: churn sin equipos adicionales ni UltraWiFi → AUC {_d(sens['auc'], 3)}, "
                       f"lift decil 1 {_d(sens['lift_10'], 1)}×.")

        cat1, cat2 = _csv("importancia_categoria_intencion.csv"), _csv("importancia_categoria_churn.csv")
        if cat1 is not None and cat2 is not None:
            section("Qué explica el riesgo, por categoría", "Importancia SHAP agregada por categoría del diccionario (%)")
            comp = cat1.merge(cat2, on="categoria", how="outer", suffixes=(" intención", " churn")).fillna(0)
            comp = comp.rename(columns={"pct_importancia intención": "Intención", "pct_importancia churn": "Churn"})
            st.altair_chart(barras_agrupadas(comp, "categoria", ["Intención", "Churn"], [BLUE, ORANGE],
                                             "% de la importancia", alto=380), width="stretch")
        bal = _csv("balance_importancia_targets.csv")
        if bal is not None:
            with st.expander("Balance de importancia entre los dos objetivos"):
                st.dataframe(bal, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 4. Voz del cliente
with tabs[3]:
    if nlp:
        section("Voz del cliente", "Análisis de las llamadas de cancelación: motivo, urgencia y sentimiento")
        kpis_row([
            ("Llamadas del Cluster 3", _n(nlp.get("llamadas_c3", 0)), f"de {_n(nlp.get('llamadas_validas', 0))} válidas", ""),
            ("Precio y facturación", _pct(nlp.get("motivos_c3_pct", {}).get("precio_facturacion", 0) / 100),
             "motivo principal", "blue"),
            ("Urgencia alta", _pct(nlp.get("urgencia_c3_pct", {}).get("alta", 0) / 100), "pide la baja ya", "orange"),
            ("Sentimiento empeora", _pct(nlp.get("sentimiento_c3", {}).get("empeora_pct", 0) / 100),
             "del inicio al final de la llamada", ""),
            ("Reclamo previo mencionado", _pct(nlp.get("reincidencia_c3_pct", 0) / 100), "reincidencia en la llamada", ""),
        ])
        mot = _csv("nlp_motivos_c3_vs_otros.csv")
        if mot is not None:
            section("Motivos: Cluster 3 frente al resto")
            col1, col2 = st.columns([3, 2])
            with col1:
                st.altair_chart(barras_agrupadas(mot, "motivo", ["Cluster 3", "Otros clústeres"], [BLUE, ORANGE],
                                                 "% de llamadas"), width="stretch")
            with col2:
                st.dataframe(mot, hide_index=True, width="stretch")
        st.caption(f"Método: {nlp.get('metodo_final', 'baseline_reglas')} · "
                   f"preprocesamiento: {json.dumps(nlp.get('preprocesamiento', {}), ensure_ascii=False)}")
        sub = _csv("nlp_submotivos_c3.csv")
        if sub is not None:
            with st.expander("Submotivos"):
                st.dataframe(sub, hide_index=True, width="stretch")
        puente = _csv("puente_llamadas_dataset.csv")
        if puente is not None:
            with st.expander("Puente llamadas → dataset (cruce agregado, no hay llave común)"):
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
        section("Accionables priorizados", "Impacto anual en tres escenarios de éxito de retención")
        kpis_row([
            ("Impacto anual · escenario base", _cop(acc["impacto_anual_base_cop"].sum()), "suma de accionables", "blue"),
            ("Escenario optimista", _cop(acc["impacto_anual_optimista_cop"].sum()), "", ""),
            ("Escenario conservador", _cop(acc["impacto_anual_conservador_cop"].sum()), "", "orange"),
            ("Proactivos / reactivos", f"{_n((acc['tipo'] == 'Proactivo').sum())} / {_n((acc['tipo'] == 'Reactivo').sum())}",
             "", ""),
        ])
        st.write("")
        col1, col2 = st.columns([2, 3])
        with col1:
            _fig("matriz_impacto_esfuerzo.png")
        with col2:
            for _, r in acc.iterrows():
                tono = "blue" if r["tipo"] == "Proactivo" else "orange"
                with st.container(border=True):
                    st.markdown(f"{pill(r['id'], 'gray')}{pill(r['tipo'], tono)}{pill(r['prioridad'], 'good')}"
                                f"<br><b>{_e(r['accionable'])}</b>", unsafe_allow_html=True)
                    st.caption(f"Impacto anual base: {_cop(r['impacto_anual_base_cop'])} · "
                               f"objetivo: {r['objetivo']}")
                    with st.expander("Evidencia y métrica"):
                        st.markdown(f"**Evidencia:** {r['evidencia']}")
                        st.markdown(f"**Métrica de seguimiento:** {r['metrica']}")
        with st.expander("Tabla completa"):
            vista = acc[["id", "prioridad", "accionable", "tipo", "n_objetivo", "impacto_anual_conservador_cop",
                         "impacto_anual_base_cop", "impacto_anual_optimista_cop"]].copy()
            for c_ in ["impacto_anual_conservador_cop", "impacto_anual_base_cop", "impacto_anual_optimista_cop"]:
                vista[c_] = vista[c_].map(_cop)
            st.dataframe(vista, hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 6. Agente
with tabs[5]:
    from cluster3.agents.graph import ask, resume  # import tardío: carga el grafo solo si se usa
    from cluster3.agents.tools import TOOLS_POR_AGENTE

    section("Tu equipo de agentes", "El orquestador elige quién responde; el crítico verifica cada cifra; "
                                    "exportar datos siempre pide aprobación humana")
    equipo = ["orquestador", "perfilado", "voz_cliente", "estrategia", "critico"]
    cols = st.columns(len(equipo))
    for col, k in zip(cols, equipo):
        nombre, desc = P.AGENTES_INFO[k]
        tools_k = [f.__name__ for f in TOOLS_POR_AGENTE.get(k, [])]
        extra = (f'<div class="at" title="{_e(", ".join(tools_k))}">🔧 {len(tools_k)} herramientas</div>'
                 if tools_k else "")
        col.markdown(f'<div class="agent"><div style="font-size:22px">{AVATAR[k]}</div><div class="an">{_e(nombre)}</div>'
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
        return AVATAR.get(ag[0], "🧭") if len(ag) == 1 else "🧭"

    def _mostrar(r: dict, k: int = 0) -> None:
        ag = r.get("agentes") or []
        crit = r.get("critica") or {}
        badges = "".join(pill(f"{AVATAR.get(a, '')} {P.AGENTES_INFO.get(a, (a,))[0]}", TONO_AGENTE.get(a, "gray"))
                         for a in ag if a != "conversacion")
        if crit and ag != ["conversacion"]:
            badges += pill("✓ Cifras verificadas" if crit.get("aprobado") else "⚠ Crítico con observaciones",
                           "good" if crit.get("aprobado") else "warn")
        if r.get("desde_cache"):
            badges += pill("⚡ Respuesta en caché", "gray")
        elif r.get("latencia_ms"):
            badges += pill(f"⏱ {_d(r['latencia_ms'] / 1000, 1)} s", "gray")
        if r.get("modelo"):
            badges += pill(f"🧠 {llm_factory.MODELOS_OPENROUTER.get(r['modelo'], r['modelo']).split(' · ')[0]}", "gray")
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
        with st.chat_message("assistant", avatar="👋"):
            st.markdown("¡Hola! Soy el asistente del Cluster 3. Pregúntame por los clientes, el modelo de ML, "
                        "lo que dicen las llamadas o qué acciones tomar. Puedes empezar con uno de los ejemplos.")

    for i, turno in enumerate(ss.chat):
        if turno["rol"] == "user":
            with st.chat_message("user", avatar="🙂"):
                st.markdown(turno["texto"])
        else:
            with st.chat_message("assistant", avatar=_avatar(turno["r"])):
                _mostrar(turno["r"], i)

    if ss.pendiente:
        with st.chat_message("assistant", avatar="🛑"):
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
        with st.chat_message("user", avatar="🙂"):
            st.markdown(pregunta)
        with st.chat_message("assistant", avatar="🧭"):
            estado = st.empty()
            vista = st.empty()
            estado.markdown(pill("🧭 El orquestador está eligiendo quién responde…", "gray"), unsafe_allow_html=True)

            def _evento(nodo: str, datos) -> None:
                if nodo == "orquestador" and isinstance(datos, dict):
                    nombres = [f"{AVATAR.get(a, '')} {P.AGENTES_INFO.get(a, (a,))[0]}" for a in datos.get("agentes") or []
                               if a != "conversacion"]
                    if nombres:
                        estado.markdown(pill("Trabajando: " + " · ".join(nombres), "blue"), unsafe_allow_html=True)
                elif nodo in ("sintesis", "directo"):
                    estado.markdown(pill("✅ El crítico está verificando las cifras…", "gray"), unsafe_allow_html=True)

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

# --------------------------------------------------------------------------- 7. Evaluación
with tabs[6]:
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
    if bench is not None:
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
