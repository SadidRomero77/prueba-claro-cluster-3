"""Genera los reportes en docs/ a partir de las salidas del pipeline (sin cifras escritas a mano)."""
from __future__ import annotations

import inspect
import json

import pandas as pd

from cluster3 import config


def _j(name: str) -> dict:
    p = config.OUT_JSON / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _t(name: str) -> pd.DataFrame:
    p = config.OUT_TAB / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _pct(x, d=1):
    return f"{x * 100:.{d}f} %".replace(".", ",")


def _n(x):
    return f"{x:,.0f}".replace(",", ".")


def _md(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> str:
    if df.empty:
        return "_(sin datos)_"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def calidad_md() -> str:
    q = _j("calidad_datos.json")
    reg = _t("registro_decisiones_variables.csv")
    auc = _t("auc_univariado_churn.csv").head(8)
    lines = [
        "# 01 · Calidad, consistencia y representatividad de los datos", "",
        f"Dataset: {_n(q['n_filas'])} filas × {q['n_columnas']} variables · duplicados: {q['duplicados']}.", "",
        "## Hallazgos y decisiones", "",
        _md(pd.DataFrame([{"hallazgo": f["hallazgo"], "decisión": f["decision"]} for f in q["consistencia"]])), "",
        "## Variables sin información", "",
        f"- Constantes: {', '.join(q['constantes'])}.",
        f"- 100 % vacías: {', '.join(q['vacias_100'])}.",
        "- `BAN_DESPOSICIONADO_*` (pedidas en el caso) son constantes en 0: no se pueden interpretar.", "",
        "## Fuga de información (AUC univariado contra churn)", "",
        _md(auc[["variable", "auc_univariado", "media_positivos", "media_negativos", "identica_al_target"]]), "",
        "## Diccionario vs dataset", "",
        f"{q['diccionario']} — detalle en `outputs/tables/diccionario_reconciliado.csv`.", "",
        "## Llamadas", "",
        *[f"- {f['id']}: {f['hallazgo']}" for f in q["llamadas"]], "",
        "## Registro de decisiones por variable", "",
        f"{len(reg)} variables con decisión documentada: " + ", ".join(f"{k} = {v}" for k, v in reg['decision'].value_counts().items())
        + ". Detalle en `outputs/tables/registro_decisiones_variables.csv`.",
    ]
    return "\n".join(lines)


def eda_md() -> str:
    k = _j("kpis_cluster3.json")
    seg = _t("segmentos_cluster3.csv")
    lines = [
        "# 02 · Perfil del Cluster 3", "",
        "| Indicador | Valor |", "|---|---|",
        f"| Clientes | {_n(k['clientes'])} |",
        f"| Churn del mes | {_pct(k['churn_tasa'], 2)} ({k['churn_n']} clientes) |",
        f"| Intención de cancelación | {_pct(k['intencion_tasa'])} ({_n(k['intencion_n'])} clientes) |",
        f"| Clientes con 3 o más intenciones | {_n(k['reincidentes_3mas'])} |",
        f"| Churn si tiene intención vs si no | {_pct(k['churn_si_intencion'], 2)} vs {_pct(k['churn_no_intencion'], 2)} |",
        f"| ARPU mediano | ${_n(k['arpu_mediano'])} COP |",
        f"| Renta mensual de clientes con intención | ${_n(k['renta_mensual_intencion'])} COP |",
        f"| Renta mensual de clientes que se fueron | ${_n(k['renta_mensual_churn'])} COP |",
        f"| Antigüedad mediana | {k['antiguedad_mediana_meses']:.0f} meses |",
        f"| Convergentes (fijo + móvil) | {_pct(k['pct_convergente'])} |",
        f"| Estratos 2 y 3 | {_pct(k['pct_estrato_2_3'])} |", "",
        "## Segmentos (churn e intención)", "",
        _md(seg, "{:.2f}"), "",
        "Gráficos: `outputs/figures/segmento_*.png`.",
    ]
    return "\n".join(lines)


def modelo_md() -> str:
    m = _j("metricas_modelo.json")
    leak = pd.DataFrame(m["comparacion_fuga"])
    bal = _t("balance_importancia_targets.csv")
    cat_i, cat_c = _t("importancia_categoria_intencion.csv"), _t("importancia_categoria_churn.csv")
    rows = []
    for key in ["intencion", "churn"]:
        x = m[key]
        rows.append({"modelo": key, "positivos": x["positivos"], "tasa base": _pct(x["tasa_base"], 2),
                     "AUC (CV 3×5)": f"{x['auc_cv_media']:.3f} ± {x['auc_cv_sd']:.3f}",
                     "IC 95 % AUC": f"{x['auc_oof_ic95'][0]:.3f}–{x['auc_oof_ic95'][1]:.3f}",
                     "PR-AUC": f"{x['pr_auc']:.3f}", "LIFT@5 %": f"{x['lift_5']:.1f}", "LIFT@10 %": f"{x['lift_10']:.1f}",
                     "LIFT@20 %": f"{x['lift_20']:.1f}"})
    lines = ["# 03 · Modelo de propensión en dos etapas", "",
             "Etapa 1 predice la **intención** (alerta temprana). Etapa 2 predice el **churn** (baja efectiva) e incluye "
             "las señales de intención, que ocurren antes de la baja.", "",
             "## Métricas (validación cruzada estratificada, fuera de muestra)", "", _md(pd.DataFrame(rows)), "",
             "## Por qué se excluyen variables: con fuga vs sin fuga", "", _md(leak), "",
             "Los planes de TV con sufijo I (`TIPO_TV_DIGITAL_PI`, `TIPO_TV_DIGITAL_BI`) se excluyen: el 100 % son "
             "cuentas no activas, así que codifican el estado de la cuenta igual que `ESTADO_FUENTE_*`.", "",
             f"Sensibilidad: el churn sin equipos adicionales ni UltraWiFi (`VAL_EQUIP_ADIC`, `VAL_UW`, más altos en "
             f"cuentas no activas) da AUC {m['sensibilidad_churn_sin_equipos']['auc']:.3f} y LIFT@10 "
             f"{m['sensibilidad_churn_sin_equipos']['lift_10']:.1f}. Ambas variables quedan como señales a validar con Claro.",
             ""]
    for key in ["intencion", "churn"]:
        x = m[key]
        mt, mn = x["matriz_top10"], x["matriz_umbral_negocio"]
        lines += [f"## Matriz de confusión · {key}", "",
                  "| Umbral | VP | FP | FN | VN | Precisión | Recall |", "|---|---|---|---|---|---|---|",
                  f"| Top 10 % ({mt['umbral']:.3f}) | {mt['VP']} | {mt['FP']} | {mt['FN']} | {mt['VN']} | {_pct(mt['precision'])} | {_pct(mt['recall'])} |",
                  f"| Negocio ({mn['umbral']:.3f}) | {mn['VP']} | {mn['FP']} | {mn['FN']} | {mn['VN']} | {_pct(mn['precision'])} | {_pct(mn['recall'])} |",
                  "", f"Umbral de negocio: maximiza ARPU × {m['supuestos_umbral']['meses_vida_extra']} meses × "
                  f"{_pct(m['supuestos_umbral']['tasa_exito'], 0)} de éxito − ${_n(m['supuestos_umbral']['costo_contacto_cop'])} por contacto.", ""]
    lines += ["## Balance de importancia entre targets (SHAP)", "",
              "Estructural = importante en ambos modelos · Negociación = solo intención · Salida = solo churn.", "",
              _md(bal), "",
              "## Importancia por categoría del diccionario (% del SHAP total)", "",
              "Intención:", "", _md(cat_i), "", "Churn:", "", _md(cat_c), "",
              "## Desbalance", "",
              "- Pesos de clase (`scale_pos_weight`) en lugar de SMOTE: con 104 positivos SMOTE fabrica clientes casi idénticos.",
              "- Métricas que no se inflan con el desbalance: PR-AUC y LIFT por decil.",
              "- Probabilidades calibradas (isotónica en intención, sigmoide en churn) para poder calcular impacto en pesos.",
              "", "Figuras: `outputs/figures/ganancia_*.png`, `outputs/figures/shap_*.png`."]
    return "\n".join(lines)


def nlp_md() -> str:
    n = _j("nlp_insights.json")
    mot = _t("nlp_motivos_c3_vs_otros.csv")
    sub = _t("nlp_submotivos_c3.csv").head(12)
    sent = _t("nlp_sentimiento_por_motivo_c3.csv")
    puente = _t("puente_llamadas_dataset.csv")
    bench = _t("nlp_benchmark.csv")
    pre = n.get("preprocesamiento", {})
    lines = ["# 04 · Voz del cliente: NLP de llamadas", "",
             f"Método de la clasificación actual: **{n.get('metodo_final')}**. "
             "Con claves de API se ejecuta con LLM y Jev (`uv run cluster3 --desde nlp --llm --jev`).", "",
             "## Preparación de las transcripciones", "",
             f"- Roles AGENT/CLIENT corregidos: {pre.get('roles_corregidos')} llamadas.",
             f"- Diarización fallida (todo en un rol): {pre.get('diarizacion_fallida')} llamadas; se usa el texto completo.",
             f"- Segunda pasada de anonimización: {pre.get('pii_reemplazos')} reemplazos en {pre.get('llamadas_con_pii_reemplazada')} llamadas.",
             f"- Sin contenido (< 1.000 caracteres): {pre.get('sin_contenido')} llamadas.", "",
             "## Motivo principal: Cluster 3 vs otros clústeres (%)", "", _md(mot), "",
             "## Submotivos del Cluster 3", "", _md(sub), "",
             "## Sentimiento, urgencia y resultado por motivo (Cluster 3)", "", _md(sent, "{:.2f}"), "",
             f"Urgencia en el Cluster 3: {n.get('urgencia_c3_pct')}. Competidores mencionados: {n.get('competidores_c3')}. "
             f"Reincidencia mencionada: {n.get('reincidencia_c3_pct')} %.", "",
             "## Cruce agregado llamadas ↔ dataset", "",
             "No hay llave común entre llamadas y clientes; el cruce se hace por motivo.", "", _md(puente, "{:.2f}"), "",
             "## Evaluación contra muestra etiquetada a mano", "", _md(bench), "",
             "Esquema JSON por llamada: `src/cluster3/nlp/schema.py`. Salidas: `outputs/json/llamadas_*.jsonl`."]
    return "\n".join(lines)


def accionables_md() -> str:
    a = _t("accionables.csv")
    s = _j("accionables.json").get("supuestos", {})
    cols = ["id", "accionable", "tipo", "prioridad", "evidencia", "metrica"]
    imp = a[["id", "impacto_anual_conservador_cop", "impacto_anual_base_cop", "impacto_anual_optimista_cop"]].copy()
    for c in imp.columns[1:]:
        imp[c] = imp[c].map(lambda v: "" if pd.isna(v) else f"${v / 1e6:,.1f} M".replace(",", "X").replace(".", ",").replace("X", "."))
    return "\n".join([
        "# 07 · Accionables priorizados e impacto económico", "",
        _md(a[cols]), "", "## Impacto anual estimado (COP)", "", _md(imp), "",
        f"Supuestos: tasa de éxito por escenario {s.get('escenarios_tasa_exito')}, horizonte {s.get('meses')} meses, "
        f"costo de contacto proactivo ${_n(s.get('costo_contacto_cop', 0))}. Los reactivos no tienen costo de contacto "
        "(el cliente ya llamó). El impacto real se mide con grupo de control.", "",
        "Matriz impacto/esfuerzo: `outputs/figures/matriz_impacto_esfuerzo.png`.",
    ])


def agentes_md() -> str:
    from cluster3.agents import prompts as P
    from cluster3.agents import tools as T

    ev = _t("agent_evals.csv")
    tool_rows = []
    for ag, fns in T.TOOLS_POR_AGENTE.items():
        for f in fns:
            sig = str(inspect.signature(f)).replace("|", "/")
            tool_rows.append({"agente": ag, "herramienta": f.__name__, "firma": sig,
                              "sensible": "sí (HITL)" if f.__name__ in T.SENSIBLES else "",
                              "descripción": (f.__doc__ or "").strip().splitlines()[0]})
    return "\n".join([
        "# 05 · Sistema multiagente", "",
        "```mermaid", "flowchart LR", "  Q[Pregunta] --> O[Orquestador]", "  O --> PF[Perfilado]", "  O --> VC[Voz del cliente]",
        "  O --> ES[Estrategia]", "  PF --> H{¿Herramienta sensible?}", "  VC --> H", "  ES --> H",
        "  H -->|sí| A[Aprobación humana]", "  H -->|no| S[Síntesis]", "  A --> S", "  S --> C[Crítico]",
        "  C -->|falla| S", "  C -->|aprueba| R[Respuesta]", "```", "",
        "## Herramientas (function calling)", "", _md(pd.DataFrame(tool_rows)), "",
        "## Human-in-the-loop", "",
        "- `generar_lista_contacto` exporta datos de clientes: el grafo se detiene con `interrupt()` hasta que una persona apruebe.",
        "- Llamadas con `confianza` < 0,6 en el NLP se envían a revisión manual.",
        "- Los accionables se aprueban antes de salir a negocio.", "",
        "## Criterios de éxito", "",
        "- 100 % de las cifras de cada respuesta respaldadas por una herramienta (control automático del crítico).",
        "- Ruta correcta del orquestador en el set de evaluación.",
        "- Juez LLM (G-Eval) ≥ 4/5 en fidelidad y pertinencia cuando hay LLM disponible.", "",
        "## Evaluación (set de preguntas doradas)", "", _md(ev), "",
        "## System prompts", "",
        *[f"### {name}\n\n```\n{txt}\n```\n" for name, txt in [("Orquestador", P.ORQUESTADOR), ("Perfilado", P.PERFILADO),
                                                             ("Voz del cliente", P.VOZ_CLIENTE), ("Estrategia", P.ESTRATEGIA),
                                                             ("Síntesis", P.SINTESIS), ("Crítico", P.CRITICO),
                                                             ("Rúbrica del juez", P.RUBRICA_JUEZ)]],
    ])


def build_reports() -> None:
    config.DOCS.mkdir(exist_ok=True)
    for fn, builder in [("01_calidad_datos.md", calidad_md), ("02_eda_cluster3.md", eda_md), ("03_modelo_churn.md", modelo_md),
                        ("04_nlp_llamadas.md", nlp_md), ("05_sistema_multiagente.md", agentes_md),
                        ("07_accionables.md", accionables_md)]:
        try:
            (config.DOCS / fn).write_text(builder(), encoding="utf-8")
        except Exception as e:
            print(f"[reportes] {fn}: {e}")
