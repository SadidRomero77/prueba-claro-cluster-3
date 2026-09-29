"""Accionables para el Cluster 3, priorizados por impacto y esfuerzo, con impacto económico.

Todas las cifras de evidencia se calculan aquí a partir de los datos (no hay números a mano).
Impacto = clientes objetivo × probabilidad de irse × tasa de éxito × ARPU × meses − costo de contacto.
Se reportan tres escenarios de tasa de éxito porque es el supuesto más incierto;
el valor real se mide con grupo de control.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cluster3 import config  # noqa: E402

ESCENARIOS = {"conservador": 0.15, "base": 0.30, "optimista": 0.45}
MESES = 12
COSTO_CONTACTO = 15_000  # COP por contacto proactivo (supuesto a validar con Claro)


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f} %".replace(".", ",")


def _n(x: float) -> str:
    return f"{x:,.0f}".replace(",", ".")


def _impact(n_obj: int, p_irse: float, arpu: float, tasa: float, costo_contacto: float = COSTO_CONTACTO) -> float:
    """Renta anual salvada neta del costo de contacto (proactivo). En reactivos el cliente ya llama: costo 0."""
    return n_obj * p_irse * tasa * arpu * MESES - n_obj * costo_contacto


def _rentables(df: pd.DataFrame) -> pd.DataFrame:
    """Clientes a los que conviene contactar: valor esperado positivo con su propia probabilidad fuera de muestra.

    p_churn_oof × tasa de éxito (escenario base) × ARPU × meses > costo de contacto. Es el mismo criterio del umbral
    de negocio del modelo: contactar a todo un segmento incluye clientes cuyo contacto cuesta más de lo que salva.
    """
    ev = df["p_churn_oof"] * ESCENARIOS["base"] * df[config.ARPU_COL] * MESES
    return df[ev > COSTO_CONTACTO]


def build_actions(sc: pd.DataFrame, nlp: dict | None = None) -> pd.DataFrame:
    """sc: dataset limpio con scores (clientes_c3_scores.parquet)."""
    C, I, A = config.TARGET_CHURN, config.TARGET_INTENCION, config.ARPU_COL  # noqa: E741
    acts = []

    # 1 · Alerta de fin de promoción / subida de renta
    up = sc[sc["VAL_VAR_RENTA"] > 5]
    flat = sc[sc["VAL_VAR_RENTA"] <= 5]
    obj = _rentables(up[up["decil_churn"] <= 2])
    acts.append(dict(
        id="A1", accionable="Alerta de subida de factura: contactar antes de que llegue el incremento", tipo="Proactivo",
        evidencia=f"Con renta +5 % vs 6 meses el churn es {_pct(up[C].mean(), 2)} vs {_pct(flat[C].mean(), 2)}; "
                  f"{_n(len(up))} clientes con subida",
        objetivo="Renta en alza, deciles 1–2 de riesgo de churn y valor esperado del contacto positivo",
        n_objetivo=len(obj), p_irse=obj[C].mean(), arpu=obj[A].median(), impacto_score=4, esfuerzo_score=3,
        metrica="Churn a 60 días en clientes con incremento; tasa de aceptación del ajuste",
    ))

    # 2 · Ajuste de plan por uso en lugar de descuento
    disc = sc[(sc[I] == 1) & (sc["VAL_RENTA_M6"] > sc[A])]
    perdida = float((disc["VAL_RENTA_M6"] - disc[A]).sum())
    extra_equip = sc[sc["VAL_EQUIP_ADIC"] > 0]
    acts.append(dict(
        id="A2", accionable="Ajustar el plan a lo que el cliente usa (retirar TV, decos o adicionales) en lugar de descontar",
        tipo="Reactivo",
        evidencia=f"{_n(len(disc))} clientes con intención tienen renta menor que hace 6 meses "
                  f"(−${f'{perdida / 1e6:.1f}'.replace('.', ',')} M/mes); con equipos adicionales el churn es "
                  f"{_pct(extra_equip[C].mean())}",
        objetivo="Clientes con intención cuyo motivo es precio o servicios no usados",
        n_objetivo=len(disc), p_irse=np.nan, arpu=disc[A].median(), impacto_score=4, esfuerzo_score=3,
        metrica="Renta retenida neta de descuentos; % retenidos con ajuste vs con descuento",
        ahorro_mensual_base=perdida * 0.20,
    ))

    # 3 · Oferta por motivo; sin descuento extra a cazadores de ofertas
    caza = sc[sc["BAN_CAZA_OFERTA"] == 1]
    disc_all = sc[(sc[I] == 1) & (sc["VAL_RENTA_M6"] > sc[A])]
    desc_prom = float((disc_all["VAL_RENTA_M6"] - disc_all[A]).mean())
    caza_int = caza[caza[I] == 1]
    acts.append(dict(
        id="A3", accionable="Oferta de retención por motivo (Next Best Offer) y sin descuento adicional a cazadores de ofertas",
        tipo="Reactivo",
        evidencia=f"{len(caza)} cazadores de ofertas: churn {_pct(caza[C].mean())}, intención {_pct(caza[I].mean())}; "
                  f"negocian pero no se van. Descuento promedio implícito: ${_n(desc_prom)}/mes",
        objetivo="Toda llamada de cancelación, con motivo clasificado por el NLP",
        n_objetivo=len(caza_int), p_irse=np.nan, arpu=np.nan, impacto_score=3, esfuerzo_score=2,
        metrica="Tasa de retención por motivo; costo de retención por cliente salvado",
        ahorro_mensual_base=len(caza_int) * desc_prom * 0.30,
    ))

    # 4 · Cobros no reconocidos en primer contacto
    rec = sc[sc["VAL_RECLAMOS_MES"] >= 2]
    norec = sc[sc["VAL_RECLAMOS_MES"] < 2]
    pct_precio = (nlp or {}).get("motivos_c3_pct", {}).get("precio_facturacion")
    acts.append(dict(
        id="A4", accionable="Resolver cobros no reconocidos en el primer contacto, con autonomía del agente para reversar",
        tipo="Reactivo",
        evidencia=f"Con 2+ reclamos en el mes el churn es {_pct(rec[C].mean(), 2)} vs {_pct(norec[C].mean(), 2)}"
                  + (f"; precio y facturación es el motivo del {pct_precio:.0f} % de las llamadas del Cluster 3" if pct_precio else ""),
        objetivo="Clientes con 2 o más reclamos en el mes",
        n_objetivo=len(rec), p_irse=rec[C].mean(), arpu=rec[A].median(), impacto_score=3, esfuerzo_score=3,
        metrica="Resolución en primer contacto; reclamos repetidos a 30 días", costo=0,
    ))

    # 5 · Lista semanal por riesgo (modelo)
    top = sc[sc["decil_churn"] == 1]
    obj5 = _rentables(top)
    acts.append(dict(
        id="A5", accionable="Lista semanal del decil de mayor riesgo de churn para contacto proactivo", tipo="Proactivo",
        evidencia=f"En validación cruzada, el decil 1 del modelo concentra {top[C].sum()} de {sc[C].sum()} bajas "
                  f"({_pct(top[C].sum() / sc[C].sum(), 0)}) contactando al 10 % de la base; se contacta a los "
                  f"{_n(len(obj5))} con valor esperado positivo, que concentran {obj5[C].sum()} bajas",
        objetivo="Decil 1 de riesgo de churn con valor esperado del contacto positivo", n_objetivo=len(obj5),
        p_irse=obj5[C].mean(), arpu=obj5[A].median(),
        impacto_score=4, esfuerzo_score=2, metrica="Churn del decil contactado vs grupo de control",
    ))

    # 6 · Coaching con trayectoria de sentimiento
    emp = (nlp or {}).get("sentimiento_c3", {}).get("empeora_pct")
    acts.append(dict(
        id="A6", accionable="Coaching de agentes de retención con la trayectoria de sentimiento de sus llamadas", tipo="Reactivo",
        evidencia=f"En el {emp:.0f} % de las llamadas del Cluster 3 el sentimiento empeora hacia el final" if emp else
        "Trayectoria de sentimiento por llamada disponible en el JSON",
        objetivo="Agentes del área de cancelación", n_objetivo=np.nan, p_irse=np.nan, arpu=np.nan,
        impacto_score=2, esfuerzo_score=1, metrica="% de llamadas con sentimiento que mejora; retención por agente",
    ))

    df = pd.DataFrame(acts)
    for esc, tasa in ESCENARIOS.items():
        vals = []
        for _, r in df.iterrows():
            if pd.notna(r.p_irse):
                costo = r.get("costo", COSTO_CONTACTO)
                vals.append(_impact(r.n_objetivo, r.p_irse, r.arpu, tasa, COSTO_CONTACTO if pd.isna(costo) else costo))
            elif pd.notna(r.get("ahorro_mensual_base", np.nan)):
                vals.append(r["ahorro_mensual_base"] / 0.30 * tasa * MESES)
            else:
                vals.append(np.nan)
        df[f"impacto_anual_{esc}_cop"] = vals
    df["prioridad"] = np.select(
        [(df.impacto_score >= 4) & (df.esfuerzo_score <= 2), df.impacto_score >= 4, df.esfuerzo_score <= 2],
        ["1 · Quick win", "2 · Estratégico", "3 · Complementario"], "4 · Evaluar")
    df = df.sort_values(["prioridad", "impacto_score"], ascending=[True, False]).reset_index(drop=True)
    df["id"] = [f"A{i + 1}" for i in range(len(df))]  # la numeración sigue la prioridad
    return df


def plot_matrix(df: pd.DataFrame) -> None:
    from cluster3.eda.profile import BLUE, GRID, INK, INK2, ORANGE

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.axhline(3, color=GRID, lw=1)
    ax.axvline(2.5, color=GRID, lw=1)
    rng = np.random.default_rng(1)
    for _, r in df.iterrows():
        x, y = r.esfuerzo_score + rng.uniform(-0.12, 0.12), r.impacto_score + rng.uniform(-0.12, 0.12)
        color = BLUE if r.tipo == "Proactivo" else ORANGE
        ax.scatter(x, y, s=260, color=color, edgecolor="white", linewidth=2, zorder=3)
        ax.text(x, y, r.id, ha="center", va="center", fontsize=8, color="white", zorder=4, weight="bold")
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Esfuerzo →", color=INK2)
    ax.set_ylabel("Impacto →", color=INK2)
    ax.text(0.7, 5.3, "Quick wins", color=INK2, fontsize=9)
    ax.text(2.7, 5.3, "Estratégicos", color=INK2, fontsize=9)
    ax.scatter([], [], color=BLUE, label="Proactivo")
    ax.scatter([], [], color=ORANGE, label="Reactivo")
    ax.legend(frameon=False, loc="lower right")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Matriz impacto / esfuerzo de accionables · Cluster 3", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / "matriz_impacto_esfuerzo.png", dpi=150)
    plt.close(fig)


def run_actions() -> pd.DataFrame:
    sc = pd.read_parquet(config.T_SCORES)
    nlp_file = config.OUT_JSON / "nlp_insights.json"
    nlp = json.loads(nlp_file.read_text(encoding="utf-8")) if nlp_file.exists() else None
    df = build_actions(sc, nlp)
    df.to_csv(config.OUT_TAB / "accionables.csv", index=False)
    plot_matrix(df)
    summary = {
        "supuestos": {"escenarios_tasa_exito": ESCENARIOS, "meses": MESES, "costo_contacto_cop": COSTO_CONTACTO},
        "accionables": df.replace({np.nan: None}).to_dict(orient="records"),
    }
    (config.OUT_JSON / "accionables.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return df
