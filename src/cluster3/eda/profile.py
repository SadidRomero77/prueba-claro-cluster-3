"""Perfil del Cluster 3: KPIs, segmentos y gráficos.

Cada segmento reporta tamaño, tasa de churn, tasa de intención y ARPU mediano,
para leer juntas las dos variables objetivo.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cluster3 import config  # noqa: E402

# Paleta de referencia (validada para daltonismo); texto en tinta neutra, nunca en color de serie.
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1"


def kpis(d: pd.DataFrame) -> dict:
    arpu = d[config.ARPU_COL]
    return {
        "clientes": int(len(d)),
        "churn_n": int(d[config.TARGET_CHURN].sum()),
        "churn_tasa": float(d[config.TARGET_CHURN].mean()),
        "intencion_n": int(d[config.TARGET_INTENCION].sum()),
        "intencion_tasa": float(d[config.TARGET_INTENCION].mean()),
        "reincidentes_3mas": int((d["CANTIDAD_INTENCIONES"] >= 3).sum()),
        "arpu_mediano": float(arpu.median()),
        "arpu_medio": float(arpu.mean()),
        "renta_mensual_total": float(arpu.sum()),
        "renta_mensual_churn": float(d.loc[d[config.TARGET_CHURN] == 1, config.ARPU_COL].sum()),
        "renta_mensual_intencion": float(d.loc[d[config.TARGET_INTENCION] == 1, config.ARPU_COL].sum()),
        "churn_si_intencion": float(d.loc[d[config.TARGET_INTENCION] == 1, config.TARGET_CHURN].mean()),
        "churn_no_intencion": float(d.loc[d[config.TARGET_INTENCION] == 0, config.TARGET_CHURN].mean()),
        "antiguedad_mediana_meses": float(d["ANTIGUEDAD_MESES"].median()),
        "pct_convergente": float(d["CONVERGENTE_CEDULA_SI"].mean()),
        "pct_fibra": float(d["TIPO_RED_FTT"].mean()),
        "pct_estrato_2_3": float(d["ESTRATO"].isin([2, 3]).mean()),
        "pct_tv": float(d["BAN_TV"].mean()),
        "pct_internet": float(d["BAN_INTERNET"].mean()),
        "pct_voz": float(d["BAN_VOZ"].mean()),
    }


def _segment_defs(d: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "Variación de renta vs 6 meses": pd.cut(
            d["VAL_VAR_RENTA"], [-101, -30, -5, 5, 30, 1e6], labels=["< -30 %", "-30 a -5 %", "±5 %", "5 a 30 %", "> 30 %"]
        ),
        "Score crediticio (quintil)": pd.qcut(d["SCORE_CREDITICIO_FIX"], 5, labels=["Q1 bajo", "Q2", "Q3", "Q4", "Q5 alto"]),
        "Downtime": pd.cut(d["VAL_DOWNTIME"], [-1, 0, 1000, 5000, 1e9], labels=["0", "1–1.000", "1.000–5.000", "> 5.000"]),
        "Llamadas técnicas negativas": pd.cut(d["VAL_LLAM_TEC_NEGATIVAS"], [-1, 0, 1, 3, 1e3], labels=["0", "1", "2–3", "4+"]),
        "Reclamos del mes": pd.cut(d["VAL_RECLAMOS_MES"], [-1, 0, 1, 3, 1e3], labels=["0", "1", "2–3", "4+"]),
        "Equipos adicionales": pd.cut(d["VAL_EQUIP_ADIC"], [-1, 0, 1, 1e3], labels=["0", "1", "2+"]),
        "Cazador de ofertas": d["BAN_CAZA_OFERTA"].map({0: "No", 1: "Sí"}),
        "Convergente (fijo + móvil)": d["CONVERGENTE_CEDULA_SI"].map({0: "No", 1: "Sí"}),
        "Red de fibra (FTTH)": d["TIPO_RED_FTT"].map({0: "No", 1: "Sí"}),
        "Estrato": d["ESTRATO"],
        "Antigüedad": pd.cut(d["ANTIGUEDAD_MESES"], [-1, 12, 36, 72, 120, 1e4], labels=["≤ 1 año", "1–3", "3–6", "6–10", "> 10 años"]),
        "Competencia en la zona": np.select(
            [d["CLASIF_OKK_ALTACOMP"] == 1, d["CLASIF_OKK_MEJORCOMP"] == 1, d["CLASIF_OKK_IGUALCONDICION"] == 1],
            ["Alta competencia", "Competencia mejor", "Igual condición"],
            default="Somos mejores / únicos",
        ),
    }


def segment_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, seg in _segment_defs(d).items():
        g = d.groupby(seg, observed=True)
        t = pd.DataFrame(
            {
                "clientes": g.size(),
                "churn_pct": g[config.TARGET_CHURN].mean() * 100,
                "intencion_pct": g[config.TARGET_INTENCION].mean() * 100,
                "arpu_mediano": g[config.ARPU_COL].median(),
            }
        ).reset_index(names="nivel")
        t.insert(0, "segmento", name)
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out["nivel"] = out["nivel"].astype(str)
    return out.round({"churn_pct": 2, "intencion_pct": 1, "arpu_mediano": 0})


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_segments(seg: pd.DataFrame, out_dir=config.OUT_FIG) -> list[str]:
    """Un gráfico por segmento: intención (%) y churn (%) en paneles separados (sin doble eje)."""
    files = []
    for name, t in seg.groupby("segmento", sort=False):
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(t) + 1.4), sharey=True)
        for ax, col, color, label in [
            (axes[0], "intencion_pct", BLUE, "Intención de cancelación (%)"),
            (axes[1], "churn_pct", ORANGE, "Churn (%)"),
        ]:
            y = np.arange(len(t))
            ax.barh(y, t[col], color=color, height=0.55)
            ax.set_yticks(y, t["nivel"])
            for yi, v, n in zip(y, t[col], t["clientes"]):
                ax.text(v, yi, (f"  {v:.2f}" if col == "churn_pct" else f"  {v:.1f}") + (f"  (n={n:,})".replace(",", ".") if col == "churn_pct" else ""),
                        va="center", fontsize=8, color=INK2)
            _style(ax, label)
            ax.set_xlim(0, max(t[col].max() * 1.35, 1))
        axes[0].invert_yaxis()
        fig.suptitle(name, x=0.01, ha="left", fontsize=13, color=INK)
        fig.tight_layout()
        fn = out_dir / f"segmento_{_slug(name)}.png"
        fig.savefig(fn, dpi=150)
        plt.close(fig)
        files.append(fn.name)
    return files


def _slug(s: str) -> str:
    import re
    import unicodedata

    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def run_eda(d: pd.DataFrame) -> dict:
    config.ensure_dirs()
    k = kpis(d)
    seg = segment_table(d)
    seg.to_csv(config.OUT_TAB / "segmentos_cluster3.csv", index=False)
    figs = plot_segments(seg)
    (config.OUT_JSON / "kpis_cluster3.json").write_text(json.dumps(k, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"kpis": k, "segmentos": seg, "figuras": figs}
