"""Pipeline NLP: limpieza → clasificación (baseline / LLM / Jev) → JSON → insights → cruce agregado.

Uso:
    python -m cluster3.nlp.run                 # baseline (sin API)
    python -m cluster3.nlp.run --llm           # + LLM (requiere clave del proveedor)
    python -m cluster3.nlp.run --llm --jev     # + Jev (requiere TYPESAFE_API_KEY)
    python -m cluster3.nlp.run --llm --solo-muestra   # solo la muestra etiquetada (para comparar)
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cluster3 import config  # noqa: E402
from cluster3.data.load import load_llamadas  # noqa: E402
from cluster3.nlp.baseline import classify_all  # noqa: E402
from cluster3.nlp.evaluate import evaluate  # noqa: E402
from cluster3.nlp.preprocess import preprocess_calls  # noqa: E402
from cluster3.nlp.rag import LABELS_FILE  # noqa: E402


def flatten(analyses) -> pd.DataFrame:
    rows = []
    for a in analyses:
        d = a.model_dump(by_alias=True)
        s = d.pop("sentimiento")
        o = d.pop("oferta_retencion")
        d.update({
            "sent_inicio": s["inicio"], "sent_fin": s["fin"], "sent_global": s["global"],
            "sent_delta": round(s["fin"] - s["inicio"], 3), "emociones": ",".join(s["emociones"]),
            "oferta_ofrecida": o["ofrecida"], "oferta_tipo": o["tipo"],
            "motivos_secundarios": ",".join(d["motivos_secundarios"]),
            "sentimiento_por_aspecto": json.dumps(d["sentimiento_por_aspecto"], ensure_ascii=False),
        })
        rows.append(d)
    return pd.DataFrame(rows)


def write_jsonl(analyses, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for a in analyses:
            f.write(a.model_dump_json(by_alias=True) + "\n")


def make_label_template(pre: pd.DataFrame, base: pd.DataFrame, n_c3: int = 40, n_other: int = 10) -> None:
    """Muestra estratificada para etiquetar a mano. No se sobrescribe si ya existe."""
    if LABELS_FILE.exists():
        return
    ok = pre[pre["calidad_transcripcion"] != "sin_contenido"].merge(base[["id_llamada", "motivo"]], on="id_llamada")
    c3 = ok[ok["cluster"] == config.CLUSTER_CRITICO]
    frac = n_c3 / len(c3)
    s3 = c3.groupby("motivo").sample(frac=frac, random_state=config.SEED)
    if len(s3) < n_c3:
        s3 = pd.concat([s3, c3.drop(s3.index).sample(n_c3 - len(s3), random_state=config.SEED)])
    s3 = s3.head(n_c3)
    so = ok[ok["cluster"] != config.CLUSTER_CRITICO].sample(n_other, random_state=config.SEED)
    sample = pd.concat([s3, so])[["id_llamada", "cluster", "motivo"]].rename(columns={"motivo": "sugerencia_baseline"})
    for c in ["motivo_humano", "submotivo_humano", "urgencia_humana", "sentimiento_humano_-1_0_1", "notas"]:
        sample[c] = ""
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.sort_values(["cluster", "id_llamada"]).to_csv(LABELS_FILE, index=False)


def insights(final: pd.DataFrame) -> dict:
    ok = final[final["calidad_transcripcion"] != "sin_contenido"].copy()
    ok["grupo"] = np.where(ok["cluster"] == config.CLUSTER_CRITICO, "Cluster 3", "Otros clústeres")
    mot = (pd.crosstab(ok["motivo"], ok["grupo"], normalize="columns") * 100).round(1)
    mot["diferencia_pp"] = (mot["Cluster 3"] - mot["Otros clústeres"]).round(1)
    mot = mot.sort_values("Cluster 3", ascending=False)
    mot.to_csv(config.OUT_TAB / "nlp_motivos_c3_vs_otros.csv")
    c3 = ok[ok["grupo"] == "Cluster 3"]
    sub = c3.groupby(["motivo", "submotivo"]).size().rename("llamadas").reset_index().sort_values("llamadas", ascending=False)
    sub["pct"] = (sub["llamadas"] / len(c3) * 100).round(1)
    sub.to_csv(config.OUT_TAB / "nlp_submotivos_c3.csv", index=False)
    sent = c3.groupby("motivo").agg(llamadas=("id_llamada", "size"), sent_inicio=("sent_inicio", "mean"),
                                    sent_fin=("sent_fin", "mean"), sent_delta=("sent_delta", "mean"),
                                    urgencia_alta_pct=("urgencia", lambda s: (s == "alta").mean() * 100),
                                    reincidencia_pct=("reincidencia_mencionada", lambda s: s.mean() * 100),
                                    retenido_pct=("resultado", lambda s: (s == "retenido").mean() * 100)).round(2)
    sent.sort_values("llamadas", ascending=False).to_csv(config.OUT_TAB / "nlp_sentimiento_por_motivo_c3.csv")
    comp = c3["competidor_mencionado"].value_counts().to_dict()
    res = {
        "llamadas_validas": int(len(ok)),
        "llamadas_c3": int(len(c3)),
        "motivos_c3_pct": mot["Cluster 3"].to_dict(),
        "motivos_otros_pct": mot["Otros clústeres"].to_dict(),
        "urgencia_c3_pct": (c3["urgencia"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "resultado_c3_pct": (c3["resultado"].value_counts(normalize=True) * 100).round(1).to_dict(),
        "sentimiento_c3": {"inicio": round(float(c3["sent_inicio"].mean()), 3), "fin": round(float(c3["sent_fin"].mean()), 3),
                           "empeora_pct": round(float((c3["sent_delta"] < 0).mean() * 100), 1)},
        "competidores_c3": {str(k): int(v) for k, v in comp.items()},
        "reincidencia_c3_pct": round(float(c3["reincidencia_mencionada"].mean() * 100), 1),
        "oferta_ofrecida_c3_pct": round(float(c3["oferta_ofrecida"].mean() * 100), 1),
    }
    _plot_motivos(mot)
    return res


def _plot_motivos(mot: pd.DataFrame) -> None:
    from cluster3.eda.profile import BLUE, ORANGE, _style

    m = mot.iloc[::-1]
    y = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(m) + 1.4))
    ax.barh(y + 0.2, m["Cluster 3"], height=0.38, color=BLUE, label="Cluster 3")
    ax.barh(y - 0.2, m["Otros clústeres"], height=0.38, color=ORANGE, label="Otros clústeres")
    ax.set_yticks(y, m.index)
    for yi, v in zip(y, m["Cluster 3"]):
        ax.text(v, yi + 0.2, f"  {v:.0f} %", va="center", fontsize=8)
    ax.legend(frameon=False, loc="lower right")
    _style(ax, "Motivo principal de cancelación (% de llamadas)")
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / "nlp_motivos_c3_vs_otros.png", dpi=150)
    plt.close(fig)


# Puente agregado llamadas ↔ dataset (no existe llave individual).
PUENTE = {
    "precio_facturacion": ("Renta subió más de 5 % vs hace 6 meses", lambda d: d["VAL_VAR_RENTA"] > 5),
    "servicios_no_usados": ("Tiene equipos adicionales", lambda d: d["VAL_EQUIP_ADIC"] > 0),
    "falla_tecnica": ("Llamadas técnicas negativas o downtime > 0", lambda d: (d["VAL_LLAM_TEC_NEGATIVAS"] > 0) | (d["VAL_DOWNTIME"] > 0)),
    "competencia": ("Zona de alta competencia o competencia mejor", lambda d: (d["CLASIF_OKK_ALTACOMP"] == 1) | (d["CLASIF_OKK_MEJORCOMP"] == 1)),
    "traslado_cobertura": ("Tiene marca de traslado", lambda d: d["BAN_TRASLADO"] == 1),
    "atencion_servicio": ("2 o más reclamos en el mes", lambda d: d["VAL_RECLAMOS_MES"] >= 2),
    "situacion_economica": ("Score crediticio en el quintil más bajo", lambda d: d["SCORE_CREDITICIO_FIX"] <= d["SCORE_CREDITICIO_FIX"].quantile(0.2)),
}


def bridge(final: pd.DataFrame, clientes: pd.DataFrame) -> pd.DataFrame:
    c3 = final[(final["cluster"] == config.CLUSTER_CRITICO) & (final["calidad_transcripcion"] != "sin_contenido")]
    rows = []
    for motivo, (desc, fn) in PUENTE.items():
        mask = fn(clientes).fillna(False)
        rows.append({
            "motivo_llamadas": motivo,
            "pct_llamadas_c3": round((c3["motivo"] == motivo).mean() * 100, 1),
            "variable_dataset": desc,
            "pct_clientes_c3": round(mask.mean() * 100, 1),
            "intencion_con_senal_pct": round(clientes.loc[mask, config.TARGET_INTENCION].mean() * 100, 1),
            "intencion_sin_senal_pct": round(clientes.loc[~mask, config.TARGET_INTENCION].mean() * 100, 1),
            "churn_con_senal_pct": round(clientes.loc[mask, config.TARGET_CHURN].mean() * 100, 2),
            "churn_sin_senal_pct": round(clientes.loc[~mask, config.TARGET_CHURN].mean() * 100, 2),
        })
    out = pd.DataFrame(rows)
    out.to_csv(config.OUT_TAB / "puente_llamadas_dataset.csv", index=False)
    return out


def run(use_llm: bool = False, use_jev: bool = False, solo_muestra: bool = False) -> dict:
    config.ensure_dirs()
    pre = preprocess_calls(load_llamadas())
    pre.to_parquet(config.T_LLAMADAS_LIMPIAS, index=False)

    base_an = classify_all(pre)
    write_jsonl(base_an, config.OUT_JSON / "llamadas_baseline.jsonl")
    base = flatten(base_an)
    make_label_template(pre, base)
    preds = {"baseline": base[["id_llamada", "motivo", "urgencia"]]}
    final = base

    target = pre
    if solo_muestra and LABELS_FILE.exists():
        target = pre[pre["id_llamada"].isin(pd.read_csv(LABELS_FILE)["id_llamada"])]

    if use_llm:
        from cluster3.nlp.llm_classifier import LLMClassifier

        clf = LLMClassifier(pre)
        llm_an, metas = [], []
        for _, r in target.iterrows():
            try:
                a, meta = clf.classify(r)
                llm_an.append(a)
                metas.append(meta)
            except Exception as e:  # una llamada fallida no detiene el lote
                metas.append({"id_llamada": int(r["id_llamada"]), "error": str(e)[:200]})
        write_jsonl(llm_an, config.OUT_JSON / "llamadas_llm.jsonl")
        pd.DataFrame(metas).to_csv(config.OUT_TAB / "llm_trazas.csv", index=False)
        llm_df = flatten(llm_an)
        preds["llm"] = llm_df[["id_llamada", "motivo", "urgencia"]]
        if not solo_muestra:
            final = llm_df

    if use_jev:
        from cluster3.nlp.jev_classifier import JevClassifier

        jev = JevClassifier()
        jrows = []
        for _, r in target.iterrows():
            try:
                jrows.append(jev.classify(r))
            except Exception as e:
                jrows.append({"id_llamada": int(r["id_llamada"]), "error": str(e)[:200]})
        jdf = pd.DataFrame(jrows)
        jdf.to_csv(config.OUT_TAB / "jev_resultados.csv", index=False)
        if "jev_motivo" in jdf:
            preds["jev"] = jdf.rename(columns={"jev_motivo": "motivo", "jev_urgencia": "urgencia"})[["id_llamada", "motivo", "urgencia"]]
            if not solo_muestra:
                final = final.merge(jdf, on="id_llamada", how="left")

    final.to_parquet(config.T_LLAMADAS_ANALISIS, index=False)
    bench = evaluate(preds)
    ins = insights(final)
    ins["metodo_final"] = final["metodo"].iloc[0]
    ins["preprocesamiento"] = {
        "roles_corregidos": int(pre["roles_corregidos"].sum()),
        "pii_reemplazos": int(pre["pii_reemplazos"].sum()),
        "llamadas_con_pii_reemplazada": int((pre["pii_reemplazos"] > 0).sum()),
        "sin_contenido": int((pre["calidad_transcripcion"] == "sin_contenido").sum()),
        "diarizacion_fallida": int((~pre["diarizacion_ok"]).sum()),
    }
    ins["benchmark"] = bench.to_dict(orient="records")
    clientes = pd.read_parquet(config.T_CLIENTES_LIMPIO) if config.T_CLIENTES_LIMPIO.exists() else None
    if clientes is not None:
        ins["puente"] = bridge(final, clientes).to_dict(orient="records")
    (config.OUT_JSON / "nlp_insights.json").write_text(json.dumps(ins, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--jev", action="store_true")
    ap.add_argument("--solo-muestra", action="store_true")
    a = ap.parse_args()
    ins = run(a.llm, a.jev, a.solo_muestra)
    print(json.dumps({k: ins[k] for k in ["metodo_final", "motivos_c3_pct", "urgencia_c3_pct"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
