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


def _competidor(v) -> str | None:
    """Normaliza el competidor (texto libre en el LLM): operador por nombre, 'otro operador (sin nombre)' o None."""
    import re

    if not isinstance(v, str) or not v.strip():
        return None
    t = v.strip().lower()
    m = re.search(r"\b(tigo|movistar|wom|etb|une|directv|hughesnet)\b", t)
    if m:
        return m.group(1)
    if re.fullmatch(r"(claro|no_identificado|ninguno|n/?a|none|null|\[nombre\])", t):
        return None
    return "otro operador (sin nombre)"


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
    comp = c3["competidor_mencionado"].map(_competidor).value_counts().to_dict()
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


def hibrido(llm_df: pd.DataFrame, jdf: pd.DataFrame) -> pd.DataFrame:
    """Método final elegido con el benchmark sobre 50 llamadas etiquetadas a mano (nlp_benchmark.csv):
    Jev para motivo y sentimiento global (mejor kappa y correlación), LLM para urgencia, submotivo, evidencia
    textual, trayectoria de sentimiento y acción sugerida (Jev no genera texto)."""
    from cluster3.nlp.taxonomy import MOTIVOS

    d = llm_df.merge(jdf[["id_llamada", "jev_motivo", "jev_motivo_prob", "jev_sentimiento"]], on="id_llamada", how="left")
    d["motivo_llm"] = d["motivo"]
    usa_jev = d["jev_motivo"].notna() & (d["calidad_transcripcion"] != "sin_contenido")
    d.loc[usa_jev, "motivo"] = d.loc[usa_jev, "jev_motivo"]
    # El submotivo del LLM solo se conserva si pertenece al motivo final
    ok_sub = [s in MOTIVOS.get(m, {}).get("submotivos", {}) for m, s in zip(d["motivo"], d["submotivo"])]
    d.loc[~pd.Series(ok_sub, index=d.index), "submotivo"] = "no_identificado"
    d.loc[d["jev_sentimiento"].notna(), "sent_global"] = d["jev_sentimiento"]
    # Filas sin resultado del LLM (respaldo con reglas) quedan marcadas para poder contarlas y reportarlas
    d["metodo"] = np.where(d["metodo"] == "baseline_reglas", "hibrido_jev_reglas", "hibrido_jev_llm")
    return d


def _clasificar_llm(clf, target: pd.DataFrame, hilos: int = 5):
    """Clasifica con el LLM en paralelo (las llamadas son independientes) y conserva el orden."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    hechos, lock = [0], Lock()

    def uno(r):
        try:
            out = clf.classify(r)
        except Exception as e:  # una llamada fallida no detiene el lote
            out = (None, {"id_llamada": int(r["id_llamada"]), "error": str(e)[:200]})
        with lock:
            hechos[0] += 1
            print(f"[llm] llamada {hechos[0]}/{len(target)}", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=hilos) as ex:
        res = list(ex.map(uno, [r for _, r in target.iterrows()]))
    return [a for a, _ in res if a is not None], [m for _, m in res]


def run(use_llm: bool = False, use_jev: bool = False, solo_muestra: bool = False) -> dict:
    config.ensure_dirs()
    pre = preprocess_calls(load_llamadas())
    pre.to_parquet(config.T_LLAMADAS_LIMPIAS, index=False)

    base_an = classify_all(pre)
    write_jsonl(base_an, config.OUT_JSON / "llamadas_baseline.jsonl")
    base = flatten(base_an)
    make_label_template(pre, base)
    preds = {"baseline": base[["id_llamada", "motivo", "urgencia", "sent_global"]].rename(columns={"sent_global": "sentimiento"})}
    final = base

    target = pre
    if solo_muestra and LABELS_FILE.exists():
        target = pre[pre["id_llamada"].isin(pd.read_csv(LABELS_FILE)["id_llamada"])]

    if use_llm:
        from cluster3.nlp.llm_classifier import LLMClassifier

        from cluster3.nlp.schema import CallAnalysis

        clf = LLMClassifier(pre)
        # Reanudar: en la corrida completa se reutilizan las llamadas que el LLM ya clasificó con éxito
        # (por ejemplo, si una corrida se cortó por falta de créditos); solo se clasifican las que faltan.
        previas, trazas_prev = [], pd.DataFrame()
        f_llm, f_traz = config.OUT_JSON / "llamadas_llm.jsonl", config.OUT_TAB / "llm_trazas.csv"
        if not solo_muestra and f_llm.exists():
            previas = [CallAnalysis.model_validate_json(x) for x in f_llm.read_text(encoding="utf-8").splitlines() if x]
            if f_traz.exists():
                trazas_prev = pd.read_csv(f_traz)
                if "error" in trazas_prev:
                    trazas_prev = trazas_prev[trazas_prev["error"].isna()]
        hechas = {a.id_llamada for a in previas}
        faltan = target[~target["id_llamada"].isin(hechas)]
        if hechas:
            print(f"[llm] {len(hechas)} llamadas ya clasificadas; faltan {len(faltan)}", flush=True)
        nuevas, metas = _clasificar_llm(clf, faltan) if len(faltan) else ([], [])
        llm_an = previas + nuevas
        write_jsonl(llm_an, f_llm)
        pd.concat([trazas_prev, pd.DataFrame(metas)], ignore_index=True).to_csv(f_traz, index=False)
        llm_df = flatten(llm_an)
        preds["llm"] = llm_df[["id_llamada", "motivo", "urgencia", "sent_global"]].rename(columns={"sent_global": "sentimiento"})
        if not solo_muestra:
            # Respaldo: una llamada sin resultado del LLM conserva la fila de reglas (no se descarta)
            sin_llm = base[~base["id_llamada"].isin(llm_df["id_llamada"])]
            final = pd.concat([llm_df, sin_llm], ignore_index=True)

    if use_jev:
        from cluster3.nlp.jev_classifier import JevClassifier

        jev = JevClassifier()
        jrows = []
        for i, (_, r) in enumerate(target.iterrows(), 1):
            if i % 25 == 0 or i == len(target):
                print(f"[jev] llamada {i}/{len(target)}", flush=True)
            try:
                jrows.append(jev.classify(r))
            except Exception as e:
                jrows.append({"id_llamada": int(r["id_llamada"]), "error": str(e)[:200]})
        jdf = pd.DataFrame(jrows)
        jdf.to_csv(config.OUT_TAB / "jev_resultados.csv", index=False)
        if "jev_motivo" in jdf:
            preds["jev"] = jdf.rename(columns={"jev_motivo": "motivo", "jev_urgencia": "urgencia", "jev_sentimiento": "sentimiento"})[["id_llamada", "motivo", "urgencia", "sentimiento"]]
            if not solo_muestra:
                final = hibrido(final, jdf) if use_llm else final.merge(jdf, on="id_llamada", how="left")

    if use_llm and use_jev and not solo_muestra:
        preds["hibrido"] = final[["id_llamada", "motivo", "urgencia", "sent_global"]].rename(
            columns={"sent_global": "sentimiento"})
        final.to_parquet(config.T_LLAMADAS_HIBRIDO, index=False)
        final.to_json(config.OUT_JSON / "llamadas_final.jsonl", orient="records", lines=True, force_ascii=False)
    elif not use_llm and not use_jev and config.T_LLAMADAS_HIBRIDO.exists():
        print("[nlp] Se reutiliza la clasificación híbrida (Jev + LLM) ya calculada; "
              "para recalcularla: python -m cluster3.nlp.run --llm --jev", flush=True)
        final = pd.read_parquet(config.T_LLAMADAS_HIBRIDO)
    final.to_parquet(config.T_LLAMADAS_ANALISIS, index=False)
    bench = evaluate(preds)
    ins = insights(final)
    ins["metodo_final"] = final["metodo"].mode()[0]
    ins["metodos_por_llamada"] = {str(k): int(v) for k, v in final["metodo"].value_counts().items()}
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
