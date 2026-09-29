"""Pipeline completo con un solo comando.

    uv run cluster3                  # todo, NLP con baseline (sin API)
    uv run cluster3 --llm --jev      # NLP con LLM y Jev (requiere claves en .env)
    uv run cluster3 --desde modelo   # retomar desde una etapa

Etapas: calidad → limpieza → eda → modelo → nlp → accionables → reportes → base_agente
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from cluster3 import config

ETAPAS = ["calidad", "limpieza", "eda", "modelo", "nlp", "accionables", "reportes", "base_agente"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Cluster 3 · pipeline reproducible")
    ap.add_argument("--llm", action="store_true", help="Clasificar llamadas con LLM")
    ap.add_argument("--jev", action="store_true", help="Clasificar llamadas con Jev (TypeSafe)")
    ap.add_argument("--desde", choices=ETAPAS, default="calidad")
    a = ap.parse_args(argv)
    config.ensure_dirs()
    run = ETAPAS[ETAPAS.index(a.desde):]
    t0 = time.time()

    from cluster3.data.load import load_clientes, load_diccionario, load_llamadas

    raw = load_clientes()
    if "calidad" in run:
        from cluster3.data.quality import run_quality

        _log("Calidad de datos")
        run_quality(raw, load_llamadas(), load_diccionario())
    if "limpieza" in run:
        from cluster3.data.clean import run_clean

        _log("Limpieza y registro de decisiones")
        run_clean(raw)
    d = pd.read_parquet(config.T_CLIENTES_LIMPIO)
    if "eda" in run:
        from cluster3.eda.profile import run_eda

        _log("EDA del Cluster 3")
        run_eda(d)
    if "modelo" in run:
        from cluster3.models.churn import run_models

        _log("Modelo en dos etapas (intención y churn)")
        run_models(d, raw, pd.read_csv(config.OUT_TAB / "diccionario_reconciliado.csv"))
    if "nlp" in run:
        from cluster3.nlp.run import run as run_nlp

        _log("NLP de llamadas" + (" con LLM" if a.llm else "") + (" y Jev" if a.jev else ""))
        run_nlp(use_llm=a.llm, use_jev=a.jev)
    if "accionables" in run:
        from cluster3.business.actions import run_actions

        _log("Accionables e impacto económico")
        run_actions()
    if "reportes" in run:
        from cluster3.report import build_reports

        _log("Reportes en docs/")
        build_reports()
    if "base_agente" in run:
        from cluster3.agents.tools import build_duckdb

        _log("Base DuckDB para el agente")
        build_duckdb()
    _log(f"Listo en {time.time() - t0:.0f} s. Resultados en outputs/ y docs/")


def _log(msg: str) -> None:
    print(f"[cluster3] {msg}", flush=True)


if __name__ == "__main__":
    main()
