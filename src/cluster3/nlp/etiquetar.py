"""Etiquetado manual de la muestra de 50 llamadas (una sola vez, ~1 hora).

Muestra el texto del cliente de cada llamada pendiente y guarda las etiquetas en
data/labels/muestra_etiquetada.csv después de cada llamada, así se puede cortar y retomar.

    uv run python -m cluster3.nlp.etiquetar
    uv run python -m cluster3.nlp.etiquetar --ver-sugerencia   # muestra la sugerencia de reglas (sesga menos si no)

Después: uv run python -m cluster3.nlp.run --llm --jev --solo-muestra
"""
from __future__ import annotations

import argparse
import textwrap

import pandas as pd

from cluster3 import config
from cluster3.nlp.rag import LABELS_FILE
from cluster3.nlp.taxonomy import MOTIVOS, URGENCIA

MOTIVOS_LISTA = list(MOTIVOS)
URG = {"a": "alta", "m": "media", "b": "baja"}


def _ask(prompt: str, valid: set[str] | None = None, allow_empty: bool = False) -> str:
    while True:
        v = input(prompt).strip()
        if v == "q":
            raise KeyboardInterrupt
        if (allow_empty and v == "") or valid is None or v in valid:
            return v
        print(f"  opciones válidas: {', '.join(sorted(valid))} (q para salir)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver-sugerencia", action="store_true")
    a = ap.parse_args()

    lab = pd.read_csv(LABELS_FILE)
    for col in ["motivo_humano", "submotivo_humano", "urgencia_humana", "sentimiento_humano_-1_0_1", "notas"]:
        lab[col] = lab[col].astype("object")
    calls = pd.read_parquet(config.T_LLAMADAS_LIMPIAS).set_index("id_llamada")
    pend = lab[lab["motivo_humano"].isna()].index
    print(f"{len(lab) - len(pend)} de {len(lab)} llamadas etiquetadas. Escribe q para salir y guardar.\n")
    print("Motivos:")
    for i, m in enumerate(MOTIVOS_LISTA, 1):
        print(f"  {i}. {m}")
    print("Urgencia: " + " · ".join(f"{k[0]}={k} ({v})" for k, v in URGENCIA.items()) + "\n")

    try:
        for n, idx in enumerate(pend, 1):
            r = lab.loc[idx]
            c = calls.loc[r["id_llamada"]]
            print("=" * 100)
            print(f"[{n}/{len(pend)}] llamada {r['id_llamada']} · cluster {r['cluster']} · "
                  f"calidad {c['calidad_transcripcion']}")
            if a.ver_sugerencia:
                print(f"Sugerencia de reglas: {r['sugerencia_baseline']}")
            texto = c["texto_cliente"] if c["diarizacion_ok"] else c["texto_anonimizado"]
            print(textwrap.fill(str(texto)[:2500], 100))
            m = _ask("motivo (1-8): ", {str(i) for i in range(1, len(MOTIVOS_LISTA) + 1)})
            lab.at[idx, "motivo_humano"] = MOTIVOS_LISTA[int(m) - 1]
            lab.at[idx, "submotivo_humano"] = _ask("submotivo (opcional, Enter para omitir): ", allow_empty=True) or None
            lab.at[idx, "urgencia_humana"] = URG[_ask("urgencia (a/m/b): ", set(URG))]
            lab.at[idx, "sentimiento_humano_-1_0_1"] = int(_ask("sentimiento (-1/0/1): ", {"-1", "0", "1"}))
            lab.at[idx, "notas"] = _ask("notas (opcional): ", allow_empty=True) or None
            lab.to_csv(LABELS_FILE, index=False)
    except (KeyboardInterrupt, EOFError):
        print("\nGuardado. Puedes retomar cuando quieras.")
    lab.to_csv(LABELS_FILE, index=False)
    print(f"Etiquetadas: {lab['motivo_humano'].notna().sum()} de {len(lab)}")


if __name__ == "__main__":
    main()
