"""Etiquetado manual de la muestra de 50 llamadas (una sola vez, ~1 hora).

Muestra el texto del cliente de cada llamada pendiente y guarda las etiquetas en
data/labels/muestra_etiquetada.csv después de cada llamada, así se puede cortar y retomar.

    uv run python -m cluster3.nlp.etiquetar
    uv run python -m cluster3.nlp.etiquetar --ver-sugerencia   # muestra la sugerencia de reglas (sesga menos si no)

Alternativa en Excel (sin consola):
    uv run python -m cluster3.nlp.etiquetar --exportar-excel   # crea data/interim/etiquetado_50_llamadas.xlsx
    uv run python -m cluster3.nlp.etiquetar --importar-excel   # lleva las etiquetas del Excel al CSV

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


EXCEL = config.DATA_INTERIM / "etiquetado_50_llamadas.xlsx"  # contiene texto de llamadas: fuera de git
SENTIMIENTOS = {"-1 · negativo": -1, "0 · neutral": 0, "1 · positivo": 1}
COLS_EXCEL = ["id_llamada", "cluster", "calidad", "texto_cliente", "motivo", "submotivo", "urgencia", "sentimiento",
              "notas"]


def exportar_excel(path=EXCEL) -> None:
    """Excel con las llamadas pendientes, listas desplegables y una hoja de instrucciones."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    lab = pd.read_csv(LABELS_FILE)
    calls = pd.read_parquet(config.T_LLAMADAS_LIMPIAS).set_index("id_llamada")
    wb = Workbook()
    ws = wb.active
    ws.title = "Etiquetar"
    ws.append(COLS_EXCEL)
    for _, r in lab.iterrows():
        c = calls.loc[r["id_llamada"]]
        texto = c["texto_cliente"] if c["diarizacion_ok"] else c["texto_anonimizado"]
        sent = {v: k for k, v in SENTIMIENTOS.items()}.get(r["sentimiento_humano_-1_0_1"]) \
            if pd.notna(r["sentimiento_humano_-1_0_1"]) else None
        ws.append([int(r["id_llamada"]), int(r["cluster"]), c["calidad_transcripcion"], str(texto)[:32000],
                   r["motivo_humano"] if pd.notna(r["motivo_humano"]) else None,
                   r["submotivo_humano"] if pd.notna(r["submotivo_humano"]) else None,
                   r["urgencia_humana"] if pd.notna(r["urgencia_humana"]) else None, sent,
                   r["notas"] if pd.notna(r["notas"]) else None])

    # Hoja auxiliar con las opciones de las listas desplegables
    op = wb.create_sheet("Opciones")
    subs = [s for m in MOTIVOS.values() for s in m["submotivos"]]
    for i, col in enumerate([MOTIVOS_LISTA, subs, list(URGENCIA), list(SENTIMIENTOS)], 1):
        for j, v in enumerate(col, 1):
            op.cell(row=j, column=i, value=v)
    n = len(lab) + 1
    for letra, col_op, largo in [("E", "A", len(MOTIVOS_LISTA)), ("F", "B", len(subs)), ("G", "C", len(URGENCIA)),
                                 ("H", "D", len(SENTIMIENTOS))]:
        dv = DataValidation(type="list", formula1=f"=Opciones!${col_op}$1:${col_op}${largo}", allow_blank=True,
                            showErrorMessage=True, errorTitle="Valor no válido", error="Elige una opción de la lista")
        ws.add_data_validation(dv)
        dv.add(f"{letra}2:{letra}{n}")
    op.sheet_state = "hidden"

    # Formato: encabezado oscuro, texto ajustado, columnas a llenar resaltadas, encabezado fijo
    head = PatternFill("solid", fgColor="0B0B0B")
    llenar = PatternFill("solid", fgColor="E9F1FB")
    for cell in ws[1]:
        cell.font, cell.fill = Font(bold=True, color="FFFFFF"), head
        cell.alignment = Alignment(vertical="center")
    for fila in ws.iter_rows(min_row=2, max_row=n):
        for cell in fila:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for cell in fila[4:]:
            cell.fill = llenar
    for letra, ancho in zip("ABCDEFGHI", [11, 8, 11, 110, 22, 26, 11, 15, 30]):
        ws.column_dimensions[letra].width = ancho
    for i in range(2, n + 1):
        ws.row_dimensions[i].height = 260
    ws.freeze_panes = "E2"

    ins = wb.create_sheet("Instrucciones", 0)
    filas = [("Cómo etiquetar", ""),
             ("1", "Lee lo que dice el cliente (columna texto_cliente) y llena las columnas azules con las listas."),
             ("2", "No mires otras clasificaciones: la etiqueta humana es la referencia para medir reglas, LLM y Jev."),
             ("3", "Un solo motivo: el que más pesa en la decisión de cancelar. Si hay varios, anótalos en notas."),
             ("4", "Guarda el archivo y avisa. Se importa con: uv run python -m cluster3.nlp.etiquetar --importar-excel"),
             ("", ""), ("Motivo", "Cuándo usarlo · submotivos")]
    filas += [(m, f"{v['definicion']} · {', '.join(v['submotivos'])}") for m, v in MOTIVOS.items()]
    filas += [("", ""), ("Urgencia", "")] + [(k, v) for k, v in URGENCIA.items()]
    filas += [("", ""), ("Sentimiento", "Cómo se siente el cliente en toda la llamada"),
              ("-1 · negativo", "Molesto, frustrado, enojado"), ("0 · neutral", "Tono informativo, sin carga"),
              ("1 · positivo", "Conforme o agradecido")]
    for f in filas:
        ins.append(list(f))
    for r_ in (1, 7, 7 + len(MOTIVOS) + 2, 7 + len(MOTIVOS) + 2 + len(URGENCIA) + 2):
        ins.cell(row=r_, column=1).font = Font(bold=True, size=12 if r_ == 1 else 11)
    ins.column_dimensions["A"].width = 22
    ins.column_dimensions["B"].width = 120
    for fila in ins.iter_rows():
        for cell in fila:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    print(f"Excel creado: {path}")


def importar_excel(path=EXCEL) -> None:
    """Lleva las etiquetas del Excel a data/labels/muestra_etiquetada.csv y valida los valores."""
    x = pd.read_excel(path, sheet_name="Etiquetar")
    lab = pd.read_csv(LABELS_FILE)
    for col in ["motivo_humano", "submotivo_humano", "urgencia_humana", "sentimiento_humano_-1_0_1", "notas"]:
        lab[col] = lab[col].astype("object")
    def _sent(v):
        """Acepta la opción de la lista ("-1 · negativo") o el número escrito a mano (-1, 0, 1)."""
        if pd.isna(v):
            return None
        if v in SENTIMIENTOS:
            return SENTIMIENTOS[v]
        try:
            n = int(float(str(v).split("·")[0].strip()))
        except ValueError:
            return "invalido"
        return n if n in (-1, 0, 1) else "invalido"

    errores = []
    for _, r in x.iterrows():
        if pd.isna(r["motivo"]):
            continue
        idx = lab.index[lab["id_llamada"] == r["id_llamada"]]
        sent = _sent(r["sentimiento"])
        if r["motivo"] not in MOTIVOS or (pd.notna(r["urgencia"]) and r["urgencia"] not in URGENCIA) \
                or sent == "invalido":
            errores.append(int(r["id_llamada"]))
            continue
        lab.loc[idx, "motivo_humano"] = r["motivo"]
        lab.loc[idx, "submotivo_humano"] = r["submotivo"] if pd.notna(r["submotivo"]) else None
        lab.loc[idx, "urgencia_humana"] = r["urgencia"] if pd.notna(r["urgencia"]) else None
        lab.loc[idx, "sentimiento_humano_-1_0_1"] = sent
        lab.loc[idx, "notas"] = r["notas"] if pd.notna(r["notas"]) else None
    lab.to_csv(LABELS_FILE, index=False)
    print(f"Etiquetadas: {lab['motivo_humano'].notna().sum()} de {len(lab)}"
          + (f" · con valores no válidos (revisar): {errores}" if errores else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver-sugerencia", action="store_true")
    ap.add_argument("--exportar-excel", action="store_true")
    ap.add_argument("--importar-excel", nargs="?", const=str(EXCEL), metavar="RUTA",
                    help="Excel etiquetado (por defecto data/interim/etiquetado_50_llamadas.xlsx)")
    a = ap.parse_args()
    if a.exportar_excel:
        return exportar_excel()
    if a.importar_excel:
        from pathlib import Path

        return importar_excel(Path(a.importar_excel))

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
