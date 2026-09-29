"""Sincroniza los artefactos del proyecto con un Volume de Unity Catalog (Databricks).

El job de Databricks corre el pipeline en disco local del cómputo y sube los artefactos al Volume
(`push`). La app de Databricks Apps los descarga al arrancar (`pull`). No importa `cluster3.config`
para poder ejecutarse antes de fijar CLUSTER3_ROOT.

    python -m cluster3.deploy.volume_sync push /Volumes/workspace/cluster3/artefactos --root /tmp/cluster3
    python -m cluster3.deploy.volume_sync pull /Volumes/workspace/cluster3/artefactos --root /tmp/cluster3
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Lo que la app y el agente necesitan. Nunca se sincroniza data/raw.
ARTEFACTOS = ["outputs", "models", "data/processed", "data/labels", "docs"]


def _local_copy(src: Path, dst: Path) -> int:
    n = 0
    for f in src.rglob("*"):
        if f.is_file():
            t = dst / f.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)
            n += 1
    return n


def push(volume: str, root: str) -> int:
    """Copia artefactos de root → volume (en un cluster o job, /Volumes está montado como disco)."""
    n = 0
    for sub in ARTEFACTOS:
        src = Path(root) / sub
        if src.exists():
            n += _local_copy(src, Path(volume) / sub)
    return n


def pull(volume: str, root: str) -> int:
    """Descarga artefactos volume → root. Usa /Volumes si está montado; si no, la API de archivos del SDK
    (caso Databricks Apps, que se autentica con el service principal de la app)."""
    if Path(volume).exists():
        n = 0
        for sub in ARTEFACTOS:
            src = Path(volume) / sub
            if src.exists():
                n += _local_copy(src, Path(root) / sub)
        return n

    from databricks.sdk import WorkspaceClient  # pip install databricks-sdk

    w = WorkspaceClient()
    n = 0

    def walk(remote: str):
        nonlocal n
        for e in w.files.list_directory_contents(remote):
            if e.is_directory:
                walk(e.path)
            else:
                rel = e.path[len(volume.rstrip("/")) + 1:]
                t = Path(root) / rel
                t.parent.mkdir(parents=True, exist_ok=True)
                with w.files.download(e.path).contents as r, open(t, "wb") as f:
                    shutil.copyfileobj(r, f)
                n += 1

    for sub in ARTEFACTOS:
        try:
            walk(f"{volume.rstrip('/')}/{sub}")
        except Exception as e:  # subcarpeta ausente
            print(f"[volume_sync] {sub}: {str(e)[:100]}")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=["push", "pull"])
    ap.add_argument("volume")
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    fn = push if a.accion == "push" else pull
    print(f"[volume_sync] {a.accion}: {fn(a.volume, a.root)} archivos")
