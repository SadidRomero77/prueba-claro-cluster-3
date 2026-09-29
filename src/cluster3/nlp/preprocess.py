"""Limpieza de transcripciones antes de cualquier modelo.

1. Segunda pasada de anonimización (nombres que quedaron, teléfonos, cédulas, correos).
2. Corrección de roles AGENT/CLIENT cuando la diarización quedó invertida.
3. Marca de llamadas sin contenido.
4. Separación de los turnos del cliente (inicio y fin) para medir la trayectoria del sentimiento.
"""
from __future__ import annotations

import re

import pandas as pd

AGENT_SCRIPT = [
    r"[áa]rea de cancelaci", r"con qui[ée]n tengo el gusto", r"en qu[ée] (?:le|te) puedo (?:ayudar|colaborar)",
    r"permítame|perm[íi]teme", r"por su amable espera", r"mi nombre es", r"le habla", r"validando", r"me confirma",
    r"n[úu]mero de (?:c[ée]dula|documento)", r"le ofrezco|te ofrezco", r"beneficio", r"en l[íi]nea",
]
TURN_RE = re.compile(r"^(AGENT|CLIENT):\s?", re.M)

PII_PATTERNS = [
    (re.compile(r"(?i)\b(mi nombre es|me llamo|hablas con|habla con|le habla|te habla)\s+(?!\[)([a-záéíóúñ]{3,})"), r"\1 [NOMBRE]"),
    (re.compile(r"\b\d{7,}\b"), "[NUMERO]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
]


def split_turns(text: str) -> list[tuple[str, str]]:
    parts = TURN_RE.split(text)
    turns = []
    for i in range(1, len(parts) - 1, 2):
        turns.append((parts[i], parts[i + 1].strip()))
    return turns


def _agent_score(t: str) -> int:
    return sum(bool(re.search(p, t, re.I)) for p in AGENT_SCRIPT)


def fix_roles(turns: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], bool]:
    """Invierte los roles si el guion del agente aparece más en los turnos marcados CLIENT."""
    a = sum(_agent_score(t) for r, t in turns if r == "AGENT")
    c = sum(_agent_score(t) for r, t in turns if r == "CLIENT")
    if c > a:
        return [("CLIENT" if r == "AGENT" else "AGENT", t) for r, t in turns], True
    return turns, False


def anonymize(text: str) -> tuple[str, int]:
    n = 0
    for pat, rep in PII_PATTERNS:
        text, k = pat.subn(rep, text)
        n += k
    return text, n


def strip_placeholders(text: str) -> str:
    """Quita los [NOMBRE] que el anonimizador puso al inicio de casi cada turno (ruido para los modelos)."""
    text = re.sub(r"\[NOMBRE\][,.]?\s*", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def preprocess_calls(calls: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in calls.iterrows():
        text, n_pii = anonymize(r["transcription"])
        turns, swapped = fix_roles(split_turns(text))
        client = [strip_placeholders(t) for role, t in turns if role == "CLIENT"]
        client = [t for t in client if t]
        # Diarización fallida: ambos hablantes quedaron en un solo rol. Se usa todo el texto.
        diarizacion_ok = len(client) >= 3
        if not diarizacion_ok:
            client = [x for x in (strip_placeholders(t) for _, t in turns) if x]
        third = max(1, len(client) // 3)
        n_chars = len(r["transcription"])
        if n_chars < 1000:
            calidad = "sin_contenido"
        elif not diarizacion_ok or n_chars <= 2500:
            calidad = "baja"
        else:
            calidad = "alta" if n_chars > 5000 else "media"
        rows.append(
            {
                "id_llamada": int(r["ID"]),
                "cluster": int(r["cluster"]),
                "n_caracteres": n_chars,
                "n_turnos": len(turns),
                "n_turnos_cliente": len(client),
                "roles_corregidos": swapped,
                "diarizacion_ok": diarizacion_ok,
                "pii_reemplazos": n_pii,
                "calidad_transcripcion": calidad,
                "texto_anonimizado": "\n".join(f"{role}: {t}" for role, t in turns),
                "texto_cliente": " ".join(client),
                "cliente_inicio": " ".join(client[:third]),
                "cliente_fin": " ".join(client[-third:]),
            }
        )
    return pd.DataFrame(rows)
