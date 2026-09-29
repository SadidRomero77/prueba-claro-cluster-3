"""Taxonomía de motivos y submotivos de cancelación (versionada).

La usan tres clasificadores:
- baseline por reglas (patrones) — corre sin API, sirve de piso de comparación;
- LLM (definiciones + ejemplos recuperados por RAG);
- Jev de TypeSafe (criterios de cada opción).
"""
from __future__ import annotations

TAXONOMY_VERSION = "v1.1"  # v1.1: operadores como palabra completa ("contigo" contaba como Tigo)

MOTIVOS: dict[str, dict] = {
    "precio_facturacion": {
        "definicion": "El cliente cancela por el valor de la factura: incrementos, cobros que no reconoce o fin de un descuento.",
        "submotivos": {
            "incremento_tarifa": r"(?:subi[óo]|increment|aument|m[áa]s caro|ahora me cobran|me lleg[óo] (?:m[áa]s|muy) alt)",
            "cobro_no_reconocido": r"(?:no (?:lo |la )?ped[íi]|cobr\w+ (?:algo|cosas|servicios) que no|cobro indebido|me est[áa]n cobrando|doble cobro|cobran de m[áa]s)",
            "fin_promocion_descuento": r"(?:se (?:me )?acab[óo] (?:el|la) (?:descuento|promoci)|termin[óo] (?:el|la) (?:descuento|promoci)|ya no tengo (?:el )?descuento)",
            "factura_alta": r"(?:factura|caro|precio|tarifa|valor|pagando mucho)",
        },
    },
    "servicios_no_usados": {
        "definicion": "Paga por servicios o equipos que no usa o no pidió: televisión, decodificadores, adicionales.",
        "submotivos": {
            "tv_no_usada": r"(?:no (?:veo|uso|utilizo) (?:la )?(?:tele|televisi|tv)|solo (?:quiero|necesito|uso) (?:el )?internet|solamente (?:el )?internet)",
            "decodificadores_equipos": r"(?:decodificador|deco\b|devolver (?:el|los) (?:equipo|deco))",
            "adicionales_no_solicitados": r"(?:adicional|paquete internacional|no lo ped[íi]|no los ped[íi])",
        },
    },
    "falla_tecnica": {
        "definicion": "Problemas del servicio: caídas, intermitencia, baja velocidad, visitas técnicas que no resuelven.",
        "submotivos": {
            "intermitencia_caidas": r"(?:se cae|se va (?:el|la) (?:internet|se[ñn]al)|intermiten|se desconecta|se corta)",
            "velocidad_baja": r"(?:lent[oa]|velocidad|no carga|megas)",
            "visita_tecnica_fallida": r"(?:t[ée]cnico|visita|no vino|no han venido|no me han solucionado)",
            "sin_servicio": r"(?:sin (?:servicio|internet|se[ñn]al)|no (?:tengo|hay) (?:servicio|internet|se[ñn]al)|no funciona)",
        },
    },
    "competencia": {
        "definicion": "Se va a otro operador o tiene una oferta mejor de la competencia.",
        "submotivos": {
            "oferta_competidor": r"(?:\btigo\b|\bmovistar\b|\bwom\b|\betb\b|\bune\b|otra (?:empresa|compa[ñn][íi]a|operador)|me ofrecen|mejor oferta|m[áa]s barato)",
            "portabilidad": r"(?:portabilidad|portar|pasarme a)",
        },
    },
    "traslado_cobertura": {
        "definicion": "Se muda y no hay cobertura, o el traslado del servicio no se pudo hacer.",
        "submotivos": {
            "traslado_sin_cobertura": r"(?:no hay cobertura|sin cobertura|no llega (?:la )?(?:red|fibra|cobertura))",
            "cambio_domicilio": r"(?:traslad|me mudo|me voy a vivir|cambio de (?:casa|apartamento|domicilio)|me cambio de)",
        },
    },
    "atencion_servicio": {
        "definicion": "Mala experiencia con la atención: reclamos repetidos, promesas incumplidas, trámites que no avanzan.",
        "submotivos": {
            "reclamo_repetido": r"(?:varias veces|(?:tres|cuatro|cinco|muchas) veces|otra vez|de nuevo|siempre (?:lo mismo|me dicen))",
            "promesa_incumplida": r"(?:me dijeron que|me prometieron|quedaron en|nunca (?:lleg|llam|vin|me))",
            "mala_atencion": r"(?:p[ée]sim|mala atenci|nadie me|no me (?:dan|dieron) soluci|cansad|hart)",
        },
    },
    "situacion_economica": {
        "definicion": "No puede pagar o necesita reducir gastos por su situación económica.",
        "submotivos": {
            "reduccion_gastos": r"(?:reducir gastos|recortar|no me alcanza|ahorrar|la situaci[óo]n est[áa])",
            "desempleo_ingresos": r"(?:desemple|sin trabajo|me qued[ée] sin|no tengo (?:plata|dinero|con qu[ée]))",
        },
    },
    "otro": {
        "definicion": "Motivo no identificado o llamada sin contenido suficiente.",
        "submotivos": {"no_identificado": r"$^"},
    },
}

# Orden de prioridad cuando hay empate de evidencia (más específico primero).
PRIORIDAD = [
    "competencia", "traslado_cobertura", "situacion_economica", "servicios_no_usados",
    "falla_tecnica", "precio_facturacion", "atencion_servicio", "otro",
]

URGENCIA = {
    "alta": "Pide la baja hoy, menciona otro operador o portabilidad, o dice que ya reclamó varias veces.",
    "media": "Amenaza con cancelar sin fecha, está dispuesto a escuchar una oferta.",
    "baja": "Consulta o negocia; no hay intención firme de irse.",
}

PATRONES_URGENCIA_ALTA = r"(?:hoy mismo|ya mismo|de una vez|inmediat|ya no m[áa]s|no puedo m[áa]s|quiero cancelar ya|portabilidad|\btigo\b|\bmovistar\b|\bwom\b|\betb\b)"
PATRONES_URGENCIA_BAJA = r"(?:solo quer[íi]a (?:saber|preguntar)|informaci[óo]n|cu[áa]nto (?:cuesta|vale)|qu[ée] opciones)"

LEXICO_NEGATIVO = [
    "mal", "malo", "pésim", "pesim", "terrible", "horrible", "molest", "cansad", "hart", "indign", "abuso", "robo",
    "estafa", "triste", "no sirve", "no funciona", "no puedo más", "no puedo mas", "queja", "reclamo", "peor", "nunca",
    "grosero", "burla", "decepcion", "decepción", "frustr", "fastidi", "rabia",
]
# Sin "gracias", "listo" ni "bien": son cortesía al cierre de casi toda llamada e inflan el final.
LEXICO_POSITIVO = [
    "excelente", "perfecto", "amable", "genial", "de acuerdo", "agradezco mucho", "muy bueno", "me parece bien",
    "acepto", "claro que sí", "súper", "me sirve", "me conviene", "buena opción",
]

EMOCIONES = {
    "frustracion": r"(?:cansad|hart|frustr|no puedo m[áa]s|otra vez|varias veces)",
    "enojo": r"(?:rabia|indign|abuso|robo|estafa|grosero|burla)",
    "desconfianza": r"(?:no les creo|siempre dicen|mentira|enga[ñn])",
    "resignacion": r"(?:ya qu[ée]|da igual|como sea|ni modo)",
    "tristeza": r"(?:triste|qu[ée] tristeza)",
}

COMPETIDORES = r"(\btigo\b|\bmovistar\b|\bwom\b|\betb\b|\bune\b|directv|hughesnet)"


def taxonomy_text() -> str:
    """Taxonomía en texto para el prompt del LLM."""
    lines = [f"Taxonomía {TAXONOMY_VERSION}:"]
    for m, v in MOTIVOS.items():
        lines.append(f"- {m}: {v['definicion']} Submotivos: {', '.join(v['submotivos'])}.")
    lines.append("Urgencia: " + " | ".join(f"{k}: {v}" for k, v in URGENCIA.items()))
    return "\n".join(lines)
