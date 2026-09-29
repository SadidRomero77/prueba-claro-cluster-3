"""Esquema de salida por llamada (JSON). Lo usan los tres clasificadores y el agente."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Motivo = Literal[
    "precio_facturacion", "servicios_no_usados", "falla_tecnica", "competencia",
    "traslado_cobertura", "atencion_servicio", "situacion_economica", "otro",
]


class Sentimiento(BaseModel):
    inicio: float = Field(..., ge=-1, le=1, description="Sentimiento del cliente en el primer tercio de la llamada")
    fin: float = Field(..., ge=-1, le=1, description="Sentimiento del cliente en el último tercio")
    global_: float = Field(..., ge=-1, le=1, alias="global", description="Sentimiento promedio del cliente")
    emociones: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class OfertaRetencion(BaseModel):
    ofrecida: bool
    tipo: Optional[str] = Field(None, description="descuento | cambio_plan | beneficio | visita_tecnica | otro")


class CallAnalysis(BaseModel):
    id_llamada: int
    cluster: int
    metodo: str = Field(..., description="baseline_reglas | llm | jev | llm+jev")
    calidad_transcripcion: Literal["alta", "media", "baja", "sin_contenido"]
    motivo: Motivo
    submotivo: str
    motivos_secundarios: list[str] = Field(default_factory=list)
    evidencia: str = Field(..., description="Cita textual del cliente que sustenta el motivo (debe existir en la transcripción)")
    sentimiento: Sentimiento
    sentimiento_por_aspecto: dict[str, float] = Field(default_factory=dict)
    urgencia: Literal["alta", "media", "baja"]
    reincidencia_mencionada: bool
    competidor_mencionado: Optional[str] = None
    oferta_retencion: OfertaRetencion
    resultado: Literal["retenido", "no_retenido", "pendiente", "no_determinado"]
    accion_sugerida: str
    confianza: float = Field(..., ge=0, le=1)
    version_taxonomia: str
