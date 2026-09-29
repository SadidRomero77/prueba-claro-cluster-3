"""Preprocesamiento de transcripciones, reglas y esquema de salida."""
import pytest
from pydantic import ValidationError

from cluster3.nlp.preprocess import anonymize, fix_roles, split_turns
from cluster3.nlp.schema import CallAnalysis
from cluster3.nlp.taxonomy import MOTIVOS, TAXONOMY_VERSION


def test_anonymize_nombres_numeros_y_correos():
    txt, n = anonymize("Hola, mi nombre es Carolina, mi cédula es 1020304050 y mi correo ana.p@mail.com")
    assert "Carolina" not in txt and "1020304050" not in txt and "ana.p@mail.com" not in txt
    assert "[NOMBRE]" in txt and "[NUMERO]" in txt and "[EMAIL]" in txt
    assert n == 3


def test_fix_roles_invierte_diarizacion():
    texto = ("AGENT: quiero cancelar el servicio, me llegó muy cara la factura\n"
             "CLIENT: bienvenido al área de cancelación, con quién tengo el gusto, en qué le puedo ayudar\n"
             "AGENT: con Pedro\n"
             "CLIENT: permítame validando, le ofrezco un beneficio")
    turns, swapped = fix_roles(split_turns(texto))
    assert swapped
    assert turns[0][0] == "CLIENT" and "cancelar" in turns[0][1]


def _base(**kw):
    d = dict(
        id_llamada=1, cluster=3, metodo="baseline_reglas", calidad_transcripcion="alta", motivo="precio_facturacion",
        submotivo="factura_alta", evidencia="me llegó muy cara la factura",
        sentimiento={"inicio": -0.5, "fin": 0.0, "global": -0.2, "emociones": ["frustracion"]},
        urgencia="media", reincidencia_mencionada=False, oferta_retencion={"ofrecida": True, "tipo": "descuento"},
        resultado="retenido", accion_sugerida="ajustar plan", confianza=0.8, version_taxonomia=TAXONOMY_VERSION,
    )
    d.update(kw)
    return d


def test_schema_valida_salida_correcta():
    ca = CallAnalysis(**_base())
    assert ca.sentimiento.global_ == -0.2
    assert ca.motivo in MOTIVOS


@pytest.mark.parametrize("campo,valor", [("motivo", "no_existe"), ("urgencia", "urgente"), ("confianza", 1.5)])
def test_schema_rechaza_valores_fuera_de_taxonomia(campo, valor):
    with pytest.raises(ValidationError):
        CallAnalysis(**_base(**{campo: valor}))
