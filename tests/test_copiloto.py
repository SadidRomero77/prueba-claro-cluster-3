"""Analizador de llamadas y copiloto del asesor (modo sin claves: reglas)."""
import pytest

from cluster3.agents.graph import es_transcripcion, route_rules
from cluster3.nlp.analizador import analizar_llamada, copiloto, normalizar, turnos_de_llamada

LLAMADA = """Asesor: Buenas tardes, área de cancelaciones, ¿en qué le puedo ayudar?
Cliente: Quiero cancelar el servicio. Me llegó la factura con un cobro de un paquete que yo nunca pedí,
y ya es la tercera vez que llamo por lo mismo.
Asesor: Entiendo, permítame revisar.
Cliente: No, igual quiero cancelar, siempre me dicen que lo quitan y vuelve a aparecer."""


def test_normalizar_roles():
    assert normalizar("me cobraron algo que no pedí").startswith("CLIENT: ")
    t = normalizar("Asesor: hola\nCliente: quiero cancelar")
    assert "AGENT: hola" in t and "CLIENT: quiero cancelar" in t


def test_analisis_sin_claves():
    r = analizar_llamada(LLAMADA, usar_llm=False, usar_jev=False)
    assert r["intencion_cancelar_prob"] >= 0.5
    assert r["urgencia"] in {"alta", "media", "baja"}
    assert r["oferta_sugerida"]
    assert any("insiste en cancelar" in a for a in r["alertas"])  # se respeta la decisión del cliente
    assert r["fuentes"]["motivo"] == "reglas"


def test_llamada_muy_corta():
    with pytest.raises(ValueError):
        analizar_llamada("hola", usar_llm=False, usar_jev=False)


def test_copiloto_pregunta_si_el_motivo_no_es_claro():
    r = copiloto([{"rol": "asesor", "texto": "Buenas, área de cancelaciones."},
                  {"rol": "cliente", "texto": "Quiero cancelar, ya no quiero seguir con ustedes."}],
                 usar_llm=False, usar_jev=False)
    assert r["motivo"] == "otro" and r["pregunta_sugerida"]
    assert r["guion"]


def test_turnos_de_llamada():
    t = turnos_de_llamada("AGENT: hola\nCLIENT: quiero cancelar\nCLIENT: ")
    assert t == [{"rol": "asesor", "texto": "hola"}, {"rol": "cliente", "texto": "quiero cancelar"}]


def test_transcripcion_va_a_voz_del_cliente():
    assert es_transcripcion(LLAMADA)
    assert route_rules(LLAMADA)[0] == ["voz_cliente"]
    assert not es_transcripcion("¿Cuál es la tasa de churn?")


def test_si_insiste_no_se_intenta_retener():
    r = copiloto([{"rol": "cliente", "texto": "Pago mucho por televisión que no veo."},
                  {"rol": "asesor", "texto": "Podemos dejarle solo internet."},
                  {"rol": "cliente", "texto": "No, gracias, igual quiero cancelar. Ya tomé la decisión."}],
                 usar_llm=True, usar_jev=False)  # aun con LLM disponible, el guion es la plantilla fija
    assert r["fuentes"]["guion"].startswith("reglas")
    assert "respeto" in r["guion"].lower() and "cancelación" in r["guion"].lower()


@pytest.mark.parametrize("frase,esperado", [
    ("Quiero saber qué necesito para llevarme mi número y cuánto pago si termino el contrato antes.", True),
    ("Me voy a otra ciudad, no creo que necesite más el servicio. ¿Dónde entrego los equipos?", True),
    ("Ya no quiero saber nada más de ustedes, que vengan a recoger el módem.", True),
    ("Desde ayer el internet está lento en las noches, ¿me agendan una visita técnica?", False),
])
def test_intencion_implicita_sin_api(frase, esperado):
    r = analizar_llamada(f"Cliente: {frase}", usar_llm=False, usar_jev=False)
    assert (r["intencion_cancelar_prob"] >= 0.5) == esperado


def test_copiloto_chat_sin_llm():
    from cluster3.nlp.analizador import SALUDO_CHAT, copiloto_chat

    m = [{"rol": "asesor", "texto": "hola"}]
    assert copiloto_chat(m, usar_llm=False, usar_jev=False)["respuesta"] == SALUDO_CHAT
    m = [{"rol": "asesor", "texto": "El cliente dice que paga por televisión que no ve y quiere dejar solo internet"}]
    out = copiloto_chat(m, usar_llm=False, usar_jev=False)
    assert out["analisis"] and "Qué ofrecer" in out["respuesta"]
    m += [{"rol": "copiloto", "texto": out["respuesta"]},
          {"rol": "asesor", "texto": "Ya le ofrecí dejarle solo internet pero insiste en cancelar"}]
    out = copiloto_chat(m, usar_llm=True, usar_jev=False)  # aun con LLM, si insiste la respuesta es fija
    assert out["fuente"].startswith("reglas") and "respeto" in out["respuesta"]
