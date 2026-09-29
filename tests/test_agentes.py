"""Guardarraíles del sistema multiagente: SQL de solo lectura, crítico de cifras, enrutamiento y HITL."""
import pytest

from cluster3 import config
from cluster3.agents.graph import numbers_supported, route_rules
from cluster3.agents.tools import SENSIBLES, TOOLS_POR_AGENTE, consultar_sql

hay_base = pytest.mark.skipif(not config.DUCKDB_PATH.exists(), reason="Correr primero el pipeline (uv run cluster3)")


@pytest.mark.parametrize("q", [
    "DROP TABLE clientes",
    "SELECT 1; DELETE FROM clientes",
    "WITH x AS (SELECT 1) INSERT INTO clientes SELECT * FROM x",
    "COPY clientes TO 'fuera.csv'",
    "ATTACH 'otra.db'",
])
def test_sql_rechaza_escritura(q):
    assert consultar_sql(q).startswith("ERROR")


@hay_base
def test_sql_lectura_funciona():
    out = consultar_sql("SELECT COUNT(*) AS n FROM clientes")
    assert out.splitlines()[0] == "n" and int(out.splitlines()[1]) == 20000


def test_critico_detecta_cifras_inventadas():
    ok, faltan = numbers_supported("El churn es 0,52 % con 104 bajas y 12345 clientes", ["churn_tasa 0.0052, churn_n 104"])
    assert not ok and faltan == ["12345"]


def test_critico_acepta_formatos_y_redondeo():
    ev = ['{"auc": 0.97012, "renta": 1928935647, "lift_20": 4.903846}', "motivo,pct\nprecio_facturacion,46.2"]
    ok, faltan = numbers_supported("AUC 0,970; renta $ 1.928,9 M; lift 4.904; precio 46,2 %", ev)
    assert ok, faltan


@pytest.mark.parametrize("pregunta,esperado", [
    ("¿Cuál es la tasa de churn?", "perfilado"),
    ("¿Qué dicen los clientes en las llamadas?", "voz_cliente"),
    ("¿Qué accionables recomiendas?", "estrategia"),
    ("Hola, buenas tardes", "conversacion"),
    ("¿Ya tienen el modelo de machine learning?", "perfilado"),
    ("¿Qué insights identificaron?", "estrategia"),
])
def test_enrutamiento_por_reglas(pregunta, esperado):
    agentes, _ = route_rules(pregunta)
    assert esperado in agentes


def test_herramienta_sensible_solo_en_estrategia():
    assert SENSIBLES == {"generar_lista_contacto"}
    for agente, tools in TOOLS_POR_AGENTE.items():
        nombres = {t.__name__ for t in tools}
        assert (agente == "estrategia") == bool(nombres & SENSIBLES)
