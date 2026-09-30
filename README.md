# Claro · Cluster Crítico (Cluster 3)

Prueba técnica de IA Senior Engineer. Análisis de churn e intención de cancelación del Cluster 3
(20.000 clientes, periodo 202508) y de 500 llamadas de cancelación, con un sistema multiagente para
conversar con los datos y los modelos.

| Entregable | Dónde |
|---|---|
| Calidad y limpieza de datos | `docs/01_calidad_datos.md` |
| Perfil y segmentos del Cluster 3 | `docs/02_eda_cluster3.md` |
| Modelo de propensión (intención y churn) | `docs/03_modelo_churn.md` |
| Voz del cliente (NLP de llamadas) | `docs/04_nlp_llamadas.md` |
| Sistema multiagente | `docs/05_sistema_multiagente.md` |
| Arquitectura Databricks, MLOps y LLMOps | `docs/06_arquitectura_databricks_llmops.md` |
| Accionables e impacto económico | `docs/07_accionables.md` |
| Analizador de llamadas, copiloto del asesor y siguientes pasos | `docs/08_copiloto_y_siguientes_pasos.md` |
| Supuestos y decisiones | `docs/supuestos_y_decisiones.md` |
| App (dashboard + chat con agentes) | `src/cluster3/app/streamlit_app.py` |
| Presentación ejecutiva (21 diapositivas) | https://claude.ai/artifact/Sa8uh8pebEQDgmyNWiEjaA (privada: compartir desde Share) |
| Roadmap y arquitectura (FigJam) | https://www.figma.com/board/iqur8skOGvZNs1DfuwLUyT |

## Resultados principales

| | Intención de cancelar | Churn |
|---|---|---|
| Tasa base | 19,9 % (3.981) | 0,52 % (104) |
| AUC (CV 3×5) | 0,797 ± 0,009 | 0,974 ± 0,012 |
| PR-AUC | 0,566 | 0,588 |
| LIFT decil 1 | 3,4× | 9,2× (96 de 104 bajas) |

- Se excluyeron variables con fuga (estado de la cuenta, planes de TV con sufijo I, reincidencias, reacciones de
  retención, saldo en churn). Con ellas ambos modelos dan AUC 1,0.
- Llamadas clasificadas con un método híbrido elegido contra 50 llamadas etiquetadas a mano (Jev para motivo y
  sentimiento, LLM para urgencia y evidencia): precio y facturación es el motivo identificable principal (24,3 % de
  las llamadas del Cluster 3) y el 28,2 % no tiene un motivo claro.
- Seis accionables priorizados por impacto y esfuerzo, con impacto anual en tres escenarios.

## Inicio rápido

Requisitos: Python 3.11+ y [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
# Datos (no están en git): copiar los tres Excel a data/raw/
#   Clientes_Cluster_3.xlsx · Llamadas.xlsx · Diccionario_Datos_Cluster3.xlsx
uv run cluster3                      # pipeline completo, ~2 min, sin claves de API
uv run streamlit run src/cluster3/app/streamlit_app.py
```

Con claves en `.env` (copiar `.env.example`):

```bash
uv run cluster3 --llm --jev          # clasifica llamadas también con LLM y con Jev
uv run cluster3 --desde modelo       # retoma desde una etapa
```

Etapas: `calidad → limpieza → eda → modelo → nlp → accionables → reportes → base_agente`.

## Sistema multiagente

LangGraph: orquestador → especialistas (perfilado, voz del cliente, estrategia) → aprobación humana si la acción
exporta datos → síntesis → crítico que verifica cada cifra contra la evidencia de las herramientas.

```bash
uv run python -m cluster3.agents.graph "¿Qué variables explican la intención de cancelación?"
uv run python -m cluster3.agents.evals            # 9 preguntas doradas (+ juez LLM si hay clave)
uv run python -m cluster3.agents.evals --offline
```

Sin clave de LLM el grafo funciona en modo offline (enrutamiento y herramientas por reglas).

## Evaluación del NLP

| Campo | Reglas | LLM (Sonnet 5 + RAG) | Jev | Híbrido (final) |
|---|---|---|---|---|
| Motivo · exactitud / kappa | 38 % / 0,26 | 44 % / 0,35 | 62 % / 0,55 | **62 % / 0,55** |
| Urgencia · exactitud / kappa | 38 % / 0,12 | 70 % / 0,40 | 54 % / 0,11 | **70 % / 0,40** |
| Sentimiento · correlación | 0,56 | 0,47 | 0,75 | **0,75** |

Fuente: `outputs/tables/nlp_benchmark.csv` (50 llamadas etiquetadas a mano: 40 del Cluster 3 y 10 de otros).

```bash
uv run python -m cluster3.nlp.etiquetar --exportar-excel     # plantilla de etiquetado en Excel
uv run python -m cluster3.nlp.etiquetar --importar-excel RUTA # etiquetas → data/labels/muestra_etiquetada.csv
uv run python -m cluster3.nlp.run --llm --jev --solo-muestra  # reglas vs LLM vs Jev → outputs/tables/nlp_benchmark.csv
uv run python -m cluster3.nlp.run --llm --jev                 # 500 llamadas con el método híbrido (reanuda si se corta)
```

## Pruebas

```bash
uv run pytest
uv run ruff check src tests
```

## Despliegue

- Databricks Free Edition: `databricks.yml`, `app.yaml`, `notebooks/databricks/pipeline_job.py` → `deploy/databricks/README.md`
- AWS (EC2 + Docker + Caddy + Cloudflare + SSM): `Dockerfile`, `deploy/aws/` → `deploy/aws/README.md`

## Estructura

```
src/cluster3/
  config.py          rutas, targets, listas de fuga y supuestos
  pipeline.py        orquesta las etapas
  data/              carga, calidad, limpieza
  eda/               KPIs y segmentos
  models/            modelo en dos etapas, SHAP, lift, MLflow
  nlp/               preprocesamiento, taxonomía, reglas, LLM, Jev, evaluación
  business/          accionables e impacto económico
  agents/            herramientas, prompts, grafo LangGraph, evaluación
  app/               Streamlit
  deploy/            sincronización con Volumes de Databricks
docs/                informes (generados por report.py + estáticos)
outputs/             tablas, figuras y JSON agregados
tests/               pruebas sin datos reales
```

## Confidencialidad

Datos internos de Claro, sin datos personales, con uso autorizado. `data/raw`, `data/processed`, los modelos y las
listas de contacto no se versionan. Al cerrar el proceso se eliminan los datos locales, el repositorio y los
recursos en la nube (ver cada guía de despliegue).
