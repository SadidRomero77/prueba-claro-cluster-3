# CLAUDE.md

Contexto para Claude Code en este repositorio.

## Qué es

Prueba técnica de Claro Colombia para el cargo de IA Senior Engineer. Caso: Cluster Crítico (Cluster 3),
20.000 clientes del periodo 202508 y 500 transcripciones de llamadas de cancelación.

Entregables: calidad de datos, EDA, modelo predictivo de churn e intención, NLP de llamadas (sentimiento, motivos,
urgencia), accionables con impacto económico, sistema multiagente y propuesta de arquitectura en Databricks.

- **Entrega: viernes 2 de octubre de 2026, 11:59 p. m.** Fecha de sustentación por confirmar.
- La audiencia es gerencial: textos cortos, cifras concretas, sin explicaciones largas.
- Idioma de código, comentarios, docs y app: español. Números en formato colombiano (1.234,5).
- Jev (TypeSafe AI) debe aparecer en la solución: clasificador alternativo de llamadas (`nlp/jev_classifier.py`).
- Despliegue en dos nubes: Databricks Free Edition y AWS (patrón EC2 + Docker + Caddy + Cloudflare + SSM).

## Comandos

```bash
uv sync --all-extras
uv run cluster3                                   # pipeline completo (~2 min, sin claves)
uv run cluster3 --desde accionables               # retomar desde una etapa
uv run cluster3 --llm --jev                       # con LLM y Jev (requiere .env)
uv run streamlit run src/cluster3/app/streamlit_app.py
uv run python -m cluster3.agents.evals --offline  # 9 preguntas doradas
uv run python -m cluster3.nlp.etiquetar           # etiquetar la muestra de 50 llamadas
uv run python -m cluster3.nlp.run --llm --jev --solo-muestra
uv run pytest && uv run ruff check src tests
```

Etapas del pipeline, en orden: `calidad → limpieza → eda → modelo → nlp → accionables → reportes → base_agente`.
Los Excel originales van en `data/raw/` (no están en git).

## Estructura

| Ruta | Contenido |
|---|---|
| `src/cluster3/config.py` | Rutas, targets, **listas de variables con fuga**, supuestos. Toda decisión vive aquí. |
| `src/cluster3/data/` | `load.py`, `quality.py` (hallazgos, AUC univariado, diccionario), `clean.py` (registro de decisiones por variable) |
| `src/cluster3/eda/profile.py` | KPIs, 12 segmentos, figuras, paleta |
| `src/cluster3/models/churn.py` | Dos etapas LightGBM, CV 3×5, calibración, SHAP por fold, lift, umbral de negocio, MLflow |
| `src/cluster3/nlp/` | preprocesamiento (PII, roles), taxonomía v1.1, reglas, LLM con RAG few-shot, Jev, evaluación, etiquetado, `analizador.py` (analizar una llamada y copiloto del asesor) |
| `src/cluster3/business/actions.py` | Accionables A1–A6, impacto en tres escenarios, matriz impacto/esfuerzo |
| `src/cluster3/agents/` | `tools.py` (DuckDB solo lectura), `prompts.py`, `graph.py` (LangGraph), `evals.py` |
| `src/cluster3/app/streamlit_app.py` | Dashboard + chat con HITL + evaluación |
| `src/cluster3/report.py` | Genera `docs/01–05` y `docs/07` desde los outputs |
| `src/cluster3/deploy/volume_sync.py` | Artefactos ↔ Volume de Unity Catalog |
| `databricks.yml`, `app.yaml`, `notebooks/databricks/` | Asset Bundle, Databricks Apps, job |
| `Dockerfile`, `deploy/aws/` | Imagen y compose con Caddy |

## Convenciones

- **Ninguna cifra a mano.** Todo número en docs, app o accionables se calcula desde los datos. Si cambia el dato,
  cambia el texto.
- `docs/01–05` y `docs/07` se **generan**: editar `report.py`, no el markdown. `docs/06` y
  `docs/supuestos_y_decisiones.md` son estáticos: actualizarlos a mano si cambian cifras.
- Formato de números: helpers `_pct`, `_n`, `_miles`, `_cop`, `_d`. Nunca `.replace(",", ".")` sobre un número ya
  formateado con decimales.
- Gráficos: paleta BLUE `#2a78d6`, ORANGE `#eb6834`, INK `#0b0b0b`, INK2 `#52514e`, GRID `#e6e5e1`. Sin doble eje.
- Los deciles y listas usan probabilidades fuera de muestra (`p_*_oof`), nunca las de entrenamiento.
- El crítico del grafo (`numbers_supported`) exige que cada cifra de la respuesta esté en la evidencia de las
  herramientas. Si se agrega una herramienta, su salida debe incluir las cifras que el agente va a citar.
- La herramienta `generar_lista_contacto` es sensible: siempre pasa por `interrupt` (aprobación humana).
- Todo lo que llame a un LLM debe funcionar también sin clave (modo offline o método de reglas).
- Prompts versionados (`PROMPT_VERSION`); si se cambia un prompt, subir la versión.
- Probar la app sin navegador con `streamlit.testing.v1.AppTest`.

## Hallazgos y decisiones clave

- **Fuga de información** (detalle en `docs/supuestos_y_decisiones.md`):
  - `ESTADO_FUENTE_*`: `ESTADO_FUENTE_C` es idéntica a `BAN_CHURN`.
  - `TIPO_TV_DIGITAL_PI/BI`: los 103 clientes con sufijo I tienen cuenta no activa; codifica estado, no plan.
    En la primera versión se tomó como hallazgo de negocio ("plan TV PI con 64 % de churn") y se corrigió.
    No volver a usarlo como palanca sin confirmar con Claro qué significa la I.
  - Reincidencias y `CANTIDAD_INTENCIONES`: derivadas de la intención (se excluyen solo del modelo de intención).
  - Reacciones de retención y campañas: posteriores al evento.
  - `VAL_SALDO_ACTUAL`: 0 en los 104 churn (solo se excluye del modelo de churn).
- `VAL_SCORE_CREDITICIO` perdió el separador decimal → `SCORE_CREDITICIO_FIX` (÷10 hasta < 1000).
- `VAL_EQUIP_ADIC` y `VAL_UW` son más altos en cuentas no activas: se mantienen como señales a validar.
- Un solo periodo: sin validación temporal (limitación declarada).
- Resultados actuales: intención AUC 0,797, LIFT@10 3,4; churn AUC 0,974, LIFT@10 9,2 (96 de 104 bajas en el decil 1).
- `BAN_CAMPANA_VENTA` se excluyó (misma campaña del mes que las demás `BAN_CAMPANA_*`): intención AUC 0,813 → 0,797.
- Accionables proactivos: solo se contacta a clientes con valor esperado positivo (`_rentables` en `actions.py`).
- NLP híbrido (Jev motivo y sentimiento + LLM urgencia, submotivo y evidencia), elegido con 50 llamadas etiquetadas a
  mano: precio y facturación 24,3 % del Cluster 3, sin motivo claro 28,2 %, urgencia alta 54,2 %.
- "Tigo en 34 llamadas" era un falso positivo de la regla ("contigo"): no hay operadores nombrados en las
  transcripciones. No volver a usarlo.

## Estado

Hecho: pipeline completo, modelos, NLP híbrido validado contra 50 etiquetas humanas, accionables, docs, grafo
multiagente (conversación, streaming, selector de modelo) con evals (offline 100 %; con LLM: ruta 100 %,
cifras respaldadas 92 %, juez 4,46/5 en fidelidad; 13 preguntas), app Streamlit guiada, repo público, deck en Artifact Slides.

Fase 1 del copiloto hecha: pestaña Copiloto (analizar llamada + copiloto en chat, `copiloto_chat`) y herramienta
`analizar_transcripcion` en el chat. Fase 2 (agente de retención) documentada en `docs/08`.

Pendiente, en este orden:
1. Desplegar en Databricks Free Edition (`deploy/databricks/README.md`) y en AWS (`deploy/aws/README.md`).
2. Preparar la presentación para la sustentación (la app es la demo en vivo).
3. Validar con Claro: significado del sufijo I, `MOTIVO_LLAM_CANCELA`, supuestos económicos.

## Confidencialidad

Datos internos de Claro, sin datos personales; la gerencia autorizó su uso en la nube.
- Nunca versionar `data/raw`, `data/processed`, `models/*.pkl`, `outputs/listas/` ni `outputs/json/llamadas_*.jsonl`.
- Repo público (aprobado por la gerencia): solo código y resultados agregados; ningún insumo original ni texto por llamada.
- **Al cerrar el proceso se borran:** datos locales, repo, recursos de Databricks (`databricks bundle destroy`) y
  de AWS (instancia, DNS, bucket temporal).
