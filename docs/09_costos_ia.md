# 09 · Costo de la IA

La IA cuesta centavos de dólar por uso. Cada tarea va al modelo más barato que la hace bien: Jev clasifica,
Claude Haiku 4.5 enruta y revisa, y Claude Sonnet 5 solo redacta, cita y razona con herramientas.

## Costo por uso

| Uso | Modelos | Costo medio | US$ por 1.000 usos | Cómo se obtuvo |
|---|---|---|---|---|
| Clasificar una llamada | Jev (motivo, sentimiento) + Sonnet 5 (urgencia, submotivo, cita, acción) | US$0,0125 | 12,5 | Estimado con el texto real de las 500 llamadas |
| Pregunta al agente | Haiku 4.5 (orquestador y crítico) + Sonnet 5 (especialistas y síntesis) | US$0,026 (0,007–0,038) | 25,6 | Medido: 13 preguntas doradas, sin caché |
| Turno del copiloto | Jev (análisis) + Sonnet 5 (respuesta) | US$0,006 | 6,4 | Medido: 3 turnos de una conversación |

Las 500 llamadas del caso costaron unos US$6,3. Reprocesarlas en Databricks cuesta US$0: el job reutiliza la
clasificación guardada en el Volume `raw/insumos`.

## Cómo se calculó

- Precios públicos de OpenRouter y TypeSafe AI al 30 de septiembre de 2026 (US$ por millón de tokens, entrada/salida):
  Sonnet 5 2/10 · Haiku 4.5 1/5 · Opus 5.5 4/20 · DeepSeek V4 Pro 0,26/0,51 · GPT-6 Luna 0,10/0,50 · Jev 0,042/gratis.
- Clasificación: cada llamada envía unos 3.100 tokens al LLM (prompt, 3 ejemplos del RAG y el texto del cliente) y
  recibe unos 280 (el JSON). El 36 % de las llamadas tuvo un reintento por la verificación de la cita (factor 1,36).
  Jev recibe la llamada completa (unos 1.900 tokens) más las preguntas (se supusieron 400 tokens).
- Agente y copiloto: tokens reales de cada modelo, medidos con `get_usage_metadata_callback` de LangChain el
  1 de octubre de 2026. En una pregunta al agente, Haiku lee 2.884 tokens y escribe 267 (US$0,004); Sonnet lee 5.292 y
  escribe 1.079 (US$0,021).

## Comparación de configuraciones

| Configuración | Clasificar 1.000 llamadas (US$) | 1.000 preguntas al agente (US$) |
|---|---|---|
| **Diseño actual** | **12,5** | **25,6** |
| Todo con Claude Opus 5.5 | 24,9 | 59,6 |
| Todo con Claude Sonnet 5 | 12,4 | 29,8 |
| Todo con Claude Haiku 4.5 | 6,2 | 14,9 |
| Todo con DeepSeek V4 Pro | 1,3 | — |
| Todo con GPT-6 Luna | 0,6 | — |
| Solo Jev (sin texto generado) | 0,10 | — |

## Por qué el diseño es eficiente

1. Jev cuesta unas 130 veces menos que Sonnet y acierta más el motivo (62 % frente a 44 % contra las etiquetas humanas).
2. El LLM caro solo hace lo que Jev no puede: juzgar la urgencia (70 % frente a 54 %), redactar y citar.
3. Haiku procesa el 35 % del texto de una pregunta al agente y es el 16 % de su costo; hacerlo todo con Sonnet subiría
   la pregunta de US$0,026 a US$0,030.
4. Opus duplicaría el costo sin una mejora medida.
5. Caché de respuestas en el agente y reutilización de la clasificación en Databricks: nada se paga dos veces.
6. Todo funciona sin clave (reglas): si se corta la API, el sistema sigue.

## Pendiente

Modelos 10 a 20 veces más baratos (DeepSeek V4 Pro, GPT-6 Luna) y Haiku para todo el agente no se evaluaron. El paso
siguiente es correrlos contra las mismas 50 etiquetas y las 13 preguntas doradas, y quedarse con el más barato que
mantenga el 70 % en urgencia y las cifras respaldadas.
