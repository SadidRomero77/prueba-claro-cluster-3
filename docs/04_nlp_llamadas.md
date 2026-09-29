# 04 · Voz del cliente: NLP de llamadas

Método de la clasificación actual: **hibrido_jev_llm**. Con claves de API se ejecuta con LLM y Jev (`uv run cluster3 --desde nlp --llm --jev`).

## Preparación de las transcripciones

- Roles AGENT/CLIENT corregidos: 158 llamadas.
- Diarización fallida (todo en un rol): 78 llamadas; se usa el texto completo.
- Segunda pasada de anonimización: 81 reemplazos en 74 llamadas.
- Sin contenido (< 1.000 caracteres): 19 llamadas.

## Motivo principal: Cluster 3 vs otros clústeres (%)

| motivo | Cluster 3 | Otros clústeres | diferencia_pp |
|---|---|---|---|
| otro | 28.2 | 31.7 | -3.5 |
| precio_facturacion | 24.3 | 21.1 | 3.2 |
| falla_tecnica | 10.3 | 6.1 | 4.2 |
| servicios_no_usados | 10 | 10.6 | -0.6 |
| atencion_servicio | 8.3 | 5.6 | 2.7 |
| traslado_cobertura | 7.6 | 11.7 | -4.1 |
| situacion_economica | 6.3 | 7.2 | -0.9 |
| competencia | 5 | 6.1 | -1.1 |

## Submotivos del Cluster 3

| motivo | submotivo | llamadas | pct |
|---|---|---|---|
| otro | no_identificado | 85 | 28.2 |
| precio_facturacion | incremento_tarifa | 25 | 8.3 |
| precio_facturacion | cobro_no_reconocido | 19 | 6.3 |
| atencion_servicio | no_identificado | 16 | 5.3 |
| servicios_no_usados | no_identificado | 14 | 4.7 |
| traslado_cobertura | cambio_domicilio | 14 | 4.7 |
| situacion_economica | no_identificado | 11 | 3.7 |
| falla_tecnica | visita_tecnica_fallida | 11 | 3.7 |
| precio_facturacion | factura_alta | 11 | 3.7 |
| falla_tecnica | intermitencia_caidas | 9 | 3 |
| competencia | no_identificado | 9 | 3 |
| precio_facturacion | fin_promocion_descuento | 9 | 3 |

## Sentimiento, urgencia y resultado por motivo (Cluster 3)

| motivo | llamadas | sent_inicio | sent_fin | sent_delta | urgencia_alta_pct | reincidencia_pct | retenido_pct |
|---|---|---|---|---|---|---|---|
| otro | 85 | -0.24 | -0.12 | 0.12 | 48.24 | 47.06 | 16.47 |
| precio_facturacion | 73 | -0.37 | -0.08 | 0.28 | 46.58 | 63.01 | 45.21 |
| falla_tecnica | 31 | -0.46 | -0.23 | 0.23 | 77.42 | 93.55 | 35.48 |
| servicios_no_usados | 30 | -0.26 | -0.07 | 0.19 | 43.33 | 50.00 | 33.33 |
| atencion_servicio | 25 | -0.42 | -0.18 | 0.24 | 72.00 | 88.00 | 28.00 |
| traslado_cobertura | 23 | -0.23 | -0.10 | 0.13 | 56.52 | 52.17 | 17.39 |
| situacion_economica | 19 | -0.31 | -0.06 | 0.25 | 42.11 | 47.37 | 42.11 |
| competencia | 15 | -0.24 | -0.19 | 0.05 | 80.00 | 73.33 | 13.33 |

Urgencia en el Cluster 3: {'alta': 54.2, 'media': 42.9, 'baja': 3.0}. Competidores mencionados: {'otro operador (sin nombre)': 9}. Reincidencia mencionada: 61.1 %.

## Cruce agregado llamadas ↔ dataset

No hay llave común entre llamadas y clientes; el cruce se hace por motivo.

| motivo_llamadas | pct_llamadas_c3 | variable_dataset | pct_clientes_c3 | intencion_con_senal_pct | intencion_sin_senal_pct | churn_con_senal_pct | churn_sin_senal_pct |
|---|---|---|---|---|---|---|---|
| precio_facturacion | 24.30 | Renta subió más de 5 % vs hace 6 meses | 25.90 | 14.30 | 21.90 | 0.87 | 0.40 |
| servicios_no_usados | 10.00 | Tiene equipos adicionales | 3.80 | 20.80 | 19.90 | 7.78 | 0.23 |
| falla_tecnica | 10.30 | Llamadas técnicas negativas o downtime > 0 | 16.30 | 14.10 | 21.00 | 0.25 | 0.57 |
| competencia | 5.00 | Zona de alta competencia o competencia mejor | 71.40 | 19.90 | 19.90 | 0.56 | 0.42 |
| traslado_cobertura | 7.60 | Tiene marca de traslado | 1.10 | 16.10 | 19.90 | 0.46 | 0.52 |
| atencion_servicio | 8.30 | 2 o más reclamos en el mes | 22.30 | 11.70 | 22.30 | 1.25 | 0.31 |
| situacion_economica | 6.30 | Score crediticio en el quintil más bajo | 19.80 | 32.70 | 16.70 | 0.53 | 0.52 |

## Evaluación contra muestra etiquetada a mano

| metodo | campo | n | exactitud | f1_macro | kappa | correlacion_spearman |
|---|---|---|---|---|---|---|
| baseline | motivo | 50 | 0.38 | 0.267 | 0.26 |  |
| baseline | urgencia | 50 | 0.38 | 0.344 | 0.124 |  |
| baseline | sentimiento | 50 | 0.66 | 0.581 | 0.441 | 0.558 |
| llm | motivo | 50 | 0.44 | 0.379 | 0.346 |  |
| llm | urgencia | 50 | 0.7 | 0.52 | 0.4 |  |
| llm | sentimiento | 50 | 0.62 | 0.623 | 0.293 | 0.471 |
| jev | motivo | 50 | 0.62 | 0.605 | 0.547 |  |
| jev | urgencia | 50 | 0.54 | 0.395 | 0.111 |  |
| jev | sentimiento | 50 | 0.62 | 0.504 | 0.33 | 0.751 |
| hibrido | motivo | 50 | 0.62 | 0.605 | 0.547 |  |
| hibrido | urgencia | 50 | 0.7 | 0.52 | 0.4 |  |
| hibrido | sentimiento | 50 | 0.62 | 0.504 | 0.33 | 0.751 |

Esquema JSON por llamada: `src/cluster3/nlp/schema.py`. Salidas: `outputs/json/llamadas_*.jsonl`.