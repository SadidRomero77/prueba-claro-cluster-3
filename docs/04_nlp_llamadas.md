# 04 · Voz del cliente: NLP de llamadas

Método de la clasificación actual: **baseline_reglas**. Con claves de API se ejecuta con LLM y Jev (`uv run cluster3 --desde nlp --llm --jev`).

## Preparación de las transcripciones

- Roles AGENT/CLIENT corregidos: 158 llamadas.
- Diarización fallida (todo en un rol): 78 llamadas; se usa el texto completo.
- Segunda pasada de anonimización: 81 reemplazos en 74 llamadas.
- Sin contenido (< 1.000 caracteres): 19 llamadas.

## Motivo principal: Cluster 3 vs otros clústeres (%)

| motivo | Cluster 3 | Otros clústeres | diferencia_pp |
|---|---|---|---|
| precio_facturacion | 46.2 | 38.9 | 7.3 |
| otro | 19.9 | 29.4 | -9.5 |
| falla_tecnica | 15 | 7.2 | 7.8 |
| traslado_cobertura | 6.6 | 8.3 | -1.7 |
| atencion_servicio | 6 | 10 | -4 |
| competencia | 3.7 | 2.8 | 0.9 |
| servicios_no_usados | 2 | 2.8 | -0.8 |
| situacion_economica | 0.7 | 0.6 | 0.1 |

## Submotivos del Cluster 3

| motivo | submotivo | llamadas | pct |
|---|---|---|---|
| precio_facturacion | factura_alta | 126 | 41.9 |
| otro | no_identificado | 60 | 19.9 |
| falla_tecnica | visita_tecnica_fallida | 32 | 10.6 |
| traslado_cobertura | cambio_domicilio | 18 | 6 |
| competencia | oferta_competidor | 8 | 2.7 |
| atencion_servicio | reclamo_repetido | 8 | 2.7 |
| precio_facturacion | incremento_tarifa | 8 | 2.7 |
| atencion_servicio | promesa_incumplida | 7 | 2.3 |
| precio_facturacion | cobro_no_reconocido | 5 | 1.7 |
| falla_tecnica | velocidad_baja | 5 | 1.7 |
| falla_tecnica | intermitencia_caidas | 4 | 1.3 |
| servicios_no_usados | adicionales_no_solicitados | 4 | 1.3 |

## Sentimiento, urgencia y resultado por motivo (Cluster 3)

| motivo | llamadas | sent_inicio | sent_fin | sent_delta | urgencia_alta_pct | reincidencia_pct | retenido_pct |
|---|---|---|---|---|---|---|---|
| precio_facturacion | 139 | -0.05 | -0.02 | 0.03 | 36.69 | 20.14 | 4.32 |
| otro | 60 | -0.01 | -0.02 | -0.01 | 0.00 | 0.00 | 0.00 |
| falla_tecnica | 45 | -0.19 | -0.02 | 0.17 | 33.33 | 24.44 | 2.22 |
| traslado_cobertura | 20 | -0.08 | -0.02 | 0.07 | 45.00 | 20.00 | 0.00 |
| atencion_servicio | 18 | -0.16 | 0.00 | 0.17 | 66.67 | 66.67 | 0.00 |
| competencia | 11 | 0.02 | -0.11 | -0.13 | 72.73 | 9.09 | 0.00 |
| servicios_no_usados | 6 | -0.26 | 0.08 | 0.34 | 16.67 | 16.67 | 0.00 |
| situacion_economica | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Urgencia en el Cluster 3: {'media': 62.8, 'alta': 31.9, 'baja': 5.3}. Competidores mencionados: {'tigo': 34}. Reincidencia mencionada: 18.9 %.

## Cruce agregado llamadas ↔ dataset

No hay llave común entre llamadas y clientes; el cruce se hace por motivo.

| motivo_llamadas | pct_llamadas_c3 | variable_dataset | pct_clientes_c3 | intencion_con_senal_pct | intencion_sin_senal_pct | churn_con_senal_pct | churn_sin_senal_pct |
|---|---|---|---|---|---|---|---|
| precio_facturacion | 46.20 | Renta subió más de 5 % vs hace 6 meses | 25.90 | 14.30 | 21.90 | 0.87 | 0.40 |
| servicios_no_usados | 2.00 | Tiene equipos adicionales | 3.80 | 20.80 | 19.90 | 7.78 | 0.23 |
| falla_tecnica | 15.00 | Llamadas técnicas negativas o downtime > 0 | 16.30 | 14.10 | 21.00 | 0.25 | 0.57 |
| competencia | 3.70 | Zona de alta competencia o competencia mejor | 71.40 | 19.90 | 19.90 | 0.56 | 0.42 |
| traslado_cobertura | 6.60 | Tiene marca de traslado | 1.10 | 16.10 | 19.90 | 0.46 | 0.52 |
| atencion_servicio | 6.00 | 2 o más reclamos en el mes | 22.30 | 11.70 | 22.30 | 1.25 | 0.31 |
| situacion_economica | 0.70 | Score crediticio en el quintil más bajo | 19.80 | 32.70 | 16.70 | 0.53 | 0.52 |

## Evaluación contra muestra etiquetada a mano

_(sin datos)_

Esquema JSON por llamada: `src/cluster3/nlp/schema.py`. Salidas: `outputs/json/llamadas_*.jsonl`.