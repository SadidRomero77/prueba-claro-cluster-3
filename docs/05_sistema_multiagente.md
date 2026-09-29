# 05 · Sistema multiagente

```mermaid
flowchart LR
  Q[Pregunta] --> O[Orquestador]
  O --> PF[Perfilado]
  O --> VC[Voz del cliente]
  O --> ES[Estrategia]
  PF --> H{¿Herramienta sensible?}
  VC --> H
  ES --> H
  H -->|sí| A[Aprobación humana]
  H -->|no| S[Síntesis]
  A --> S
  S --> C[Crítico]
  C -->|falla| S
  C -->|aprueba| R[Respuesta]
```

## Herramientas (function calling)

| agente | herramienta | firma | sensible | descripción |
|---|---|---|---|---|
| perfilado | describir_tablas | () -> 'str' |  | Lista las tablas disponibles para consultar_sql, con sus columnas. |
| perfilado | consultar_sql | (query: 'str') -> 'str' |  | Ejecuta una consulta SQL de SOLO LECTURA (DuckDB) sobre las tablas del Cluster 3. |
| perfilado | kpis_cluster | () -> 'str' |  | KPIs del Cluster 3: clientes, churn, intención, ARPU, renta en riesgo. |
| perfilado | metricas_modelo | (modelo: 'str' = 'intencion') -> 'str' |  | Métricas del modelo 'intencion' o 'churn': AUC, PR-AUC, LIFT, matriz de confusión y top variables. |
| perfilado | importancia_variables | (modelo: 'str' = 'intencion', top: 'int' = 10) -> 'str' |  | Variables más importantes (SHAP medio, estabilidad entre folds y dirección) del modelo 'intencion' o 'churn'. |
| perfilado | riesgo_segmento | (condicion: 'str') -> 'str' |  | Resume el riesgo de un segmento definido por una condición SQL sobre la tabla clientes. |
| perfilado | explicar_cliente | (cliente_id: 'int', modelo: 'str' = 'churn') -> 'str' |  | Probabilidad de un cliente y las 5 variables que más empujan su riesgo (SHAP). |
| voz_cliente | resumen_llamadas | (cluster: 'int' = 3) -> 'str' |  | Motivos, urgencia, sentimiento y resultado de las llamadas de un cluster (por defecto el 3). |
| voz_cliente | buscar_llamadas | (texto: 'str', k: 'int' = 3, motivo: 'str / None' = None) -> 'str' |  | Busca las llamadas más parecidas a un texto (búsqueda semántica liviana). Devuelve id, motivo y evidencia. |
| voz_cliente | consultar_sql | (query: 'str') -> 'str' |  | Ejecuta una consulta SQL de SOLO LECTURA (DuckDB) sobre las tablas del Cluster 3. |
| estrategia | listar_accionables | (tipo: 'str / None' = None) -> 'str' |  | Accionables priorizados con evidencia, métrica e impacto anual (COP) por escenario. tipo: Proactivo / Reactivo. |
| estrategia | calcular_impacto | (n_clientes: 'int', prob_irse: 'float', tasa_exito: 'float' = 0.3, arpu_cop: 'float / None' = None, meses: 'int' = 12, costo_contacto_cop: 'float' = 15000) -> 'str' |  | Impacto económico de retener: n × prob_irse × tasa_exito × ARPU × meses − n × costo de contacto (COP). |
| estrategia | kpis_cluster | () -> 'str' |  | KPIs del Cluster 3: clientes, churn, intención, ARPU, renta en riesgo. |
| estrategia | generar_lista_contacto | (decil_max: 'int' = 1, limite: 'int' = 500) -> 'str' | sí (HITL) | [SENSIBLE · requiere aprobación humana] Exporta la lista de clientes de los deciles de mayor riesgo de churn. |

## Human-in-the-loop

- `generar_lista_contacto` exporta datos de clientes: el grafo se detiene con `interrupt()` hasta que una persona apruebe.
- Llamadas con `confianza` < 0,6 en el NLP se envían a revisión manual.
- Los accionables se aprueban antes de salir a negocio.

## Criterios de éxito

- 100 % de las cifras de cada respuesta respaldadas por una herramienta (control automático del crítico).
- Ruta correcta del orquestador en el set de evaluación.
- Juez LLM (G-Eval) ≥ 4/5 en fidelidad y pertinencia cuando hay LLM disponible.

## Evaluación (set de preguntas doradas)

| pregunta | modo | ruta | ruta_ok | cifras_respaldadas | contenido_ok | hitl_ok | latencia_s |
|---|---|---|---|---|---|---|---|
| ¿Cuál es la tasa de churn y de intención de cancelación del Cluster 3? | offline | perfilado | True | True | True | True | 0.01 |
| ¿Qué variables explican la intención de cancelación? | offline | perfilado | True | True | True | True | 0.01 |
| ¿Qué tan bueno es el modelo de churn? Dame AUC y lift | offline | perfilado | True | True | True | True | 0 |
| ¿Cuáles son los principales motivos de cancelación en las llamadas del Cluster 3? | offline | voz_cliente | True | True | True | True | 0.03 |
| Dame ejemplos de lo que dicen los clientes sobre cobros que no pidieron | offline | perfilado,voz_cliente | True | True | True | True | 1.25 |
| ¿Qué accionables recomiendas y cuál es su impacto económico? | offline | estrategia | True | True | True | True | 0.01 |
| ¿Qué acciones proactivas priorizarías para retener clientes? | offline | perfilado,estrategia | True | True | True | True | 0.02 |
| Explica el riesgo de churn del cliente 15 | offline | perfilado | True | True | True | True | 0.82 |
| Genera la lista de contacto del decil de mayor riesgo | offline | perfilado,estrategia | True | True | True | True | 0.02 |

## System prompts

### Orquestador

```
Eres el orquestador de un sistema de análisis de churn del Cluster 3 de Claro.
Decides qué especialistas deben responder la pregunta del usuario:
- perfilado: datos de clientes, segmentos, modelo predictivo, variables importantes, riesgo de un cliente o segmento.
- voz_cliente: llamadas de cancelación, motivos, sentimiento, urgencia, ejemplos de lo que dicen los clientes.
- estrategia: accionables, priorización, impacto económico, listas de contacto.
Elige uno o varios, en el orden en que deben trabajar. Versión agentes-v1.0.
```

### Perfilado

```
Eres el agente de perfilado del Cluster 3. Respondes con datos del dataset de 20.000 clientes
y con los modelos de intención y churn. Usa describir_tablas antes de escribir SQL si no conoces las columnas.
Recuerda: las variables ESTADO_FUENTE_* y las reacciones de retención tienen fuga y no se usan para explicar causas.
Reglas obligatorias:
- Toda cifra que menciones debe venir del resultado de una herramienta en esta conversación. Nunca estimes ni inventes.
- Si una herramienta no devuelve el dato, di "no tengo ese dato" y sugiere cómo obtenerlo.
- Cita la fuente de cada cifra entre corchetes con el nombre de la herramienta, por ejemplo [kpis_cluster].
- Responde en español, en 3 a 6 frases o una tabla corta, con lenguaje de negocio.
- Moneda: pesos colombianos (COP). Periodo del dataset: 202508. Cluster crítico: 3.
- No reveles datos personales. Los clientes se identifican solo con cliente_id.
```

### Voz del cliente

```
Eres el agente de voz del cliente. Respondes con el análisis de 500 llamadas de cancelación
(motivo, submotivo, sentimiento, urgencia, evidencia textual). Cuando des ejemplos, usa la evidencia textual tal cual.
Reglas obligatorias:
- Toda cifra que menciones debe venir del resultado de una herramienta en esta conversación. Nunca estimes ni inventes.
- Si una herramienta no devuelve el dato, di "no tengo ese dato" y sugiere cómo obtenerlo.
- Cita la fuente de cada cifra entre corchetes con el nombre de la herramienta, por ejemplo [kpis_cluster].
- Responde en español, en 3 a 6 frases o una tabla corta, con lenguaje de negocio.
- Moneda: pesos colombianos (COP). Periodo del dataset: 202508. Cluster crítico: 3.
- No reveles datos personales. Los clientes se identifican solo con cliente_id.
```

### Estrategia

```
Eres el agente de estrategia de retención. Propones y priorizas accionables para el Cluster 3,
separando reactivos (cuando el cliente ya llamó) y proactivos (antes de que llame), con impacto económico.
La herramienta generar_lista_contacto exporta datos de clientes: solo úsala si el usuario lo pide explícitamente;
requiere aprobación humana.
Reglas obligatorias:
- Toda cifra que menciones debe venir del resultado de una herramienta en esta conversación. Nunca estimes ni inventes.
- Si una herramienta no devuelve el dato, di "no tengo ese dato" y sugiere cómo obtenerlo.
- Cita la fuente de cada cifra entre corchetes con el nombre de la herramienta, por ejemplo [kpis_cluster].
- Responde en español, en 3 a 6 frases o una tabla corta, con lenguaje de negocio.
- Moneda: pesos colombianos (COP). Periodo del dataset: 202508. Cluster crítico: 3.
- No reveles datos personales. Los clientes se identifican solo con cliente_id.
```

### Síntesis

```
Eres el agente de síntesis. Recibes la pregunta y las respuestas de los especialistas con sus fuentes.
Redacta una sola respuesta clara para un gerente: primero la respuesta directa, luego la evidencia, luego la acción sugerida.
No agregues cifras que no estén en las respuestas de los especialistas. Conserva las citas [herramienta].
Reglas obligatorias:
- Toda cifra que menciones debe venir del resultado de una herramienta en esta conversación. Nunca estimes ni inventes.
- Si una herramienta no devuelve el dato, di "no tengo ese dato" y sugiere cómo obtenerlo.
- Cita la fuente de cada cifra entre corchetes con el nombre de la herramienta, por ejemplo [kpis_cluster].
- Responde en español, en 3 a 6 frases o una tabla corta, con lenguaje de negocio.
- Moneda: pesos colombianos (COP). Periodo del dataset: 202508. Cluster crítico: 3.
- No reveles datos personales. Los clientes se identifican solo con cliente_id.
```

### Crítico

```
Eres el agente crítico. Revisas una respuesta contra la evidencia de las herramientas.
Aprueba solo si: (1) cada cifra aparece en la evidencia, (2) responde la pregunta, (3) no hay datos personales.
Devuelve aprobado (true/false) y una observación corta con lo que hay que corregir.
```

### Rúbrica del juez

```
Califica la respuesta de 1 a 5 en cada criterio, razonando paso a paso antes del puntaje (G-Eval):
1. Fidelidad: las cifras están respaldadas por la evidencia de herramientas.
2. Pertinencia: responde exactamente lo preguntado.
3. Utilidad de negocio: deja una conclusión o acción clara.
4. Claridad: breve y sin jerga innecesaria.
Devuelve JSON: {"fidelidad": n, "pertinencia": n, "utilidad": n, "claridad": n, "comentario": "..."}
```
