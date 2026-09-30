# 08 · Analizador de llamadas, copiloto del asesor y siguientes pasos

## Qué hay hoy (fase 1)

Dos capacidades nuevas sobre el mismo método híbrido validado con 50 llamadas etiquetadas a mano
(`outputs/tables/nlp_benchmark.csv`): Jev para motivo, sentimiento e intención; LLM con RAG para urgencia,
submotivo y cita textual; reglas cuando no hay claves de API.

| Capacidad | Qué recibe | Qué devuelve | Dónde |
|---|---|---|---|
| **Analizador de llamadas** | Una transcripción completa (con o sin `Cliente:` / `Asesor:`) | Intención de cancelar (probabilidad), motivo, submotivo, urgencia, sentimiento inicio → fin, emociones, cita verificada, oferta sugerida, accionable relacionado, alertas y contexto del Cluster 3 | App → pestaña **Copiloto** · chat (pegar la llamada) · `nlp/analizador.py: analizar_llamada` |
| **Copiloto del asesor** | Los turnos de la llamada en curso | Lo mismo, actualizado con cada turno, más una **pregunta para el cliente** cuando el motivo no está claro y un **guion** para el asesor | App → pestaña **Copiloto → Copiloto en vivo** · `copiloto(turnos)` |

### Cómo decide el copiloto

1. **Análisis rápido** con cada turno: Jev (≈0,2 s) para motivo, sentimiento e intención; reglas para urgencia,
   reincidencia e insistencia en cancelar.
2. **¿El motivo está claro?** Si Jev duda entre dos motivos (diferencia menor a 20 puntos) o su confianza es menor
   a 55 %, propone una pregunta para distinguirlos (por ejemplo: "¿El problema es que la factura subió o que está
   pagando por cosas que no usa?"). El asesor la hace y registra la respuesta como un turno más.
3. **Oferta sugerida** por motivo y submotivo, alineada con el plan de accionables (A3 ajustar el plan, A6 reversar
   cobros, A4 oferta por motivo). Es una propuesta: el catálogo y la política reales los define Claro.
4. **Guion** de 2 a 3 frases con un modelo rápido (Haiku 4.5), tratando al cliente de usted, sin prometer nada fuera
   de la oferta sugerida. Sin clave de API, el guion sale de una plantilla.
5. **Alertas**: urgencia alta (resolver sin transferir), reclamo previo (no repetir la misma promesa) y **cliente que
   insiste en cancelar: respetar su decisión y gestionar la baja**.

El copiloto sugiere; decide el asesor. No ejecuta acciones ni ofrece nada por su cuenta.

### Evaluación

- Métodos de clasificación: los del benchmark (motivo 62 %, urgencia 70 %, sentimiento 0,75 contra etiquetas humanas).
- Pruebas automáticas: `tests/test_copiloto.py` (roles, alerta de insistencia, pregunta con motivo incierto,
  ruteo de una llamada pegada al agente de voz del cliente).
- Pregunta dorada con una llamada pegada en `agents/evals.py`.
- Tiempos medidos: análisis con reglas 0,1 s; con Jev ≈2 s; híbrido completo ≈25 s; copiloto con guion ≈3 s.

## Fase 2 · Agente de retención del Cluster 3 (siguiente paso)

| Etapa | Qué hace | Condición para avanzar |
|---|---|---|
| **2a · Copiloto en producción** | Recibe la transcripción en streaming (ASR) y muestra la guía en el escritorio del asesor | Llave de cruce llamada–cliente para sumar el riesgo del modelo; catálogo de ofertas de Claro |
| **2b · Piloto con grupo de control** | Asesores con copiloto vs sin copiloto | Retención, resolución en primer contacto, reincidencia a 30 días, duración de la llamada |
| **2c · Agente de chat para cancelaciones** | Atiende por texto, identifica el motivo y ofrece solo lo que la política permite | Resultados del piloto; guardarraíles aprobados por Legal y Cumplimiento |

**Guardarraíles del agente directo**
- Nunca obstaculiza la cancelación: la regulación de la CRC protege el derecho del usuario a terminar el contrato.
- Solo ofertas del catálogo aprobado; montos por encima de un umbral pasan a un humano.
- Traspaso a un asesor si el cliente lo pide, insiste o el sentimiento cae.
- Datos personales enmascarados antes del LLM; trazas y prompts versionados en MLflow.

**Qué falta para construirlo**
- Llave de cruce entre llamada y cliente (para usar su riesgo y su historial).
- Catálogo de ofertas y política de retención.
- Clientes simulados construidos con las 500 llamadas para evaluar el agente antes de exponerlo.
- Métricas de éxito y grupo de control definidos con Retención.
