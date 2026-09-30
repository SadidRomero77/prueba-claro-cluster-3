# Guion de la demo · analizador de llamadas y copiloto del asesor

Llamadas escritas para la demo (no son de clientes reales). App → pestaña **📞 Copiloto**.

## 1 · Analizar una llamada

Modo **Analizar una llamada** → pegar el texto → **Analizar llamada**.

```
Asesor: Buenas tardes, bienvenido al área de cancelaciones, ¿con quién tengo el gusto?
Cliente: Buenas tardes. Llamo porque quiero cancelar el servicio del hogar.
Asesor: Lamento escuchar eso. ¿Me puede contar qué ha pasado?
Cliente: Mire, la factura de este mes me llegó por ciento ochenta mil pesos, y yo venía pagando ciento diez. Nadie me avisó que iba a subir.
Asesor: Entiendo, déjeme revisar su plan.
Cliente: Y además me están cobrando dos decodificadores adicionales que ni siquiera uso, uno está guardado en una caja desde hace meses.
Cliente: Esto ya lo reclamé el mes pasado y me dijeron que lo iban a corregir, y mire, sigue igual.
Asesor: Tiene razón, veo el ajuste de tarifa por fin de la promoción y los dos equipos adicionales.
Cliente: La verdad estoy cansado. Si no me lo arreglan hoy, me voy.
```

Qué debería mostrar (probado con Jev + LLM, unos 10 s):

| Campo | Resultado esperado |
|---|---|
| Intención de cancelar | ≈96 % |
| Motivo · submotivo | precio_facturacion · fin_promocion_descuento |
| Urgencia | alta |
| Sentimiento | empeora (≈ −0,4 → −0,7) |
| Cita | "la factura de este mes me llegó por ciento ochenta mil pesos, y yo venía pagando ciento diez" |
| Oferta | explicar el fin de la promoción y ajustar el plan a lo que usa (A3) |
| Alertas | urgencia alta (resolver sin transferir) · ya había reclamado (no repetir la promesa) |
| Contexto | el motivo es el 24,3 % de las llamadas del Cluster 3 y se retiene al 45,2 % |

La misma llamada se puede pegar en el chat (pestaña **💬 Agente**) con "Analiza esta llamada:" al inicio.

## 2 · Copiloto en vivo

Modo **Copiloto en vivo**. Registrar cada turno con **Quién habla** + **Texto del turno** → **Agregar turno**.
El panel de la derecha se actualiza cuando habla el cliente (≈2–3 s).

| # | Quién habla | Texto | Qué muestra el copiloto |
|---|---|---|---|
| 1 | Asesor | Buenas tardes, área de cancelaciones, ¿en qué le puedo ayudar? | (espera a que hable el cliente) |
| 2 | Cliente | Hola, quiero cancelar el servicio, ya no quiero seguir con ustedes. | Intención ≈99 %, motivo **no claro** → **🔎 Pregúntale al cliente:** "¿Me cuenta qué lo lleva a querer cancelar el servicio?" |
| 3 | Asesor | Lamento escucharlo. ¿Me cuenta qué lo lleva a querer cancelar? | — |
| 4 | Cliente | Es que pago mucho por un paquete de televisión que casi no veo, en la casa solo usamos internet. | Motivo **servicios_no_usados** · oferta: retirar la TV y los decos que no usa (A3) · guion para proponerlo |
| 5 | Asesor | Entiendo. Podemos revisar su plan y dejarle solo lo que usa. | — |
| 6 | Cliente | No, gracias, igual quiero cancelar. Ya tomé la decisión. | ⚠️ **El cliente insiste: respetar su decisión** · guion fijo: "Entiendo su decisión y la respeto. Ya mismo le gestiono la cancelación…" |

Mensaje para el comité: el copiloto identifica el motivo haciendo la pregunta correcta, propone la oferta que
corresponde y **nunca presiona** cuando el cliente ya decidió.

Alternativa sin escribir: **Simular con una llamada real** → elegir una llamada → **Siguiente turno** varias veces.
