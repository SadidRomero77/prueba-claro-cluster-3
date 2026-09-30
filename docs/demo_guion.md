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

### 1b · Intención sin decir "cancelar"

Prueba de que el sistema entiende el sentido y no solo palabras clave. Ninguna de estas llamadas contiene la palabra
"cancelar"; las dos últimas son controles donde el cliente **no** quiere irse (resultados con Jev + LLM):

| Llamada (resumen) | Intención | Motivo |
|---|---|---|
| Otra empresa le ofrece la mitad; pregunta cómo llevarse su número y cuánto paga si termina antes | 95 % | competencia |
| Se muda a otra ciudad: "no creo que necesite más el servicio", ¿dónde entrego los equipos? | 97 % | traslado |
| "Ya no quiero saber nada más de ustedes… que vengan a recoger el módem" | 98 % | falla técnica |
| Control: solo pide una visita técnica por internet lento | 3 % | falla técnica |
| Control: quiere un plan más barato "pero quedarme con ustedes" | 30 % | precio |

Texto para pegar (caso de competencia):

```
Asesor: Buenas tardes, ¿en qué le puedo ayudar?
Cliente: Hola, mire, me llegó una propuesta de otra empresa: internet de 500 megas y televisión por casi la mitad de lo que pago con ustedes.
Asesor: Entiendo, ¿qué necesita?
Cliente: Quiero saber qué necesito para llevarme mi número fijo, y cuánto me toca pagar si termino el contrato antes de tiempo.
Cliente: La verdad ya les di muchas oportunidades.
```

## 2 · Copiloto (chat)

Modo **Copiloto (chat)**. Escribirle al copiloto como a un colega (probado con Sonnet 5):

| # | El asesor escribe | Qué responde el copiloto |
|---|---|---|
| 1 | Oye, tengo un cliente que dice que quiere otro plan porque no usa los datos, que le sobran muchos megas | Intención de cancelar baja (≈34 %): quiere ajustar el plan, no irse. Motivo servicios no usados; sugiere preguntar qué otros servicios tiene activos y ajustar el plan en lugar de descontar |
| 2 | Me dice que sí, que en la casa solo usan internet para trabajar y que la tele casi no la ven. ¿Qué le ofrezco? | Motivo confirmado; ofrecer retirar la TV y los decos y dejar solo internet (A3), con una frase lista para el cliente |
| 3 | Ya le ofrecí dejarle solo internet pero insiste en cancelar | Respuesta fija e inmediata: gestionar la baja sin más ofertas, "Entiendo su decisión y la respeto…" |

Otros arranques: "Tengo un cliente que llama porque el internet se le cae todas las noches" · "La cliente dice que la
factura le subió y no sabe por qué". Cada respuesta trae chips (intención, motivo, urgencia, accionable) y la ficha
completa del caso.

Mensaje para el comité: el asesor conversa como con un colega; el copiloto entiende la intención aunque nadie diga
"cancelar", propone lo que corresponde y **nunca presiona** cuando el cliente ya decidió.
